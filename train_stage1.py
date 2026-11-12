"""Stage 1 training: single-epoch model."""
import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import EpochDataset, load_index
from model import VOSAStage1, load_weights
from utils import Logger, set_seed

parser = argparse.ArgumentParser(description="VOSA Stage 1 training")
parser.add_argument("--train_set", nargs="+", required=True, help="index names, e.g. SHHS2_train")
parser.add_argument("--data_dir", default="./data")
parser.add_argument("--img_root", default=None, help="prefix for relative image paths")
parser.add_argument("--save_dir", default="./runs")
parser.add_argument("--exp_name", default="shhs_stage1")
parser.add_argument("--epochs", type=int, default=15)
parser.add_argument("--bs", type=int, default=128)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--aug", default="lineout", choices=["None", "lineout"])
parser.add_argument("--num_signals", type=int, default=25)
parser.add_argument("--n_workers", type=int, default=8)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--resume", default=None, help="checkpoint to resume training from")
parser.add_argument("--wandb", action="store_true")
args = parser.parse_args()

set_seed(args.seed)
logger = Logger(args.wandb, "VOSA_stage1", vars(args))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ---------------- data ----------------
filenames, labels = load_index(args.train_set, args.data_dir, args.img_root)
print(f"train epochs: {len(filenames)}")
train_loader = DataLoader(EpochDataset(filenames, labels, args.aug), batch_size=args.bs, shuffle=True,
                          num_workers=args.n_workers, drop_last=True)

# ---------------- model ----------------
model = VOSAStage1(num_signals=args.num_signals)
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs")
    model = nn.DataParallel(model)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-8)
criterion = nn.CrossEntropyLoss()

start_epoch = 0
if args.resume:
    ckpt = torch.load(args.resume, map_location=device)
    load_weights(model, args.resume, device)
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    start_epoch = ckpt["epoch"] + 1

ckpt_dir = os.path.join(args.save_dir, args.exp_name)
os.makedirs(ckpt_dir, exist_ok=True)

# ---------------- train ----------------
model.train()
for epoch in range(start_epoch, args.epochs):
    total_loss = 0
    pbar = tqdm(train_loader, desc=f"epoch {epoch}")
    for it, (x, y) in enumerate(pbar, 1):
        x, y = x.to(device), y.long().to(device)
        optimizer.zero_grad()
        out = model(x)                                  # [B, 30, 3]
        loss = criterion(out.reshape(-1, 3), y.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        logger.log({"loss": loss.item()})
        pbar.set_postfix(loss=total_loss / it)

    logger.log({"epoch_loss": total_loss / len(train_loader)})
    torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()},
               os.path.join(ckpt_dir, f"epoch_{epoch}.pt"))

logger.finish()
