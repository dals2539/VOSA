"""Stage 2 training: multi-epoch head on a frozen Stage-1 encoder."""
import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import ContextDataset, load_index
from model import Stage1Encoder, VOSAStage1, VOSAStage2, load_weights
from utils import Logger, set_seed

parser = argparse.ArgumentParser(description="VOSA Stage 2 training")
parser.add_argument("--train_set", nargs="+", required=True)
parser.add_argument("--stage1_ckpt", required=True, help="trained Stage-1 checkpoint (kept frozen)")
parser.add_argument("--data_dir", default="./data")
parser.add_argument("--img_root", default=None)
parser.add_argument("--save_dir", default="./runs")
parser.add_argument("--exp_name", default="shhs_stage2")
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--bs", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--aug", default="lineout", choices=["None", "lineout"])
parser.add_argument("--num_signals", type=int, default=25)
parser.add_argument("--n_workers", type=int, default=8)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--wandb", action="store_true")
args = parser.parse_args()

set_seed(args.seed)
logger = Logger(args.wandb, "VOSA_stage2", vars(args))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ---------------- data ----------------
filenames, labels = load_index(args.train_set, args.data_dir, args.img_root)
print(f"train epochs: {len(filenames)}")
train_loader = DataLoader(ContextDataset(filenames, labels, args.aug), batch_size=args.bs, shuffle=True,
                          num_workers=args.n_workers, drop_last=True)

# ---------------- model ----------------
stage1 = load_weights(VOSAStage1(num_signals=args.num_signals), args.stage1_ckpt)
encoder = Stage1Encoder(stage1).eval()
for p in encoder.parameters():
    p.requires_grad = False
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs for the frozen Stage-1 encoder")
    encoder = nn.DataParallel(encoder)
encoder.to(device)

model = VOSAStage2(num_signals=args.num_signals).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-8)
criterion = nn.CrossEntropyLoss()

ckpt_dir = os.path.join(args.save_dir, args.exp_name)
os.makedirs(ckpt_dir, exist_ok=True)


@torch.no_grad()
def encode(x_prev, x, x_next):
    feats = [encoder(v.to(device)) for v in (x_prev, x, x_next)]
    return torch.stack(feats, dim=1)                    # [B*30, 3, P+1, 768]


# ---------------- train ----------------
model.train()
for epoch in range(args.epochs):
    total_loss = 0
    pbar = tqdm(train_loader, desc=f"epoch {epoch}")
    for it, (x_prev, x, x_next, y) in enumerate(pbar, 1):
        y = y.long().to(device)
        feats = encode(x_prev, x, x_next)
        optimizer.zero_grad()
        out = model(feats)                              # [B, 30, 3]
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
