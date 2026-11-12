"""Per-second evaluation of a Stage-1 or Stage-2 model.
Several --ckpt files (e.g. runs/shhs_stage2/epoch_*.pt) can be compared in one run."""
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import ContextDataset, EpochDataset, load_index
from model import Stage1Encoder, VOSAStage1, VOSAStage2, load_weights
from utils import Logger, segment_metrics

parser = argparse.ArgumentParser(description="VOSA segment-level test")
parser.add_argument("--stage", type=int, choices=[1, 2], default=2)
parser.add_argument("--test_set", nargs="+", required=True, help="index names, e.g. SHHS2_test")
parser.add_argument("--ckpt", nargs="+", required=True, help="Stage-1 ckpt(s) if --stage 1, Stage-2 ckpt(s) if --stage 2")
parser.add_argument("--stage1_ckpt", default=None, help="frozen Stage-1 ckpt (required for --stage 2)")
parser.add_argument("--data_dir", default="./data")
parser.add_argument("--img_root", default=None)
parser.add_argument("--bs", type=int, default=64)
parser.add_argument("--num_signals", type=int, default=25)
parser.add_argument("--n_workers", type=int, default=8)
parser.add_argument("--wandb", action="store_true")
args = parser.parse_args()

logger = Logger(args.wandb, "VOSA_test", vars(args))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
multi_gpu = torch.cuda.device_count() > 1

filenames, labels = load_index(args.test_set, args.data_dir, args.img_root)
DS = EpochDataset if args.stage == 1 else ContextDataset
loader = DataLoader(DS(filenames, labels, "None"), batch_size=args.bs, shuffle=False,
                    num_workers=args.n_workers, drop_last=True)
print(f"test epochs: {len(filenames)}")

if args.stage == 2:
    assert args.stage1_ckpt, "--stage1_ckpt is required for --stage 2"
    encoder = Stage1Encoder(load_weights(VOSAStage1(num_signals=args.num_signals), args.stage1_ckpt)).eval()
    encoder = (nn.DataParallel(encoder) if multi_gpu else encoder).to(device)


@torch.no_grad()
def evaluate(model):
    y_true, y_pred = [], []
    for batch in tqdm(loader):
        y = batch[-1]
        if args.stage == 1:
            out = model(batch[0].to(device))
        else:
            feats = torch.stack([encoder(v.to(device)) for v in batch[:3]], dim=1)
            out = model(feats)
        y_pred.append(out.argmax(-1).cpu().numpy())
        y_true.append(y.numpy())
    return segment_metrics(np.concatenate(y_true), np.concatenate(y_pred))


for ck in args.ckpt:
    model = VOSAStage1(num_signals=args.num_signals) if args.stage == 1 else VOSAStage2(num_signals=args.num_signals)
    model = load_weights(model, ck).eval()
    if multi_gpu and args.stage == 1:
        model = nn.DataParallel(model)
    model.to(device)

    m = evaluate(model)
    print(f"\n===== {ck} =====")
    print(m["report"])
    print(m["confusion"])
    print("macro_f1 {macro_f1:.5f} | weighted_f1 {weighted_f1:.5f} | accuracy {accuracy:.5f} | macro_recall {macro_recall:.5f}".format(**m))
    logger.log({k: v for k, v in m.items() if k not in ("report", "confusion")})

logger.finish()
