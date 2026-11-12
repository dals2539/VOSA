"""Patient-level AHI estimation and OSA severity.

Per recording: predict per-second labels, drop events < 10 s,
AHI = #events / TST (h), then compare with the reference AHI.
"""
import argparse
import json
import math
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from sklearn import metrics
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import ContextDataset, load_index, recording_id
from model import Stage1Encoder, VOSAStage1, VOSAStage2, load_weights
from utils import SEVERITY_NAMES, ahi_to_severity, count_events, remove_short_events

parser = argparse.ArgumentParser(description="VOSA AHI estimation")
parser.add_argument("--test_set", nargs="+", required=True, help="index names, e.g. MrOS_test")
parser.add_argument("--stage1_ckpt", required=True)
parser.add_argument("--stage2_ckpt", required=True)
parser.add_argument("--ahi_json", required=True, help='{"<recording_id>": reference AHI}')
parser.add_argument("--tst_json", required=True, help='{"<recording_id>": total sleep time in minutes}')
parser.add_argument("--data_dir", default="./data")
parser.add_argument("--img_root", default=None)
parser.add_argument("--out_dir", default="./results")
parser.add_argument("--bs", type=int, default=64)
parser.add_argument("--num_signals", type=int, default=25)
parser.add_argument("--n_workers", type=int, default=8)
parser.add_argument("--save_patient_wise", action="store_true", help="save per-second gt/pred per recording")
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
multi_gpu = torch.cuda.device_count() > 1

# ---------------- data ----------------
filenames, labels = load_index(args.test_set, args.data_dir, args.img_root)
records = OrderedDict()
for i, fn in enumerate(filenames):
    records.setdefault(recording_id(fn), []).append(i)
ann_ahi = json.load(open(args.ahi_json))
ann_tst = json.load(open(args.tst_json))
print(f"{len(filenames)} epochs, {len(records)} recordings")

# ---------------- model ----------------
encoder = Stage1Encoder(load_weights(VOSAStage1(num_signals=args.num_signals), args.stage1_ckpt)).eval()
encoder = (nn.DataParallel(encoder) if multi_gpu else encoder).to(device)
head = load_weights(VOSAStage2(num_signals=args.num_signals), args.stage2_ckpt).to(device).eval()

os.makedirs(args.out_dir, exist_ok=True)
if args.save_patient_wise:
    os.makedirs(os.path.join(args.out_dir, "patient"), exist_ok=True)

# ---------------- inference ----------------
results = {}
for rec, idx in tqdm(records.items()):
    if rec not in ann_ahi or rec not in ann_tst:
        continue
    tst, ahi_ref = ann_tst[rec], ann_ahi[rec]
    if tst is None or ahi_ref is None or (isinstance(tst, float) and math.isnan(tst)):
        continue

    ds = ContextDataset([filenames[i] for i in idx], labels[idx], "None")
    loader = DataLoader(ds, batch_size=args.bs, shuffle=False, num_workers=args.n_workers, drop_last=True)

    y_true, y_pred = [], []
    with torch.no_grad():
        for x_prev, x, x_next, y in loader:
            feats = torch.stack([encoder(v.to(device)) for v in (x_prev, x, x_next)], dim=1)
            y_pred.append(head(feats).argmax(-1).cpu().numpy())
            y_true.append(y.numpy())
    if not y_pred:
        continue
    y_true, y_pred = np.concatenate(y_true).flatten(), np.concatenate(y_pred).flatten()

    if args.save_patient_wise:
        np.save(os.path.join(args.out_dir, "patient", f"{rec}_gt.npy"), y_true)
        np.save(os.path.join(args.out_dir, "patient", f"{rec}_pred.npy"), y_pred)

    ahi_pred = count_events(remove_short_events(y_pred)) / (tst / 60)
    results[rec] = {"ahi_pred": ahi_pred, "ahi_ref": float(ahi_ref)}

with open(os.path.join(args.out_dir, "ahi_results.json"), "w") as f:
    json.dump(results, f, indent=2)

# ---------------- metrics ----------------
pred = np.array([r["ahi_pred"] for r in results.values()])
ref = np.array([r["ahi_ref"] for r in results.values()])
sev_pred = [ahi_to_severity(a) for a in pred]
sev_ref = [ahi_to_severity(a) for a in ref]

print(f"\n#recordings: {len(results)}")
print(f"AHI  Pearson r = {stats.pearsonr(pred, ref)[0]:.4f} | MAE ={np.abs(pred - ref).mean():.3f}")
print(metrics.classification_report(sev_ref, sev_pred, labels=[0, 1, 2, 3],
                                    target_names=SEVERITY_NAMES, digits=4, zero_division=0))
print(metrics.confusion_matrix(sev_ref, sev_pred, labels=[0, 1, 2, 3]))
for th in (5, 15, 30):
    p, r = pred >= th, ref >= th
    print(f"AHI>={th:<2d}  sens={np.mean(p[r]):.4f}  spec={np.mean(~p[~r]):.4f}  acc={np.mean(p == r):.4f}")
