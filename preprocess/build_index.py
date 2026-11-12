"""Builds <name>_filenames.pt and <name>_labels.pt from exported images and per-second labels.

    <img_dir>/<recording>.edf_done/0001.png, ...   graph exporter output
    <label_dir>/<recording>.npy                    one label per second

Usage:
    python preprocess/build_index.py --img_dir /data/MrOS_Image --label_dir ./data/MrOS/labels \
        --split splits/MrOS_test.txt --name MrOS_test --out_dir ./data
"""
import argparse
import os
import re

import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--img_dir", required=True)
parser.add_argument("--label_dir", required=True)
parser.add_argument("--split", required=True, help="text file with one recording ID per line")
parser.add_argument("--name", required=True)
parser.add_argument("--out_dir", default="./data")
args = parser.parse_args()

recordings = [l.strip() for l in open(args.split) if l.strip()]
folders = {re.sub(r"\.edf_done$", "", d, flags=re.I): os.path.join(args.img_dir, d) for d in os.listdir(args.img_dir)}

filenames, labels = [], []
for rec in recordings:
    folder = folders.get(rec)
    if folder is None:
        print(f"[skip] no image folder for {rec}")
        continue
    label_path = os.path.join(args.label_dir, rec + ".npy")
    if not os.path.exists(label_path):
        print(f"[skip] no label for {rec}")
        continue
    imgs = sorted(f for f in os.listdir(folder) if f.endswith(".png"))
    lab = np.load(label_path).astype(np.int64).flatten()

    n = min(len(imgs), len(lab) // 30)          # image i covers seconds [30i, 30i+30)
    if n == 0:
        continue
    filenames += [os.path.join(folder, f) for f in imgs[:n]]
    labels.append(torch.from_numpy(lab[: n * 30].reshape(n, 30)))

os.makedirs(args.out_dir, exist_ok=True)
torch.save(filenames, os.path.join(args.out_dir, f"{args.name}_filenames.pt"))
torch.save(torch.cat(labels).float(), os.path.join(args.out_dir, f"{args.name}_labels.pt"))
print(f"{args.name}: {len(filenames)} epochs from {len(labels)} recordings")
