"""Per-second apnea/hypopnea labels from NSRR Profusion XML annotations.

Modified from DRIVEN (alignment to the exported images, NSRR presets, AHI/TST export):

    DRIVEN is (c) 2024, University of Luxembourg
    DRIVEN is published and distributed under the Academic Software License v1.0 (ASL).
    DRIVEN is distributed in the hope that it will be useful for non-commercial academic
    research, but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the ASL for more details.
    The ASL is published at https://turbogap.fi/wiki/index.php/Academic_Software_Licence.
    You may contact the original licensor at LCSB-innovation@uni.lu.

Labels (0 Normal, 1 Hypopnea, 2 Apnea):
  2  obstructive / central apnea during sleep
  1  hypopnea during sleep followed by a >=3 % desaturation within 45 s or an arousal within 5 s

Outputs in --out_dir: labels/<recording_id>.npy, <name>_ahi.json, <name>_tst.json (TST in minutes)

Usage:
  python preprocess/make_labels.py --dataset mros --nsrr_root /path/to/mros --out_dir ./data/MrOS
"""
import argparse
import json
import math
import os
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from tqdm import tqdm

DS_DISTANCE = 45   # s after hypopnea end, desaturation
AR_DISTANCE = 5    # s after hypopnea end, arousal
MIN_DESAT = 3      # %

# NSRR file layout and the CSV columns for reference AHI / TST
PRESETS = {
    "shhs2": dict(
        csv="datasets/shhs2-dataset-0.20.0.csv",
        rec=lambda r: f"shhs2-{int(r['nsrrid'])}",
        xml="polysomnography/annotations-events-profusion/shhs2/{rec}-profusion.xml",
        edf="polysomnography/edfs/shhs2/{rec}.edf",
        ahi="ahi_a0h3a", tst="slpprdp", keep=lambda r: True),
    "mros": dict(
        csv="datasets/mros-visit1-harmonized-0.6.0.csv",
        rec=lambda r: f"mros-visit1-{str(r['nsrrid']).lower()}",
        xml="polysomnography/annotations-events-profusion/visit1/{rec}-profusion.xml",
        edf="polysomnography/edfs/visit1/{rec}.edf",
        ahi="nsrr_ahi_hp3r_aasm15", tst="nsrr_ttldursp_f1", keep=lambda r: r["visit"] == 1),
}


def round30s_skip(edf_path):
    """Seconds the graph exporter skips so that epochs start on a 30-s clock boundary.
    The start time is read from the EDF header (bytes 176-184, 'hh.mm.ss')."""
    with open(edf_path, "rb") as f:
        start = f.read(256)[176:184].decode("ascii", errors="ignore").strip()
    try:
        sec = int(start.split(".")[2])
    except (IndexError, ValueError):
        sec = 0
    return (60 - sec) % 30


def parse_profusion(xml_path):
    """Per-second sleep stages and scored events (in file order)."""
    root = ET.parse(xml_path).getroot()
    stages = np.repeat([int(s.text) for s in root.findall("SleepStages/SleepStage")], 30)
    events = []
    for ev in root.findall("ScoredEvents/ScoredEvent"):
        name = ev.findtext("Name")
        start, dur = float(ev.findtext("Start")), float(ev.findtext("Duration"))
        stage = stages[int(start)] if int(start) < len(stages) else stages[-1]
        desat = float(ev.findtext("Desaturation") or 0) if name == "SpO2 desaturation" else 0.
        events.append((name, start, dur, stage, desat))
    return stages, events


def select_ahi_events(events):
    """Apneas during sleep, and hypopneas confirmed by a desaturation or an arousal."""
    selected, pending = [], None      # pending hypopnea: (name, start, dur, desat_until, arousal_until)
    for name, start, dur, stage, desat in events:
        if name in ("Obstructive Apnea", "Central Apnea"):
            if stage > 0:
                selected.append((name, start, dur))
        elif name in ("Hypopnea", "Unsure"):
            if stage > 0:
                pending = (name, start, dur, start + dur + DS_DISTANCE, start + dur + AR_DISTANCE)
        elif name == "SpO2 desaturation":
            if desat >= MIN_DESAT and pending and pending[1] <= start <= pending[3]:
                selected.append(pending[:3])
                pending = None
        elif "Arousal" in name:
            if pending and pending[1] <= start <= pending[4]:
                selected.append(pending[:3])
                pending = None
    return selected


def label_seconds(selected, n_sec):
    labels = np.zeros(n_sec, dtype=np.int8)
    for name, start, dur in selected:
        labels[int(start):int(start + dur)] = 2 if "Apnea" in name else 1
    return labels


def make_labels(xml_path, edf_path=None):
    stages, events = parse_profusion(xml_path)
    labels = label_seconds(select_ahi_events(events), len(stages))
    if edf_path is not None and os.path.exists(edf_path):
        labels = labels[round30s_skip(edf_path):]   # align with the first image
    return labels[: len(labels) // 30 * 30]         # only full epochs are exported


def _valid(v):
    try:
        return not math.isnan(float(v))
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(PRESETS))
    parser.add_argument("--nsrr_root", required=True, help="root of the NSRR download (contains datasets/, polysomnography/)")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--name", default=None, help="prefix of the json files (default: SHHS2 / MrOS)")
    args = parser.parse_args()

    p = PRESETS[args.dataset]
    name = args.name or {"shhs2": "SHHS2", "mros": "MrOS"}[args.dataset]
    os.makedirs(os.path.join(args.out_dir, "labels"), exist_ok=True)
    df = pd.read_csv(os.path.join(args.nsrr_root, p["csv"]), low_memory=False, encoding="iso-8859-1")

    ahi, tst = {}, {}
    for _, row in tqdm(df.iterrows(), total=len(df)):
        if not p["keep"](row):
            continue
        rec = p["rec"](row)
        xml_path = os.path.join(args.nsrr_root, p["xml"].format(rec=rec))
        edf_path = os.path.join(args.nsrr_root, p["edf"].format(rec=rec))
        if not os.path.exists(xml_path):
            continue
        try:
            labels = make_labels(xml_path, edf_path)
        except ET.ParseError:
            print(f"[skip] cannot parse {xml_path}")
            continue
        np.save(os.path.join(args.out_dir, "labels", f"{rec}.npy"), labels)
        if _valid(row[p["ahi"]]) and _valid(row[p["tst"]]):
            ahi[rec], tst[rec] = float(row[p["ahi"]]), float(row[p["tst"]])

    json.dump(ahi, open(os.path.join(args.out_dir, f"{name}_ahi.json"), "w"), indent=2)
    json.dump(tst, open(os.path.join(args.out_dir, f"{name}_tst.json"), "w"), indent=2)
    print(f"{name}: {len(ahi)} recordings with reference AHI/TST -> {args.out_dir}")
