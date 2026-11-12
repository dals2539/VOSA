# VOSA

**Per-Second, Interpretable Obstructive Sleep Apnea Detection from Multimodal Time-Series using Vision Transformer**

Official PyTorch implementation of **VOSA**, a two-stage Vision Transformer that classifies every second of a polysomnography (PSG) recording as normal, hypopnea or apnea, and estimates the apnea–hypopnea index (AHI) from these predictions.

This repository contains the training and testing code. Pretrained weights and data splits are on [Google Drive](https://drive.google.com/drive/folders/1DYa0GGwMJV8UbQmwQrRwszO2Ic3QxiKm?usp=sharing). PSG recordings are converted into images with the graph exporter from [SleepXViT](https://github.com/sarahhyojin/SleepXViT) (see [Preprocessing](#preprocessing)).

---

## Framework

* **Stage 1 (`VOSAStage1`)**: an intra-epoch model trained on single 30-s epochs. For each second, a signal-wise Transformer runs over the 25 signal rows plus a CLS token. A time-wise Transformer (3 layers, 64-d) then runs over the 30 seconds and classifies each second.
* **Stage 2 (`VOSAStage2`)**: an inter-epoch model. The Stage-1 signal-wise encoder is frozen and applied to epochs *t-1*, *t* and *t+1*. A new time-wise Transformer attends over the combined 90 s and predicts the 30 labels of the centre epoch.
* **AHI estimation**: predicted events shorter than 10 s are removed. The number of remaining apnea/hypopnea events is divided by the total sleep time in hours.

---

## Datasets

- **[KISS](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=210)**: Korean standardized image-based PSG database. KISS can only be accessed in a restricted environment through AI Hub. To use KISS, prepare it in the [dataset format](#dataset-format) below.
- **[SHHS](https://sleepdata.org/datasets/shhs)** (visit 2) and **[MrOS](https://sleepdata.org/datasets/mros)** (visit 1): available from the [National Sleep Research Resource (NSRR)](https://sleepdata.org/).

## Requirements

```bash
conda create -n vosa python=3.8 -y && conda activate vosa
pip install -r requirements.txt
```

Tested with Python 3.8, PyTorch 2.0.0 and CUDA 11.7 on NVIDIA RTX A6000 GPUs. When more than one GPU is visible, the scripts use `nn.DataParallel`.

---

## Checkpoints and splits

Download the folders from [Google Drive](https://drive.google.com/drive/folders/1DYa0GGwMJV8UbQmwQrRwszO2Ic3QxiKm?usp=sharing) and put them in the repository root:

```
VOSA/
├── checkpoints/
│   ├── kiss/  stage1.pt  stage2.pt
│   ├── shhs/  stage1.pt  stage2.pt
│   └── mros/  stage1.pt  stage2.pt
└── splits/
    ├── KISS_{train,valid,test}.txt
    ├── SHHS2_{train,valid,test}.txt
    └── MrOS_{train,valid,test}.txt
```
* Each checkpoint stores `{"model_state_dict": ...}` with `num_signals=25`. 

Quick inference:

```python
from model import VOSA, load_weights

model = VOSA()
load_weights(model.encoder.stage1, "checkpoints/shhs/stage1.pt")
load_weights(model.head,           "checkpoints/shhs/stage2.pt")
model.eval()

# x_prev, x, x_next: [B, 1080, 1920] float tensors (raw 0-255 pixel values)
logits = model(x_prev, x, x_next)       # [B, 30, 3]
pred = logits.argmax(-1)                # 0 Normal / 1 Hypopnea / 2 Apnea
```

---

## Dataset format

### 1. PSG images

Each 30-s epoch is stored as one grayscale **1920 × 1080 PNG**. The time axis is 64 px per second, and the image is made of **25 signal rows of 43 px** (1075 px used). A channel that is missing from a dataset is left blank.

Images are stored in one folder per recording and numbered in time order:

```
<IMG_ROOT>/mros-visit1-aa0001.edf_done/0001.png, 0002.png, ...
```

### 2. Labels

Labels have **one value per second**:

| value | class |
|---|---|
| `0` | Normal |
| `1` | Hypopnea |
| `2` | Apnea (obstructive / central) |

Image `k` of a recording covers seconds `[30(k-1), 30k)`, so its label is a vector of length 30.

For the NSRR datasets, the labels are derived from the Profusion XML annotations with [`preprocess/make_labels.py`](preprocess/make_labels.py). 
Only events that count toward the AHI are labelled:

* **Apnea (2)**: an obstructive or central apnea scored during sleep.
* **Hypopnea (1)**: a hypopnea scored during sleep that is followed by an SpO2 desaturation of at least 3 % starting within 45 s of its end, or by an arousal starting within 5 s of its end.

### 3. Index files (`./data`)

All scripts read pairs of index files by **name**, for example `--train_set SHHS2_train`:

| file | type | content |
|---|---|---|
| `data/<name>_filenames.pt` | `list[str]`, length N | image paths, grouped by recording and ordered by epoch |
| `data/<name>_labels.pt` | `torch.FloatTensor [N, 30]` | per-second labels of each image |

### 4. Patient-level reference (AHI evaluation only)

| file | content |
|---|---|
| `<NAME>_ahi.json` | `{"<recording_id>": reference AHI, ...}` |
| `<NAME>_tst.json` | `{"<recording_id>": total sleep time in minutes, ...}` |

For the NSRR datasets, `make_labels.py` takes these values from the dataset CSVs:

| dataset | AHI column | TST column |
|---|---|---|
| SHHS2 | `ahi_a0h3a` | `slpprdp` |
| MrOS | `nsrr_ahi_hp3r_aasm15` | `nsrr_ttldursp_f1` |

---

## Preprocessing

### Step 1. PSG images

PSG-to-image conversion uses the **graph exporter of [SleepXViT](https://github.com/sarahhyojin/SleepXViT/tree/main/data)**. 
The exporter configs we used are in [`preprocess/graph_exporter/`](preprocess/graph_exporter):

| config | dataset | EDF source |
|---|---|---|
| `config_shhs2.json` | SHHS visit 2 | `shhs/polysomnography/edfs/shhs2` |
| `config_mros.json` | MrOS visit 1 | `mros/polysomnography/edfs/visit1` |

### Step 2. Labels

```bash
python preprocess/make_labels.py --dataset mros --nsrr_root /path/to/mros --out_dir ./data/MrOS
# -> data/MrOS/labels/<recording_id>.npy, data/MrOS/MrOS_ahi.json, data/MrOS/MrOS_tst.json
```

### Step 3. Index files

```bash
for split in train valid test; do
  python preprocess/build_index.py --img_dir /path/to/MrOS_Image --label_dir ./data/MrOS/labels \
      --split splits/MrOS_$split.txt --name MrOS_$split --out_dir ./data
done
```

---

## Usage

### Train

```bash
# Stage 1: single-epoch model 
python train_stage1.py --train_set SHHS2_train --exp_name shhs_stage1

# pick the best Stage-1 epoch on the validation set
python test.py --stage 1 --test_set SHHS2_valid --ckpt runs/shhs_stage1/epoch_*.pt

# Stage 2: multi-epoch head on the frozen Stage-1 encoder 
python train_stage2.py --train_set SHHS2_train --exp_name shhs_stage2 \
    --stage1_ckpt runs/shhs_stage1/epoch_5.pt
```

### Test: per-second event detection

```bash
python test.py --stage 2 --test_set SHHS2_test \
    --stage1_ckpt checkpoints/shhs/stage1.pt --ckpt checkpoints/shhs/stage2.pt
```

### Test: AHI estimation and severity

```bash
python test_ahi.py --test_set SHHS2_test \
    --stage1_ckpt checkpoints/shhs/stage1.pt --stage2_ckpt checkpoints/shhs/stage2.pt \
    --ahi_json data/SHHS2/SHHS2_ahi.json --tst_json data/SHHS2/SHHS2_tst.json --out_dir results/shhs
```
---

## Repository structure

```
VOSA/
├── model/vosa.py            # VOSA_stage1, VOSA_stage2 
├── datasets.py              # index loading, EpochDataset (Stage 1), ContextDataset (Stage 2)
├── utils.py                 # metrics, AHI post-processing
├── train_stage1.py
├── train_stage2.py
├── test.py                  # per-second evaluation
├── test_ahi.py              # patient-level AHI / severity evaluation
└── preprocess/
    ├── graph_exporter/      # graph exporter configs 
    ├── make_labels.py       # NSRR XML -> per-second labels, reference AHI / TST
    └── build_index.py       # images + labels -> index files
```

## License

This project is released under the [MIT License](LICENSE). 

## Acknowledgement

PSG image generation is based on [SleepXViT](https://github.com/sarahhyojin/SleepXViT). The SHHS and MrOS data were provided by the [National Sleep Research Resource](https://sleepdata.org/). The KISS data were provided through [AI Hub](https://www.aihub.or.kr).

## Citation

```bibtex
@article{vosa,
  title   = {Per-Second, Interpretable Obstructive Sleep Apnea Detection from Multimodal Time-Series using Vision Transformer},
  author  = {TBA},
  journal = {TBA},
  year    = {TBA}
}
```
