# VOSA

**Per-Second, Interpretable Obstructive Sleep Apnea Detection from Multimodal Time-Series using Vision Transformer**

Official PyTorch implementation of **VOSA**, a two-stage Vision Transformer that classifies every second of a polysomnography (PSG) recording as normal, hypopnea or apnea, and estimates the apnea–hypopnea index (AHI) from these predictions.

This repository contains the training and testing code. Pretrained weights and data splits are on [Google Drive](https://drive.google.com/drive/folders/1DYa0GGwMJV8UbQmwQrRwszO2Ic3QxiKm?usp=sharing). PSG recordings are converted into images with the graph exporter from [SleepXViT](https://github.com/sarahhyojin/SleepXViT) (see [Preprocessing](#preprocessing)).

---

## Framework

```

```

* **Stage 1 (`VOSAStage1`)**: an intra-epoch model trained on single 30-s epochs. For each second, a signal-wise Transformer runs over the 25 signal rows plus a CLS token. A time-wise Transformer (3 layers, 64-d) then runs over the 30 seconds and classifies each second.
* **Stage 2 (`VOSAStage2`)**: an inter-epoch model. The Stage-1 signal-wise encoder is frozen and applied to epochs *t-1*, *t* and *t+1*. A new time-wise Transformer attends over the combined 90 s and predicts the 30 labels of the centre epoch.
* **AHI estimation**: predicted events shorter than 10 s are removed. The number of remaining apnea/hypopnea events is divided by the total sleep time in hours.

---

## Datasets

- **[KISS](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=210)**: Korean standardized image-based PSG database. KISS can only be accessed in a restricted environment through AI Hub. Because it is private data, no KISS preprocessing code is included. To use KISS, prepare it in the [dataset format](#dataset-format) below.
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

| training set | checkpoints | train / valid / test recordings |
|---|---|---|
| KISS | `checkpoints/kiss/stage{1,2}.pt` | 6,203 / 767 / 775 |
| SHHS2 | `checkpoints/shhs/stage{1,2}.pt` | 1,908 / 477 / 266 |
| MrOS (visit 1) | `checkpoints/mros/stage{1,2}.pt` | 2,325 / 291 / 291 |

* Each split file lists one recording ID per line.
* Each checkpoint stores `{"model_state_dict": ...}` with `num_signals=25`. Stage 1 files are about 89 MB and Stage 2 files about 2 MB.

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

| Row(s) | Pixel rows | Signal | SHHS2 | MrOS |
|---|---|---|---|---|
| 1–3 | 0–129 | Position x / y / z | – | – |
| 4–7 | 129–301 | EEG ×4 | EEG(sec), EEG (×2) | C3, C4, A1, A2 |
| 8–9 | 301–387 | EOG L / R | EOG(L), EOG(R) | LOC, ROC |
| 10 | 387–430 | Chin EMG | EMG | L Chin |
| 11 | 430–473 | ECG | ECG | ECG |
| 12 (×3 height) | 473–602 | Nasal pressure flow | AIRFLOW | Cannula Flow |
| 13 | 602–645 | Thermistor | AIRFLOW | Airflow |
| 14–15 | 645–731 | Thoracic / Abdominal effort | THOR RES, ABDO RES | Thoracic, Abdominal |
| 16 | 731–774 | Snore | – | – |
| 17 | 774–817 | Sound (dB) | – | – |
| 18–19 | 817–903 | Leg EMG L / R | – | Leg L, Leg R |
| 20 (×2 height) | 903–989 | SpO2 (85–100 %) | SaO2 | SaO2 |
| 21 (×2 height) | 989–1075 | SpO2 (40–100 %) | SaO2 | SaO2 |

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

Image `k` of a recording (1-indexed) covers seconds `[30(k-1), 30k)`, so its label is a vector of length 30.

For the NSRR datasets, the labels are derived from the Profusion XML annotations with [`preprocess/make_labels.py`](preprocess/make_labels.py). Only events that count toward the AHI are labelled:

* **Apnea (2)**: an obstructive or central apnea scored during sleep.
* **Hypopnea (1)**: a hypopnea scored during sleep that is followed by an SpO2 desaturation of at least 3 % starting within 45 s of its end, or by an arousal starting within 5 s of its end.

The label sequence is shifted so that it lines up with the exported images. The exporter starts epochs on a 30-s boundary of the clock and skips the first `(60 - start_second) % 30` seconds of the EDF, so the same number of seconds is removed from the start of the labels.

### 3. Index files (`./data`)

All scripts read pairs of index files by **name**, for example `--train_set SHHS2_train`:

| file | type | content |
|---|---|---|
| `data/<name>_filenames.pt` | `list[str]`, length N | image paths, grouped by recording and ordered by epoch |
| `data/<name>_labels.pt` | `torch.FloatTensor [N, 30]` | per-second labels of each image |

* The list order matters. Stage 2 takes epochs *t-1* and *t+1* from the neighbouring entries.
* The recording ID is the name of the folder that holds the image, with `.edf_done` removed (for example `mros-visit1-aa0001`).
* If the paths are relative, pass `--img_root`.

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

Preprocessing code is provided for the public NSRR datasets (SHHS2, MrOS). It has three steps: export the images, make the labels, and build the index files. The examples use MrOS. Paths assume the standard NSRR download layout (`<root>/datasets/*.csv` and `<root>/polysomnography/...`).

### Step 1. PSG images

PSG-to-image conversion uses the **graph exporter of [SleepXViT](https://github.com/sarahhyojin/SleepXViT/tree/main/data)**. Please follow its README for EDF conversion and the Docker setup. The exporter configs we used are in [`preprocess/graph_exporter/`](preprocess/graph_exporter):

| config | dataset | EDF source |
|---|---|---|
| `config_shhs2.json` | SHHS visit 2 | `shhs/polysomnography/edfs/shhs2` |
| `config_mros.json` | MrOS visit 1 | `mros/polysomnography/edfs/visit1` |

1. Copy `preprocess/graph_exporter/*.json` into `SleepXViT/data/graph_exporter/config/`. `config_shhs2.json` replaces the original file.
2. Register the MrOS type in `SleepXViT/data/graph_exporter/edf/index.js`:

   ```js
   else if (inputType === 'mros') config = require('../config/config_mros.json');
   ```
3. Run the exporter from `SleepXViT/data/graph_exporter`. Use `TYPE=shhs2` for SHHS.

   ```bash
   docker build -t graph-exporter-convert .
   docker run -it --rm -e TYPE=mros \
       -v /path/to/mros/polysomnography/edfs/visit1:/input \
       -v /path/to/MrOS_Image:/output graph-exporter-convert
   ```

### Step 2. Labels and reference AHI/TST

```bash
python preprocess/make_labels.py --dataset mros --nsrr_root /path/to/mros --out_dir ./data/MrOS
# -> data/MrOS/labels/<recording_id>.npy, data/MrOS/MrOS_ahi.json, data/MrOS/MrOS_tst.json
```

`--dataset` is `shhs2` or `mros`. The EDF files must be present, because the alignment step reads the recording start time from the EDF header.

### Step 3. Index files

```bash
for split in train valid test; do
  python preprocess/build_index.py --img_dir /path/to/MrOS_Image --label_dir ./data/MrOS/labels \
      --split splits/MrOS_$split.txt --name MrOS_$split --out_dir ./data
done
```

---

## Usage

The examples use SHHS2. To use another dataset, change the index names and checkpoint paths. Add `--wandb` to any script to log to Weights & Biases.

### Train

```bash
# Stage 1: single-epoch model (15 epochs, bs 128, lr 1e-4, lineout augmentation)
python train_stage1.py --train_set SHHS2_train --exp_name shhs_stage1

# pick the best Stage-1 epoch on the validation set
python test.py --stage 1 --test_set SHHS2_valid --ckpt runs/shhs_stage1/epoch_*.pt

# Stage 2: multi-epoch head on the frozen Stage-1 encoder (10 epochs, bs 64)
python train_stage2.py --train_set SHHS2_train --exp_name shhs_stage2 \
    --stage1_ckpt runs/shhs_stage1/epoch_5.pt
```

Checkpoints are saved to `runs/<exp_name>/epoch_<k>.pt`.

### Test: per-second event detection

```bash
python test.py --stage 2 --test_set SHHS2_test \
    --stage1_ckpt checkpoints/shhs/stage1.pt --ckpt checkpoints/shhs/stage2.pt
```

This prints per-class precision, recall and F1, macro-F1 and the confusion matrix (Normal / Hypopnea / Apnea).

### Test: AHI estimation and severity

```bash
python test_ahi.py --test_set SHHS2_test \
    --stage1_ckpt checkpoints/shhs/stage1.pt --stage2_ckpt checkpoints/shhs/stage2.pt \
    --ahi_json data/SHHS2/SHHS2_ahi.json --tst_json data/SHHS2/SHHS2_tst.json --out_dir results/shhs
```

This prints the AHI correlation and MAE, the 4-class severity report, and sensitivity and specificity at AHI ≥ 5 / 15 / 30. Per-recording results are saved to `results/shhs/ahi_results.json`. Add `--save_patient_wise` to also save per-second predictions.

For cross-dataset evaluation, pass a model trained on another dataset and, if needed, several index names, for example `--test_set MrOS_train MrOS_valid MrOS_test`.

---

## Repository structure

```
VOSA/
├── model/vosa.py            # VOSAStage1, VOSAStage2, VOSA, load_weights
├── datasets.py              # index loading, EpochDataset (Stage 1), ContextDataset (Stage 2)
├── utils.py                 # metrics, AHI post-processing
├── train_stage1.py
├── train_stage2.py
├── test.py                  # per-second evaluation
├── test_ahi.py              # patient-level AHI / severity evaluation
└── preprocess/
    ├── graph_exporter/      # SleepXViT exporter configs (SHHS2, MrOS)
    ├── make_labels.py       # NSRR XML -> per-second labels, reference AHI / TST
    └── build_index.py       # images + labels -> index files
```

## License

This project is released under the [MIT License](LICENSE). The one exception is `preprocess/make_labels.py`, which is adapted from DRIVEN (© 2024 University of Luxembourg) and remains under the Academic Software License v1.0. See the header of that file.

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
