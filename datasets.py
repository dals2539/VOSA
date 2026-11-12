"""Index loading, augmentation and datasets."""
import os
import random
import re

import torch
import torchvision


# ---------------------------------------------------------------------------
# Index files
# ---------------------------------------------------------------------------
def load_index(names, data_dir="./data", img_root=None):
    """Concatenates <name>_filenames.pt (list[str]) and <name>_labels.pt ([N, 30]) for each name."""
    filenames, labels = [], []
    for name in names:
        fns = torch.load(os.path.join(data_dir, f"{name}_filenames.pt"))
        lbs = torch.as_tensor(torch.load(os.path.join(data_dir, f"{name}_labels.pt")))
        assert len(fns) == len(lbs), f"{name}: {len(fns)} filenames vs {len(lbs)} labels"
        filenames += list(fns)
        labels.append(lbs.reshape(-1, 30))
    if img_root is not None:
        filenames = [os.path.join(img_root, fn) for fn in filenames]
    return filenames, torch.cat(labels, 0)


def recording_id(path):
    """Folder name of the image, e.g. /.../mros-visit1-aa0001.edf_done/0001.png -> mros-visit1-aa0001"""
    rec = os.path.basename(os.path.dirname(path))
    return re.sub(r"(\.edf_done|\.edf|_standard)$", "", rec, flags=re.IGNORECASE)


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------
def lineout(image):
    """Blanks one random signal group with p=0.5 (training augmentation)."""
    r = random.random()
    if r > 0.5:
        return image
    elif r < 1 / 8:
        image[:, :130, :] = 0       # rows 1-3:   position
    elif r < 1 / 4:
        image[:, -263:, :] = 0      # rows 18-21: leg EMG, SpO2
    elif r < 3 / 8:
        image[:, :431, :] = 0       # rows 1-10:  position, EEG, EOG, chin EMG
    else:
        image[:, 431:, :] = 0       # rows 11-21: ECG, respiratory, leg EMG, SpO2
    return image


AUGMENTATIONS = {"None": None, "lineout": lineout}


def _read(path, aug):
    img = torchvision.io.read_image(path)
    if aug is not None:
        img = aug(img)
    return img[0].float()           # grayscale -> [1080, 1920]


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
class EpochDataset(torch.utils.data.Dataset):
    """Stage 1: image of epoch t -> labels [30]"""

    def __init__(self, filenames, labels, aug="None"):
        self.filenames, self.labels = filenames, labels
        self.aug = AUGMENTATIONS[aug]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return _read(self.filenames[idx], self.aug), self.labels[idx]


class ContextDataset(torch.utils.data.Dataset):
    """Stage 2: images of epochs t-1, t, t+1 -> labels [30] of epoch t.
    Neighbours are the adjacent entries of the index list."""

    def __init__(self, filenames, labels, aug="None"):
        self.filenames, self.labels = filenames, labels
        self.aug = AUGMENTATIONS[aug]

    def __len__(self):
        return len(self.labels) - 1

    def __getitem__(self, idx):
        prev_img = _read(self.filenames[idx - 1], self.aug)
        img = _read(self.filenames[idx], self.aug)
        next_img = _read(self.filenames[idx + 1], self.aug)
        return prev_img, img, next_img, self.labels[idx]
