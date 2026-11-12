"""Metrics and AHI post-processing."""
import random

import numpy as np
import torch
from sklearn import metrics

CLASS_NAMES = ["Normal", "Hypopnea", "Apnea"]
SEVERITY_NAMES = ["Normal", "Mild", "Moderate", "Severe"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Logger:
    """Optional wandb logging."""

    def __init__(self, enabled, project, config):
        self.run = None
        if enabled:
            import wandb
            self.run = wandb.init(project=project, config=config)

    def log(self, d):
        if self.run is not None:
            self.run.log(d)

    def finish(self):
        if self.run is not None:
            self.run.finish()


def segment_metrics(y_true, y_pred):
    """Per-second classification metrics."""
    y_true, y_pred = np.asarray(y_true).flatten(), np.asarray(y_pred).flatten()
    return {
        "macro_f1": metrics.f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": metrics.f1_score(y_true, y_pred, average="weighted"),
        "accuracy": metrics.accuracy_score(y_true, y_pred),
        "macro_recall": metrics.recall_score(y_true, y_pred, average="macro"),
        "report": metrics.classification_report(y_true, y_pred, labels=[0, 1, 2],
                                                target_names=CLASS_NAMES, digits=4, zero_division=0),
        "confusion": metrics.confusion_matrix(y_true, y_pred, labels=[0, 1, 2]),
    }


# ---------------------------------------------------------------------------
# AHI post-processing
# ---------------------------------------------------------------------------
def remove_short_events(arr, min_len=10):
    """Sets hypopnea/apnea runs shorter than ``min_len`` s to normal."""
    arr = np.array(arr, copy=True)
    count, start = 0, None
    for i in range(len(arr)):
        if arr[i] in (1, 2):
            if start is None:
                start = i
            count += 1
        else:
            if 0 < count < min_len:
                arr[start:i] = 0
            start, count = None, 0
    return arr


def count_events(arr):
    """Number of hypopnea/apnea runs (a class change starts a new event)."""
    return sum(1 for i in range(len(arr)) if arr[i] in (1, 2) and arr[i - 1] != arr[i])


def ahi_to_severity(ahi):
    """0 Normal (<5), 1 Mild (5-15), 2 Moderate (15-30), 3 Severe (>=30)"""
    return 0 if ahi < 5 else 1 if ahi < 15 else 2 if ahi < 30 else 3
