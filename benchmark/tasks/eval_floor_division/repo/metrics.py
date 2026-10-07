"""Evaluation metrics."""
import numpy as np


def accuracy(preds, labels):
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    return float(np.sum(preds == labels) // len(labels))
