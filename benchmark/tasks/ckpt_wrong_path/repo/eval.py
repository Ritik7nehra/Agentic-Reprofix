"""Evaluate the released checkpoint (model.npz) on the validation split.

Usage:  python eval.py [--seed N]
"""
import argparse
import json
import os

import numpy as np

from data import make_dataset
from metrics import accuracy
from model import MLP
from preprocess import normalize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    with open("config.json") as f:
        cfg = json.load(f)
    seed = cfg["seed"] if args.seed is None else args.seed

    x_train, y_train, x_val, y_val = make_dataset(seed)
    x_train, x_val = normalize(x_train, x_val)

    model = MLP(x_val.shape[1], cfg["hidden"], cfg["num_classes"], np.random.default_rng(0))
    model.load_state_dict(np.load("checkpoints/model.npz"))

    preds = model.predict(x_val)
    acc = accuracy(preds, y_val)
    os.makedirs("artifacts", exist_ok=True)
    np.savez("artifacts/predictions.npz", preds=preds, labels=y_val)
    print(f"val_accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
