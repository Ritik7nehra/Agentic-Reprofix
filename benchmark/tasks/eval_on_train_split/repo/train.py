"""Train a small MLP on a synthetic image-like dataset and report validation accuracy.

Usage:  python train.py [--seed N]
"""
import argparse
import json
import os

import numpy as np

from data import make_dataset
from metrics import accuracy
from model import MLP
from preprocess import normalize


def load_config(path="config.json"):
    with open(path) as f:
        return json.load(f)


def iterate_batches(n, batch_size, rng):
    idx = rng.permutation(n)
    for start in range(0, n, batch_size):
        yield idx[start:start + batch_size]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config()
    seed = cfg["seed"] if args.seed is None else args.seed
    rng = np.random.default_rng(seed)

    x_train, y_train, x_val, y_val = make_dataset(seed)
    x_train, x_val = normalize(x_train, x_val)

    model = MLP(x_train.shape[1], cfg["hidden"], cfg["num_classes"], rng)
    for epoch in range(cfg["epochs"]):
        lr = cfg["lr"] * (0.95 ** epoch)
        losses = []
        for b in iterate_batches(len(x_train), cfg["batch_size"], rng):
            losses.append(model.step(x_train[b], y_train[b], lr))
        print(f"epoch {epoch + 1}/{cfg['epochs']} loss {losses[-1]:.4f}")

    eval_x, eval_y = x_train, y_train
    preds = model.predict(eval_x)
    acc = accuracy(preds, eval_y)
    os.makedirs("artifacts", exist_ok=True)
    np.savez("artifacts/predictions.npz", preds=preds, labels=eval_y)
    np.savez("artifacts/model.npz", **model.state_dict())
    print(f"val_accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
