"""Train a small PyTorch MLP on a synthetic image-like dataset and report validation accuracy.

Usage:  python train.py [--seed N]
"""
import argparse
import json
import os

import numpy as np
import torch
from torch import nn

from data import make_dataset
from metrics import accuracy
from model import MLP
from preprocess import normalize


def load_config(path="config.json"):
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config()
    seed = cfg["seed"] if args.seed is None else args.seed
    torch.manual_seed(seed)
    torch.set_num_threads(1)  # a model this small is faster on one thread, and runs repeat exactly

    x_train, y_train, x_val, y_val = make_dataset(seed)
    x_train, x_val = normalize(x_train, x_val)
    x_train, x_val = torch.from_numpy(x_train).float(), torch.from_numpy(x_val).float()
    y_train = torch.from_numpy(y_train).long()

    model = MLP(x_train.shape[1], cfg["hidden"], cfg["num_classes"], cfg["dropout"])
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg["lr"])
    loss_fn = nn.CrossEntropyLoss()
    shuffle = torch.Generator().manual_seed(seed)

    for epoch in range(cfg["epochs"]):
        for group in optimizer.param_groups:
            group["lr"] = cfg["lr"] * (0.95 ** epoch)
        model.train()
        order = torch.randperm(len(x_train), generator=shuffle)
        for start in range(0, len(order), cfg["batch_size"]):
            batch = order[start:start + cfg["batch_size"]]
            optimizer.zero_grad()
            loss = loss_fn(model(x_train[batch]), y_train[batch])
            loss.backward()
            optimizer.step()
        print(f"epoch {epoch + 1}/{cfg['epochs']} loss {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        preds = model(x_val).argmax(dim=1).numpy()
    acc = accuracy(preds, y_val)
    os.makedirs("artifacts", exist_ok=True)
    np.savez("artifacts/predictions.npz", preds=preds, labels=y_val)
    torch.save(model.state_dict(), "artifacts/model.pt")
    print(f"val_accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
