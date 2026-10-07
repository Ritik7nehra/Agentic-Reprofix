"""Generate ReproBench: broken ML repositories with reference fixes.

Every task starts from the same small, working project (an MLP on a synthetic image-like dataset),
then injects ONE realistic bug (the demo task injects three). The reference fix is stored as exact
search/replace edits, so `reprofix bench validate` can prove mechanically that
  (a) the broken repo really fails its verification, and
  (b) applying the reference fix really passes it.

Framework: 29 tasks are numpy-only so they run offline in seconds and reproduce anywhere. One task
(`torch_double_softmax`) is a real PyTorch project: it needs `pip install torch` (several GB and
minutes) and is the only task that does. There is no GPU anywhere in the benchmark: the single
"device" task is a CPU-side simulation (device.py) and is labelled that way in its task.json.

Usage:
  python benchmark/build_tasks.py                 add the tasks that are missing; the committed ones are
                                                  verified to be exactly what this file produces and are never
                                                  rewritten (so published results stay valid)
  python benchmark/build_tasks.py --check         only verify the committed tasks, write nothing
  python benchmark/build_tasks.py --measure       run every broken state and print what it does, write nothing
  python benchmark/build_tasks.py --table         print the markdown task table used in docs/benchmark.md
  python benchmark/build_tasks.py --rebuild       wipe benchmark/tasks and regenerate everything from fresh
                                                  calibration (this changes the tasks; results must be re-run)
  --torch-deps DIR   a directory with torch installed (pip install --target DIR torch), needed only to
                     measure or first-generate the PyTorch task
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TASKS = ROOT / "tasks"

# --------------------------------------------------------------------------- good template
DATA_PY = '''"""Synthetic image-like dataset: 10 classes of 3x8x8 "images" with pixel values in [0, 255]."""
import numpy as np

NUM_CLASSES = 10
IMAGE_SHAPE = (3, 8, 8)

# Dataset statistics of the training split. The README says inputs are normalised with these.
PIXEL_MEAN = @MEAN@
PIXEL_STD = @STD@


def make_dataset(seed, n_train=@NTRAIN@, n_val=2000):
    proto_rng = np.random.default_rng(1234)  # class prototypes are identical for every seed
    protos = 127.0 + @SIGNAL@ * proto_rng.uniform(-1.0, 1.0, size=(NUM_CLASSES, int(np.prod(IMAGE_SHAPE))))
    rng = np.random.default_rng(seed)

    def sample(n):
        y = rng.integers(0, NUM_CLASSES, size=n)
        x = protos[y] + rng.normal(0, @NOISE@, size=(n, protos.shape[1]))
        return np.clip(x, 0, 255).astype(np.float32), y

    x_train, y_train = sample(n_train)
    x_val, y_val = sample(n_val)
    return x_train, y_train, x_val, y_val
'''

PREPROCESS_PY = '''"""Input normalisation."""
from data import PIXEL_MEAN, PIXEL_STD


def _norm(x):
    return (x - PIXEL_MEAN) / PIXEL_STD


def normalize(x_train, x_val):
    """Normalise both splits with the dataset statistics documented in the README."""
    return _norm(x_train), _norm(x_val)
'''

MODEL_PY = '''"""A two-layer MLP with manual backprop (numpy only)."""
import numpy as np


class MLP:
    def __init__(self, in_dim, hidden, num_classes, rng):
        self.W1 = rng.normal(0, np.sqrt(2.0 / in_dim), (in_dim, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, np.sqrt(2.0 / hidden), (hidden, num_classes))
        self.b2 = np.zeros(num_classes)

    def forward(self, x):
        self.h = np.maximum(0.0, x @ self.W1 + self.b1)
        return self.h @ self.W2 + self.b2

    def predict(self, x):
        return np.argmax(self.forward(x), axis=1)

    def step(self, x, y, lr):
        logits = self.forward(x)
        shifted = logits - logits.max(axis=1, keepdims=True)
        probs = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
        n = len(y)
        loss = -np.log(probs[np.arange(n), y] + 1e-12).mean()
        dlogits = probs.copy()
        dlogits[np.arange(n), y] -= 1.0
        dlogits /= n
        dW2 = self.h.T @ dlogits
        db2 = dlogits.sum(axis=0)
        dh = dlogits @ self.W2.T
        dh[self.h <= 0] = 0.0
        dW1 = x.T @ dh
        db1 = dh.sum(axis=0)
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        return loss

    def state_dict(self):
        return {"w1": self.W1, "b1": self.b1, "w2": self.W2, "b2": self.b2}

    def load_state_dict(self, state):
        w1, b1, w2, b2 = state["w1"], state["b1"], state["w2"], state["b2"]
        if w1.shape != self.W1.shape or w2.shape != self.W2.shape:
            raise ValueError(f"checkpoint shapes {w1.shape}/{w2.shape} do not match model {self.W1.shape}/{self.W2.shape}")
        self.W1, self.b1, self.W2, self.b2 = w1, b1, w2, b2
'''

METRICS_PY = '''"""Evaluation metrics."""
import numpy as np


def accuracy(preds, labels):
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    return float(np.sum(preds == labels) / len(labels))
'''

TRAIN_PY = '''"""Train a small MLP on a synthetic image-like dataset and report validation accuracy.

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

    eval_x, eval_y = x_val, y_val
    preds = model.predict(eval_x)
    acc = accuracy(preds, eval_y)
    os.makedirs("artifacts", exist_ok=True)
    np.savez("artifacts/predictions.npz", preds=preds, labels=eval_y)
    np.savez("artifacts/model.npz", **model.state_dict())
    print(f"val_accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
'''

EVAL_PY = '''"""Evaluate the released checkpoint (model.npz) on the validation split.

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
    model.load_state_dict(np.load("model.npz"))

    preds = model.predict(x_val)
    acc = accuracy(preds, y_val)
    os.makedirs("artifacts", exist_ok=True)
    np.savez("artifacts/predictions.npz", preds=preds, labels=y_val)
    print(f"val_accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
'''

CONFIG_JSON = '''{
  "seed": 0,
  "epochs": 30,
  "batch_size": 64,
  "lr": 0.1,
  "hidden": 64,
  "num_classes": 10
}
'''

REQUIREMENTS = "numpy>=1.26\npytest>=8\n"

TEST_PY = '''import numpy as np

from data import NUM_CLASSES, make_dataset
from metrics import accuracy
from model import MLP
from preprocess import normalize


def test_dataset_shapes():
    x_tr, y_tr, x_va, y_va = make_dataset(0)
    assert x_tr.shape[1] == 192 and len(x_tr) == len(y_tr) and len(x_va) == len(y_va)


def test_labels_are_learnable():
    # A nearest-class-mean classifier on the validation set must beat chance by a wide margin.
    x_tr, y_tr, x_va, y_va = make_dataset(0)
    means = np.stack([x_tr[y_tr == c].mean(axis=0) for c in range(NUM_CLASSES)])
    pred = np.argmin(((x_va[:, None, :] - means[None]) ** 2).sum(-1), axis=1)
    assert (pred == y_va).mean() > 0.5


def test_normalization_statistics():
    x_tr, _, x_va, _ = make_dataset(0)
    n_tr, n_va = normalize(x_tr, x_va)
    assert abs(n_tr.mean()) < 0.2 and 0.8 < n_tr.std() < 1.2
    assert abs(n_va.mean()) < 0.2 and 0.8 < n_va.std() < 1.2


def test_forward_shape():
    m = MLP(192, 64, 10, np.random.default_rng(0))
    assert m.forward(np.zeros((4, 192))).shape == (4, 10)


def test_accuracy():
    assert accuracy(np.array([1, 2, 3, 4]), np.array([1, 2, 0, 0])) == 0.5
'''

README_TRAIN = '''# Synthetic image classifier

A small MLP trained on a synthetic 10-class, 3x8x8 image-like dataset (numpy only).

## Reproduce

```
python train.py
```

Documented result: **val_accuracy: @ACC@** (validation accuracy after 30 epochs, hidden=64, lr=0.1 with
per-epoch decay of 0.95).

## Notes

- Inputs are pixel values in [0, 255]. They are normalised with the dataset statistics defined in
  `data.py` (`PIXEL_MEAN`, `PIXEL_STD`).
- Hyperparameters live in `config.json`.
- Tests: `pytest -q`.
'''

README_EVAL = '''# Synthetic image classifier (released checkpoint)

A small MLP trained on a synthetic 10-class, 3x8x8 image-like dataset (numpy only). The trained weights
are released as `model.npz` (hidden=64).

## Reproduce

```
python eval.py
```

Documented result: **val_accuracy: @ACC@** for the released checkpoint on the validation split.

## Notes

- Inputs are pixel values in [0, 255]. They are normalised with the dataset statistics defined in
  `data.py` (`PIXEL_MEAN`, `PIXEL_STD`).
- Hyperparameters live in `config.json`.
- Tests: `pytest -q`.
'''

# --------------------------------------------------------------------------- PyTorch variant (one task)
# Same data, preprocessing and metric modules as the numpy tasks; only the model and the training loop are PyTorch.
TORCH_MODEL_PY = '''"""A two-layer MLP with dropout (PyTorch)."""
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_dim, hidden, num_classes, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)
'''

TORCH_TRAIN_PY = '''"""Train a small PyTorch MLP on a synthetic image-like dataset and report validation accuracy.

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
'''

TORCH_CONFIG_JSON = '''{
  "seed": 0,
  "epochs": 30,
  "batch_size": 64,
  "lr": 0.1,
  "hidden": 64,
  "num_classes": 10,
  "dropout": 0.3
}
'''

TORCH_REQUIREMENTS = "numpy>=1.26\ntorch>=2.2\npytest>=8\n"

TORCH_TEST_PY = '''import numpy as np
import torch

from data import make_dataset
from metrics import accuracy
from model import MLP
from preprocess import normalize


def test_dataset_shapes():
    x_tr, y_tr, x_va, y_va = make_dataset(0)
    assert x_tr.shape[1] == 192 and len(x_tr) == len(y_tr) and len(x_va) == len(y_va)


def test_normalization_statistics():
    x_tr, _, x_va, _ = make_dataset(0)
    n_tr, n_va = normalize(x_tr, x_va)
    assert abs(n_tr.mean()) < 0.2 and 0.8 < n_tr.std() < 1.2
    assert abs(n_va.mean()) < 0.2 and 0.8 < n_va.std() < 1.2


def test_forward_shape():
    m = MLP(192, 64, 10, 0.3)
    assert m(torch.zeros(4, 192)).shape == (4, 10)


def test_accuracy():
    assert accuracy(np.array([1, 2, 3, 4]), np.array([1, 2, 0, 0])) == 0.5
'''

README_TORCH = '''# Synthetic image classifier (PyTorch)

A small MLP with dropout, trained with PyTorch on a synthetic 10-class, 3x8x8 image-like dataset.

## Reproduce

```
pip install -r requirements.txt
python train.py
```

Documented result: **val_accuracy: @ACC@** (validation accuracy after 30 epochs, hidden=64, dropout=0.3,
SGD with lr=0.1 and per-epoch decay of 0.95). The run takes a few seconds on a CPU.

## Notes

- Inputs are pixel values in [0, 255]. They are normalised with the dataset statistics defined in
  `data.py` (`PIXEL_MEAN`, `PIXEL_STD`).
- Hyperparameters live in `config.json`.
- Tests: `pytest -q`.
'''


def torch_files(mean: float, std: float, noise: float, acc: float | None, signal: float = 34.0, ntrain: int = 1000) -> dict[str, str]:
    base = good_files("train", mean, std, noise, acc, signal, ntrain)
    return {
        "data.py": base["data.py"], "preprocess.py": base["preprocess.py"], "metrics.py": base["metrics.py"],
        "model.py": TORCH_MODEL_PY, "train.py": TORCH_TRAIN_PY, "config.json": TORCH_CONFIG_JSON,
        "requirements.txt": TORCH_REQUIREMENTS, "tests/test_smoke.py": TORCH_TEST_PY,
        "README.md": README_TORCH.replace("@ACC@", f"{acc:.3f}" if acc is not None else "0.000"),
    }


def good_files(variant: str, mean: float, std: float, noise: float, acc: float | None, signal: float = 34.0, ntrain: int = 1000) -> dict[str, str]:
    if variant == "torch":
        return torch_files(mean, std, noise, acc, signal, ntrain)
    files = {
        "data.py": DATA_PY.replace("@MEAN@", repr(mean)).replace("@STD@", repr(std)).replace("@NOISE@", repr(noise)).replace("@SIGNAL@", repr(signal)).replace("@NTRAIN@", repr(ntrain)),
        "preprocess.py": PREPROCESS_PY, "model.py": MODEL_PY, "metrics.py": METRICS_PY,
        "config.json": CONFIG_JSON, "requirements.txt": REQUIREMENTS, "tests/test_smoke.py": TEST_PY,
        "train.py": TRAIN_PY, "eval.py": EVAL_PY if variant == "eval" else None,
        "README.md": (README_EVAL if variant == "eval" else README_TRAIN).replace("@ACC@", f"{acc:.3f}" if acc is not None else "0.000"),
    }
    if variant == "train":
        files.pop("eval.py")
    else:
        files.pop("train.py")
    return files


# --------------------------------------------------------------------------- bug catalogue
def E(path: str, search: str, replace: str) -> dict:
    return {"path": path, "search": search, "replace": replace}


BUGS: list[dict] = [
    # ---- dependency
    dict(id="dep_conflicting_pins", category="dependency", variant="train", difficulty="easy",
         title="Contradictory numpy pins in requirements.txt",
         inject=[E("requirements.txt", "numpy>=1.26\n", "numpy==2.1.0\nnumpy<2\n")],
         fix=[E("requirements.txt", "numpy==2.1.0\nnumpy<2\n", "numpy>=1.26\n")],
         symptom=dict(kind="install_failure"), truth_files=["requirements.txt"],
         description="requirements.txt pins numpy==2.1.0 and numpy<2 at once, so pip cannot resolve it."),
    dict(id="dep_nonexistent_version", category="dependency", variant="train", difficulty="easy",
         title="Pinned numpy version that does not exist",
         inject=[E("requirements.txt", "numpy>=1.26\n", "numpy==9.9.9\n")],
         fix=[E("requirements.txt", "numpy==9.9.9\n", "numpy>=1.26\n")],
         symptom=dict(kind="install_failure"), truth_files=["requirements.txt"],
         description="numpy==9.9.9 does not exist on the package index."),
    dict(id="dep_missing_requirement", category="dependency", variant="train", difficulty="medium",
         title="train.py imports tqdm but requirements.txt omits it",
         inject=[E("train.py", "import numpy as np\n", "import numpy as np\nfrom tqdm import tqdm\n"),
                 E("train.py", "        for b in iterate_batches(len(x_train), cfg[\"batch_size\"], rng):",
                   "        for b in tqdm(list(iterate_batches(len(x_train), cfg[\"batch_size\"], rng)), disable=True):")],
         fix=[E("requirements.txt", "pytest>=8\n", "pytest>=8\ntqdm>=4.60\n")],
         symptom=dict(kind="crash", pattern="No module named 'tqdm'"), truth_files=["requirements.txt"],
         description="tqdm is imported by train.py but never declared as a requirement."),
    # ---- architecture
    dict(id="arch_classifier_dim", category="architecture", variant="train", difficulty="easy",
         title="Output layer input size does not match the hidden layer",
         inject=[E("model.py", "self.W2 = rng.normal(0, np.sqrt(2.0 / hidden), (hidden, num_classes))",
                   "self.W2 = rng.normal(0, np.sqrt(2.0 / hidden), (hidden // 2, num_classes))")],
         fix=[E("model.py", "(hidden // 2, num_classes)", "(hidden, num_classes)")],
         symptom=dict(kind="crash", pattern="core dimension|shape"), truth_files=["model.py"],
         description="W2 is allocated with hidden//2 input features but receives `hidden` features."),
    dict(id="arch_num_classes", category="architecture", variant="train", difficulty="easy",
         title="num_classes in config smaller than the number of classes in the data",
         inject=[E("config.json", '"num_classes": 10', '"num_classes": 5')],
         fix=[E("config.json", '"num_classes": 5', '"num_classes": 10')],
         symptom=dict(kind="crash", pattern="IndexError|out of bounds"), truth_files=["config.json"],
         description="The output layer has 5 units but labels run 0-9."),
    # ---- data
    dict(id="data_label_shuffle", category="data", variant="train", difficulty="medium",
         title="Training labels shuffled independently of the images",
         inject=[E("data.py", "    x_train, y_train = sample(n_train)\n",
                   "    x_train, y_train = sample(n_train)\n    y_train = rng.permutation(y_train)  # shuffle for randomness\n")],
         fix=[E("data.py", "    y_train = rng.permutation(y_train)  # shuffle for randomness\n", "")],
         symptom=dict(kind="wrong_result"), truth_files=["data.py"],
         description="Permuting y_train alone breaks the image/label pairing, so the model learns noise."),
    dict(id="data_val_labels", category="data", variant="train", difficulty="medium",
         title="Validation labels drawn from a second, unrelated sample",
         inject=[E("data.py", "    x_val, y_val = sample(n_val)\n", "    x_val, _ = sample(n_val)\n    _, y_val = sample(n_val)\n")],
         fix=[E("data.py", "    x_val, _ = sample(n_val)\n    _, y_val = sample(n_val)\n", "    x_val, y_val = sample(n_val)\n")],
         symptom=dict(kind="wrong_result"), truth_files=["data.py"],
         description="sample() is called twice, so validation labels no longer belong to the validation images."),
    # ---- preprocessing
    dict(id="pre_wrong_normalization", category="preprocessing", variant="train", difficulty="medium",
         title="Generic 0.5/0.5 normalisation instead of the dataset statistics",
         inject=[E("preprocess.py", "    return (x - PIXEL_MEAN) / PIXEL_STD", "    return (x - 0.5) / 0.5")],
         fix=[E("preprocess.py", "    return (x - 0.5) / 0.5", "    return (x - PIXEL_MEAN) / PIXEL_STD")],
         symptom=dict(kind="wrong_result"), truth_files=["preprocess.py"],
         description="Pixels in [0,255] are normalised with mean=0.5/std=0.5 rather than PIXEL_MEAN/PIXEL_STD."),
    dict(id="pre_val_not_normalized", category="preprocessing", variant="train", difficulty="medium",
         title="Validation split is not normalised",
         inject=[E("preprocess.py", "    return _norm(x_train), _norm(x_val)", "    return _norm(x_train), x_val")],
         fix=[E("preprocess.py", "    return _norm(x_train), x_val", "    return _norm(x_train), _norm(x_val)")],
         symptom=dict(kind="wrong_result"), truth_files=["preprocess.py"],
         description="Train/validation skew: only the training split goes through normalisation."),
    # ---- training
    dict(id="train_model_reinit", category="training", variant="train", difficulty="medium",
         title="Model re-initialised at the start of every epoch",
         inject=[E("train.py", "    for epoch in range(cfg[\"epochs\"]):\n        lr =",
                   "    for epoch in range(cfg[\"epochs\"]):\n        model = MLP(x_train.shape[1], cfg[\"hidden\"], cfg[\"num_classes\"], rng)\n        lr =")],
         fix=[E("train.py", "        model = MLP(x_train.shape[1], cfg[\"hidden\"], cfg[\"num_classes\"], rng)\n        lr =", "        lr =")],
         symptom=dict(kind="wrong_result"), truth_files=["train.py"],
         description="Constructing the model inside the epoch loop throws away all learning except the final epoch."),
    dict(id="train_update_sign", category="training", variant="train", difficulty="hard",
         title="Gradient ascent on the output weights",
         inject=[E("model.py", "self.W2 -= lr * dW2", "self.W2 += lr * dW2")],
         fix=[E("model.py", "self.W2 += lr * dW2", "self.W2 -= lr * dW2")],
         symptom=dict(kind="wrong_result"), truth_files=["model.py"],
         description="W2 is updated with the wrong sign, so the loss increases for the output layer."),
    # ---- evaluation
    dict(id="eval_on_train_split", category="evaluation", variant="train", difficulty="medium",
         title="'Validation' accuracy computed on the training split",
         inject=[E("train.py", "eval_x, eval_y = x_val, y_val", "eval_x, eval_y = x_train, y_train")],
         fix=[E("train.py", "eval_x, eval_y = x_train, y_train", "eval_x, eval_y = x_val, y_val")],
         symptom=dict(kind="wrong_result"), truth_files=["train.py"],
         description="The reported metric is measured on training data, which inflates it."),
    dict(id="eval_floor_division", category="evaluation", variant="train", difficulty="easy",
         title="Accuracy computed with floor division",
         inject=[E("metrics.py", "np.sum(preds == labels) / len(labels)", "np.sum(preds == labels) // len(labels)")],
         fix=[E("metrics.py", "np.sum(preds == labels) // len(labels)", "np.sum(preds == labels) / len(labels)")],
         symptom=dict(kind="wrong_result"), truth_files=["metrics.py"],
         description="`//` truncates the accuracy to 0 (or 1)."),
    # ---- configuration
    dict(id="cfg_key_mismatch", category="configuration", variant="train", difficulty="easy",
         title="config.json uses 'learning_rate' but train.py reads 'lr'",
         inject=[E("config.json", '"lr": 0.1', '"learning_rate": 0.1')],
         fix=[E("config.json", '"learning_rate": 0.1', '"lr": 0.1')],
         symptom=dict(kind="crash", pattern="KeyError"), truth_files=["config.json"],
         description="The config key was renamed without updating the code that reads it."),
    dict(id="cfg_epochs", category="configuration", variant="train", difficulty="medium",
         title="epochs set to 1 although the README documents 30",
         inject=[E("config.json", '"epochs": 30', '"epochs": 1')],
         fix=[E("config.json", '"epochs": 1', '"epochs": 30')],
         symptom=dict(kind="wrong_result"), truth_files=["config.json"],
         description="The model is under-trained because config.json no longer matches the documented setup."),
    # ---- checkpoint
    dict(id="ckpt_key_names", category="checkpoint", variant="eval", difficulty="medium",
         title="Checkpoint keys are lowercase but the loader asks for uppercase",
         inject=[E("model.py", 'w1, b1, w2, b2 = state["w1"], state["b1"], state["w2"], state["b2"]',
                   'w1, b1, w2, b2 = state["W1"], state["B1"], state["W2"], state["B2"]')],
         fix=[E("model.py", 'state["W1"], state["B1"], state["W2"], state["B2"]', 'state["w1"], state["b1"], state["w2"], state["b2"]')],
         symptom=dict(kind="crash", pattern="KeyError|not a file in the archive"), truth_files=["model.py"],
         description="load_state_dict looks up W1/B1/W2/B2 while the released checkpoint stores w1/b1/w2/b2."),
    dict(id="ckpt_hidden_mismatch", category="checkpoint", variant="eval", difficulty="medium",
         title="config hidden size differs from the released checkpoint",
         inject=[E("config.json", '"hidden": 64', '"hidden": 32')],
         fix=[E("config.json", '"hidden": 32', '"hidden": 64')],
         symptom=dict(kind="crash", pattern="do not match model|shape"), truth_files=["config.json"],
         description="The checkpoint was trained with hidden=64 but config.json says 32."),
    # ---- code
    dict(id="code_none_shuffle", category="code", variant="train", difficulty="easy",
         title="rng.shuffle() result used as an index array",
         inject=[E("train.py", "    idx = rng.permutation(n)\n", "    idx = rng.shuffle(np.arange(n))\n")],
         fix=[E("train.py", "    idx = rng.shuffle(np.arange(n))\n", "    idx = rng.permutation(n)\n")],
         symptom=dict(kind="crash", pattern="NoneType"), truth_files=["train.py"],
         description="Generator.shuffle works in place and returns None."),
    dict(id="code_off_by_one", category="code", variant="train", difficulty="easy",
         title="Off-by-one when printing the last loss",
         inject=[E("train.py", "losses[-1]", "losses[len(losses)]")],
         fix=[E("train.py", "losses[len(losses)]", "losses[-1]")],
         symptom=dict(kind="crash", pattern="IndexError"), truth_files=["train.py"],
         description="losses[len(losses)] is one past the end of the list."),
    # ---- the three-problem demo from the project brief
    dict(id="demo_broken_image_classifier", category="multi", variant="train", difficulty="hard",
         title="Demo: dependency conflict + classifier dimension + wrong normalisation",
         inject=[E("requirements.txt", "numpy>=1.26\n", "numpy==2.1.0\nnumpy<2\n"),
                 E("model.py", "self.W2 = rng.normal(0, np.sqrt(2.0 / hidden), (hidden, num_classes))",
                   "self.W2 = rng.normal(0, np.sqrt(2.0 / hidden), (hidden // 2, num_classes))"),
                 E("preprocess.py", "    return (x - PIXEL_MEAN) / PIXEL_STD", "    return (x - 0.5) / 0.5")],
         fix=[E("requirements.txt", "numpy==2.1.0\nnumpy<2\n", "numpy>=1.26\n"),
              E("model.py", "(hidden // 2, num_classes)", "(hidden, num_classes)"),
              E("preprocess.py", "    return (x - 0.5) / 0.5", "    return (x - PIXEL_MEAN) / PIXEL_STD")],
         symptom=dict(kind="install_failure"), truth_files=["requirements.txt", "model.py", "preprocess.py"],
         stages=[dict(category="dependency", statement="requirements.txt pins numpy==2.1.0 and numpy<2 at once, so installation fails."),
                 dict(category="architecture", statement="The output layer W2 is allocated with hidden//2 inputs, so the classifier dimension does not match the hidden layer."),
                 dict(category="preprocessing", statement="Inputs are normalised with generic mean 0.5/std 0.5 instead of the documented dataset statistics.")],
         description="Three independent faults that surface one after another: install error, crash, then a wrong metric."),
]

# --------------------------------------------------------------------------- tasks 21-30
# Added after the first release. The 20 tasks above are byte-for-byte what they were, so results already
# published against them stay valid. Every symptom below was MEASURED with `--measure` before being kept:
# candidates that did not break this dataset (e.g. removing the 1/batch_size factor, which still scores
# 0.8825 here because a larger effective learning rate happens to help) were dropped, not kept for the count.
DEVICE_PY = '''"""Device selection. Training can only use the devices this machine exposes."""

AVAILABLE_DEVICES = ("cpu",)


def resolve_device(name):
    if name not in AVAILABLE_DEVICES:
        raise RuntimeError(f"device {name!r} was requested but is not available (found: {', '.join(AVAILABLE_DEVICES)})")
    return name
'''

BUGS += [
    dict(id="device_cuda_config", category="device", variant="train", difficulty="easy",
         title="Simulated device mismatch: config asks for an accelerator the machine does not have",
         inject=[E("train.py", "from data import make_dataset\n", "from data import make_dataset\nfrom device import resolve_device\n"),
                 E("train.py", "    cfg = load_config()\n",
                   "    cfg = load_config()\n    device = resolve_device(cfg[\"device\"])\n    print(f\"training on {device}\")\n"),
                 E("config.json", '  "num_classes": 10\n}', '  "num_classes": 10,\n  "device": "cuda"\n}')],
         add_files={"device.py": DEVICE_PY},
         fix=[E("config.json", '"device": "cuda"', '"device": "cpu"')],
         symptom=dict(kind="crash", pattern="not available"), truth_files=["config.json"],
         notes="SIMULATED: device.py stands in for an accelerator check. No GPU, CUDA or PyTorch is involved; "
               "this tests the agent's handling of a config/hardware mismatch, not real GPU behaviour.",
         description="config.json requests device 'cuda' but only 'cpu' exists on the machine running the experiment."),
    dict(id="train_lr_too_high", category="training", variant="train", difficulty="easy",
         title="Learning rate 1000x the documented value",
         inject=[E("config.json", '"lr": 0.1', '"lr": 100.0')],
         fix=[E("config.json", '"lr": 100.0', '"lr": 0.1')],
         symptom=dict(kind="wrong_result"), truth_files=["config.json"],
         description="lr is 100.0 instead of the documented 0.1; the weights overflow to NaN and accuracy collapses."),
    dict(id="train_lr_schedule", category="training", variant="train", difficulty="medium",
         title="Learning-rate decay factor 0.5 instead of the documented 0.95",
         inject=[E("train.py", "(0.95 ** epoch)", "(0.5 ** epoch)")],
         fix=[E("train.py", "(0.5 ** epoch)", "(0.95 ** epoch)")],
         symptom=dict(kind="wrong_result"), truth_files=["train.py"],
         description="The learning rate halves every epoch, so it is effectively zero after a few epochs and the model under-trains."),
    dict(id="train_init_scale", category="training", variant="train", difficulty="medium",
         title="He initialisation formula inverted for the first layer",
         inject=[E("model.py", "np.sqrt(2.0 / in_dim)", "np.sqrt(2.0 * in_dim)")],
         fix=[E("model.py", "np.sqrt(2.0 * in_dim)", "np.sqrt(2.0 / in_dim)")],
         symptom=dict(kind="wrong_result"), truth_files=["model.py"],
         description="W1 is drawn with std sqrt(2*fan_in) instead of sqrt(2/fan_in), about 200x too large."),
    dict(id="eval_inverted_metric", category="evaluation", variant="train", difficulty="easy",
         title="Error rate reported as accuracy",
         inject=[E("metrics.py", "np.sum(preds == labels) / len(labels)", "np.sum(preds != labels) / len(labels)")],
         fix=[E("metrics.py", "np.sum(preds != labels) / len(labels)", "np.sum(preds == labels) / len(labels)")],
         symptom=dict(kind="wrong_result"), truth_files=["metrics.py"],
         description="accuracy() counts mismatches instead of matches, so the printed number is 1 - accuracy."),
    dict(id="data_label_offset", category="data", variant="train", difficulty="medium",
         title="Training labels shifted by one; validation labels are not",
         inject=[E("data.py", "    x_train, y_train = sample(n_train)\n",
                   "    x_train, y_train = sample(n_train)\n    y_train = (y_train + 1) % NUM_CLASSES  # class ids in the label file start at 1\n")],
         fix=[E("data.py", "    y_train = (y_train + 1) % NUM_CLASSES  # class ids in the label file start at 1\n", "")],
         symptom=dict(kind="wrong_result"), truth_files=["data.py"],
         description="Only the training split is shifted to 1-based class ids, so the model learns a permutation of the real classes."),
    dict(id="pre_double_scaling", category="preprocessing", variant="train", difficulty="medium",
         title="Pixels rescaled to [0, 1] and then standardised with [0, 255] statistics",
         inject=[E("preprocess.py", "    return (x - PIXEL_MEAN) / PIXEL_STD", "    return (x / 255.0 - PIXEL_MEAN) / PIXEL_STD")],
         fix=[E("preprocess.py", "    return (x / 255.0 - PIXEL_MEAN) / PIXEL_STD", "    return (x - PIXEL_MEAN) / PIXEL_STD")],
         symptom=dict(kind="wrong_result"), truth_files=["preprocess.py"],
         description="Dividing by 255 before subtracting a mean that is on the 0-255 scale squashes every input to nearly the same value."),
    dict(id="cfg_wrong_dtype", category="configuration", variant="train", difficulty="easy",
         title="Hidden size stored as a string in config.json",
         inject=[E("config.json", '"hidden": 64', '"hidden": "64"')],
         fix=[E("config.json", '"hidden": "64"', '"hidden": 64')],
         symptom=dict(kind="crash", pattern="TypeError"), truth_files=["config.json"],
         description='"hidden" is the string "64", which numpy cannot use as an array dimension.'),
    dict(id="ckpt_wrong_path", category="checkpoint", variant="eval", difficulty="easy",
         title="Evaluation script loads the checkpoint from a directory that does not exist",
         inject=[E("eval.py", 'np.load("model.npz")', 'np.load("checkpoints/model.npz")')],
         fix=[E("eval.py", 'np.load("checkpoints/model.npz")', 'np.load("model.npz")')],
         symptom=dict(kind="crash", pattern="FileNotFoundError|No such file"), truth_files=["eval.py"],
         description="eval.py reads checkpoints/model.npz, but the released checkpoint is model.npz at the repository root."),
    # ---- the one real PyTorch task (needs `pip install torch`: several GB and minutes; see docs/benchmark.md)
    dict(id="torch_double_softmax", category="training", variant="torch", difficulty="medium", timeout_s=1200,
         title="PyTorch: model returns probabilities but the loss expects logits",
         inject=[E("model.py", "from torch import nn\n", "import torch\nfrom torch import nn\n"),
                 E("model.py", "        return self.net(x)\n", "        return torch.softmax(self.net(x), dim=1)\n")],
         fix=[E("model.py", "import torch\nfrom torch import nn\n", "from torch import nn\n"),
              E("model.py", "        return torch.softmax(self.net(x), dim=1)\n", "        return self.net(x)\n")],
         symptom=dict(kind="wrong_result"), truth_files=["model.py"],
         notes="REAL PyTorch on CPU. Needs the PyPI torch wheel (about 5 GB installed on Linux x86-64 with the CUDA runtime bundled, "
               "several minutes to install); it runs on the CPU and needs no GPU. Installing its requirements is the only large download in the benchmark.",
         description="MLP.forward applies softmax, then CrossEntropyLoss applies log-softmax again, so the gradients are tiny and the model barely learns."),
]


def apply_edits(files: dict[str, str], edits: list[dict]) -> dict[str, str]:
    out = dict(files)
    for e in edits:
        text = out[e["path"]]
        if text.count(e["search"]) != 1:
            raise SystemExit(f"edit does not match exactly once: {e['path']!r} {e['search'][:60]!r} (count={text.count(e['search'])})")
        out[e["path"]] = text.replace(e["search"], e["replace"], 1)
    return out


def run_py(cwd: Path, *args: str, timeout: int = 120, env_extra: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    env = {**os.environ, **(env_extra or {})}
    return subprocess.run([sys.executable, *args], cwd=cwd, capture_output=True, text=True, timeout=timeout, env=env)


def write_tree(root: Path, files: dict[str, str | bytes]) -> None:
    for rel, content in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            p.write_bytes(content)
        else:
            p.write_text(content)


def metric_of(proc: subprocess.CompletedProcess) -> float | None:
    m = re.findall(r"val_accuracy:\s*([0-9.]+)", proc.stdout)
    return float(m[-1]) if m else None


@dataclass
class Calibration:
    """Constants that make the numpy template land near the documented accuracy, plus the released checkpoint."""
    mean: float
    std: float
    noise: float
    signal: float
    ntrain: int
    acc: float                 # numpy template accuracy, seed 0 (what the numpy tasks document)
    ckpt: bytes                # model.npz shipped with the checkpoint-evaluation tasks
    torch_acc: float | None = None   # PyTorch template accuracy, seed 0 (None until that task exists or is measured)


def calibrate(noise: float, signal: float, ntrain: int) -> Calibration:
    """Derive the constants from the working numpy template (used by --rebuild and for a first-ever build)."""
    ns: dict = {}
    exec(DATA_PY.replace("@MEAN@", "0.0").replace("@STD@", "1.0").replace("@NOISE@", repr(noise)).replace("@SIGNAL@", repr(signal)).replace("@NTRAIN@", repr(ntrain)), ns)
    x_tr, *_ = ns["make_dataset"](0)
    mean, std = round(float(x_tr.mean()), 1), round(float(x_tr.std()), 1)
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        write_tree(d, good_files("train", mean, std, noise, None, signal, ntrain))
        p = run_py(d, "train.py")
        if p.returncode != 0:
            raise SystemExit("good template failed:\n" + p.stderr)
        acc = metric_of(p)
        assert acc is not None
        ckpt = (d / "artifacts" / "model.npz").read_bytes()
    return Calibration(mean, std, noise, signal, ntrain, acc, ckpt)


def committed_calibration() -> Calibration | None:
    """Read the constants back from the committed tasks, so adding tasks never changes the existing ones."""
    ref = TASKS / "dep_conflicting_pins"           # its bug only touches requirements.txt, so data.py is the template's
    ckpt_file = TASKS / "ckpt_key_names" / "repo" / "model.npz"
    if not ((ref / "repo" / "data.py").is_file() and (ref / "task.json").is_file() and ckpt_file.is_file()):
        return None
    data = (ref / "repo" / "data.py").read_text()

    def grab(pattern: str) -> str:
        m = re.search(pattern, data)
        if not m:
            raise SystemExit(f"cannot read the calibration constant {pattern!r} from {ref / 'repo' / 'data.py'}; use --rebuild")
        return m.group(1)

    spec = json.loads((ref / "task.json").read_text())
    torch_acc = None
    tspec = TASKS / "torch_double_softmax" / "task.json"
    if tspec.is_file():
        torch_acc = json.loads(tspec.read_text())["metric"]["expected"]
    return Calibration(mean=float(grab(r"PIXEL_MEAN = ([0-9.]+)")), std=float(grab(r"PIXEL_STD = ([0-9.]+)")),
                       noise=float(grab(r"rng\.normal\(0, ([0-9.]+),")), signal=float(grab(r"127\.0 \+ ([0-9.]+) \*")),
                       ntrain=int(grab(r"n_train=([0-9]+)")), acc=spec["metric"]["expected"], ckpt=ckpt_file.read_bytes(),
                       torch_acc=torch_acc)


def measure_torch_acc(cal: Calibration, torch_deps: Path | None) -> float:
    """Run the GOOD PyTorch template once to learn the accuracy it documents. Needs torch importable."""
    if torch_deps is None:
        raise SystemExit("the PyTorch task is not built yet: pass --torch-deps DIR (a directory made with "
                         "`pip install --target DIR torch`) so its documented accuracy can be measured, or --only to skip it")
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        write_tree(d, good_files("torch", cal.mean, cal.std, cal.noise, None, cal.signal, cal.ntrain))
        p = run_py(d, "train.py", timeout=600, env_extra={"PYTHONPATH": str(torch_deps)})
        acc = metric_of(p)
        if p.returncode != 0 or acc is None:
            raise SystemExit("the good PyTorch template failed:\n" + p.stderr[-2000:])
    return acc


def acc_for(bug: dict, cal: Calibration) -> float:
    if bug["variant"] == "torch":
        assert cal.torch_acc is not None
        return cal.torch_acc
    return cal.acc


def render(bug: dict, cal: Calibration) -> tuple[dict[str, str | bytes], dict]:
    """The repository files and task.json for one bug, exactly as they are written to disk."""
    acc = acc_for(bug, cal)
    files = good_files(bug["variant"], cal.mean, cal.std, cal.noise, acc, cal.signal, cal.ntrain)
    repo_files: dict[str, str | bytes] = dict(apply_edits(files, bug["inject"]))
    for rel, content in bug.get("add_files", {}).items():
        if rel in repo_files:
            raise SystemExit(f"{bug['id']}: add_files would overwrite {rel}")
        repo_files[rel] = content
    if bug["variant"] == "eval":
        repo_files["model.npz"] = cal.ckpt
    task = {
        "id": bug["id"], "category": bug["category"], "title": bug["title"], "difficulty": bug["difficulty"],
        "goal": ("Reproduce the documented experiment and determine why the validation accuracy does not match "
                 "the documented result. Fix the repository so it reproduces the documented result."),
        "command": "python eval.py" if bug["variant"] == "eval" else "python train.py",
        "metric": {"name": "val_accuracy", "expected": round(acc, 3), "tolerance": 0.03},
        "symptom": bug["symptom"],
        "ground_truth": {"category": bug["category"], "files": bug["truth_files"], "description": bug["description"]},
        "fix": bug["fix"],
        **({"stages": bug["stages"]} if "stages" in bug else {}),
        "hidden": {"seed": 1234, "tolerance": 0.06},
        **({"timeout_s": bug["timeout_s"]} if "timeout_s" in bug else {}),
        **({"notes": bug["notes"]} if "notes" in bug else {}),
    }
    return repo_files, task


def write_task(bug: dict, cal: Calibration) -> None:
    repo_files, task = render(bug, cal)
    out = TASKS / bug["id"]
    write_tree(out / "repo", repo_files)
    (out / "task.json").write_text(json.dumps(task, indent=2) + "\n")


def differences(bug: dict, cal: Calibration) -> list[str]:
    """How the committed task differs from what this generator would write (empty = identical, byte for byte)."""
    repo_files, task = render(bug, cal)
    out = TASKS / bug["id"]
    problems = []
    expected = {f"repo/{k}": (v if isinstance(v, bytes) else v.encode()) for k, v in repo_files.items()}
    expected["task.json"] = (json.dumps(task, indent=2) + "\n").encode()
    actual = {p.relative_to(out).as_posix(): p.read_bytes() for p in out.rglob("*") if p.is_file()}
    for name in sorted(expected.keys() - actual.keys()):
        problems.append(f"missing {name}")
    for name in sorted(actual.keys() - expected.keys()):
        problems.append(f"unexpected extra file {name}")
    for name in sorted(expected.keys() & actual.keys()):
        if expected[name] != actual[name]:
            problems.append(f"{name} differs")
    return problems


def markdown_table() -> str:
    rows = ["| task | category | difficulty | first symptom | what is wrong |", "|---|---|---|---|---|"]
    for d in sorted(p for p in TASKS.iterdir() if (p / "task.json").is_file()):
        t = json.loads((d / "task.json").read_text())
        rows.append(f"| `{t['id']}` | {t['category']} | {t['difficulty']} | {t['symptom']['kind'].replace('_', ' ')} | {t['title']} |")
    return "\n".join(rows) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--measure", action="store_true", help="run every broken state, print the outcome, write nothing")
    ap.add_argument("--check", action="store_true", help="verify the committed tasks are exactly what this file produces")
    ap.add_argument("--rebuild", action="store_true", help="wipe benchmark/tasks and regenerate all of it (changes existing tasks)")
    ap.add_argument("--table", action="store_true", help="print the markdown task table")
    ap.add_argument("--only", help="comma-separated task ids (default: all)")
    ap.add_argument("--torch-deps", type=Path, help="directory with torch installed, for measuring/first-building the PyTorch task")
    ap.add_argument("--noise", type=float, default=70.0)
    ap.add_argument("--signal", type=float, default=34.0)
    ap.add_argument("--ntrain", type=int, default=1000)
    args = ap.parse_args()

    if args.table:
        print(markdown_table(), end="")
        return
    bugs = [b for b in BUGS if not args.only or b["id"] in set(args.only.split(","))]
    if args.only and len(bugs) != len(set(args.only.split(","))):
        raise SystemExit(f"unknown task id in --only; known ids: {', '.join(b['id'] for b in BUGS)}")

    if args.rebuild:
        cal = calibrate(args.noise, args.signal, args.ntrain)
        print(f"calibration: PIXEL_MEAN={cal.mean} PIXEL_STD={cal.std} good val_accuracy={cal.acc:.4f} "
              f"(noise={cal.noise} signal={cal.signal} ntrain={cal.ntrain})")
    else:
        cal = committed_calibration()
        if cal is None:                                  # first-ever build: nothing committed to read back
            cal = calibrate(args.noise, args.signal, args.ntrain)
            print(f"calibration (fresh): PIXEL_MEAN={cal.mean} PIXEL_STD={cal.std} good val_accuracy={cal.acc:.4f}")
        else:
            print(f"calibration (read from the committed tasks): PIXEL_MEAN={cal.mean} PIXEL_STD={cal.std} good val_accuracy={cal.acc:.3f}")
    torch_wanted = any(b["variant"] == "torch" for b in bugs)
    if torch_wanted and cal.torch_acc is None:
        if args.check:
            print("the PyTorch task is not built yet: skipped by --check")
            bugs = [b for b in bugs if b["variant"] != "torch"]
        else:
            cal.torch_acc = round(measure_torch_acc(cal, args.torch_deps), 3)
            print(f"PyTorch template: good val_accuracy={cal.torch_acc:.3f} (measured now)")

    if args.measure:
        env = {"PYTHONPATH": str(args.torch_deps)} if args.torch_deps else None
        rows = []
        for bug in bugs:
            if bug["variant"] == "torch" and env is None:
                rows.append((bug["id"], "-", None, "skipped: pass --torch-deps"))
                continue
            repo_files, _ = render(bug, cal)
            with tempfile.TemporaryDirectory() as td:
                d = Path(td)
                write_tree(d, repo_files)
                p = run_py(d, "eval.py" if bug["variant"] == "eval" else "train.py", timeout=600, env_extra=env if bug["variant"] == "torch" else None)
                rows.append((bug["id"], p.returncode, metric_of(p), (p.stderr.strip().splitlines() or [""])[-1][:90]))
        print(f"{'task':32} {'exit':>4} {'val_acc':>8}  last stderr line")
        for r in rows:
            print(f"{r[0]:32} {r[1]:>4} {('%.4f' % r[2]) if r[2] is not None else '-':>8}  {r[3]}")
        return

    if args.rebuild:
        if args.only:
            raise SystemExit("--rebuild regenerates everything; it cannot be combined with --only")
        if TASKS.exists():
            shutil.rmtree(TASKS)
        TASKS.mkdir(exist_ok=True)
        for bug in bugs:
            write_task(bug, cal)
        print(f"rebuilt {len(bugs)} tasks in {TASKS}")
        return

    TASKS.mkdir(exist_ok=True)
    same, drifted, todo = [], {}, []
    for bug in bugs:                                     # 1. compare every committed task with what would be generated
        if (TASKS / bug["id"]).exists():
            problems = differences(bug, cal)
            if problems:
                drifted[bug["id"]] = problems
            else:
                same.append(bug["id"])
        else:
            todo.append(bug)
    for tid, problems in drifted.items():
        print(f"DIFFERS {tid}: {'; '.join(problems[:4])}")
    if drifted:                                          # 2. never write anything next to a committed task that has drifted
        print(f"{len(same)} committed task(s) identical to this generator; {len(drifted)} differing: nothing was written")
        raise SystemExit(1)
    if args.check:
        print(f"{len(same)} committed task(s) identical to this generator; {len(todo)} not generated yet")
        raise SystemExit(1 if todo else 0)
    for bug in todo:                                     # 3. only then add the missing ones
        write_task(bug, cal)
    print(f"{len(same)} committed task(s) identical to this generator; {len(todo)} added {[b['id'] for b in todo] if todo else ''}")

if __name__ == "__main__":
    main()
