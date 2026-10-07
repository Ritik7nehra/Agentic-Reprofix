import numpy as np

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
