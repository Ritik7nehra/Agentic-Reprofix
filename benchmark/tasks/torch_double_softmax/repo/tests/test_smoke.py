import numpy as np
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
