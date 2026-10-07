"""A two-layer MLP with dropout (PyTorch)."""
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_dim, hidden, num_classes, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return torch.softmax(self.net(x), dim=1)
