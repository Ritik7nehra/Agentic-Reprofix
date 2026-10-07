"""A two-layer MLP with manual backprop (numpy only)."""
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
