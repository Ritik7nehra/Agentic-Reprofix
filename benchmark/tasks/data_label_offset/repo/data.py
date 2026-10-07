"""Synthetic image-like dataset: 10 classes of 3x8x8 "images" with pixel values in [0, 255]."""
import numpy as np

NUM_CLASSES = 10
IMAGE_SHAPE = (3, 8, 8)

# Dataset statistics of the training split. The README says inputs are normalised with these.
PIXEL_MEAN = 127.2
PIXEL_STD = 67.7


def make_dataset(seed, n_train=1000, n_val=2000):
    proto_rng = np.random.default_rng(1234)  # class prototypes are identical for every seed
    protos = 127.0 + 34.0 * proto_rng.uniform(-1.0, 1.0, size=(NUM_CLASSES, int(np.prod(IMAGE_SHAPE))))
    rng = np.random.default_rng(seed)

    def sample(n):
        y = rng.integers(0, NUM_CLASSES, size=n)
        x = protos[y] + rng.normal(0, 70.0, size=(n, protos.shape[1]))
        return np.clip(x, 0, 255).astype(np.float32), y

    x_train, y_train = sample(n_train)
    y_train = (y_train + 1) % NUM_CLASSES  # class ids in the label file start at 1
    x_val, y_val = sample(n_val)
    return x_train, y_train, x_val, y_val
