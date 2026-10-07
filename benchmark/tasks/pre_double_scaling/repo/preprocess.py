"""Input normalisation."""
from data import PIXEL_MEAN, PIXEL_STD


def _norm(x):
    return (x / 255.0 - PIXEL_MEAN) / PIXEL_STD


def normalize(x_train, x_val):
    """Normalise both splits with the dataset statistics documented in the README."""
    return _norm(x_train), _norm(x_val)
