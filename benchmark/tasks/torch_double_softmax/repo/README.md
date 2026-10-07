# Synthetic image classifier (PyTorch)

A small MLP with dropout, trained with PyTorch on a synthetic 10-class, 3x8x8 image-like dataset.

## Reproduce

```
pip install -r requirements.txt
python train.py
```

Documented result: **val_accuracy: 0.939** (validation accuracy after 30 epochs, hidden=64, dropout=0.3,
SGD with lr=0.1 and per-epoch decay of 0.95). The run takes a few seconds on a CPU.

## Notes

- Inputs are pixel values in [0, 255]. They are normalised with the dataset statistics defined in
  `data.py` (`PIXEL_MEAN`, `PIXEL_STD`).
- Hyperparameters live in `config.json`.
- Tests: `pytest -q`.
