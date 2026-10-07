# Synthetic image classifier

A small MLP trained on a synthetic 10-class, 3x8x8 image-like dataset (numpy only).

## Reproduce

```
python train.py
```

Documented result: **val_accuracy: 0.881** (validation accuracy after 30 epochs, hidden=64, lr=0.1 with
per-epoch decay of 0.95).

## Notes

- Inputs are pixel values in [0, 255]. They are normalised with the dataset statistics defined in
  `data.py` (`PIXEL_MEAN`, `PIXEL_STD`).
- Hyperparameters live in `config.json`.
- Tests: `pytest -q`.
