# Synthetic image classifier (released checkpoint)

A small MLP trained on a synthetic 10-class, 3x8x8 image-like dataset (numpy only). The trained weights
are released as `model.npz` (hidden=64).

## Reproduce

```
python eval.py
```

Documented result: **val_accuracy: 0.881** for the released checkpoint on the validation split.

## Notes

- Inputs are pixel values in [0, 255]. They are normalised with the dataset statistics defined in
  `data.py` (`PIXEL_MEAN`, `PIXEL_STD`).
- Hyperparameters live in `config.json`.
- Tests: `pytest -q`.
