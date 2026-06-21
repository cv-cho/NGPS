# Data Format

NGPS expects locally prepared volumes saved as NumPy arrays with shape:

```text
(D, H, W)
```

The values should be `float32` and normalized to `[0, 1]`.

Use this folder structure:

```text
DATA_ROOT/
  train/
    noisy/
      case001.npy
      case002.npy
    clean/        # optional for evaluation
      case001.npy
      case002.npy
  val/
    noisy/
      case101.npy
    clean/
      case101.npy
```

Clean volumes are not used during NGPS training. They are only used by
`scripts/evaluate_volume.py`.

Medical datasets are not included in this repository. Please download and
prepare datasets according to their original license and access terms.
