# NGPS: Neighbor-Guided Patch Sampling

PyTorch implementation of **Neighbor-Guided Patch Sampling (NGPS)** for
volumetric self-supervised denoising.

NGPS trains a 2D slice denoiser with neighboring slices while accounting for
through-plane misalignment. During training, it builds pseudo-targets by:

1. generating an edge-preserving BF+MF guide,
2. detecting direction-aware static/dynamic regions,
3. matching local guide patches across adjacent slices,
4. retrieving Top-K raw neighboring pixels as dynamic-region targets, and
5. optimizing a hybrid reconstruction and regional-consistency objective.

At inference time, the trained model is a standard denoiser forward pass. There
is no guide generation or patch search during deployment.

## Repository Contents

```text
configs/                  YAML configs for NGPS, NS-N2N-style baseline, and demo
datasets/                 Local .npy volume loaders and synthetic demo generator
docs/                     Data format and method-code mapping
examples/                 One-command demo scripts
ngps/                     NGPS modules, NAFNet backbone, losses, metrics
scripts/                  Train, infer, evaluate, and demo-data scripts
```

## Installation

```bash
git clone https://github.com/<your-org>/<your-repo>.git
cd <your-repo>
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

For CUDA-enabled training, install the PyTorch build matching your CUDA version
from the official PyTorch instructions.

## Quick Synthetic Demo

The demo creates small synthetic volumes with simple moving structures and
Gaussian noise. It does not require medical data.

```bash
python scripts/make_synthetic_demo_data.py --output-root data/synthetic
python scripts/train_ngps.py \
  --data-root data/synthetic \
  --config configs/synthetic_demo.yaml \
  --output-dir runs/synthetic_ngps \
  --device cpu
python scripts/infer_volume.py \
  --checkpoint runs/synthetic_ngps/model_final.pth \
  --input data/synthetic/val/noisy/val_case_000.npy \
  --output outputs/synthetic_val_case_000_ngps.npy \
  --device cpu
python scripts/evaluate_volume.py \
  --pred outputs/synthetic_val_case_000_ngps.npy \
  --clean data/synthetic/val/clean/val_case_000.npy
```

PowerShell users can run:

```powershell
.\examples\train_synthetic_demo.ps1
```

## Data Preparation

Prepare volumes as `.npy` arrays with shape `(D, H, W)`, dtype `float32`, and
values normalized to `[0, 1]`.

```text
DATA_ROOT/
  train/
    noisy/
      case001.npy
      case002.npy
    clean/        # optional, used only for evaluation
      case001.npy
      case002.npy
  val/
    noisy/
      case101.npy
    clean/
      case101.npy
```

Clean references are not used by `scripts/train_ngps.py`. They are only needed
when running `scripts/evaluate_volume.py`.

Medical datasets and derived volumes are not redistributed here. Please obtain
AAPM-Mayo, LIDC-IDRI, IXI, or other datasets from their official sources and
follow their license terms.

## Training NGPS

```bash
python scripts/train_ngps.py \
  --data-root /path/to/DATA_ROOT \
  --config configs/ngps_nafnet_default.yaml \
  --output-dir runs/ngps \
  --device cuda
```

The default config follows the paper setting:

```text
patch size      7 x 7
search window   15 x 15
Top-K           4
mask threshold  0.05
RC weight       0.5
optimizer       AdamW, lr=2e-4, weight_decay=1e-5
backbone        NAFNet width 32
```

## Same-Backbone NS-N2N-Style Baseline

For objective-level comparisons with the same denoiser backbone:

```bash
python scripts/train_nsn2n.py \
  --data-root /path/to/DATA_ROOT \
  --config configs/nsn2n_nafnet_default.yaml \
  --output-dir runs/nsn2n \
  --device cuda
```

## Inference

```bash
python scripts/infer_volume.py \
  --checkpoint runs/ngps/model_final.pth \
  --input /path/to/noisy_volume.npy \
  --output outputs/denoised_volume.npy \
  --device cuda
```

Inference uses only the trained NAFNet denoiser.

## Evaluation

```bash
python scripts/evaluate_volume.py \
  --pred outputs/denoised_volume.npy \
  --clean /path/to/clean_volume.npy \
  --output-csv outputs/metrics.csv
```

The evaluation script reports PSNR, SSIM, FSIM, HFEN, and GMSD over slices and
optionally writes per-slice metrics to CSV.

## Method-Code Mapping

See [docs/METHOD_MAPPING.md](docs/METHOD_MAPPING.md).

## Notes

- Store data, checkpoints, and generated outputs outside Git tracking.
- Keep split manifests and preprocessing details with your experiment logs.
- The synthetic demo is for installation and pipeline checks, not for benchmark
  claims.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{ngps2026,
  title     = {NGPS: Structure-Preserving Self-Supervised Denoising via Neighbor-Guided Patch Sampling},
  author    = {Anonymous / update after camera-ready},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

Please update the BibTeX entry with the final camera-ready author and venue
metadata.
