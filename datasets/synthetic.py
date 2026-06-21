from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.ndimage


def _one_volume(seed: int, depth: int, size: int, noise_sigma: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    volume = []
    for z in range(depth):
        shift = np.sin(z / max(depth - 1, 1) * np.pi * 2.0)
        cx = size * (0.50 + 0.08 * shift)
        cy = size * (0.52 + 0.06 * np.cos(z / max(depth - 1, 1) * np.pi))
        ellipse = (((xx - cx) / (0.24 * size)) ** 2 + ((yy - cy) / (0.18 * size)) ** 2) < 1.0
        vessel = np.abs((yy - (0.25 * size + 0.35 * xx + 3.0 * shift))) < 2.0
        ring = np.logical_xor(
            ((xx - 0.35 * size) ** 2 + (yy - 0.35 * size) ** 2) < (0.13 * size) ** 2,
            ((xx - 0.35 * size) ** 2 + (yy - 0.35 * size) ** 2) < (0.09 * size) ** 2,
        )
        clean = 0.08 + 0.45 * ellipse.astype(np.float32) + 0.22 * vessel.astype(np.float32) + 0.18 * ring.astype(np.float32)
        clean = scipy.ndimage.gaussian_filter(clean, sigma=0.7)
        volume.append(clean)
    clean_vol = np.clip(np.stack(volume, axis=0), 0.0, 1.0).astype(np.float32)
    noisy = np.clip(clean_vol + rng.normal(0.0, noise_sigma, clean_vol.shape).astype(np.float32), 0.0, 1.0)
    return noisy.astype(np.float32), clean_vol


def make_synthetic_dataset(
    output_root: str | Path,
    train_cases: int = 4,
    val_cases: int = 2,
    depth: int = 24,
    size: int = 96,
    noise_sigma: float = 0.08,
    seed: int = 42,
) -> None:
    """Create a tiny non-medical volume dataset for smoke tests."""

    root = Path(output_root)
    for split, count in [("train", train_cases), ("val", val_cases)]:
        (root / split / "noisy").mkdir(parents=True, exist_ok=True)
        (root / split / "clean").mkdir(parents=True, exist_ok=True)
        for idx in range(count):
            noisy, clean = _one_volume(seed + idx + (0 if split == "train" else 1000), depth, size, noise_sigma)
            name = f"{split}_case_{idx:03d}.npy"
            np.save(root / split / "noisy" / name, noisy)
            np.save(root / split / "clean" / name, clean)
