from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np


@dataclass(frozen=True)
class VolumeItem:
    case_id: str
    noisy_path: Path
    clean_path: Optional[Path] = None


def list_volume_items(data_root: str | Path, split: str) -> List[VolumeItem]:
    """List volumes from DATA_ROOT/split/noisy/*.npy.

    If DATA_ROOT/split/clean contains a matching file, it is attached for
    evaluation. Clean volumes are never required for NGPS training.
    """

    root = Path(data_root)
    noisy_dir = root / split / "noisy"
    clean_dir = root / split / "clean"
    if not noisy_dir.exists():
        raise FileNotFoundError(f"No noisy directory found: {noisy_dir}")

    items = []
    for noisy_path in sorted(noisy_dir.glob("*.npy")):
        clean_path = clean_dir / noisy_path.name
        items.append(VolumeItem(noisy_path.stem, noisy_path, clean_path if clean_path.exists() else None))
    if not items:
        raise RuntimeError(f"No .npy volumes found in {noisy_dir}")
    return items


def load_volume(path: str | Path) -> np.ndarray:
    """Load a float32 volume shaped (D, H, W), clipped to [0, 1]."""

    volume = np.load(path).astype(np.float32)
    if volume.ndim != 3:
        raise ValueError(f"Expected volume shaped (D, H, W), got {volume.shape}: {path}")
    return np.clip(volume, 0.0, 1.0)
