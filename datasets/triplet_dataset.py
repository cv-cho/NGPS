from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from ngps.config import NGPSConfig
from ngps.guide import guide_filter_volume

from .volume_folder import VolumeItem, load_volume


def _center_indices(depth: int, central_fraction: float, margin: int) -> List[int]:
    lo = margin
    hi = depth - margin
    if not (0 < central_fraction <= 1.0):
        raise ValueError("central_fraction must be in (0, 1].")
    if central_fraction < 1.0:
        keep = max(1, int((hi - lo) * central_fraction))
        center = (lo + hi) // 2
        start = max(lo, center - keep // 2)
        end = min(hi, start + keep)
        start = max(lo, end - keep)
        return list(range(start, end))
    return list(range(lo, hi))


def _crop(arrays: Sequence[np.ndarray], crop_size: int) -> List[np.ndarray]:
    arrays = list(arrays)
    if crop_size <= 0:
        return arrays
    height, width = arrays[0].shape
    if height <= crop_size or width <= crop_size:
        return arrays
    top = np.random.randint(0, height - crop_size + 1)
    left = np.random.randint(0, width - crop_size + 1)
    return [arr[top : top + crop_size, left : left + crop_size] for arr in arrays]


class VolumeTripletDataset(Dataset):
    """Adjacent-slice triplets for neighboring-slice self-supervision."""

    def __init__(
        self,
        items: Sequence[VolumeItem],
        cfg: NGPSConfig,
        crop_size: int = 256,
        central_fraction: float = 1.0,
        cache_guides: bool = True,
    ) -> None:
        self.items = list(items)
        self.cfg = cfg
        self.crop_size = crop_size
        self.cache_guides = cache_guides
        self.samples: List[Tuple[int, int]] = []
        self._raw_cache: Dict[Path, np.ndarray] = {}
        self._guide_cache: Dict[Path, np.ndarray] = {}

        for item_idx, item in enumerate(self.items):
            depth = int(np.load(item.noisy_path, mmap_mode="r").shape[0])
            for z in _center_indices(depth, central_fraction=central_fraction, margin=1):
                self.samples.append((item_idx, z))
        if not self.samples:
            raise RuntimeError("No valid slice triplets were found.")

    def __len__(self) -> int:
        return len(self.samples)

    def _raw(self, item: VolumeItem) -> np.ndarray:
        if item.noisy_path not in self._raw_cache:
            self._raw_cache[item.noisy_path] = load_volume(item.noisy_path)
        return self._raw_cache[item.noisy_path]

    def _guide(self, item: VolumeItem, raw: np.ndarray) -> np.ndarray:
        if not self.cache_guides:
            return guide_filter_volume(raw, self.cfg)
        if item.noisy_path not in self._guide_cache:
            self._guide_cache[item.noisy_path] = guide_filter_volume(raw, self.cfg)
        return self._guide_cache[item.noisy_path]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, ...]:
        item_idx, z = self.samples[index]
        item = self.items[item_idx]
        raw = self._raw(item)
        guide = self._guide(item, raw)
        prev_z, next_z = z - 1, z + 1

        arrays = _crop(
            [
                raw[z],
                raw[prev_z],
                raw[next_z],
                guide[z],
                guide[prev_z],
                guide[next_z],
            ],
            self.crop_size,
        )
        return tuple(torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0).float() for arr in arrays)
