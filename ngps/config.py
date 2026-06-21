from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml


@dataclass(frozen=True)
class NGPSConfig:
    """Hyperparameters used by guide generation, matching, and the backbone."""

    patch_size: int = 7
    search_window: int = 15
    topk: int = 4
    mask_threshold: float = 0.05
    bilateral_d: int = 5
    bilateral_sigma_color: float = 35.0
    bilateral_sigma_space: float = 50.0
    median_kernel: int = 5
    rc_weight: float = 0.5
    nafnet_width: int = 32
    nafnet_enc_blocks: Tuple[int, ...] = (2, 2, 4, 8)
    nafnet_middle_blocks: int = 8
    nafnet_dec_blocks: Tuple[int, ...] = (2, 2, 2, 2)

    def __post_init__(self) -> None:
        if self.patch_size % 2 != 1:
            raise ValueError("patch_size must be odd.")
        if self.search_window % 2 != 1:
            raise ValueError("search_window must be odd.")
        if self.topk < 1 or self.topk > self.search_window * self.search_window:
            raise ValueError("topk must be in [1, search_window ** 2].")
        if self.median_kernel % 2 != 1:
            raise ValueError("median_kernel must be odd.")


@dataclass(frozen=True)
class TrainingConfig:
    """Common training options."""

    epochs: int = 10
    batch_size: int = 4
    crop_size: int = 256
    lr: float = 2e-4
    weight_decay: float = 1e-5
    num_workers: int = 0
    seed: int = 42


def _update_dataclass(cls: type, values: Dict[str, Any]):
    names = {f.name for f in fields(cls)}
    selected = {k: v for k, v in values.items() if k in names}
    if "nafnet_enc_blocks" in selected:
        selected["nafnet_enc_blocks"] = tuple(selected["nafnet_enc_blocks"])
    if "nafnet_dec_blocks" in selected:
        selected["nafnet_dec_blocks"] = tuple(selected["nafnet_dec_blocks"])
    return cls(**selected)


def load_config(path: str | Path) -> tuple[NGPSConfig, TrainingConfig, Dict[str, Any]]:
    """Load a YAML config and return typed method/training configs plus raw dict."""

    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    method = _update_dataclass(NGPSConfig, raw.get("ngps", {}))
    train = _update_dataclass(TrainingConfig, raw.get("training", {}))
    return method, train, raw
