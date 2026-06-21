from __future__ import annotations

import numpy as np
import torch

from .config import NGPSConfig

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover - import stays optional for torch-only use.
    cv2 = None


def guide_filter_slice(image: np.ndarray, cfg: NGPSConfig = NGPSConfig()) -> np.ndarray:
    """Build a BF+MF structural guide for one normalized slice.

    NGPS matches patches on this noise-suppressed guide, while the retrieved
    supervision value is still taken from the raw neighboring slice.
    """

    if cv2 is None:
        raise ImportError("OpenCV is required for guide generation. Install opencv-python.")
    image_u8 = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    guide = cv2.bilateralFilter(
        image_u8,
        d=cfg.bilateral_d,
        sigmaColor=cfg.bilateral_sigma_color,
        sigmaSpace=cfg.bilateral_sigma_space,
    )
    guide = cv2.medianBlur(guide, cfg.median_kernel)
    return guide.astype(np.float32) / 255.0


def guide_filter_volume(volume: np.ndarray, cfg: NGPSConfig = NGPSConfig()) -> np.ndarray:
    """Apply BF+MF to a volume shaped ``(D, H, W)``."""

    if volume.ndim != 3:
        raise ValueError("volume must have shape (D, H, W).")
    return np.stack([guide_filter_slice(slice_, cfg) for slice_ in volume], axis=0)


def make_misalignment_mask(
    guide_center: torch.Tensor,
    guide_neighbor: torch.Tensor,
    threshold: float = 0.05,
) -> torch.Tensor:
    """Direction-aware mask where 1 means dynamic/misaligned."""

    return (torch.abs(guide_center - guide_neighbor) > threshold).to(guide_center.dtype)
