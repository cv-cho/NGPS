from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import NGPSConfig
from .guide import make_misalignment_mask


@dataclass
class NGPSTargets:
    target_prev: torch.Tensor
    target_next: torch.Tensor
    mask_prev: torch.Tensor
    mask_next: torch.Tensor


class NeighborGuidedPatchSampler(nn.Module):
    """Dense local Top-K retrieval for Neighbor-Guided Patch Sampling.

    For every pixel p in the center slice, the sampler compares the guide patch
    around p to guide patches in a local window of the neighboring slice. The
    Top-K matched coordinates are then used to retrieve raw center-pixel values
    from that neighbor. This keeps structural matching and raw noisy target
    retrieval decoupled.
    """

    def __init__(self, cfg: NGPSConfig = NGPSConfig()) -> None:
        super().__init__()
        self.cfg = cfg
        self.pad = cfg.search_window // 2
        self.num_shifts = cfg.search_window * cfg.search_window
        kernel = torch.ones((self.num_shifts, 1, cfg.patch_size, cfg.patch_size), dtype=torch.float32)
        self.register_buffer("ssd_kernel", kernel, persistent=False)

    @staticmethod
    def _validate_image(x: torch.Tensor, name: str) -> None:
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(f"{name} must have shape (B, 1, H, W).")

    def _shift_stack(self, image: torch.Tensor) -> torch.Tensor:
        """Stack all local-window shifts along the channel dimension."""

        self._validate_image(image, "image")
        _, _, height, width = image.shape
        padded = F.pad(image, (self.pad, self.pad, self.pad, self.pad), mode="reflect")
        shifts = []
        for dy in range(self.cfg.search_window):
            for dx in range(self.cfg.search_window):
                shifts.append(padded[:, :, dy : dy + height, dx : dx + width])
        return torch.cat(shifts, dim=1)

    def _ssd_stack(self, guide_center: torch.Tensor, guide_neighbor: torch.Tensor) -> torch.Tensor:
        shifted_guides = self._shift_stack(guide_neighbor)
        diff_sq = (guide_center - shifted_guides).pow(2)
        kernel = self.ssd_kernel.to(device=diff_sq.device, dtype=diff_sq.dtype)
        return F.conv2d(diff_sq, kernel, padding=self.cfg.patch_size // 2, groups=self.num_shifts)

    @torch.no_grad()
    def forward_direction(
        self,
        guide_center: torch.Tensor,
        guide_neighbor: torch.Tensor,
        raw_neighbor: torch.Tensor,
        return_cost: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Return the NGPS target for one neighbor direction.

        If ``return_cost`` is true, the second return value is the Top-K mean SSD
        cost divided by patch area. This is useful for reliability diagnostics
        or calibrated match-cost gating.
        """

        ssd = self._ssd_stack(guide_center, guide_neighbor)
        top_values, top_indices = torch.topk(ssd, k=self.cfg.topk, dim=1, largest=False)
        shifted_raw = self._shift_stack(raw_neighbor)
        top_raw = torch.gather(shifted_raw, dim=1, index=top_indices)
        target = top_raw.mean(dim=1, keepdim=True).detach()
        if not return_cost:
            return target
        cost = top_values.mean(dim=1, keepdim=True) / float(self.cfg.patch_size**2)
        return target, cost.detach()

    @torch.no_grad()
    def forward(
        self,
        guide_center: torch.Tensor,
        guide_prev: torch.Tensor,
        guide_next: torch.Tensor,
        raw_prev: torch.Tensor,
        raw_next: torch.Tensor,
    ) -> NGPSTargets:
        target_prev = self.forward_direction(guide_center, guide_prev, raw_prev)
        target_next = self.forward_direction(guide_center, guide_next, raw_next)
        mask_prev = make_misalignment_mask(guide_center, guide_prev, self.cfg.mask_threshold)
        mask_next = make_misalignment_mask(guide_center, guide_next, self.cfg.mask_threshold)
        return NGPSTargets(target_prev, target_next, mask_prev.detach(), mask_next.detach())
