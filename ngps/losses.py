from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn


@dataclass
class LossOutput:
    total: torch.Tensor
    recon: torch.Tensor
    n2n: torch.Tensor
    dynamic: torch.Tensor
    rc: torch.Tensor
    ic: torch.Tensor

    def as_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "loss": self.total,
            "recon": self.recon,
            "n2n": self.n2n,
            "dynamic": self.dynamic,
            "rc": self.rc,
            "ic": self.ic,
        }


def masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    normalize_by_active_pixels: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    mask = mask.to(device=pred.device, dtype=pred.dtype)
    if mask.shape != pred.shape:
        mask = mask.expand_as(pred)
    sq = (pred - target).pow(2) * mask
    if normalize_by_active_pixels:
        return sq.sum() / mask.sum().clamp_min(eps)
    return sq.mean()


class NGPSHybridLoss(nn.Module):
    """Static N2N + NGPS dynamic target + regional consistency."""

    def __init__(self, lambda_rc: float = 0.5, normalize_by_active_pixels: bool = True) -> None:
        super().__init__()
        self.lambda_rc = lambda_rc
        self.normalize_by_active_pixels = normalize_by_active_pixels

    def _mse(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return masked_mse(pred, target, mask, self.normalize_by_active_pixels)

    def forward(
        self,
        out_center: torch.Tensor,
        raw_prev: torch.Tensor,
        raw_next: torch.Tensor,
        target_prev: torch.Tensor,
        target_next: torch.Tensor,
        mask_prev: torch.Tensor,
        mask_next: torch.Tensor,
        out_prev: Optional[torch.Tensor] = None,
        out_next: Optional[torch.Tensor] = None,
    ) -> LossOutput:
        static_prev = 1.0 - mask_prev
        static_next = 1.0 - mask_next
        n2n = 0.5 * (self._mse(out_center, raw_prev, static_prev) + self._mse(out_center, raw_next, static_next))
        dynamic = 0.5 * (self._mse(out_center, target_prev, mask_prev) + self._mse(out_center, target_next, mask_next))
        recon = n2n + dynamic

        if out_prev is None or out_next is None:
            rc = out_center.new_zeros(())
        else:
            rc = 0.5 * (self._mse(out_center, out_prev, static_prev) + self._mse(out_center, out_next, static_next))
        total = recon + self.lambda_rc * rc
        zero = out_center.new_zeros(())
        return LossOutput(total, recon, n2n, dynamic, rc, zero)


class NSN2NLoss(nn.Module):
    """Same-backbone NS-N2N-style masking baseline.

    The dynamic region is excluded from reconstruction. Regional consistency is
    applied on static pixels; the optional IC term follows the common
    interpolation-consistency form used by masking-based neighboring-slice SSL.
    """

    def __init__(
        self,
        lambda_rc: float = 0.5,
        lambda_ic: float = 1.0,
        normalize_by_active_pixels: bool = True,
    ) -> None:
        super().__init__()
        self.lambda_rc = lambda_rc
        self.lambda_ic = lambda_ic
        self.normalize_by_active_pixels = normalize_by_active_pixels

    def _mse(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return masked_mse(pred, target, mask, self.normalize_by_active_pixels)

    def forward(
        self,
        model: nn.Module,
        raw_center: torch.Tensor,
        raw_prev: torch.Tensor,
        raw_next: torch.Tensor,
        mask_prev: torch.Tensor,
        mask_next: torch.Tensor,
    ) -> LossOutput:
        out_center = model(raw_center)
        out_prev = model(raw_prev)
        out_next = model(raw_next)
        static_prev = 1.0 - mask_prev
        static_next = 1.0 - mask_next

        n2n = 0.5 * (self._mse(out_center, raw_prev, static_prev) + self._mse(out_center, raw_next, static_next))
        rc = 0.5 * (self._mse(out_center, out_prev, static_prev) + self._mse(out_center, out_next, static_next))

        avg_in = 0.5 * (raw_prev + raw_next)
        avg_out = 0.5 * (out_prev.detach() + out_next.detach())
        ic = torch.nn.functional.mse_loss(model(avg_in), avg_out)
        recon = n2n
        total = recon + self.lambda_rc * rc + self.lambda_ic * ic
        zero = out_center.new_zeros(())
        return LossOutput(total, recon, n2n, zero, rc, ic)
