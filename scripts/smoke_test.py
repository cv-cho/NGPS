from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ngps import NGPSConfig, NGPSHybridLoss, NeighborGuidedPatchSampler, build_nafnet


def main() -> None:
    cfg = NGPSConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_nafnet(cfg).to(device)
    sampler = NeighborGuidedPatchSampler(cfg).to(device)
    criterion = NGPSHybridLoss(lambda_rc=cfg.rc_weight)

    raw_center = torch.rand(1, 1, 64, 64, device=device)
    raw_prev = torch.rand(1, 1, 64, 64, device=device)
    raw_next = torch.rand(1, 1, 64, 64, device=device)
    guide_center = raw_center.detach()
    guide_prev = raw_prev.detach()
    guide_next = raw_next.detach()

    targets = sampler(guide_center, guide_prev, guide_next, raw_prev, raw_next)
    out_center = model(raw_center)
    out_prev = model(raw_prev)
    out_next = model(raw_next)
    loss = criterion(
        out_center,
        raw_prev,
        raw_next,
        targets.target_prev,
        targets.target_next,
        targets.mask_prev,
        targets.mask_next,
        out_prev,
        out_next,
    )
    loss.total.backward()
    print({k: float(v.detach().cpu()) for k, v in loss.as_dict().items()})


if __name__ == "__main__":
    main()
