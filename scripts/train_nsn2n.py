from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.triplet_dataset import VolumeTripletDataset
from datasets.volume_folder import list_volume_items
from ngps import NSN2NLoss, build_nafnet, load_config, make_misalignment_mask
from ngps.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a same-backbone NS-N2N-style baseline.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--config", default="configs/nsn2n_nafnet_default.yaml")
    parser.add_argument("--output-dir", default="runs/nsn2n")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--central-fraction", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=5)
    args = parser.parse_args()

    cfg, train_cfg, raw_cfg = load_config(args.config)
    seed_everything(train_cfg.seed)
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    items = list_volume_items(args.data_root, "train")
    dataset = VolumeTripletDataset(items, cfg, crop_size=train_cfg.crop_size, central_fraction=args.central_fraction)
    loader = DataLoader(dataset, batch_size=train_cfg.batch_size, shuffle=True, num_workers=train_cfg.num_workers)

    model = build_nafnet(cfg).to(device)
    criterion = NSN2NLoss(lambda_rc=cfg.rc_weight, lambda_ic=float(raw_cfg.get("nsn2n", {}).get("lambda_ic", 1.0)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    for epoch in range(1, train_cfg.epochs + 1):
        model.train()
        losses = []
        loop = tqdm(loader, desc=f"NS-N2N epoch {epoch}/{train_cfg.epochs}")
        for raw_center, raw_prev, raw_next, guide_center, guide_prev, guide_next in loop:
            raw_center = raw_center.to(device)
            raw_prev = raw_prev.to(device)
            raw_next = raw_next.to(device)
            guide_center = guide_center.to(device)
            guide_prev = guide_prev.to(device)
            guide_next = guide_next.to(device)
            mask_prev = make_misalignment_mask(guide_center, guide_prev, cfg.mask_threshold)
            mask_next = make_misalignment_mask(guide_center, guide_next, cfg.mask_threshold)

            loss_out = criterion(model, raw_center, raw_prev, raw_next, mask_prev, mask_next)
            optimizer.zero_grad(set_to_none=True)
            loss_out.total.backward()
            optimizer.step()
            losses.append(float(loss_out.total.detach().cpu()))
            loop.set_postfix(loss=f"{sum(losses) / len(losses):.4f}")

        if epoch % args.save_every == 0 or epoch == train_cfg.epochs:
            torch.save(
                {"model": model.state_dict(), "ngps_config": asdict(cfg), "training_config": asdict(train_cfg)},
                out_dir / f"model_E{epoch}.pth",
            )
    torch.save({"model": model.state_dict(), "ngps_config": asdict(cfg), "training_config": asdict(train_cfg)}, out_dir / "model_final.pth")
    print(f"Saved checkpoint to {out_dir / 'model_final.pth'}")


if __name__ == "__main__":
    main()
