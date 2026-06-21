from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from ngps.metrics import fsim, gmsd, hfen, psnr, ssim


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate denoised .npy volumes.")
    parser.add_argument("--pred", required=True)
    parser.add_argument("--clean", required=True)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    pred = np.clip(np.load(args.pred).astype(np.float32), 0.0, 1.0)
    clean = np.clip(np.load(args.clean).astype(np.float32), 0.0, 1.0)
    if pred.shape != clean.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape}, clean={clean.shape}")

    rows = []
    for z in range(pred.shape[0]):
        rows.append(
            {
                "slice": z,
                "PSNR": psnr(clean[z], pred[z]),
                "SSIM": ssim(clean[z], pred[z]),
                "FSIM": fsim(clean[z], pred[z]),
                "HFEN": hfen(clean[z], pred[z]),
                "GMSD": gmsd(clean[z], pred[z]),
            }
        )
    summary = {k: float(np.mean([row[k] for row in rows])) for k in ["PSNR", "SSIM", "FSIM", "HFEN", "GMSD"]}
    print(summary)
    if args.output_csv:
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.output_csv).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
