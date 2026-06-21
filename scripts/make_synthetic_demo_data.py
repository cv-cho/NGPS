from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.synthetic import make_synthetic_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a tiny synthetic volume dataset.")
    parser.add_argument("--output-root", default="data/synthetic")
    parser.add_argument("--train-cases", type=int, default=4)
    parser.add_argument("--val-cases", type=int, default=2)
    parser.add_argument("--depth", type=int, default=24)
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--noise-sigma", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    make_synthetic_dataset(
        args.output_root,
        train_cases=args.train_cases,
        val_cases=args.val_cases,
        depth=args.depth,
        size=args.size,
        noise_sigma=args.noise_sigma,
        seed=args.seed,
    )
    print(f"Synthetic dataset written to {args.output_root}")


if __name__ == "__main__":
    main()
