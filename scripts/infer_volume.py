from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from tqdm import tqdm

from ngps import build_nafnet
from ngps.config import NGPSConfig


def _to_bchw(slice_: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(slice_.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Denoise one .npy volume with a trained NGPS/NAFNet checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    state = torch.load(args.checkpoint, map_location=device)
    cfg_state = state.get("ngps_config", {})
    cfg = NGPSConfig(**cfg_state) if isinstance(cfg_state, dict) else cfg_state
    model = build_nafnet(cfg).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    volume = np.clip(np.load(args.input).astype(np.float32), 0.0, 1.0)
    outputs = []
    with torch.no_grad():
        for z in tqdm(range(volume.shape[0]), desc="Infer"):
            pred = model(_to_bchw(volume[z], device)).squeeze().cpu().numpy()
            outputs.append(np.clip(pred.astype(np.float32), 0.0, 1.0))
    output = np.stack(outputs, axis=0)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, output)
    print(f"Saved denoised volume to {args.output}")


if __name__ == "__main__":
    main()
