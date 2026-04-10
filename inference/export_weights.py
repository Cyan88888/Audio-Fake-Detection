"""
Export only ``detect_model`` weights (+ arch dict) from a Lightning checkpoint for deployment
(FrameTransformerDetector / SSL frame features).
Usage:
  python inference/export_weights.py --ckpt Exps/TransformerSpoof19_hubert_e30/checkpoints/last.ckpt \\
      --out inference/exports/detector.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to .ckpt")
    parser.add_argument("--config", default=None, help="config.yaml (default: parent/parent of ckpt)")
    parser.add_argument("--out", required=True, help="Output .pt path")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    cfg_path = Path(args.config) if args.config else ckpt_path.parent.parent / "config.yaml"
    cfg = OmegaConf.load(cfg_path)

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    prefix = "detect_model."
    sub = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
    if not sub:
        raise SystemExit(f"No {prefix} weights in checkpoint")

    arch = OmegaConf.to_container(cfg.detect_model, resolve=True)
    arch.pop("_target_", None)

    bundle = {
        "arch": arch,
        "state_dict": sub,
        "meta": {"source_ckpt": str(ckpt_path), "config": str(cfg_path)},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, args.out)
    print(f"Saved {args.out} ({len(sub)} tensors)")


if __name__ == "__main__":
    main()
