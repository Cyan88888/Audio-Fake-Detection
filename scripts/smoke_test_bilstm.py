#!/usr/bin/env python3
"""Quick smoke test for BiLSTM supplement config (no full training)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conf_dir",
        default="config/ablation_supplement/bilstm.yaml",
        help="Config yaml to smoke-test",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--synthetic-only",
        action="store_true",
        help="Skip dataloader; use fake batch (fast, no HuBERT npy IO)",
    )
    args = parser.parse_args()

    from omegaconf import OmegaConf
    import hydra

    cfg = OmegaConf.load(args.conf_dir)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"[1/5] Config: {args.conf_dir}")
    print(f"      exp.name = {cfg.exp.name}")
    print(f"      detect_model = {cfg.detect_model._target_}")
    print(f"      device = {device}")

    print("[2/5] Instantiate detect_model ...")
    detect_model = hydra.utils.instantiate(cfg.detect_model)
    detect_model = detect_model.to(device)
    detect_model.eval()

    print("[3/5] Forward pass (random tensor) ...")
    x = torch.randn(2, 768, 201, device=device)
    with torch.no_grad():
        logits, feat = detect_model(x)
    print(f"      logits {tuple(logits.shape)}, feature {tuple(feat.shape)}")

    if args.synthetic_only:
        print("[4/5] Synthetic train batch (B=4, C=768, T=201) ...")
        bsz = 4
        feat = torch.randn(bsz, 768, 201, device=device)
        target = torch.randint(0, 2, (bsz,), device=device)
        batch = (None, feat, target, ["smoke.flac"] * bsz)
    else:
        print("[4/5] Instantiate datamodule + one train batch ...")
        datamodule = hydra.utils.instantiate(cfg.datamodule)
        datamodule.num_workers = 0
        datamodule.setup("fit")
        batch = next(iter(datamodule.train_dataloader()))
    print(f"      batch len: {len(batch)}")

    print("[5/5] training_step on real batch ...")
    system = hydra.utils.instantiate(cfg.system, detect_model=detect_model)
    system = system.to(device)
    system.train()
    loss = system.training_step(batch, 0)
    if isinstance(loss, dict):
        print(f"      loss keys: {list(loss.keys())}")
    else:
        print(f"      loss: {float(loss):.6f}")

    print("\n[OK] BiLSTM smoke test passed — safe to run full train.py on this config.")


if __name__ == "__main__":
    main()
