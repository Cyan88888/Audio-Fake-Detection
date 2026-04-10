"""
CLI: predict spoof/bonafide for one audio file using frame features + Transformer detector.

Run from repo root:
  python -m inference.predict --audio path.wav --ckpt path.ckpt
  python -m inference.predict --audio path.wav --ckpt path.ckpt --feat hubert
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from inference.featurizer_factory import create_featurizer
from inference.load_model import load_detector_auto


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, help="Path to .wav / .flac")
    parser.add_argument(
        "--ckpt",
        default=None,
        help="Lightning .ckpt or exported .pt bundle (from inference/export_weights.py)",
    )
    parser.add_argument(
        "--feat",
        choices=("wavlm", "hubert"),
        default=os.environ.get("SAFEAR_FEAT", "wavlm"),
        help="Frame backend: WavLM (default) or fairseq HuBERT",
    )
    parser.add_argument("--hubert_ckpt", default=str(_ROOT / "model_zoos" / "hubert_base_ls960.pt"))
    parser.add_argument("--wavlm_model", default="microsoft/wavlm-base")
    parser.add_argument("--max_len", type=int, default=64600)
    parser.add_argument("--device", default=None)
    parser.add_argument("--json", action="store_true", help="Print JSON only")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    if not args.ckpt:
        raise SystemExit("--ckpt is required (trained checkpoint or exported .pt)")

    model = load_detector_auto(args.ckpt, map_location=str(device))
    model.eval()
    model.to(device)

    featurizer = create_featurizer(
        device,
        feat_kind=args.feat,
        hubert_ckpt=args.hubert_ckpt,
        wavlm_model=args.wavlm_model,
    )
    feat = featurizer.file_to_feat(args.audio, max_len=args.max_len).to(device)

    with torch.no_grad():
        logits, emb = model(feat)
        prob = torch.softmax(logits, dim=-1)[0]
        p_bonafide = float(prob[0].item())
        p_spoof = float(prob[1].item())
        pred = int(torch.argmax(logits, dim=-1).item())
        label = "bonafide" if pred == 0 else "spoof"

    out = {
        "path": args.audio,
        "prob_bonafide": p_bonafide,
        "prob_spoof": p_spoof,
        "pred_label": label,
        "pred_class": pred,
    }
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
