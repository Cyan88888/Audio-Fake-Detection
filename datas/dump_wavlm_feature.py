"""
Offline WavLM frame features for ASVspoof training (.npy next to each .flac).

Layout matches HuBERT dump: each file is saved as (T, C) float32; the dataset loader
permutes to (C, T). Time length is interpolated to T = crop_audio_len // 320.

Usage (from repo root):
  python datas/dump_wavlm_feature.py \\
    datas/datasets/ASVSpoof2019/LA/ASVspoof2019_LA_train/flac \\
    datas/datasets/ASVSpoof2019_WavLM_base/LA/ASVspoof2019_LA_train/flac \\
    --model_name microsoft/wavlm-base
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import tqdm
import torch

from inference.wavlm_featurizer import WavLMFeaturizer

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("dump_wavlm_feature")


def dump_wavlm_to_dir(
    audio_dir: Path,
    save_dir: Path,
    model_name: str,
    device: torch.device,
    max_len: int = 64600,
):
    fe = WavLMFeaturizer(model_name=model_name, device=device)
    audio_files = sorted(audio_dir.glob("**/*.flac"))
    for audio_file in tqdm.tqdm(audio_files, desc="WavLM dump"):
        rel = audio_file.relative_to(audio_dir).with_suffix(".npy")
        out_path = save_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        feat = fe.file_to_feat(str(audio_file), max_len=max_len)
        row = fe.feat_to_dump_layout(feat)
        np.save(out_path, row.cpu().numpy().astype("float32"))
    logger.info("finished successfully")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("audio_dir", type=str, help="Directory tree containing .flac")
    parser.add_argument("save_dir", type=str, help="Output directory (mirrors flac layout)")
    parser.add_argument(
        "--model_name",
        default="microsoft/wavlm-base",
        help="Hugging Face model id for WavLM",
    )
    parser.add_argument("--max_len", type=int, default=64600, help="Waveform crop length (samples)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device=%s model=%s", device, args.model_name)
    dump_wavlm_to_dir(
        Path(args.audio_dir),
        Path(args.save_dir),
        args.model_name,
        device,
        max_len=args.max_len,
    )


if __name__ == "__main__":
    main()
