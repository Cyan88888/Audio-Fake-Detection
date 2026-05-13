"""
Offline wav2vec 2.0 frame features for ASVspoof training.

The output directory mirrors the input ``.flac`` tree and stores one ``.npy``
per utterance in (T, C) float32 layout, compatible with the ASVspoof loaders.
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
import torch
import tqdm

from inference.wav2vec2_featurizer import Wav2Vec2Featurizer

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("dump_wav2vec2_feature")


def dump_wav2vec2_to_dir(
    audio_dir: Path,
    save_dir: Path,
    model_name: str,
    device: torch.device,
    max_len: int = 64600,
    preserve_length: bool = True,
    skip_exists: bool = False,
):
    featurizer = Wav2Vec2Featurizer(model_name=model_name, device=device)
    audio_files = sorted(audio_dir.glob("**/*.flac"))
    for audio_file in tqdm.tqdm(audio_files, desc="wav2vec2 dump"):
        rel = audio_file.relative_to(audio_dir).with_suffix(".npy")
        out_path = save_dir / rel
        if skip_exists and out_path.exists():
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        feat = featurizer.file_to_feat(
            str(audio_file),
            max_len=max_len,
            preserve_length=preserve_length,
        )
        row = featurizer.feat_to_dump_layout(feat)
        np.save(out_path, row.cpu().numpy().astype("float32"))
    logger.info("finished successfully")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("audio_dir", type=str, help="Directory tree containing .flac files")
    parser.add_argument("save_dir", type=str, help="Output directory that mirrors the input tree")
    parser.add_argument(
        "--model_name",
        default="facebook/wav2vec2-base",
        help="Hugging Face wav2vec 2.0 model id",
    )
    parser.add_argument("--max_len", type=int, default=64600, help="Waveform crop length in samples")
    parser.add_argument(
        "--fixed_length",
        action="store_true",
        help="Force all dumped features to max_len // 320 frames. By default full utterance length is preserved.",
    )
    parser.add_argument("--device", default=None, help="Torch device, e.g. cuda, cuda:0, or cpu")
    parser.add_argument("--skip_exists", action="store_true", help="Skip output files that already exist")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("device=%s model=%s", device, args.model_name)
    dump_wav2vec2_to_dir(
        Path(args.audio_dir),
        Path(args.save_dir),
        args.model_name,
        device,
        max_len=args.max_len,
        preserve_length=not args.fixed_length,
        skip_exists=args.skip_exists,
    )


if __name__ == "__main__":
    main()
