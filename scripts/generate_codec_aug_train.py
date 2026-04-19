from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torchaudio
import torchaudio.functional as AF


def maybe_codec(audio: torch.Tensor, sr: int, codec: str) -> torch.Tensor:
    if codec == "gsm":
        x = AF.resample(audio, sr, 8000)
        x = AF.apply_codec(x, 8000, codec)
        return AF.resample(x, 8000, sr)
    return AF.apply_codec(audio, sr, codec)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate offline codec-augmented train audio.")
    parser.add_argument(
        "--src_dir",
        type=str,
        default="datas/datasets/ASVSpoof2019/LA/ASVspoof2019_LA_train/flac",
    )
    parser.add_argument(
        "--dst_dir",
        type=str,
        default="datas/datasets/ASVSpoof2019_aug/LA/ASVspoof2019_LA_train_mix/flac",
    )
    parser.add_argument("--p_aug", type=float, default=0.5)
    parser.add_argument(
        "--codecs",
        type=str,
        default="ogg",
        help="Comma-separated codec list, e.g. 'ogg,gsm'",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_exists", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    codecs = [c.strip() for c in args.codecs.split(",") if c.strip()]
    if not codecs:
        raise ValueError("No valid codecs provided.")

    src = Path(args.src_dir)
    dst = Path(args.dst_dir)
    dst.mkdir(parents=True, exist_ok=True)

    files = sorted(src.glob("*.flac"))
    print(f"source files: {len(files)}")
    ok_aug = 0
    fail_aug = 0
    copied = 0

    for idx, f in enumerate(files, 1):
        out_path = dst / f.name
        if args.skip_exists and out_path.exists():
            continue

        wav, sr = torchaudio.load(str(f))
        out = wav
        do_aug = random.random() < args.p_aug
        if do_aug:
            codec = random.choice(codecs)
            try:
                out = maybe_codec(wav, sr, codec)
                ok_aug += 1
            except Exception:
                out = wav
                fail_aug += 1
        else:
            copied += 1

        torchaudio.save(str(out_path), out, sr)
        if idx % 5000 == 0:
            print(f"{idx}/{len(files)}")

    print("done")
    print(f"aug_ok={ok_aug} aug_fail={fail_aug} copied={copied}")


if __name__ == "__main__":
    main()
