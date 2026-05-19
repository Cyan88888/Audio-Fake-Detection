from __future__ import annotations

import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = Path(__file__).resolve().parent


# Web 推理默认使用 HuBERT 最优 checkpoint（与 run.md 一致）
DEFAULT_HUBERT_CKPT = (
    ROOT_DIR
    / "Exps"
    / "Search_PoolMax_S3_ls002_hubert"
    / "checkpoints"
    / "epoch=4-val_eer=0.0009.ckpt"
)


def get_device_name() -> str:
    """
    Device for web inference.
    - ``SAFEAR_DEVICE`` if set (``cuda`` / ``cpu``)
    - else ``cuda`` when available, otherwise ``cpu`` (无 GPU 时自动 CPU，开 GPU 后重启即用 CUDA)
    """
    import torch

    override = os.environ.get("SAFEAR_DEVICE", "").strip().lower()
    if override:
        if override.startswith("cuda") and not torch.cuda.is_available():
            return "cpu"
        return override
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_ckpt_path() -> str:
    return os.environ.get("SAFEAR_CKPT", str(DEFAULT_HUBERT_CKPT))


def get_feat_kind() -> str:
    return os.environ.get("SAFEAR_FEAT", "hubert")


def get_hubert_path() -> str:
    return os.environ.get("SAFEAR_HUBERT", str(ROOT_DIR / "model_zoos" / "hubert_base_ls960.pt"))


def get_wavlm_model() -> str:
    return os.environ.get("SAFEAR_WAVLM", "microsoft/wavlm-base")


def get_web_fixed_threshold_spoof() -> float:
    """
    Fixed spoof decision threshold used by web APIs.
    Any client-provided threshold should be ignored.
    """
    raw = os.environ.get("SAFEAR_WEB_FIXED_THRESHOLD_SPOOF", "0.5")
    try:
        th = float(raw)
    except ValueError:
        return 0.5
    return max(0.0, min(1.0, th))

