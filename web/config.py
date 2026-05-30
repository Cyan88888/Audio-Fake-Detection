from __future__ import annotations

import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = Path(__file__).resolve().parent


DEFAULT_HUBERT_CKPT = (
    ROOT_DIR
    / "Exps"
    / "Search_PoolMax_S3_ls002_hubert"
    / "checkpoints"
    / "epoch=4-val_eer=0.0009.ckpt"
)


def _env(primary: str, legacy: str, default: str = "") -> str:
  """Read env var with legacy fallback (SAFEAR_* → SPOOFDET_*)."""
  return os.environ.get(primary) or os.environ.get(legacy, default)


def get_device_name() -> str:
    import torch

    override = _env("SPOOFDET_DEVICE", "SAFEAR_DEVICE", "").strip().lower()
    if override:
        if override.startswith("cuda") and not torch.cuda.is_available():
            return "cpu"
        return override
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_ckpt_path() -> str:
    return _env("SPOOFDET_CKPT", "SAFEAR_CKPT", str(DEFAULT_HUBERT_CKPT))


def get_feat_kind() -> str:
    return _env("SPOOFDET_FEAT", "SAFEAR_FEAT", "hubert")


def get_hubert_path() -> str:
    return _env("SPOOFDET_HUBERT", "SAFEAR_HUBERT", str(ROOT_DIR / "model_zoos" / "hubert_base_ls960.pt"))


def get_wavlm_model() -> str:
    return _env("SPOOFDET_WAVLM", "SAFEAR_WAVLM", "microsoft/wavlm-base")


def get_web_fixed_threshold_spoof() -> float:
    raw = _env("SPOOFDET_WEB_FIXED_THRESHOLD_SPOOF", "SAFEAR_WEB_FIXED_THRESHOLD_SPOOF", "0.5")
    try:
        th = float(raw)
    except ValueError:
        return 0.5
    return max(0.0, min(1.0, th))
