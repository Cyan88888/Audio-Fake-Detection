from __future__ import annotations

import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = Path(__file__).resolve().parent
STORAGE_DIR = WEB_DIR / "storage"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)


def get_device_name() -> str:
    return os.environ.get("SAFEAR_DEVICE", "cuda")


def get_ckpt_path() -> str:
    default_ckpt = ROOT_DIR / "Exps" / "Search_PoolMax_S3_ls002" / "checkpoints" / "epoch=6-val_eer=0.0266.ckpt"
    return os.environ.get("SAFEAR_CKPT", str(default_ckpt))


def get_feat_kind() -> str:
    return os.environ.get("SAFEAR_FEAT", "wavlm")


def get_hubert_path() -> str:
    return os.environ.get("SAFEAR_HUBERT", str(ROOT_DIR / "model_zoos" / "hubert_base_ls960.pt"))


def get_wavlm_model() -> str:
    return os.environ.get("SAFEAR_WAVLM", "microsoft/wavlm-base")


def get_auth_user() -> str:
    return os.environ.get("SAFEAR_WEB_USER", "admin")


def get_auth_password() -> str:
    return os.environ.get("SAFEAR_WEB_PASSWORD", "safeear123")


def get_auth_token() -> str:
    return os.environ.get("SAFEAR_WEB_TOKEN", "safeear-demo-token")


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

