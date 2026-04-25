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
    return os.environ.get("SAFEAR_CKPT", "")


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

