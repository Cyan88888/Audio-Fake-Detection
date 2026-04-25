from __future__ import annotations

from fastapi import APIRouter

from .. import config
from ..services.inference_service import inference_service

router = APIRouter(prefix="", tags=["health"])


@router.get("/health")
def health():
    ready = inference_service.is_ready()
    return {
        "status": "ok" if ready else "no_model",
        "ckpt": config.get_ckpt_path(),
        "feat": config.get_feat_kind(),
        "model_version": inference_service.model_version,
    }
