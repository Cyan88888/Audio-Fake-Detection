"""Select HuBERT (fairseq) or WavLM (transformers) frame featurizer via env or arguments."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent


def create_featurizer(
    device: torch.device,
    feat_kind: Optional[str] = None,
    hubert_ckpt: Optional[str] = None,
    wavlm_model: Optional[str] = None,
):
    """
    Args:
        device: torch device for models.
        feat_kind: ``"wavlm"`` | ``"hubert"``. If None, uses env ``SAFEAR_FEAT`` (default ``wavlm``).
        hubert_ckpt: Path to fairseq HuBERT checkpoint (HuBERT only).
        wavlm_model: Hugging Face model id (WavLM only); default ``microsoft/wavlm-base``.
    """
    kind = (feat_kind or os.environ.get("SAFEAR_FEAT", "wavlm")).strip().lower()
    if kind in ("hubert", "h", "fairseq_hubert"):
        from inference.hubert_featurizer import HubertFeaturizer

        ckpt = hubert_ckpt or os.environ.get(
            "SAFEAR_HUBERT", str(_REPO_ROOT / "model_zoos" / "hubert_base_ls960.pt")
        )
        return HubertFeaturizer(ckpt_path=ckpt, device=device)
    if kind in ("wavlm", "wavlm-base", "w", ""):
        from inference.wavlm_featurizer import WavLMFeaturizer

        name = wavlm_model or os.environ.get("SAFEAR_WAVLM", "microsoft/wavlm-base")
        return WavLMFeaturizer(model_name=name, device=device)
    raise ValueError(f"Unknown feat_kind / SAFEAR_FEAT={kind!r}; use 'wavlm' or 'hubert'")
