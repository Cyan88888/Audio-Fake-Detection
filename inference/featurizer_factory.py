"""Select HuBERT, WavLM, or wav2vec 2.0 frame featurizer via env or arguments."""
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
    wav2vec2_model: Optional[str] = None,
):
    """
    Args:
        device: torch device for models.
        feat_kind: ``"wavlm"`` | ``"hubert"`` | ``"wav2vec2"``. If None, uses env ``SAFEAR_FEAT`` (default ``wavlm``).
        hubert_ckpt: Path to fairseq HuBERT checkpoint (HuBERT only).
        wavlm_model: Hugging Face model id (WavLM only); default ``microsoft/wavlm-base``.
        wav2vec2_model: Hugging Face model id (wav2vec 2.0 only); default ``facebook/wav2vec2-base``.
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
    if kind in ("wav2vec2", "wav2vec", "w2v2", "facebook_wav2vec2"):
        from inference.wav2vec2_featurizer import Wav2Vec2Featurizer

        name = wav2vec2_model or os.environ.get("SAFEAR_WAV2VEC2", "facebook/wav2vec2-base")
        return Wav2Vec2Featurizer(model_name=name, device=device)
    raise ValueError(f"Unknown feat_kind / SAFEAR_FEAT={kind!r}; use 'wavlm', 'hubert', or 'wav2vec2'")
