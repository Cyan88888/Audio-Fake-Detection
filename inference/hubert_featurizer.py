"""
On-the-fly HuBERT L9 (layer 9) features for inference, aligned with training crop length.
Requires fairseq (see project requirements / fairseq_ours) and hubert_base_ls960.pt.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

_REPO_ROOT = Path(__file__).resolve().parent.parent


class HubertFeaturizer:
    """Loads HuBERT once; extracts avg layer features matching dump_hubert_avg_feature.py."""

    def __init__(
        self,
        ckpt_path: Optional[str] = None,
        layer: int = 9,
        sample_rate: int = 16000,
        device: Optional[torch.device] = None,
        max_chunk: int = 1600000,
    ):
        ckpt_path = ckpt_path or str(_REPO_ROOT / "model_zoos" / "hubert_base_ls960.pt")
        self.layer = layer
        self.sample_rate = sample_rate
        self.max_chunk = max_chunk
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        import fairseq.checkpoint_utils as cu

        models, cfg, task = cu.load_model_ensemble_and_task([ckpt_path])
        self.model = models[0].eval().to(self.device)
        self.task = task

    def load_wav_mono(self, path: str) -> torch.Tensor:
        wav, sr = torchaudio.load(path)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        return wav.squeeze(0)

    def wav_tensor_to_feat(
        self,
        wav_1d: torch.Tensor,
        max_len: int = 64600,
    ) -> torch.Tensor:
        """Returns feats (1, 768, T_frames) with T_frames = max_len // 320 (training alignment)."""
        if wav_1d.dim() != 1:
            wav_1d = wav_1d.reshape(-1)
        if wav_1d.numel() > max_len:
            wav_1d = wav_1d[:max_len]
        elif wav_1d.numel() < max_len:
            wav_1d = F.pad(wav_1d, (0, max_len - wav_1d.numel()))

        x = wav_1d.float().to(self.device)
        if getattr(self.task.cfg, "normalize", False):
            x = F.layer_norm(x, x.shape)
        x = x.view(1, -1)

        avg_parts = []
        with torch.no_grad():
            for start in range(0, x.size(1), self.max_chunk):
                chunk = x[:, start : start + self.max_chunk]
                _, _, avg_feat_chunk = self.model.extract_features(
                    source=chunk,
                    padding_mask=None,
                    mask=False,
                    output_layer=self.layer,
                )
                avg_parts.append(avg_feat_chunk)
        avg = torch.cat(avg_parts, dim=1)
        # Normalize to (T, C): avoid (1,T,C) + transpose(0,1) -> (T,1,C) -> 4D after unsqueeze.
        while avg.dim() > 2 and avg.size(0) == 1:
            avg = avg.squeeze(0)
        if avg.dim() == 3:
            avg = avg[0]
        if avg.dim() != 2:
            raise ValueError(f"Unexpected HuBERT feature shape {tuple(avg.shape)}, expected (T, C) or (1, T, C).")

        target_t = max(1, max_len // 320)
        c_dim = avg.size(-1)
        if c_dim == 768:
            feat = avg.transpose(0, 1).unsqueeze(0)  # (1, 768, T_h)
        elif avg.size(0) == 768:
            feat = avg.unsqueeze(0)  # already (768, T_h)
        else:
            raise ValueError(f"Cannot infer layout from shape {tuple(avg.shape)} (expected 768-dim channel).")
        if feat.dim() != 3:
            raise ValueError(f"Internal error: feat must be 3D (1,C,T), got {tuple(feat.shape)}")
        if feat.size(-1) != target_t:
            feat = F.interpolate(feat, size=target_t, mode="linear", align_corners=False)
        return feat

    def file_to_feat(self, path: str, max_len: int = 64600) -> torch.Tensor:
        wav = self.load_wav_mono(path)
        return self.wav_tensor_to_feat(wav, max_len=max_len)
