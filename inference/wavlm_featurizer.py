"""
WavLM frame features for inference, aligned with training crop length.

Uses Hugging Face ``transformers`` WavLM. Time axis is linearly interpolated to
``T* = max_len // 320`` so audio--feature alignment in ``asvspoof19.py`` (which
assumes ~20 ms / 320-sample frames at 16 kHz) stays consistent without changing
the dataset loader (strategy A from project plan).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import torchaudio

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Must match ``feat_duration = max_len // 320`` in ``safeear/datas/asvspoof19.py``.
SAMPLES_PER_FRAME: int = 320


class WavLMFeaturizer:
    """Loads WavLM once; outputs (1, C, T*) with T* = max_len // SAMPLES_PER_FRAME."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        sample_rate: int = 16000,
        device: Optional[torch.device] = None,
    ):
        self.model_name = model_name or "microsoft/wavlm-base"
        self.sample_rate = sample_rate
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        from transformers import Wav2Vec2FeatureExtractor, WavLMModel

        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_name)
        self.model = WavLMModel.from_pretrained(self.model_name).eval().to(self.device)

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
        """Returns feats (1, hidden_dim, T*) with T* = max_len // SAMPLES_PER_FRAME."""
        if wav_1d.dim() != 1:
            wav_1d = wav_1d.reshape(-1)
        if wav_1d.numel() > max_len:
            wav_1d = wav_1d[:max_len]
        elif wav_1d.numel() < max_len:
            wav_1d = F.pad(wav_1d, (0, max_len - wav_1d.numel()))

        wav_np = wav_1d.float().cpu().numpy()
        inputs = self.processor(
            wav_np,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=False,
        )
        input_values = inputs.input_values.to(self.device)

        target_t = max(1, max_len // SAMPLES_PER_FRAME)

        with torch.no_grad():
            out = self.model(input_values, output_hidden_states=False)
            hidden = out.last_hidden_state  # (1, T_seq, H)

        # (1, H, T_seq)
        feat = hidden.transpose(1, 2)
        if feat.size(-1) != target_t:
            feat = F.interpolate(feat, size=target_t, mode="linear", align_corners=False)
        return feat

    def file_to_feat(self, path: str, max_len: int = 64600) -> torch.Tensor:
        wav = self.load_wav_mono(path)
        return self.wav_tensor_to_feat(wav, max_len=max_len)

    def feat_to_dump_layout(self, feat: torch.Tensor) -> torch.Tensor:
        """(1, C, T) -> (T, C) for offline .npy compatible with ASVSppof2019 loader."""
        if feat.dim() != 3 or feat.size(0) != 1:
            raise ValueError(f"Expected feat (1, C, T), got {tuple(feat.shape)}")
        return feat.squeeze(0).transpose(0, 1).contiguous()
