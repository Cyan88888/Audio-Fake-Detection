"""
wav2vec 2.0 frame features for inference and offline dumping.

Supports both Hugging Face model ids/directories and fairseq ``.pt`` checkpoints
such as ``model_zoos/wav2vec_small.pt``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import sys

import torch
import torch.nn.functional as F
import torchaudio

_REPO_ROOT = Path(__file__).resolve().parent.parent
_FAIRSEQ_ROOT = _REPO_ROOT / "fairseq_ours"
if _FAIRSEQ_ROOT.is_dir() and str(_FAIRSEQ_ROOT) not in sys.path:
    sys.path.insert(0, str(_FAIRSEQ_ROOT))

SAMPLES_PER_FRAME: int = 320


class Wav2Vec2Featurizer:
    """Loads wav2vec 2.0 once; outputs frame-level SSL features."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        sample_rate: int = 16000,
        device: Optional[torch.device] = None,
    ):
        self.model_name = model_name or "facebook/wav2vec2-base"
        self.sample_rate = sample_rate
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backend = "fairseq" if Path(self.model_name).is_file() else "transformers"
        self.task = None
        self.processor = None

        if self.backend == "fairseq":
            import fairseq.checkpoint_utils as cu

            models, cfg, task = cu.load_model_ensemble_and_task([self.model_name])
            self.model = models[0].eval().to(self.device)
            self.task = task
        else:
            from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_name)
            self.model = Wav2Vec2Model.from_pretrained(self.model_name).eval().to(self.device)

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
        preserve_length: bool = False,
    ) -> torch.Tensor:
        """Returns feats (1, hidden_dim, T*) with T* aligned to max_len // 320."""
        if wav_1d.dim() != 1:
            wav_1d = wav_1d.reshape(-1)
        if not preserve_length:
            if wav_1d.numel() > max_len:
                wav_1d = wav_1d[:max_len]
            elif wav_1d.numel() < max_len:
                wav_1d = F.pad(wav_1d, (0, max_len - wav_1d.numel()))

        target_t = max(1, wav_1d.numel() // SAMPLES_PER_FRAME) if preserve_length else max(1, max_len // SAMPLES_PER_FRAME)

        if self.backend == "fairseq":
            x = wav_1d.float().to(self.device)
            if self.task is not None and getattr(self.task.cfg, "normalize", False):
                x = F.layer_norm(x, x.shape)
            x = x.view(1, -1)
            with torch.no_grad():
                out = self.model.extract_features(source=x, padding_mask=None, mask=False)
            if isinstance(out, dict):
                hidden = out["x"]
            elif isinstance(out, tuple):
                hidden = out[0]
            else:
                raise ValueError(f"Unexpected fairseq wav2vec2 output type: {type(out)}")
        else:
            wav_np = wav_1d.float().cpu().numpy()
            inputs = self.processor(
                wav_np,
                sampling_rate=self.sample_rate,
                return_tensors="pt",
                padding=False,
            )
            input_values = inputs.input_values.to(self.device)
            with torch.no_grad():
                out = self.model(input_values, output_hidden_states=False)
                hidden = out.last_hidden_state

        feat = hidden.transpose(1, 2)
        if feat.size(-1) != target_t:
            feat = F.interpolate(feat, size=target_t, mode="linear", align_corners=False)
        return feat

    def file_to_feat(self, path: str, max_len: int = 64600, preserve_length: bool = False) -> torch.Tensor:
        wav = self.load_wav_mono(path)
        return self.wav_tensor_to_feat(wav, max_len=max_len, preserve_length=preserve_length)

    def feat_to_dump_layout(self, feat: torch.Tensor) -> torch.Tensor:
        """(1, C, T) -> (T, C), compatible with ASVSpoof2019 dataset loader."""
        if feat.dim() != 3 or feat.size(0) != 1:
            raise ValueError(f"Expected feat (1, C, T), got {tuple(feat.shape)}")
        return feat.squeeze(0).transpose(0, 1).contiguous()
