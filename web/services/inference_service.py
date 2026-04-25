from __future__ import annotations

import io
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import torch
import torch.nn as nn
import torchaudio
import torchaudio.functional as AF

from .. import config

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference.featurizer_factory import create_featurizer
from inference.load_model import load_detector_auto


class InferenceService:
    def __init__(self) -> None:
        self._detector: Optional[nn.Module] = None
        self._featurizer: Optional[Any] = None
        self._device: Optional[torch.device] = None
        self._model_version: str = "unloaded"
        # Must match ``TransformerSpoofTrainer._normalize_feat`` when training used CMVN.
        self._feat_norm_mode: str = "none"

    @property
    def model_version(self) -> str:
        return self._model_version

    @staticmethod
    def _resolve_feat_norm_mode(ckpt_path: str) -> str:
        """
        Match training-time feature normalization (see ``TransformerSpoofTrainer.feat_norm_mode``).
        Priority: env ``SAFEAR_FEAT_NORM`` -> Lightning ``hyper_parameters`` -> ``config.yaml`` -> none.
        """
        override = os.environ.get("SAFEAR_FEAT_NORM", "").strip().lower()
        if override in {"none", "utt_cmvn"}:
            return override
        if override:
            return "none"

        p = Path(ckpt_path)
        if p.suffix == ".ckpt":
            try:
                blob = torch.load(str(p), map_location="cpu")
                hp = blob.get("hyper_parameters")
                if isinstance(hp, dict):
                    v = str(hp.get("feat_norm_mode", "none")).lower()
                    if v in {"none", "utt_cmvn"}:
                        return v
            except Exception:
                pass
            cfg_path = p.parent.parent / "config.yaml"
            if cfg_path.is_file():
                try:
                    from omegaconf import OmegaConf

                    cfg = OmegaConf.load(cfg_path)
                    v = str(OmegaConf.select(cfg, "system.feat_norm_mode", default="none")).lower()
                    if v in {"none", "utt_cmvn"}:
                        return v
                except Exception:
                    pass
        return "none"

    def _normalize_frame_feat(self, feat: torch.Tensor) -> torch.Tensor:
        feat = feat.to(memory_format=torch.contiguous_format).float()
        if self._feat_norm_mode != "utt_cmvn":
            return feat
        mean = feat.mean(dim=-1, keepdim=True)
        std = feat.std(dim=-1, keepdim=True).clamp_min(1e-5)
        return (feat - mean) / std

    def load(self) -> None:
        self._device = torch.device(config.get_device_name() if torch.cuda.is_available() else "cpu")
        ckpt = config.get_ckpt_path()
        if not ckpt or not Path(ckpt).is_file():
            self._detector = None
            self._featurizer = None
            self._model_version = "unloaded"
            self._feat_norm_mode = "none"
            return

        self._feat_norm_mode = self._resolve_feat_norm_mode(ckpt)
        self._detector = load_detector_auto(ckpt, map_location=str(self._device))
        self._detector.eval()
        self._detector.to(self._device)
        self._featurizer = create_featurizer(
            self._device,
            feat_kind=config.get_feat_kind(),
            hubert_ckpt=config.get_hubert_path(),
            wavlm_model=config.get_wavlm_model(),
        )
        self._model_version = Path(ckpt).name

    def is_ready(self) -> bool:
        return self._detector is not None and self._featurizer is not None and self._device is not None

    @staticmethod
    def _wav_mel_for_plot(wav_1d: torch.Tensor, sr: int, max_wave_points: int = 4000) -> Dict[str, Any]:
        wav = wav_1d.float()
        if wav.numel() > max_wave_points:
            step = max(1, wav.numel() // max_wave_points)
            wav_ds = wav[::step][:max_wave_points]
        else:
            wav_ds = wav
        wave_list = wav_ds.cpu().numpy().tolist()

        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=1024, hop_length=160, n_mels=64, mel_scale="htk"
        )(wav.unsqueeze(0))
        mel_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80.0)(mel)
        z = mel_db.squeeze(0).numpy()
        max_t = 200
        if z.shape[1] > max_t:
            z = z[:, :max_t]
        return {"waveform": wave_list, "mel_db": z.tolist(), "sample_rate": sr}

    def _decode_audio(self, raw: bytes) -> Tuple[torch.Tensor, int]:
        wav, sr = torchaudio.load(io.BytesIO(raw))
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != 16000:
            wav = AF.resample(wav, sr, 16000)
            sr = 16000
        return wav, sr

    def predict_bytes(self, raw: bytes, filename: str, max_len: int, threshold: float) -> Dict[str, Any]:
        if not self.is_ready():
            raise RuntimeError("Model not loaded.")
        wav, sr = self._decode_audio(raw)
        wav_1d = wav.squeeze(0)
        plot_payload = self._wav_mel_for_plot(wav_1d, sr)

        st = time.perf_counter()
        feat = self._featurizer.wav_tensor_to_feat(wav_1d, max_len=max_len).to(self._device)
        feat = self._normalize_frame_feat(feat)
        with torch.no_grad():
            logits, _ = self._detector(feat)
            prob = torch.softmax(logits, dim=-1)[0]
            pred = int(torch.argmax(logits, dim=-1).item())
        elapsed_ms = (time.perf_counter() - st) * 1000.0
        prob_bonafide = float(prob[0].item())
        prob_spoof = float(prob[1].item())
        decision = "spoof" if prob_spoof >= threshold else "bonafide"

        return {
            "filename": filename,
            "prob_bonafide": prob_bonafide,
            "prob_spoof": prob_spoof,
            "pred_label": "bonafide" if pred == 0 else "spoof",
            "pred_class": pred,
            "threshold": threshold,
            "decision_by_threshold": decision,
            "confidence_gap": abs(prob_spoof - threshold),
            "inference_time_ms": elapsed_ms,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_version": self._model_version,
            **plot_payload,
        }

    def new_job_id(self) -> str:
        return uuid4().hex


inference_service = InferenceService()
