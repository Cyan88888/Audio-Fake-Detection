"""
SafeEar-lite trainer:
- keeps a frozen SpeechTokenizer as acoustic frontend
- trains only Transformer spoof detector on tokenizer acoustic features
- does NOT include privacy/content objectives
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from .transformer_trainer import TransformerSpoofTrainer
from ..models.decouple import SpeechTokenizer


_TOKENIZER_REQUIRED_KEYS = {
    "n_filters",
    "dimension",
    "strides",
    "lstm_layers",
    "bidirectional",
    "dilation_base",
    "residual_kernel_size",
    "n_residual_layers",
    "activation",
    "sample_rate",
    "n_q",
    "semantic_dimension",
    "codebook_size",
}


def _extract_tokenizer_cfg(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict):
        if _TOKENIZER_REQUIRED_KEYS.issubset(set(obj.keys())):
            return obj
        for v in obj.values():
            if isinstance(v, dict):
                maybe = _extract_tokenizer_cfg(v)
                if maybe:
                    return maybe
    return {}


def _normalize_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in state_dict.items():
        nk = k
        for prefix in ("module.", "model.", "decouple_model."):
            if nk.startswith(prefix):
                nk = nk[len(prefix) :]
        out[nk] = v
    return out


def _load_speechtokenizer(tokenizer_config_path: str, tokenizer_ckpt_path: str) -> SpeechTokenizer:
    cfg_path = Path(tokenizer_config_path)
    ckpt_path = Path(tokenizer_ckpt_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Tokenizer config not found: {cfg_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Tokenizer checkpoint not found: {ckpt_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        raw_cfg = json.load(f)
    tok_cfg = _extract_tokenizer_cfg(raw_cfg)
    if not tok_cfg:
        raise ValueError(
            f"Failed to parse tokenizer config from {cfg_path}; "
            f"required keys: {sorted(_TOKENIZER_REQUIRED_KEYS)}"
        )

    model = SpeechTokenizer(**tok_cfg)
    ckpt_obj = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(ckpt_obj, dict) and "state_dict" in ckpt_obj:
        state_dict = ckpt_obj["state_dict"]
    else:
        state_dict = ckpt_obj
    state_dict = _normalize_state_dict_keys(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        raise RuntimeError(
            f"Tokenizer checkpoint missing keys ({len(missing)}): {missing[:10]}"
        )
    if unexpected:
        # Non-fatal, but keep explicit log for debug.
        print(f"[WARN] Tokenizer unexpected keys ({len(unexpected)}), ignored: {unexpected[:10]}")
    return model


def _get_wav_target_batch(batch):
    """Returns (wav, target, audio_path_or_none)."""
    if len(batch) == 5:
        wav, _, target, audio_path, _ = batch
        return wav, target, audio_path
    if len(batch) == 4:
        wav, _, target, audio_path = batch
        return wav, target, audio_path
    wav, _, target = batch
    return wav, target, None


class SafeEarLiteTrainer(TransformerSpoofTrainer):
    def __init__(
        self,
        detect_model: torch.nn.Module,
        tokenizer_config_path: str,
        tokenizer_ckpt_path: str,
        tokenizer_layers: Optional[List[int]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(detect_model=detect_model, **kwargs)
        self.tokenizer = _load_speechtokenizer(tokenizer_config_path, tokenizer_ckpt_path)
        self.tokenizer_layers = tokenizer_layers if tokenizer_layers is not None else [0]
        self.tokenizer.eval()
        for p in self.tokenizer.parameters():
            p.requires_grad = False

    def _extract_token_feat(self, wav: torch.Tensor) -> torch.Tensor:
        """Extract acoustic token feature in shape (B, C, T) for detector."""
        self.tokenizer.eval()
        with torch.no_grad():
            # forward_feature returns list of quantized tensors in (B, D, T)
            q_list = self.tokenizer.forward_feature(wav, layers=self.tokenizer_layers)
            q = q_list[0]
            feat = q.permute(0, 2, 1)  # (B, T, D)
            feat = self.tokenizer.transform(feat)  # align to semantic_dimension
            feat = feat.permute(0, 2, 1).contiguous()  # (B, C, T)
        return feat

    def forward(self, batch, is_train: bool = True):
        wav, target, audio_path = _get_wav_target_batch(batch)
        wav = wav.to(memory_format=torch.contiguous_format).float()
        feat = self._extract_token_feat(wav)
        feat = self._prepare_feat(feat, is_train=is_train)
        target = target.long()
        logits, _ = self.detect_model(feat)
        if is_train:
            loss = F.cross_entropy(logits, target, label_smoothing=self.label_smoothing)
            return loss, logits, target
        with torch.no_grad():
            prob_bonafide = torch.softmax(logits, dim=-1)[:, 0]
        return audio_path, torch.tensor(0.0, device=feat.device), prob_bonafide, target

    def validation_step(self, batch, batch_idx):
        wav, target, _ = _get_wav_target_batch(batch)
        wav = wav.to(memory_format=torch.contiguous_format).float()
        feat = self._extract_token_feat(wav)
        feat = self._prepare_feat(feat, is_train=False)
        target = target.long()
        with torch.no_grad():
            logits, _ = self.detect_model(feat)
            prob_bonafide = torch.softmax(logits, dim=-1)[:, 0]
            val_loss = F.cross_entropy(logits, target)
        self.val_index_loader.append(target)
        self.val_score_loader.append(prob_bonafide)
        self.val_loss_loader.append(val_loss.detach())

    def test_step(self, batch, batch_idx):
        wav, target, audio_path = _get_wav_target_batch(batch)
        wav = wav.to(memory_format=torch.contiguous_format).float()
        feat = self._extract_token_feat(wav)
        feat = self._prepare_feat(feat, is_train=False)
        target = target.long()
        with torch.no_grad():
            prob_bonafide = self._tta_predict(feat, feat_lengths=None)
            eps = 1e-6
            prob_bonafide = torch.clamp(prob_bonafide, eps, 1.0 - eps)
            logits_for_loss = torch.stack(
                [torch.log(prob_bonafide), torch.log(1.0 - prob_bonafide)],
                dim=-1,
            )
            test_loss = F.cross_entropy(logits_for_loss, target)
        self.eval_index_loader.append(target)
        self.eval_score_loader.append(prob_bonafide)
        self.eval_loss_loader.append(test_loss.detach())
        self.eval_filename_loader.append(audio_path)
