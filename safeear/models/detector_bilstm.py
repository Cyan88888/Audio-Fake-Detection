"""
BiLSTM spoof detector on precomputed frame features (HuBERT / WavLM layout).

Input: feats (B, C, T). Output: (logits, utterance_feature) — same as FrameTransformerDetector.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FrameBiLSTMDetector(nn.Module):
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        num_classes: int = 2,
        num_layers: int = 2,
        dropout_rate: float = 0.15,
        pooling: str = "mean",
        bidirectional: bool = True,
    ):
        super().__init__()
        pooling = pooling.lower()
        if pooling not in ("mean", "max", "attention"):
            raise ValueError(f"BiLSTM detector supports pooling in mean/max/attention, got {pooling}")
        self.pooling = pooling
        lstm_dropout = dropout_rate if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=bidirectional,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        if pooling == "attention":
            self.attention_pool = nn.Linear(out_dim, 1)
        self.fc = nn.Linear(out_dim, num_classes)

    def forward(self, feats: torch.Tensor):
        if feats.dim() != 3:
            raise ValueError(f"Expected feats (B, C, T), got {tuple(feats.shape)}")
        x = feats.transpose(1, 2)  # (B, T, C)
        x, _ = self.lstm(x)
        if self.pooling == "mean":
            feature = x.mean(dim=1)
        elif self.pooling == "max":
            feature = x.max(dim=1).values
        else:
            weights = torch.softmax(self.attention_pool(x), dim=1)
            feature = torch.sum(weights * x, dim=1)
        logits = self.fc(feature)
        return logits, feature
