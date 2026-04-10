"""
Transformer-based spoof detector on precomputed frame features (e.g. WavLM or HuBERT).

Input: feats (B, C, T) where C=input_dim (e.g. 768), T = time frames.
Does not use SpeechTokenizer / RVQ (privacy decoupling removed for thesis focus).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .safeear import TransformerClassifier


class FrameTransformerDetector(nn.Module):
    """
    Projects frame-level sequence features (e.g. SSL embeddings) to embedding_dim, then TransformerClassifier.

    Forward:
        feats: (B, input_dim, T) — same layout as ASVspoof DataModule collate_fn output.
    """

    def __init__(
        self,
        input_dim: int = 768,
        embedding_dim: int = 768,
        num_classes: int = 2,
        num_layers: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 1.0,
        dropout_rate: float = 0.1,
        attention_dropout: float = 0.1,
        stochastic_depth_rate: float = 0.1,
        positional_embedding: str = "sine",
        sequence_length: int = 512,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        if input_dim != embedding_dim:
            self.proj = nn.Linear(input_dim, embedding_dim)
        else:
            self.proj = nn.Identity()

        self.classifier = TransformerClassifier(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout_rate=dropout_rate,
            attention_dropout=attention_dropout,
            stochastic_depth_rate=stochastic_depth_rate,
            positional_embedding=positional_embedding,
            sequence_length=sequence_length,
        )

    def forward(self, feats: torch.Tensor):
        # feats: (B, C, T)
        if feats.dim() != 3:
            raise ValueError(f"Expected feats (B, C, T), got shape {tuple(feats.shape)}")
        x = feats.transpose(1, 2)  # (B, T, C)
        x = self.proj(x)
        x = x.transpose(1, 2)  # (B, embedding_dim, T)
        return self.classifier(x)


# Backward-compatible alias (older configs / checkpoints may reference this name).
HuBERTTransformerDetector = FrameTransformerDetector
