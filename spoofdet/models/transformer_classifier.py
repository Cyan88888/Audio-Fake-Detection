"""Transformer encoder + pooling classifier for frame-level spoof detection."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Dropout, Identity, LayerNorm, Linear, Module, ModuleList, Parameter, init
from timm.models.layers import DropPath


class Attention(Module):
    def __init__(self, dim, num_heads=8, attention_dropout=0.1, projection_dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5

        self.qkv = Linear(dim, dim * 3, bias=False)
        self.attn_drop = Dropout(attention_dropout)
        self.proj = Linear(dim, dim)
        self.proj_drop = Dropout(projection_dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerEncoderLayer(Module):
    def __init__(
        self,
        d_model,
        nhead,
        atten=Attention,
        dim_feedforward=2048,
        dropout=0.1,
        attention_dropout=0.1,
        drop_path_rate=0.1,
    ):
        super().__init__()
        self.pre_norm = LayerNorm(d_model)
        self.self_attn = atten(
            dim=d_model,
            num_heads=nhead,
            attention_dropout=attention_dropout,
            projection_dropout=dropout,
        )

        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout1 = Dropout(dropout)
        self.norm1 = LayerNorm(d_model)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.dropout2 = Dropout(dropout)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else Identity()
        self.activation = F.gelu

    def forward(self, src: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        src = src + self.drop_path(self.self_attn(self.pre_norm(src)))
        src = self.norm1(src)
        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        src = src + self.drop_path(self.dropout2(src2))
        return src


class TransformerClassifier(Module):
    """Compact Transformer classifier with configurable pooling and positional encoding."""

    def __init__(
        self,
        embedding_dim=768,
        num_classes=1000,
        num_layers=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout_rate=0.1,
        attention_dropout=0.1,
        stochastic_depth_rate=0.1,
        pooling="attention",
        positional_embedding="sine",
        sequence_length=10000,
        *args,
        **kwargs,
    ):
        super().__init__()
        positional_embedding = (
            positional_embedding if positional_embedding in ["sine", "learnable", "none"] else "sine"
        )
        pooling = pooling.lower()
        if pooling not in ["attention", "mean", "max", "meanmax"]:
            raise ValueError(f"Unsupported pooling: {pooling}")
        dim_feedforward = int(embedding_dim * mlp_ratio)
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.pooling = pooling

        assert sequence_length is not None or positional_embedding == "none"

        if positional_embedding != "none":
            if positional_embedding == "learnable":
                self.positional_emb = Parameter(
                    torch.zeros(1, sequence_length, embedding_dim),
                    requires_grad=True,
                )
                init.trunc_normal_(self.positional_emb, std=0.2)
            else:
                self.positional_emb = Parameter(
                    self.sinusoidal_embedding(sequence_length, embedding_dim),
                    requires_grad=False,
                )
        else:
            self.positional_emb = None

        self.dropout = Dropout(p=dropout_rate)
        dpr = [x.item() for x in torch.linspace(0, stochastic_depth_rate, num_layers)]
        self.blocks = ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=embedding_dim,
                    nhead=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout_rate,
                    attention_dropout=attention_dropout,
                    drop_path_rate=dpr[i],
                )
                for i in range(num_layers)
            ]
        )
        self.norm = LayerNorm(embedding_dim)
        self.attention_pool = Linear(self.embedding_dim, 1)
        self.pool_proj = (
            Linear(self.embedding_dim * 2, self.embedding_dim)
            if self.pooling == "meanmax"
            else Identity()
        )
        self.fc = Linear(embedding_dim, num_classes)
        self.apply(self.init_weight)

    def forward(self, x):
        x = torch.transpose(x, -1, -2)
        seq_len = x.size(1)
        if self.positional_emb is not None:
            x = x + self.positional_emb[:, :seq_len, :]

        x = self.dropout(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if self.pooling == "attention":
            feature = torch.matmul(
                F.softmax(self.attention_pool(x), dim=1).transpose(-1, -2), x
            ).squeeze(-2)
        elif self.pooling == "mean":
            feature = x.mean(dim=1)
        elif self.pooling == "max":
            feature = x.max(dim=1).values
        else:
            feature = torch.cat([x.mean(dim=1), x.max(dim=1).values], dim=-1)
            feature = self.pool_proj(feature)
        logits = self.fc(feature)
        return logits, feature

    @staticmethod
    def init_weight(m):
        if isinstance(m, Linear):
            init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, LayerNorm):
            init.constant_(m.bias, 0)
            init.constant_(m.weight, 1.0)

    @staticmethod
    def sinusoidal_embedding(n_channels, dim):
        pe = torch.FloatTensor(
            [[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)] for p in range(n_channels)]
        )
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        return pe.unsqueeze(0)
