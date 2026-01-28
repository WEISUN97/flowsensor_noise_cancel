# src/models/crnn.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CRNNConfig:
    # Input feature shape: [B, n_mels, n_frames]
    n_mels: int = 128

    # CNN backbone
    cnn_channels: tuple[int, ...] = (32, 64, 128)
    cnn_kernel: tuple[int, int] = (3, 3)
    cnn_dropout: float = 0.1

    # Pooling along frequency axis reduces compute while preserving temporal resolution.
    pool_freq: tuple[int, ...] = (2, 2, 2)  # One value per CNN stage.

    # RNN
    rnn_hidden: int = 256
    rnn_layers: int = 2
    rnn_dropout: float = 0.1
    bidirectional: bool = True

    # Output classes (single-class detection => 1 logit per frame)
    num_classes: int = 1


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: tuple[int, int],
        dropout: float,
        pool_f: int,
    ):
        super().__init__()
        kf, kt = kernel
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size=(kf, kt), padding=(kf // 2, kt // 2)
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.dropout = nn.Dropout2d(p=dropout)
        self.pool = nn.MaxPool2d(kernel_size=(pool_f, 1))  # Pool only on frequency axis

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, F, T]
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.pool(x)  # reduce F, keep T
        return x


class CRNN(nn.Module):
    """
    CRNN for frame-level sound event detection.
    Input:  log-mel [B, n_mels, n_frames]
    Output: frame logits [B, n_frames, num_classes]
    """

    def __init__(self, cfg: CRNNConfig):
        super().__init__()
        self.cfg = cfg

        layers = []
        in_ch = 1  # Treat log-mel as 1-channel "image"
        for i, out_ch in enumerate(cfg.cnn_channels):
            pool_f = cfg.pool_freq[i] if i < len(cfg.pool_freq) else 2
            layers.append(
                ConvBlock(in_ch, out_ch, cfg.cnn_kernel, cfg.cnn_dropout, pool_f)
            )
            in_ch = out_ch
        self.cnn = nn.Sequential(*layers)

        # Compute frequency dimension after pooling to build the RNN input size.
        # Since pool is only on frequency axis, time frames remain unchanged.
        f_reduced = cfg.n_mels
        for p in cfg.pool_freq[: len(cfg.cnn_channels)]:
            f_reduced = max(1, f_reduced // p)
        self.f_reduced = f_reduced

        rnn_in = cfg.cnn_channels[-1] * self.f_reduced
        self.rnn = nn.LSTM(
            input_size=rnn_in,
            hidden_size=cfg.rnn_hidden,
            num_layers=cfg.rnn_layers,
            dropout=cfg.rnn_dropout if cfg.rnn_layers > 1 else 0.0,
            bidirectional=cfg.bidirectional,
            batch_first=True,
        )

        rnn_out = cfg.rnn_hidden * (2 if cfg.bidirectional else 1)
        self.head = nn.Linear(rnn_out, cfg.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, F, T] -> [B, 1, F, T]
        x = x.unsqueeze(1)

        x = self.cnn(x)  # [B, C, F', T]
        b, c, f, t = x.shape

        # Flatten frequency & channel to feed the RNN per time step
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, T, C, F']
        x = x.view(b, t, c * f)  # [B, T, C*F']

        x, _ = self.rnn(x)  # [B, T, rnn_out]
        logits = self.head(x)  # [B, T, num_classes]
        return logits
