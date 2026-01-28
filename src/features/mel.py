# src/features/mel.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torchaudio


@dataclass
class MelSpecConfig:
    # Audio / feature configuration for 48 kHz offline SED.
    sample_rate: int = 48_000

    # STFT parameters (40 ms window, 10 ms hop at 48 kHz).
    n_fft: int = 2048
    win_length: int = 1920  # 40 ms * 48k = 1920
    hop_length: int = 480  # 10 ms * 48k = 480

    # Mel parameters.
    n_mels: int = 128
    f_min: float = 20.0
    f_max: float = 20_000.0

    # Log-mel numerical stability.
    eps: float = 1e-10

    # Normalization: "per_sample" is robust for noise variability.
    normalize: str = "per_sample"  # {"none", "per_sample"}


class LogMelExtractor(torch.nn.Module):
    """
    Log-mel feature extractor.
    Input: waveform tensor [1, T] in float32, range typically [-1, 1]
    Output: log-mel tensor [n_mels, n_frames]
    """

    def __init__(self, cfg: MelSpecConfig):
        super().__init__()
        self.cfg = cfg
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            win_length=cfg.win_length,
            hop_length=cfg.hop_length,
            f_min=cfg.f_min,
            f_max=cfg.f_max,
            n_mels=cfg.n_mels,
            power=2.0,
            center=True,
            pad_mode="reflect",
            norm=None,
            mel_scale="htk",
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        # wav: [1, T]
        mel = self.mel(wav)  # [1, n_mels, n_frames]
        mel = mel.clamp_min(self.cfg.eps)
        log_mel = torch.log(mel)  # natural log

        # Remove channel dim: [n_mels, n_frames]
        log_mel = log_mel.squeeze(0)

        if self.cfg.normalize == "per_sample":
            # Per-sample normalization improves robustness to recording gain changes.
            mu = log_mel.mean()
            sigma = log_mel.std().clamp_min(1e-6)
            log_mel = (log_mel - mu) / sigma

        return log_mel

    def frames_to_seconds(self, n_frames: int) -> float:
        # Convert number of frames to seconds using hop length.
        return (n_frames * self.cfg.hop_length) / self.cfg.sample_rate

    def seconds_to_frame_index(self, t_sec: float) -> int:
        # Convert time in seconds to nearest frame index.
        return int(round(t_sec * self.cfg.sample_rate / self.cfg.hop_length))
