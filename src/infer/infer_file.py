# src/infer/infer_file.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torchaudio

from src.features.mel import LogMelExtractor, MelSpecConfig
from src.postprocess.hysteresis import HysteresisConfig, probs_to_events_hysteresis


@dataclass
class InferConfig:
    sample_rate: int = 48_000
    window_sec: float = 6.0
    hop_sec: float = 1.0

    # Match feature hop (10 ms) used in mel extractor.
    frame_hop_sec: float = 0.010

    device: str = "cpu"


def _load_mono_48k(path: str, target_sr: int) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if wav.dim() == 2 and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=target_sr)
    return wav


def infer_events_for_file(
    model: torch.nn.Module,
    wav_path: str,
    mel_cfg: MelSpecConfig,
    infer_cfg: InferConfig,
    pp_cfg: HysteresisConfig,
) -> List[Tuple[float, float, float, str]]:
    """
    Offline inference by sliding windows and aggregating frame probabilities.

    Returns events in absolute time of the file:
      (onset_sec, offset_sec, confidence, label)
    """
    device = torch.device(infer_cfg.device)
    model = model.to(device)
    model.eval()

    extractor = LogMelExtractor(mel_cfg).to(device)

    wav = _load_mono_48k(wav_path, mel_cfg.sample_rate).to(device)  # [1, T]
    T_samples = wav.size(-1)
    dur_sec = T_samples / mel_cfg.sample_rate

    win = infer_cfg.window_sec
    hop = infer_cfg.hop_sec

    # Determine total frames for the full file using hop_length of mel extractor.
    # We will aggregate window predictions by overlap-add (mean).
    # Compute mel frames for full file by running extractor once on the full waveform (offline).
    # This is simple and stable for offline detection.
    full_feat = extractor(wav)  # [n_mels, n_frames]
    n_frames_full = full_feat.size(-1)

    agg = torch.zeros(n_frames_full, device=device)
    cnt = torch.zeros(n_frames_full, device=device)

    # Slide windows in seconds; map to frame indices on the aggregated timeline.
    start = 0.0
    while start < dur_sec - 1e-9:
        end = min(dur_sec, start + win)

        # Load waveform segment for the window from the in-memory wav tensor.
        s0 = int(round(start * mel_cfg.sample_rate))
        s1 = int(round(end * mel_cfg.sample_rate))
        seg = wav[:, s0:s1]

        feat = extractor(seg)  # [n_mels, n_frames_win]
        feat = feat.unsqueeze(0)  # [1, n_mels, n_frames_win]

        with torch.no_grad():
            logits = model(feat)  # [1, T, 1]
            probs = torch.sigmoid(logits).squeeze(0).squeeze(-1)  # [T_win]

        # Map this window's frames onto the full timeline.
        # Use the same frame mapping as in extractor: frame_index = round(t*sr/hop_len).
        i0 = extractor.seconds_to_frame_index(start)
        i1 = min(n_frames_full, i0 + probs.numel())

        # Aggregate by mean in overlap regions.
        agg[i0:i1] += probs[: (i1 - i0)]
        cnt[i0:i1] += 1.0

        start += hop

    # Avoid division by zero (should not happen, but keep robust).
    probs_full = agg / cnt.clamp_min(1.0)

    # Post-process to events.
    pp_cfg = HysteresisConfig(
        th_on=pp_cfg.th_on,
        th_off=pp_cfg.th_off,
        min_dur_sec=pp_cfg.min_dur_sec,
        merge_gap_sec=pp_cfg.merge_gap_sec,
        frame_hop_sec=infer_cfg.frame_hop_sec,
        label=pp_cfg.label,
    )
    events = probs_to_events_hysteresis(probs_full.detach().cpu(), pp_cfg)
    return events
