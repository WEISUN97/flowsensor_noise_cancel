# src/postprocess/hysteresis.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch


@dataclass
class HysteresisConfig:
    # Two-threshold hysteresis for stable event detection.
    th_on: float = 0.6
    th_off: float = 0.4

    # Minimum duration (seconds) to keep an event.
    min_dur_sec: float = 0.30

    # Merge events if the gap between them is shorter than this.
    merge_gap_sec: float = 0.20

    # Frame hop in seconds (must match feature hop).
    frame_hop_sec: float = 0.010

    # Fixed label for single-class setting.
    label: str = "target"


def probs_to_events_hysteresis(
    probs_1d: torch.Tensor,
    cfg: HysteresisConfig,
) -> List[Tuple[float, float, float, str]]:
    """
    Convert a 1D probability sequence to event segments using hysteresis.
    Returns a list of (onset_sec, offset_sec, confidence, label).

    probs_1d: [T] probabilities in [0,1]
    """
    if probs_1d.dim() != 1:
        raise ValueError("probs_1d must be 1D [T]")

    T = probs_1d.numel()
    on_th = cfg.th_on
    off_th = cfg.th_off
    hop = cfg.frame_hop_sec

    events: List[Tuple[int, int, float]] = []  # (start_frame, end_frame, conf)

    active = False
    start = 0
    peak = 0.0

    for i in range(T):
        p = float(probs_1d[i].item())

        if not active:
            if p >= on_th:
                active = True
                start = i
                peak = p
        else:
            peak = max(peak, p)
            if p <= off_th:
                end = i
                events.append((start, end, peak))
                active = False

    # Handle trailing active event
    if active:
        events.append((start, T - 1, peak))

    # Convert to seconds and apply min duration
    segs: List[Tuple[float, float, float, str]] = []
    for s, e, conf in events:
        onset = s * hop
        offset = (e + 1) * hop  # make offset exclusive
        if (offset - onset) >= cfg.min_dur_sec:
            segs.append((onset, offset, conf, cfg.label))

    # Merge close segments
    if not segs:
        return segs

    segs.sort(key=lambda x: x[0])
    merged: List[Tuple[float, float, float, str]] = []
    cur_on, cur_off, cur_conf, cur_lab = segs[0]

    for on, off, conf, lab in segs[1:]:
        gap = on - cur_off
        if gap <= cfg.merge_gap_sec and lab == cur_lab:
            cur_off = max(cur_off, off)
            cur_conf = max(cur_conf, conf)
        else:
            merged.append((cur_on, cur_off, cur_conf, cur_lab))
            cur_on, cur_off, cur_conf, cur_lab = on, off, conf, lab

    merged.append((cur_on, cur_off, cur_conf, cur_lab))
    return merged
