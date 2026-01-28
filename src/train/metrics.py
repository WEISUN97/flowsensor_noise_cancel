# src/train/metrics.py
from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass
class FrameMetrics:
    precision: float
    recall: float
    f1: float


def frame_metrics_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-9,
) -> FrameMetrics:
    """
    Compute frame-level precision/recall/F1 for binary detection.
    logits:  [B, T] or [B, T, 1]
    targets: same shape in {0,1}
    """
    if logits.dim() == 3 and logits.size(-1) == 1:
        logits = logits.squeeze(-1)
    if targets.dim() == 3 and targets.size(-1) == 1:
        targets = targets.squeeze(-1)

    probs = torch.sigmoid(logits)
    pred = (probs >= threshold).to(torch.float32)

    tp = (pred * targets).sum().item()
    fp = (pred * (1.0 - targets)).sum().item()
    fn = ((1.0 - pred) * targets).sum().item()

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = (2.0 * precision * recall) / (precision + recall + eps)
    return FrameMetrics(precision=precision, recall=recall, f1=f1)
