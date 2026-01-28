# src/train/losses.py
from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FocalLossConfig:
    gamma: float = 2.0
    alpha: float = 0.25
    reduction: str = "mean"  # {"mean", "sum", "none"}


class FocalLoss(nn.Module):
    """
    Focal loss for binary or multi-label classification with logits.
    This implementation expects:
      logits: [B, T] or [B, T, C]
      targets: same shape with values in {0,1}

    For single-class SED, use logits [B, T] and targets [B, T].
    """

    def __init__(self, cfg: FocalLossConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Ensure shapes match
        if logits.shape != targets.shape:
            raise ValueError(
                f"Shape mismatch: logits {logits.shape} vs targets {targets.shape}"
            )

        # Compute standard BCE with logits per element
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # p_t = sigmoid(logit) if y=1 else 1-sigmoid(logit)
        prob = torch.sigmoid(logits)
        p_t = prob * targets + (1.0 - prob) * (1.0 - targets)

        # alpha factor balances positive/negative samples
        alpha_t = self.cfg.alpha * targets + (1.0 - self.cfg.alpha) * (1.0 - targets)

        focal = alpha_t * (1.0 - p_t).pow(self.cfg.gamma) * bce

        if self.cfg.reduction == "mean":
            return focal.mean()
        if self.cfg.reduction == "sum":
            return focal.sum()
        return focal


class BCEWithLogitsLoss(nn.Module):
    """
    A thin wrapper to support optional pos_weight for imbalanced data.
    """

    def __init__(self, pos_weight: float | None = None):
        super().__init__()
        self.pos_weight = (
            None
            if pos_weight is None
            else torch.tensor([pos_weight], dtype=torch.float32)
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pos_weight = (
            self.pos_weight.to(logits.device) if self.pos_weight is not None else None
        )
        return F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_weight
        )
