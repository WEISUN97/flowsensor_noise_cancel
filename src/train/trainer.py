# src/train/trainer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from src.train.metrics import frame_metrics_from_logits, FrameMetrics


@dataclass
class TrainState:
    epoch: int = 0
    step: int = 0


class Trainer:
    """
    Minimal trainer for frame-level SED.
    Designed to be extensible:
      - mixed precision
      - gradient accumulation
      - EMA
      - multi-task heads
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn,
        device: str = "cpu",
        grad_clip: float | None = 1.0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = torch.device(device)
        self.grad_clip = grad_clip
        self.state = TrainState()

        self.model.to(self.device)

    def train_one_epoch(
        self, loader: DataLoader, log_every: int = 50
    ) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_batches = 0

        # Aggregate metrics on the fly (frame-level).
        prec_sum = 0.0
        rec_sum = 0.0
        f1_sum = 0.0

        for b_idx, (feat, y_frame, meta) in enumerate(loader):
            feat = feat.to(self.device)  # [B, F, T]
            y_frame = y_frame.to(self.device)  # [B, T]

            logits = self.model(feat)  # [B, T, 1]
            logits = logits.squeeze(-1)  # [B, T]

            loss = self.loss_fn(logits, y_frame)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()

            total_loss += float(loss.item())
            total_batches += 1
            self.state.step += 1

            m = frame_metrics_from_logits(
                logits.detach(), y_frame.detach(), threshold=0.5
            )
            prec_sum += m.precision
            rec_sum += m.recall
            f1_sum += m.f1

            if log_every > 0 and (b_idx + 1) % log_every == 0:
                avg_loss = total_loss / max(1, total_batches)
                avg_f1 = f1_sum / max(1, total_batches)
                print(
                    f"[train] step={self.state.step} loss={avg_loss:.4f} frame_f1={avg_f1:.4f}"
                )

        return {
            "loss": total_loss / max(1, total_batches),
            "frame_precision": prec_sum / max(1, total_batches),
            "frame_recall": rec_sum / max(1, total_batches),
            "frame_f1": f1_sum / max(1, total_batches),
        }

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        total_batches = 0

        prec_sum = 0.0
        rec_sum = 0.0
        f1_sum = 0.0

        for feat, y_frame, meta in loader:
            feat = feat.to(self.device)
            y_frame = y_frame.to(self.device)

            logits = self.model(feat).squeeze(-1)
            loss = self.loss_fn(logits, y_frame)

            total_loss += float(loss.item())
            total_batches += 1

            m = frame_metrics_from_logits(logits, y_frame, threshold=0.5)
            prec_sum += m.precision
            rec_sum += m.recall
            f1_sum += m.f1

        return {
            "loss": total_loss / max(1, total_batches),
            "frame_precision": prec_sum / max(1, total_batches),
            "frame_recall": rec_sum / max(1, total_batches),
            "frame_f1": f1_sum / max(1, total_batches),
        }
