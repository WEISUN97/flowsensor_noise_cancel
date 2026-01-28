# src/data/samplers.py
from __future__ import annotations

from typing import Iterator, List, Tuple
import random

import torch


class BalancedClipSampler(torch.utils.data.Sampler[int]):
    """
    A simple sampler that maintains a desired positive ratio.
    Works well for rare events to avoid degenerate "all-negative" training.

    This sampler assumes dataset.__getitem__ returns (features, y_frame, meta),
    and defines a clip as positive if y_frame contains any 1s.
    """

    def __init__(self, dataset, pos_ratio: float = 0.5, seed: int = 123):
        self.dataset = dataset
        self.pos_ratio = float(pos_ratio)
        self.rng = random.Random(seed)

        # Precompute positive/negative indices.
        self.pos_indices: List[int] = []
        self.neg_indices: List[int] = []

        for i in range(len(dataset)):
            _, y_frame, _ = dataset[i]
            if float(y_frame.max().item()) > 0.0:
                self.pos_indices.append(i)
            else:
                self.neg_indices.append(i)

        if not self.pos_indices:
            raise ValueError(
                "No positive clips found. Check your annotations or windowing settings."
            )
        if not self.neg_indices:
            # It is rare but possible (e.g., fully labeled target-only dataset).
            # In that case, fall back to uniform sampling.
            self.neg_indices = self.pos_indices.copy()

    def __iter__(self) -> Iterator[int]:
        n = len(self.dataset)
        for _ in range(n):
            if self.rng.random() < self.pos_ratio:
                yield self.rng.choice(self.pos_indices)
            else:
                yield self.rng.choice(self.neg_indices)

    def __len__(self) -> int:
        return len(self.dataset)
