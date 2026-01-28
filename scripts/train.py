# scripts/train.py
from __future__ import annotations

import os

import torch
from torch.utils.data import DataLoader

from src.data.dataset import DatasetConfig, StrongSEDDataset, WindowingConfig
from src.data.samplers import BalancedClipSampler
from src.features.mel import MelSpecConfig
from src.models.crnn import CRNN, CRNNConfig
from src.train.losses import FocalLoss, FocalLossConfig
from src.train.trainer import Trainer


def main():
    # Basic paths
    dataset_root = "dataset"
    train_tsv = os.path.join(dataset_root, "annotations", "strong_train.tsv")
    val_tsv = os.path.join(dataset_root, "annotations", "strong_val.tsv")

    # Feature and window settings for 48 kHz
    mel_cfg = MelSpecConfig(
        sample_rate=48_000, n_mels=128, win_length=1920, hop_length=480
    )
    win_cfg = WindowingConfig(window_sec=6.0, hop_sec=1.0, frame_hop_sec=0.010)

    # Datasets
    train_cfg = DatasetConfig(
        dataset_root=dataset_root,
        split="train",
        strong_tsv=train_tsv,
        label_map={"target": 0},
        mel_cfg=mel_cfg,
        win_cfg=win_cfg,
        include_negative_windows=True,
        event_boundary_mode="clip",
    )
    val_cfg = DatasetConfig(
        dataset_root=dataset_root,
        split="val",
        strong_tsv=val_tsv,
        label_map={"target": 0},
        mel_cfg=mel_cfg,
        win_cfg=win_cfg,
        include_negative_windows=True,
        event_boundary_mode="clip",
    )

    train_ds = StrongSEDDataset(train_cfg)
    val_ds = StrongSEDDataset(val_cfg)

    # Balanced sampling helps rare events
    sampler = BalancedClipSampler(train_ds, pos_ratio=0.5, seed=123)

    train_loader = DataLoader(
        train_ds, batch_size=8, sampler=sampler, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=8, shuffle=False, num_workers=4, pin_memory=True
    )

    # Model
    model_cfg = CRNNConfig(n_mels=mel_cfg.n_mels, num_classes=1)
    model = CRNN(model_cfg)

    # Loss (Focal is robust for class imbalance)
    loss_fn = FocalLoss(FocalLossConfig(gamma=2.0, alpha=0.25, reduction="mean"))

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Trainer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = Trainer(model, optimizer, loss_fn, device=device, grad_clip=1.0)

    best_val_f1 = -1.0
    for epoch in range(1, 31):
        trainer.state.epoch = epoch
        tr = trainer.train_one_epoch(train_loader, log_every=50)
        va = trainer.validate(val_loader)

        print(
            f"[epoch {epoch:03d}] "
            f"train_loss={tr['loss']:.4f} train_f1={tr['frame_f1']:.4f} | "
            f"val_loss={va['loss']:.4f} val_f1={va['frame_f1']:.4f}"
        )

        # Save best checkpoint by validation frame F1 (simple baseline criterion)
        if va["frame_f1"] > best_val_f1:
            best_val_f1 = va["frame_f1"]
            os.makedirs("checkpoints", exist_ok=True)
            ckpt_path = os.path.join("checkpoints", "crnn_best.pt")
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_cfg": model_cfg.__dict__,
                    "mel_cfg": mel_cfg.__dict__,
                    "win_cfg": win_cfg.__dict__,
                    "epoch": epoch,
                    "val_f1": best_val_f1,
                },
                ckpt_path,
            )
            print(f"Saved best checkpoint to {ckpt_path} (val_f1={best_val_f1:.4f})")


if __name__ == "__main__":
    main()
