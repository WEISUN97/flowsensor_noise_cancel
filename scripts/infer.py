# scripts/infer.py
from __future__ import annotations

import os
import torch

from src.features.mel import MelSpecConfig
from src.models.crnn import CRNN, CRNNConfig
from src.postprocess.hysteresis import HysteresisConfig
from src.infer.infer_file import InferConfig
from src.infer.infer_dir import infer_directory_to_tsv


def main():
    ckpt_path = os.path.join("checkpoints", "crnn_best.pt")
    dataset_root = "dataset"
    test_audio_dir = os.path.join(dataset_root, "audio", "test")
    out_tsv = "predictions.tsv"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_cfg = CRNNConfig(**ckpt["model_cfg"])
    mel_cfg = MelSpecConfig(**ckpt["mel_cfg"])

    model = CRNN(model_cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    infer_cfg = InferConfig(
        sample_rate=48_000,
        window_sec=6.0,
        hop_sec=1.0,
        frame_hop_sec=0.010,
        device=device,
    )

    pp_cfg = HysteresisConfig(
        th_on=0.6,
        th_off=0.4,
        min_dur_sec=0.30,
        merge_gap_sec=0.20,
        frame_hop_sec=0.010,
        label="target",
    )

    infer_directory_to_tsv(
        model=model,
        audio_dir=test_audio_dir,
        out_tsv=out_tsv,
        mel_cfg=mel_cfg,
        infer_cfg=infer_cfg,
        pp_cfg=pp_cfg,
        recursive=False,
    )
    print(f"Saved predictions to {out_tsv}")


if __name__ == "__main__":
    main()
