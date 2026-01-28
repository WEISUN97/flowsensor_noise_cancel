# src/infer/infer_dir.py
from __future__ import annotations

import os
from typing import List

import pandas as pd

from src.features.mel import MelSpecConfig
from src.postprocess.hysteresis import HysteresisConfig
from src.infer.infer_file import InferConfig, infer_events_for_file


def infer_directory_to_tsv(
    model,
    audio_dir: str,
    out_tsv: str,
    mel_cfg: MelSpecConfig,
    infer_cfg: InferConfig,
    pp_cfg: HysteresisConfig,
    recursive: bool = False,
) -> None:
    """
    Run offline inference on a directory of wav files and write DCASE-style TSV:
      filename onset offset event_label confidence
    """
    wav_paths: List[str] = []
    if recursive:
        for root, _, files in os.walk(audio_dir):
            for fn in files:
                if fn.lower().endswith(".wav"):
                    wav_paths.append(os.path.join(root, fn))
    else:
        for fn in sorted(os.listdir(audio_dir)):
            if fn.lower().endswith(".wav"):
                wav_paths.append(os.path.join(audio_dir, fn))

    rows = []
    for p in wav_paths:
        events = infer_events_for_file(model, p, mel_cfg, infer_cfg, pp_cfg)
        base = os.path.basename(p)
        for on, off, conf, lab in events:
            rows.append(
                {
                    "filename": base,
                    "onset": float(on),
                    "offset": float(off),
                    "event_label": str(lab),
                    "confidence": float(conf),
                }
            )

    df = pd.DataFrame(
        rows, columns=["filename", "onset", "offset", "event_label", "confidence"]
    )
    df.to_csv(out_tsv, sep="\t", index=False)
