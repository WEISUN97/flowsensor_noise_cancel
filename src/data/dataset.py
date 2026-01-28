# src/data/dataset.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
import torchaudio

from src.features.mel import LogMelExtractor, MelSpecConfig


@dataclass
class WindowingConfig:
    # Offline windowing for training/inference.
    window_sec: float = 6.0
    hop_sec: float = 1.0

    # Frame hop is implied by mel extractor config (10 ms by default).
    # We still keep it explicit for clarity.
    frame_hop_sec: float = 0.010


@dataclass
class DatasetConfig:
    # Root folder containing "audio/" and "annotations/".
    dataset_root: str

    # Split name: "train", "val", or "test".
    split: str

    # Annotation file (DCASE strong TSV) for train/val.
    # For test, you can pass None.
    strong_tsv: Optional[str] = None

    # Label map supports future multi-class extension.
    label_map: Dict[str, int] = None

    # Audio and feature configs.
    mel_cfg: MelSpecConfig = MelSpecConfig()
    win_cfg: WindowingConfig = WindowingConfig()

    # If True, clips that contain no events are sampled as negatives too.
    include_negative_windows: bool = True

    # Controls how to treat partial events that cross window boundaries.
    # "clip": clamp event to window; "drop": drop if not fully inside window.
    event_boundary_mode: str = "clip"  # {"clip", "drop"}


def _list_wavs(folder: str) -> List[str]:
    wavs = []
    for fn in os.listdir(folder):
        if fn.lower().endswith(".wav"):
            wavs.append(fn)
    wavs.sort()
    return wavs


def _read_strong_tsv(tsv_path: str) -> pd.DataFrame:
    """
    Read DCASE strong labels TSV.
    Expected columns: filename, onset, offset, event_label
    """
    df = pd.read_csv(tsv_path, sep="\t")
    required = {"filename", "onset", "offset", "event_label"}
    missing = required.difference(set(df.columns))
    if missing:
        raise ValueError(f"Missing columns in strong TSV: {missing}")
    return df


def _group_events(df: pd.DataFrame) -> Dict[str, List[Tuple[float, float, str]]]:
    """
    Group events by filename:
      { "audio_0001.wav": [(onset, offset, label), ...], ... }
    """
    grouped: Dict[str, List[Tuple[float, float, str]]] = {}
    for row in df.itertuples(index=False):
        fn = str(row.filename)
        onset = float(row.onset)
        offset = float(row.offset)
        label = str(row.event_label)
        grouped.setdefault(fn, []).append((onset, offset, label))
    # Sort events for each file by onset.
    for fn in grouped:
        grouped[fn].sort(key=lambda x: x[0])
    return grouped


@dataclass
class ClipIndex:
    """
    An index entry describing a training clip (window) extracted from a file.
    """

    filename: str
    start_sec: float
    end_sec: float


class StrongSEDDataset(torch.utils.data.Dataset):
    """
    Offline windowed dataset for strong-labeled Sound Event Detection (SED).

    Returns:
      features: Tensor [n_mels, n_frames]
      y_frame:  Tensor [n_frames] for single-class (0/1). (For multi-class, extend to [C, n_frames].)
      meta: dict with file/window info
    """

    def __init__(self, cfg: DatasetConfig):
        if cfg.label_map is None:
            cfg.label_map = {"target": 0}

        self.cfg = cfg
        self.split = cfg.split

        self.audio_dir = os.path.join(cfg.dataset_root, "audio", cfg.split)
        if not os.path.isdir(self.audio_dir):
            raise FileNotFoundError(f"Audio directory not found: {self.audio_dir}")

        self.wav_files = _list_wavs(self.audio_dir)

        # Load strong labels if provided (train/val).
        self.events_by_file: Dict[str, List[Tuple[float, float, str]]] = {}
        if cfg.strong_tsv is not None:
            df = _read_strong_tsv(cfg.strong_tsv)
            self.events_by_file = _group_events(df)

        self.extractor = LogMelExtractor(cfg.mel_cfg)

        # Precompute clip indices for fast sampling.
        self.clips: List[ClipIndex] = self._build_clip_index()

    def _get_audio_duration_sec(self, wav_path: str) -> float:
        # Using torchaudio.info avoids loading the whole audio file.
        info = torchaudio.info(wav_path)
        return float(info.num_frames) / float(info.sample_rate)

    def _build_clip_index(self) -> List[ClipIndex]:
        """
        Build a list of fixed windows for offline training/inference.
        Includes:
          - All windows from all wav files
          - If include_negative_windows=True, windows with no events are included as negatives
          - Windows overlapping an event are included as positives candidates
        """
        win = self.cfg.win_cfg.window_sec
        hop = self.cfg.win_cfg.hop_sec

        clips: List[ClipIndex] = []

        for fn in self.wav_files:
            wav_path = os.path.join(self.audio_dir, fn)
            dur = self._get_audio_duration_sec(wav_path)

            # If the audio is shorter than window, still create one window starting at 0.
            if dur <= win:
                clips.append(ClipIndex(fn, 0.0, dur))
                continue

            # Sliding windows: [start, start+win]
            start = 0.0
            while start + win <= dur + 1e-9:
                end = start + win
                clips.append(ClipIndex(fn, start, end))
                start += hop

            # Optionally include a tail window to cover the end.
            # This helps when events occur near the end of file.
            if (dur - win) > 0 and (dur - (clips[-1].start_sec + win)) > 0.25 * hop:
                tail_start = max(0.0, dur - win)
                clips.append(ClipIndex(fn, tail_start, tail_start + win))

        # Optional filtering: keep only windows with events if negatives are not desired.
        if not self.cfg.include_negative_windows and self.events_by_file:
            filtered: List[ClipIndex] = []
            for c in clips:
                if self._window_has_any_event(c.filename, c.start_sec, c.end_sec):
                    filtered.append(c)
            clips = filtered

        return clips

    def _window_has_any_event(self, filename: str, start: float, end: float) -> bool:
        events = self.events_by_file.get(filename, [])
        for on, off, _ in events:
            if off > start and on < end:
                return True
        return False

    def _load_audio_segment(
        self, wav_path: str, start_sec: float, end_sec: float
    ) -> torch.Tensor:
        """
        Load a waveform segment [start_sec, end_sec] from a wav file.
        Returns a mono waveform tensor [1, T] at the original sample rate (expected 48k).
        """
        sr = self.cfg.mel_cfg.sample_rate
        frame_offset = int(round(start_sec * sr))
        num_frames = int(round((end_sec - start_sec) * sr))

        wav, file_sr = torchaudio.load(
            wav_path, frame_offset=frame_offset, num_frames=num_frames
        )

        # Enforce mono by averaging channels if needed.
        if wav.dim() == 2 and wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # Resample only if necessary (should not happen if your dataset is consistent).
        if file_sr != sr:
            wav = torchaudio.functional.resample(wav, orig_freq=file_sr, new_freq=sr)

        return wav

    def _make_frame_labels_single_class(
        self, filename: str, clip_start: float, clip_end: float, n_frames: int
    ) -> torch.Tensor:
        """
        Create frame-level labels for a single target class.
        y_frame: [n_frames], values in {0,1}
        """
        y = torch.zeros(n_frames, dtype=torch.float32)

        # No labels available (e.g., test split): return all zeros.
        if not self.events_by_file:
            return y

        events = self.events_by_file.get(filename, [])
        if not events:
            return y

        for onset, offset, label in events:
            if label not in self.cfg.label_map:
                # Unknown labels are ignored; this keeps the pipeline robust.
                continue

            # Check overlap with the clip.
            if offset <= clip_start or onset >= clip_end:
                continue

            if self.cfg.event_boundary_mode == "drop":
                # Drop events that are not fully inside the window.
                if onset < clip_start or offset > clip_end:
                    continue
                on_in = onset
                off_in = offset
            else:
                # Clip event boundaries to window.
                on_in = max(onset, clip_start)
                off_in = min(offset, clip_end)

            # Convert to frame indices relative to this clip.
            rel_on = on_in - clip_start
            rel_off = off_in - clip_start

            i0 = self.extractor.seconds_to_frame_index(rel_on)
            i1 = self.extractor.seconds_to_frame_index(rel_off)

            # Ensure valid indexing.
            i0 = max(0, min(n_frames - 1, i0))
            i1 = max(0, min(n_frames, i1))

            if i1 > i0:
                y[i0:i1] = 1.0

        return y

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int):
        c = self.clips[idx]
        wav_path = os.path.join(self.audio_dir, c.filename)

        wav = self._load_audio_segment(wav_path, c.start_sec, c.end_sec)
        feat = self.extractor(wav)  # [n_mels, n_frames]
        n_frames = feat.shape[-1]

        y_frame = self._make_frame_labels_single_class(
            filename=c.filename,
            clip_start=c.start_sec,
            clip_end=c.end_sec,
            n_frames=n_frames,
        )

        meta = {
            "filename": c.filename,
            "clip_start": c.start_sec,
            "clip_end": c.end_sec,
            "sample_rate": self.cfg.mel_cfg.sample_rate,
        }
        return feat, y_frame, meta
