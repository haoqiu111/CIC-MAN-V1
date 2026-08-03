"""PyTorch Dataset for CIC-MAN window indexes."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import numpy as np

from cicman.data.io import read_recording
from cicman.data.windowing import resample_signal, robust_normalize


class WindowIndexDataset:
    """Read signal windows on demand from a window-level CSV index.

    Torch is imported lazily in ``__getitem__`` so data preparation scripts can
    still run in environments where PyTorch is not installed.
    """

    def __init__(self, index_csv: str | Path, normalize: bool = True):
        self.index_csv = Path(index_csv)
        self.normalize = normalize
        with self.index_csv.open(newline="", encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))
        self._recording_cache: dict[str, np.ndarray] = {}
        self._required_lengths: dict[str, int] = {}
        for row in self.rows:
            cache_key = self._cache_key(row)
            self._required_lengths[cache_key] = max(self._required_lengths.get(cache_key, 0), int(row["target_end"]))

    @staticmethod
    def _cache_key(row: dict[str, str]) -> str:
        return f"{row['dataset_id']}::{row['recording_id']}::{row['target_sampling_rate']}"

    def __len__(self) -> int:
        return len(self.rows)

    def _load_resampled_recording(self, row: dict[str, str]) -> np.ndarray:
        cache_key = self._cache_key(row)
        cached = self._recording_cache.get(cache_key)
        if cached is not None:
            return cached

        manifest_like = {
            "dataset_id": row["dataset_id"],
            "source_file": row["source_file"],
            "archive_file": row.get("archive_file", ""),
            "recording_id": row["recording_id"],
            "signal_key": row["signal_key"],
            "speed_key": row.get("speed_key", ""),
            "sampling_rate": row["source_sampling_rate"],
        }
        with tempfile.TemporaryDirectory(prefix="cicman_dataset_") as tmp:
            recording = read_recording(manifest_like, temp_root=Path(tmp) if row["dataset_id"] == "paderborn" else None)
        signal = resample_signal(recording.signal, recording.sampling_rate, int(row["target_sampling_rate"]))
        if self.normalize:
            signal = robust_normalize(signal)
        required_length = self._required_lengths.get(cache_key, 0)
        if required_length > len(signal):
            pad_width = required_length - len(signal)
            mode = "edge" if len(signal) else "constant"
            signal = np.pad(signal, (0, pad_width), mode=mode).astype(np.float32)
        self._recording_cache[cache_key] = signal
        return signal

    def __getitem__(self, index: int):
        import torch

        row = self.rows[index]
        signal = self._load_resampled_recording(row)
        start = int(row["target_start"])
        end = int(row["target_end"])
        window = signal[start:end].astype(np.float32)
        if len(window) != int(row["window_length"]):
            raise RuntimeError(f"Window length mismatch for row {index}: got {len(window)}")
        x = torch.from_numpy(window).unsqueeze(0)
        y = torch.tensor(int(row["label_id"]), dtype=torch.long)
        return {
            "x": x,
            "y": y,
            "metadata": row,
        }
