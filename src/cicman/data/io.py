"""Low-level recording readers for the three Paper 1 datasets."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat


@dataclass
class RecordingData:
    signal: np.ndarray
    speed: np.ndarray | None
    sampling_rate: int
    source_file: str
    recording_id: str


def _as_1d_float32(array: object) -> np.ndarray:
    return np.asarray(array, dtype=np.float32).reshape(-1)


def _load_paderborn_struct(mat_path: Path, recording_id: str):
    data = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    key = recording_id
    if key not in data:
        keys = [name for name in data if not name.startswith("__")]
        if len(keys) != 1:
            raise KeyError(f"Could not identify Paderborn struct in {mat_path}; keys={keys}")
        key = keys[0]
    return data[key]


def _paderborn_channel_map(struct_obj) -> dict[str, np.ndarray]:
    channels = {}
    for channel in np.atleast_1d(struct_obj.Y):
        channels[str(channel.Name)] = _as_1d_float32(channel.Data)
    return channels


def _extract_rar_member(archive_file: Path, member_name: str, temp_dir: Path) -> Path:
    if not archive_file.exists():
        raise FileNotFoundError(f"Missing Paderborn archive: {archive_file}")
    if shutil.which("bsdtar"):
        subprocess.check_call(["bsdtar", "-xf", str(archive_file), "-C", str(temp_dir), member_name])
    elif shutil.which("tar"):
        subprocess.check_call(["tar", "-xf", str(archive_file), "-C", str(temp_dir), member_name])
    elif shutil.which("7z"):
        subprocess.check_call(["7z", "x", str(archive_file), member_name, f"-o{temp_dir}", "-y"])
    else:
        raise RuntimeError("Reading Paderborn .rar files requires bsdtar, tar, or 7z on PATH.")
    extracted = temp_dir / member_name
    if not extracted.exists():
        raise FileNotFoundError(f"RAR member extraction failed: {member_name} from {archive_file}")
    return extracted


def read_recording(row: dict[str, str], *, temp_root: Path | None = None) -> RecordingData:
    dataset = row["dataset_id"]
    if dataset == "paderborn":
        return read_paderborn_recording(row, temp_root=temp_root)
    if dataset == "ottawa":
        return read_ottawa_recording(row)
    if dataset == "hust":
        return read_hust_recording(row)
    raise ValueError(f"Unsupported dataset_id: {dataset}")


def read_paderborn_recording(row: dict[str, str], *, temp_root: Path | None = None) -> RecordingData:
    archive_file = Path(row["archive_file"])
    member_name = row["source_file"]
    recording_id = row["recording_id"]
    signal_key = row["signal_key"]
    speed_key = row.get("speed_key", "")

    if temp_root is not None:
        temp_root.mkdir(parents=True, exist_ok=True)
        mat_path = _extract_rar_member(archive_file, member_name, temp_root)
        struct_obj = _load_paderborn_struct(mat_path, recording_id)
    else:
        with tempfile.TemporaryDirectory(prefix="cicman_paderborn_") as tmp:
            mat_path = _extract_rar_member(archive_file, member_name, Path(tmp))
            struct_obj = _load_paderborn_struct(mat_path, recording_id)

    channels = _paderborn_channel_map(struct_obj)
    if signal_key not in channels:
        raise KeyError(f"Missing Paderborn signal channel {signal_key}; available={sorted(channels)}")
    signal = channels[signal_key]
    speed = channels.get(speed_key) if speed_key else None
    return RecordingData(
        signal=signal,
        speed=speed,
        sampling_rate=int(row["sampling_rate"]),
        source_file=member_name,
        recording_id=recording_id,
    )


def read_ottawa_recording(row: dict[str, str]) -> RecordingData:
    mat_path = Path(row["source_file"])
    data = loadmat(mat_path)
    signal_key = row["signal_key"]
    speed_key = row.get("speed_key", "")
    if signal_key not in data:
        raise KeyError(f"Missing Ottawa signal channel {signal_key}; keys={sorted(data)}")
    signal = _as_1d_float32(data[signal_key])
    speed = _as_1d_float32(data[speed_key]) if speed_key and speed_key in data else None
    return RecordingData(
        signal=signal,
        speed=speed,
        sampling_rate=int(row["sampling_rate"]),
        source_file=str(mat_path),
        recording_id=row["recording_id"],
    )


def read_hust_recording(row: dict[str, str]) -> RecordingData:
    mat_path = Path(row["source_file"])
    data = loadmat(mat_path)
    signal_key = row["signal_key"]
    speed_key = row.get("speed_key", "")
    if signal_key not in data:
        raise KeyError(f"Missing HUST signal channel {signal_key}; keys={sorted(data)}")
    signal = _as_1d_float32(data[signal_key])
    speed = _as_1d_float32(data[speed_key]) if speed_key and speed_key in data else None
    return RecordingData(
        signal=signal,
        speed=speed,
        sampling_rate=int(row["sampling_rate"]),
        source_file=str(mat_path),
        recording_id=row["recording_id"],
    )
