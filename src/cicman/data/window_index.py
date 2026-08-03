"""Window-level index builders.

The index stores metadata only. It does not cache signal arrays, so it remains
small and portable between macOS development and the Windows GPU workstation.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from scipy.io import loadmat

from cicman.data.windowing import estimate_window_count


WINDOW_INDEX_COLUMNS = [
    "dataset_id",
    "recording_id",
    "split",
    "source_file",
    "archive_file",
    "label_column",
    "label",
    "label_id",
    "fault_code",
    "bearing_id",
    "bearing_type_id",
    "condition_id",
    "speed_profile_id",
    "trial_id",
    "measurement_id",
    "sensor_modality",
    "signal_key",
    "speed_key",
    "source_sampling_rate",
    "target_sampling_rate",
    "source_start",
    "source_end",
    "target_start",
    "target_end",
    "window_length",
    "hop_length",
    "window_index",
    "recording_window_count",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(rows: list[dict[str, object]], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def estimate_num_samples(row: dict[str, str]) -> int:
    value = row.get("num_samples", "")
    try:
        num_samples = int(float(value))
    except (TypeError, ValueError):
        num_samples = 0
    if num_samples > 0:
        return num_samples

    # Fallbacks based on the dataset facts established in the data analysis.
    dataset_id = row["dataset_id"]
    if dataset_id == "paderborn":
        return 256000
    if dataset_id == "ottawa":
        return 2_000_000
    if dataset_id == "hust":
        mat_path = Path(row["source_file"])
        if mat_path.exists():
            data = loadmat(mat_path, variable_names=[row["signal_key"]])
            if row["signal_key"] in data:
                return int(data[row["signal_key"]].size)
        # Fallback only for path-inspection failures. HUST has variable lengths,
        # so full generation should normally use the exact length above.
        return 512000
    raise ValueError(f"Cannot estimate num_samples for dataset_id={dataset_id}")


def build_window_rows(
    split_rows: list[dict[str, str]],
    *,
    label_column: str,
    target_sampling_rate: int,
    window_length: int,
    hop_length: int,
    max_recordings: int | None = None,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    rows = split_rows[:max_recordings] if max_recordings else split_rows
    label_id_column = f"{label_column}_id"

    for row in rows:
        label = row[label_column]
        if label == "exclude":
            continue
        source_rate = int(row["sampling_rate"])
        source_samples = estimate_num_samples(row)
        count = estimate_window_count(source_samples, source_rate, target_sampling_rate, window_length, hop_length)
        ratio = source_rate / target_sampling_rate
        for idx in range(count):
            target_start = idx * hop_length
            target_end = target_start + window_length
            source_start = int(round(target_start * ratio))
            source_end = int(round(target_end * ratio))
            output.append(
                {
                    "dataset_id": row["dataset_id"],
                    "recording_id": row["recording_id"],
                    "split": row.get("split", ""),
                    "source_file": row["source_file"],
                    "archive_file": row.get("archive_file", ""),
                    "label_column": label_column,
                    "label": label,
                    "label_id": row[label_id_column],
                    "fault_code": row.get("fault_code", ""),
                    "bearing_id": row.get("bearing_id", ""),
                    "bearing_type_id": row.get("bearing_type_id", ""),
                    "condition_id": row.get("condition_id", ""),
                    "speed_profile_id": row.get("speed_profile_id", ""),
                    "trial_id": row.get("trial_id", ""),
                    "measurement_id": row.get("measurement_id", ""),
                    "sensor_modality": row.get("sensor_modality", "vibration"),
                    "signal_key": row["signal_key"],
                    "speed_key": row.get("speed_key", ""),
                    "source_sampling_rate": source_rate,
                    "target_sampling_rate": target_sampling_rate,
                    "source_start": source_start,
                    "source_end": source_end,
                    "target_start": target_start,
                    "target_end": target_end,
                    "window_length": window_length,
                    "hop_length": hop_length,
                    "window_index": idx,
                    "recording_window_count": count,
                }
            )
    return output


def summarize_window_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "windows": len(rows),
        "recordings": len({row["recording_id"] for row in rows}),
        "datasets": dict(sorted(Counter(str(row["dataset_id"]) for row in rows).items())),
        "labels": dict(sorted(Counter(str(row["label"]) for row in rows).items())),
        "splits": dict(sorted(Counter(str(row["split"]) for row in rows).items())),
        "window_length_values": sorted({int(row["window_length"]) for row in rows}),
        "target_sampling_rates": sorted({int(row["target_sampling_rate"]) for row in rows}),
    }


def write_summary(summary: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
