#!/usr/bin/env python3
"""Smoke-test dataset readers, resampling, and windowing on a few recordings."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def add_src_to_path() -> None:
    script_path = Path(__file__).resolve()
    project_dir = script_path.parents[2]
    sys.path.insert(0, str(project_dir / "src"))


add_src_to_path()

from cicman.data.io import read_recording  # noqa: E402
from cicman.data.windowing import make_windows, resample_signal, robust_normalize  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
        help="Project root containing data/paper1_cicman.",
    )
    parser.add_argument("--target-rate", type=int, default=25600)
    parser.add_argument("--window-length", type=int, default=4096)
    parser.add_argument("--hop-length", type=int, default=2048)
    parser.add_argument("--per-dataset", type=int, default=3)
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary Paderborn files extracted from RAR archives.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def choose_rows(manifest_dir: Path, per_dataset: int) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    for manifest_name in ["paderborn_manifest.csv", "ottawa_manifest.csv", "hust_manifest.csv"]:
        rows = read_csv(manifest_dir / manifest_name)
        by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            if row["task4_label"] != "exclude":
                by_label[row["task4_label"]].append(row)
            elif row["task3_label"] != "exclude":
                by_label[row["task3_label"]].append(row)
        for label in sorted(by_label)[:per_dataset]:
            selected.append(by_label[label][0])
    return selected


def main() -> None:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    data_root = project_root / "data" / "paper1_cicman"
    manifest_dir = data_root / "manifests"
    report_dir = data_root / "sanity_reports"
    temp_root = data_root / "processed" / "smoke_tmp"
    temp_root.mkdir(parents=True, exist_ok=True)

    rows = choose_rows(manifest_dir, args.per_dataset)
    results = []
    for row in rows:
        item = {
            "dataset_id": row["dataset_id"],
            "recording_id": row["recording_id"],
            "label": row["task4_label"] if row["task4_label"] != "exclude" else row["task3_label"],
            "source_rate": int(row["sampling_rate"]),
        }
        try:
            record_temp = temp_root / row["dataset_id"] / row["recording_id"]
            recording = read_recording(row, temp_root=record_temp if row["dataset_id"] == "paderborn" else None)
            resampled = resample_signal(recording.signal, recording.sampling_rate, args.target_rate)
            normalized = robust_normalize(resampled)
            windows = make_windows(normalized, args.window_length, args.hop_length)
            item.update(
                {
                    "status": "ok",
                    "raw_len": int(len(recording.signal)),
                    "resampled_len": int(len(resampled)),
                    "window_count": int(len(windows)),
                    "window_shape": list(windows.shape),
                    "raw_mean": float(recording.signal.mean()),
                    "raw_std": float(recording.signal.std()),
                    "norm_mean": float(normalized.mean()),
                    "norm_std": float(normalized.std()),
                    "speed_len": None if recording.speed is None else int(len(recording.speed)),
                }
            )
        except Exception as exc:  # noqa: BLE001
            item.update({"status": "error", "error": repr(exc)})
        results.append(item)
        print(item)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / f"reader_smoke_test_{timestamp}.json"
    md_path = report_dir / f"reader_smoke_test_{timestamp}.md"
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "target_rate": args.target_rate,
        "window_length": args.window_length,
        "hop_length": args.hop_length,
        "results": results,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Reader Smoke Test",
        "",
        f"- Target rate: `{args.target_rate}`",
        f"- Window length: `{args.window_length}`",
        f"- Hop length: `{args.hop_length}`",
        "",
        "| Dataset | Recording | Label | Status | Raw Len | Resampled Len | Windows | Speed Len |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for item in results:
        lines.append(
            f"| {item['dataset_id']} | {item['recording_id']} | {item['label']} | {item['status']} | "
            f"{item.get('raw_len', '')} | {item.get('resampled_len', '')} | "
            f"{item.get('window_count', '')} | {item.get('speed_len', '')} |"
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")

    errors = [item for item in results if item["status"] != "ok"]
    if errors:
        raise SystemExit(f"Smoke test failed for {len(errors)} recordings")

    if not args.keep_temp and temp_root.exists():
        shutil.rmtree(temp_root)


if __name__ == "__main__":
    main()
