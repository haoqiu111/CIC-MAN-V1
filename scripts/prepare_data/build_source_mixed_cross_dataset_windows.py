#!/usr/bin/env python3
"""Build source-mixed target-free cross-dataset window indexes.

The original cross-dataset_task3 indexes hold out one source dataset for
validation. This script keeps the target dataset untouched, but re-splits the
source windows at recording level so every source dataset can contribute to
training and source-only validation.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--source-protocol", default="cross_dataset_task3")
    parser.add_argument("--output-protocol", default="cross_dataset_task3_source_mixed")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(rows: list[dict[str, str]], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def split_source_rows(rows: list[dict[str, str]], *, val_fraction: float, seed: int) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    groups: dict[tuple[str, str], set[str]] = defaultdict(set)
    by_recording: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        recording_key = f"{row['dataset_id']}::{row['recording_id']}"
        groups[(row["dataset_id"], row["label"])].add(recording_key)
        by_recording[recording_key].append(row)

    rng = random.Random(seed)
    val_recordings: set[str] = set()
    for group_key, recording_keys in sorted(groups.items()):
        shuffled = sorted(recording_keys)
        rng.shuffle(shuffled)
        n_val = max(1, int(round(len(shuffled) * val_fraction))) if len(shuffled) > 1 else 1
        n_val = min(n_val, max(1, len(shuffled) - 1)) if len(shuffled) > 1 else 1
        val_recordings.update(shuffled[:n_val])

    train_rows: list[dict[str, str]] = []
    val_rows: list[dict[str, str]] = []
    for recording_key, recording_rows in by_recording.items():
        if recording_key in val_recordings:
            val_rows.extend(recording_rows)
        else:
            train_rows.extend(recording_rows)
    return train_rows, val_rows


def summarize(rows: list[dict[str, str]]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {
        "dataset_id": defaultdict(int),
        "label": defaultdict(int),
        "recordings": defaultdict(int),
    }
    recording_seen: set[tuple[str, str]] = set()
    for row in rows:
        counts["dataset_id"][row["dataset_id"]] += 1
        counts["label"][row["label"]] += 1
        key = (row["dataset_id"], row["recording_id"])
        if key not in recording_seen:
            counts["recordings"][row["dataset_id"]] += 1
            recording_seen.add(key)
    return {key: dict(sorted(value.items())) for key, value in counts.items()}


def write_summary(path: Path, target: str, train_rows: list[dict[str, str]], val_rows: list[dict[str, str]], test_rows: list[dict[str, str]]) -> None:
    lines = [
        f"# Source-Mixed Cross-Dataset Windows: {target}",
        "",
        "Target test windows are copied unchanged from the original target-free split.",
        "Train/validation windows are re-split from source-domain recordings only.",
        "",
    ]
    for name, rows in [("train", train_rows), ("val", val_rows), ("test", test_rows)]:
        lines.extend(
            [
                f"## {name}",
                "",
                "```json",
                __import__("json").dumps(summarize(rows), ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    windows_root = project_root / "data" / "paper1_cicman" / "cache" / "windows"
    source_root = windows_root / args.source_protocol
    output_root = windows_root / args.output_protocol

    for target_dir in sorted(source_root.glob("target_dataset_*")):
        source_rows = read_rows(target_dir / "train_windows.csv") + read_rows(target_dir / "val_windows.csv")
        test_rows = read_rows(target_dir / "test_windows.csv")
        if not source_rows:
            continue
        fieldnames = list(source_rows[0].keys())
        train_rows, val_rows = split_source_rows(source_rows, val_fraction=args.val_fraction, seed=args.seed)
        out_dir = output_root / target_dir.name
        write_rows(train_rows, out_dir / "train_windows.csv", fieldnames)
        write_rows(val_rows, out_dir / "val_windows.csv", fieldnames)
        write_rows(test_rows, out_dir / "test_windows.csv", fieldnames)
        write_summary(out_dir / "summary.md", target_dir.name, train_rows, val_rows, test_rows)
        print(
            f"{target_dir.name}: train={len(train_rows)} val={len(val_rows)} test={len(test_rows)} -> {out_dir}"
        )


if __name__ == "__main__":
    main()

