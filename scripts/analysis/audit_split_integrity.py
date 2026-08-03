#!/usr/bin/env python3
"""Audit split/index integrity for the source-mixed cross-dataset protocol."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


TARGETS = ["hust", "ottawa", "paderborn"]
SPLITS = ["train", "val", "test"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def audit(project_root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    base = project_root / "data" / "paper1_cicman" / "cache" / "windows" / "cross_dataset_task3_source_mixed"
    summary = []
    overlaps = []
    recordings_by_group: dict[tuple[str, str], set[str]] = {}
    rows_by_group: dict[tuple[str, str], list[dict[str, str]]] = {}

    for target in TARGETS:
        for split in SPLITS:
            path = base / f"target_dataset_{target}" / f"{split}_windows.csv"
            rows = read_rows(path)
            rows_by_group[(target, split)] = rows
            recordings = {row["recording_id"] for row in rows}
            recordings_by_group[(target, split)] = recordings
            summary.append(
                {
                    "target": target,
                    "file_split": split,
                    "num_windows": len(rows),
                    "num_recordings": len(recordings),
                    "dataset_counts": dict(Counter(row["dataset_id"] for row in rows)),
                    "label_counts": dict(Counter(row["label"] for row in rows)),
                    "internal_split_counts": dict(Counter(row.get("split", "") for row in rows)),
                    "recording_label_counts": dict(Counter(first_label_by_recording(rows).values())),
                }
            )

    for target in TARGETS:
        for left_idx, left in enumerate(SPLITS):
            for right in SPLITS[left_idx + 1 :]:
                shared = recordings_by_group[(target, left)] & recordings_by_group[(target, right)]
                if shared:
                    sample = sorted(shared)[:10]
                    overlaps.append(
                        {
                            "target": target,
                            "left_split": left,
                            "right_split": right,
                            "num_shared_recordings": len(shared),
                            "sample_recordings": ";".join(sample),
                        }
                    )
    return summary, overlaps


def first_label_by_recording(rows: list[dict[str, str]]) -> dict[str, str]:
    labels = {}
    for row in rows:
        labels.setdefault(row["recording_id"], row["label"])
    return labels


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(summary: list[dict[str, object]], overlaps: list[dict[str, object]], path: Path) -> None:
    lines = [
        "# Split Integrity Audit",
        "",
        "Scope: `cross_dataset_task3_source_mixed` train/val/test window index files.",
        "",
        "## Summary",
        "",
        "| Target | File Split | Windows | Recordings | Dataset Counts | Label Counts | Internal Split Counts | Recording Label Counts |",
        "|---|---|---:|---:|---|---|---|---|",
    ]
    for row in summary:
        lines.append(
            f"| {row['target']} | {row['file_split']} | {row['num_windows']} | {row['num_recordings']} | "
            f"`{row['dataset_counts']}` | `{row['label_counts']}` | `{row['internal_split_counts']}` | "
            f"`{row['recording_label_counts']}` |"
        )
    lines.extend(["", "## Recording Overlaps", ""])
    if overlaps:
        lines.extend(
            [
                "| Target | Split A | Split B | Shared Recordings | Sample |",
                "|---|---|---|---:|---|",
            ]
        )
        for row in overlaps:
            lines.append(
                f"| {row['target']} | {row['left_split']} | {row['right_split']} | "
                f"{row['num_shared_recordings']} | `{row['sample_recordings']}` |"
            )
    else:
        lines.append("No recording-level overlap was found within any target protocol.")
    lines.extend(
        [
            "",
            "## Adversarial Notes",
            "",
            "- `internal_split_counts` reflects the `split` column stored inside each row. If it does not match the filename split, this may be either inherited metadata or a split-generation bug.",
            "- Recording overlap is the stronger leakage check. Any nonzero overlap between train/val/test for the same target protocol should be treated as a protocol bug.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    output_dir = (args.output_dir or project_root  / "outputs" / "tables").resolve()
    summary, overlaps = audit(project_root)
    write_csv(summary, output_dir / "split_integrity_summary.csv")
    write_csv(overlaps, output_dir / "split_integrity_overlaps.csv")
    write_markdown(summary, overlaps, output_dir / "split_integrity_audit.md")
    print(f"Wrote split integrity audit to {output_dir}")


if __name__ == "__main__":
    main()
