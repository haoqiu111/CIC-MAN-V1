#!/usr/bin/env python3
"""Build window-level indexes from split CSV files."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


def add_src_to_path() -> None:
    script_path = Path(__file__).resolve()
    project_dir = script_path.parents[2]
    sys.path.insert(0, str(project_dir / "src"))


add_src_to_path()

from cicman.data.window_index import (  # noqa: E402
    WINDOW_INDEX_COLUMNS,
    build_window_rows,
    read_csv,
    summarize_window_rows,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--protocol", required=True, help="Split protocol directory name.")
    parser.add_argument("--task", required=True, help="Split task directory name.")
    parser.add_argument("--target-rate", type=int, default=25600)
    parser.add_argument("--window-length", type=int, default=4096)
    parser.add_argument("--hop-length", type=int, default=2048)
    parser.add_argument(
        "--max-recordings-per-part",
        type=int,
        default=None,
        help="Optional smoke-test limit per train/val/test split.",
    )
    return parser.parse_args()


def infer_label_column(rows: list[dict[str, str]]) -> str:
    values = {row.get("label_column", "") for row in rows if row.get("label_column")}
    if len(values) == 1:
        return next(iter(values))
    # Split CSVs currently store this column only indirectly; infer from labels.
    has_task3 = any(row.get("task3_label", "exclude") != "exclude" for row in rows)
    has_task4_rolling = any(row.get("task4_label", "exclude") == "rolling" for row in rows)
    if has_task3 and not has_task4_rolling:
        return "task3_label"
    if any(row.get("task4_label", "exclude") != "exclude" for row in rows):
        return "task4_label"
    return "task3_label"


def main() -> None:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    data_root = project_root / "data" / "paper1_cicman"
    split_task_dir = data_root / "splits" / args.protocol / args.task
    output_dir = data_root / "cache" / "windows" / args.protocol / args.task
    output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = {}
    for part in ["train", "val", "test"]:
        split_csv = split_task_dir / f"{part}.csv"
        if not split_csv.exists():
            raise FileNotFoundError(f"Missing split file: {split_csv}")
        rows = read_csv(split_csv)
        for row in rows:
            row["split"] = part
        label_column = infer_label_column(rows)
        window_rows = build_window_rows(
            rows,
            label_column=label_column,
            target_sampling_rate=args.target_rate,
            window_length=args.window_length,
            hop_length=args.hop_length,
            max_recordings=args.max_recordings_per_part,
        )
        out_csv = output_dir / f"{part}_windows.csv"
        write_csv(window_rows, out_csv, WINDOW_INDEX_COLUMNS)
        summary = summarize_window_rows(window_rows)
        all_summaries[part] = summary
        print(f"{part}: wrote {len(window_rows)} windows -> {out_csv}")

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "protocol": args.protocol,
        "task": args.task,
        "target_rate": args.target_rate,
        "window_length": args.window_length,
        "hop_length": args.hop_length,
        "max_recordings_per_part": args.max_recordings_per_part,
        "parts": all_summaries,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
