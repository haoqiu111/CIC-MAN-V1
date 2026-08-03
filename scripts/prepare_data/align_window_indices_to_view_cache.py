#!/usr/bin/env python3
"""Remove split-index rows that do not exist in the canonical view cache.

The loader historically skipped such rows silently.  This script makes the
effective dataset explicit and reproducible while preserving one backup of
every changed CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = ap.parse_args()
    root = args.project_root.resolve()
    master_path = root / "data/paper1_cicman/cache/views_v2/master.csv"
    windows_root = root / "data/paper1_cicman/cache/windows"
    keys = {
        (row["dataset_id"], row["recording_id"], int(row["window_index"]))
        for row in csv.DictReader(master_path.open(newline="", encoding="utf-8"))
    }
    report = []
    for path in sorted(windows_root.rglob("*_windows.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            rows = list(reader)
        kept = [
            row for row in rows
            if (row["dataset_id"], row["recording_id"], int(row["window_index"])) in keys
        ]
        removed = len(rows) - len(kept)
        if not removed:
            continue
        backup = path.with_suffix(path.suffix + ".pre_cache_alignment")
        if not backup.exists():
            backup.write_bytes(path.read_bytes())
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(kept)
        temporary.replace(path)
        report.append({"path": str(path), "rows_before": len(rows), "rows_after": len(kept), "removed": removed})
        print(f"aligned {path}: removed {removed}")
    report_path = root / "outputs/tables/window_cache_alignment.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {report_path}; changed files={len(report)}; removed rows={sum(r['removed'] for r in report)}")


if __name__ == "__main__":
    main()
