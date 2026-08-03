#!/usr/bin/env python3
"""Prepare the Paper 1 CIC-MAN data workspace.

This script is intentionally conservative:
- it never writes inside data/raw;
- it copies only the three dataset directories needed for Paper 1;
- it is idempotent by default and skips existing destination folders;
- it writes an inventory report for later sanity checks.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


DATASET_SOURCES = {
    "paderborn": "Paderborn University Bearing Dataset",
    "ottawa": "v43hmbwxpm-2",
    "hust": "HUST bearing",
}

WORKSPACE_DIRS = [
    "raw_copy",
    "extracted/paderborn",
    "manifests",
    "splits",
    "processed",
    "cache/windows",
    "cache/stft",
    "cache/envelope",
    "cache/wavelet",
    "sanity_reports",
]


@dataclass(frozen=True)
class TreeStats:
    files: int
    dirs: int
    bytes: int
    suffix_counts: dict[str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
        help="Project root containing the local data directory.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Delete existing raw_copy dataset folders before copying again.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions without copying files.",
    )
    return parser.parse_args()


def safe_resolve(path: Path) -> Path:
    return path.expanduser().resolve()


def ensure_not_inside(child: Path, parent: Path, message: str) -> None:
    child = safe_resolve(child)
    parent = safe_resolve(parent)
    if child == parent or parent in child.parents:
        raise ValueError(message)


def ensure_workspace_dirs(root: Path, dry_run: bool) -> None:
    for rel in WORKSPACE_DIRS:
        path = root / rel
        if dry_run:
            print(f"[dry-run] mkdir -p {path}")
        else:
            path.mkdir(parents=True, exist_ok=True)


def copy_dataset(src: Path, dst: Path, refresh: bool, dry_run: bool) -> str:
    if not src.exists():
        raise FileNotFoundError(f"Missing source dataset: {src}")
    if dst.exists():
        if not refresh:
            return "skipped_existing"
        if dry_run:
            print(f"[dry-run] remove existing {dst}")
        else:
            shutil.rmtree(dst)
    if dry_run:
        print(f"[dry-run] copytree {src} -> {dst}")
        return "dry_run"
    shutil.copytree(src, dst)
    return "copied"


def iter_tree(path: Path) -> Iterable[Path]:
    if not path.exists():
        return []
    return path.rglob("*")


def collect_stats(path: Path) -> TreeStats:
    files = 0
    dirs = 0
    bytes_total = 0
    suffix_counts: dict[str, int] = {}
    if not path.exists():
        return TreeStats(files=0, dirs=0, bytes=0, suffix_counts={})

    for item in path.rglob("*"):
        if item.is_dir():
            dirs += 1
            continue
        if item.is_file():
            files += 1
            try:
                bytes_total += item.stat().st_size
            except OSError:
                pass
            suffix = item.suffix.lower() or "[no_ext]"
            suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1
    return TreeStats(files=files, dirs=dirs, bytes=bytes_total, suffix_counts=dict(sorted(suffix_counts.items())))


def human_bytes(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{num_bytes} B"


def write_reports(report_dir: Path, payload: dict) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = report_dir / f"workspace_prepare_{timestamp}.json"
    md_path = report_dir / f"workspace_prepare_{timestamp}.md"

    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Paper 1 CIC-MAN Workspace Prepare Report",
        "",
        f"- Created at: `{payload['created_at']}`",
        f"- Project root: `{payload['project_root']}`",
        f"- Raw data root: `{payload['raw_data_root']}`",
        f"- Paper 1 data root: `{payload['paper1_data_root']}`",
        f"- Raw protection: `{payload['raw_protection']}`",
        "",
        "## Dataset Copy Results",
        "",
        "| Dataset | Source | Destination | Status | Files | Size | Suffix Counts |",
        "|---|---|---|---|---:|---:|---|",
    ]
    for name, info in payload["datasets"].items():
        suffix_counts = ", ".join(f"{k}:{v}" for k, v in info["stats"]["suffix_counts"].items())
        lines.append(
            f"| {name} | `{info['source']}` | `{info['destination']}` | {info['status']} | "
            f"{info['stats']['files']} | {info['stats']['human_size']} | {suffix_counts} |"
        )
    lines.extend(
        [
            "",
            "## Next Required Step",
            "",
            "Generate recording-level manifests for Paderborn, Ottawa, and HUST.",
            "Do not train models until manifests, splits, and window sanity checks are complete.",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


def main() -> None:
    args = parse_args()
    project_root = safe_resolve(args.project_root)
    raw_data_root = project_root / "data" / "raw"
    paper1_data_root = project_root / "data" / "paper1_cicman"
    raw_copy_root = paper1_data_root / "raw_copy"
    report_dir = paper1_data_root / "sanity_reports"

    if not raw_data_root.exists():
        raise FileNotFoundError(f"Expected raw data root does not exist: {raw_data_root}")

    ensure_not_inside(raw_data_root, paper1_data_root, "data/raw must not be inside the writable Paper 1 workspace.")
    ensure_not_inside(raw_data_root, raw_copy_root, "data/raw must not be inside raw_copy.")

    ensure_workspace_dirs(paper1_data_root, dry_run=args.dry_run)

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(project_root),
        "raw_data_root": str(raw_data_root),
        "paper1_data_root": str(paper1_data_root),
        "raw_protection": "data/raw is source-only; all writes target data/paper1_cicman",
        "datasets": {},
    }

    for dataset_name, source_name in DATASET_SOURCES.items():
        src = raw_data_root / source_name
        dst = raw_copy_root / dataset_name
        status = copy_dataset(src=src, dst=dst, refresh=args.refresh, dry_run=args.dry_run)
        stats = collect_stats(dst)
        payload["datasets"][dataset_name] = {
            "source": str(src),
            "destination": str(dst),
            "status": status,
            "stats": {
                "files": stats.files,
                "dirs": stats.dirs,
                "bytes": stats.bytes,
                "human_size": human_bytes(stats.bytes),
                "suffix_counts": stats.suffix_counts,
            },
        }
        print(f"{dataset_name}: {status} -> {dst} ({stats.files} files, {human_bytes(stats.bytes)})")

    if not args.dry_run:
        write_reports(report_dir, payload)


if __name__ == "__main__":
    main()
