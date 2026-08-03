#!/usr/bin/env python3
"""Build target-free domain-generalization splits for Paper 1."""

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

from cicman.data.splits import (  # noqa: E402
    cross_dataset_task3,
    cross_dataset_task4_ottawa_hust,
    hust_leave_bearing_type,
    ottawa_leave_speed,
    paderborn_leave_bearing,
    paderborn_leave_condition,
    read_manifest,
    write_split_result,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
        help="Project root containing data/paper1_cicman.",
    )
    parser.add_argument(
        "--paper1-data-root",
        type=Path,
        default=None,
        help="Override Paper 1 data root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    data_root = (args.paper1_data_root or (project_root / "data" / "paper1_cicman")).expanduser().resolve()
    manifest_dir = data_root / "manifests"
    split_dir = data_root / "splits"
    report_dir = data_root / "sanity_reports"

    paderborn = read_manifest(manifest_dir / "paderborn_manifest.csv")
    ottawa = read_manifest(manifest_dir / "ottawa_manifest.csv")
    hust = read_manifest(manifest_dir / "hust_manifest.csv")
    fieldnames = list(paderborn[0].keys())

    split_builders = [
        paderborn_leave_condition(paderborn),
        paderborn_leave_bearing(paderborn),
        ottawa_leave_speed(ottawa),
        hust_leave_bearing_type(hust),
        cross_dataset_task3(paderborn + ottawa + hust),
        cross_dataset_task4_ottawa_hust(ottawa, hust),
    ]

    all_summaries: dict[str, object] = {}
    total = 0
    for results in split_builders:
        for result in results:
            summary = write_split_result(result, split_dir, fieldnames)
            all_summaries[f"{result.protocol}/{result.name}"] = summary
            total += 1
            print(
                f"{result.protocol}/{result.name}: "
                f"train={len(result.train)} val={len(result.val)} test={len(result.test)}"
            )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_json = report_dir / f"split_summary_{timestamp}.json"
    summary_md = report_dir / f"split_summary_{timestamp}.md"
    summary_json.write_text(json.dumps(all_summaries, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# Split Summary", "", f"- Total split tasks: `{total}`", ""]
    by_protocol: dict[str, list[tuple[str, dict[str, object]]]] = {}
    for key, summary in all_summaries.items():
        protocol, name = key.split("/", 1)
        by_protocol.setdefault(protocol, []).append((name, summary))
    for protocol, items in sorted(by_protocol.items()):
        lines.extend([f"## {protocol}", ""])
        lines.append("| Split | Train | Val | Test | Train labels | Test labels |")
        lines.append("|---|---:|---:|---:|---|---|")
        for name, summary in items:
            sizes = summary["sizes"]
            labels = summary["labels"]
            lines.append(
                f"| {name} | {sizes['train']} | {sizes['val']} | {sizes['test']} | "
                f"`{labels['train']}` | `{labels['test']}` |"
            )
        lines.append("")
    summary_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {summary_json}")
    print(f"Wrote {summary_md}")


if __name__ == "__main__":
    main()

