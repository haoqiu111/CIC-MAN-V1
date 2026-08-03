#!/usr/bin/env python3
"""Build recording-level manifests for Paderborn, Ottawa, and HUST."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


def add_src_to_path() -> None:
    script_path = Path(__file__).resolve()
    project_dir = script_path.parents[2]
    src_dir = project_dir / "src"
    sys.path.insert(0, str(src_dir))


add_src_to_path()

from cicman.data.manifest import (  # noqa: E402
    build_hust_manifest,
    build_ottawa_manifest,
    build_paderborn_manifest,
    summarize_rows,
    write_manifest,
    write_summary_markdown,
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
    raw_copy = data_root / "raw_copy"
    manifest_dir = data_root / "manifests"
    report_dir = data_root / "sanity_reports"

    builders = {
        "paderborn": (build_paderborn_manifest, raw_copy / "paderborn"),
        "ottawa": (build_ottawa_manifest, raw_copy / "ottawa"),
        "hust": (build_hust_manifest, raw_copy / "hust"),
    }

    all_summaries: dict[str, object] = {}
    for dataset, (builder, root) in builders.items():
        if not root.exists():
            raise FileNotFoundError(f"Missing raw_copy dataset folder for {dataset}: {root}")
        rows = builder(root)
        output_path = manifest_dir / f"{dataset}_manifest.csv"
        write_manifest(rows, output_path)
        stats = summarize_rows(rows)
        all_summaries[dataset] = stats
        print(f"{dataset}: wrote {len(rows)} rows -> {output_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_json = report_dir / f"manifest_summary_{timestamp}.json"
    summary_md = report_dir / f"manifest_summary_{timestamp}.md"
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(all_summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary_markdown(all_summaries, summary_md)
    print(f"Wrote {summary_json}")
    print(f"Wrote {summary_md}")


if __name__ == "__main__":
    main()

