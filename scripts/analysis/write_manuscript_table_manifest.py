#!/usr/bin/env python3
"""Freeze hashes linking manuscript tables/figures to generated result sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    root = args.project_root.resolve()
    sys.path.insert(0, str(root))
    from recording_protocol import OFFICIAL_AGGREGATION, PROTOCOL_VERSION  # noqa: PLC0415

    tables = root / "outputs/tables"
    mapping: dict[str, list[Path]] = {
        "tab:main": [tables / "v2_multiseed_main_matrix.md"],
        "tab:stats": [tables / "v2_statistical_tests.csv"],
        "tab:ablation": [tables / "v2_ablation_matrix.md"],
        "tab:shortcut": [tables / "v2_shortcut_reversal_seed42.md"],
        "tab:hard": [tables / "v2_hard_sample_official.md"],
        "fig:hero": [tables / "recording_aggregation_audit.csv", tables / "v2_shortcut_reversal_seed42.md"],
        "fig:confusion": [tables / "recording_aggregation_audit.csv"],
        "fig:perturb": sorted((tables / "v2_perturbations").glob("*_seed42.json")),
        "fig:tsnecmp": [tables / "r1_tsne_compare_metrics.json"],
        "fig:framework": [
            root / "scripts/analysis/make_framework_p1_submission.py",
            root / "outputs/figures/fig_framework_p1_submission_qa.json",
            root / "paper/figures/fig_framework_p1.pdf",
        ],
    }
    items = []
    for label, sources in mapping.items():
        if not sources:
            raise RuntimeError(f"no sources resolved for {label}")
        for source in sources:
            if not source.exists():
                raise FileNotFoundError(source)
            items.append({"manuscript_label": label, "source": str(source), "sha256": sha256(source)})
    payload = {
        "recording_protocol_version": PROTOCOL_VERSION,
        "official_aggregation": OFFICIAL_AGGREGATION,
        "official_epoch": "last",
        "manuscript": str(root / "paper/main.tex"),
        "manuscript_sha256": sha256(root / "paper/main.tex"),
        "tables": items,
    }
    output = tables / "manuscript_table_sources.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"-> {output}")


if __name__ == "__main__":
    main()
