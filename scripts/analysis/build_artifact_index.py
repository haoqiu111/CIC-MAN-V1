#!/usr/bin/env python3
"""Inventory all experiment result artifacts into outputs/ARTIFACT_INDEX.md."""

from __future__ import annotations

import argparse
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()

    out_root = args.project_root / "outputs"
    sections = {
        "tables": ["*.md", "*.csv", "*.json"],
        "figures": ["*.png", "*.pdf", "*.svg"],
        "logs": ["*.log", "*.err"],
    }

    lines = [f"# Experiment Artifact Index (generated {time.strftime('%Y-%m-%d %H:%M')})", ""]
    total = 0
    for section, patterns in sections.items():
        base = out_root / section
        files = sorted({f for pat in patterns for f in base.rglob(pat)}) if base.exists() else []
        lines.append(f"## {section}/ ({len(files)} files)")
        lines.append("")
        for f in files:
            rel = f.relative_to(out_root)
            lines.append(f"- `{rel}` ({f.stat().st_size:,} B, {time.strftime('%Y-%m-%d %H:%M', time.localtime(f.stat().st_mtime))})")
        lines.append("")
        total += len(files)

    ckpt_root = out_root / "checkpoints"
    metrics = sorted(ckpt_root.rglob("metrics.json")) + sorted(ckpt_root.rglob("shortcut_metrics.json")) + sorted(ckpt_root.rglob("view_reliability.json"))
    weights = sorted(ckpt_root.rglob("*.pt"))
    lines.append(f"## checkpoints/ ({len(metrics)} metrics files, {len(weights)} weight files)")
    lines.append("")
    for f in metrics:
        lines.append(f"- `{f.relative_to(out_root)}`")
    total += len(metrics)

    index = out_root / "ARTIFACT_INDEX.md"
    index.write_text("\n".join(lines), encoding="utf-8")
    print(f"indexed {total} result files + {len(weights)} weight files -> {index}")


if __name__ == "__main__":
    main()
