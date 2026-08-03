#!/usr/bin/env python3
"""Summarize repeated-seed mechanism-guarded inner aggregation audits."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np


TARGETS = ["hust", "ottawa", "paderborn"]
SEEDS = [42, 2025, 2026]
METRICS = [
    "target_macro_f1_posthoc",
    "target_normal_recall_posthoc",
    "target_inner_recall_posthoc",
    "target_inner_precision_posthoc",
    "target_outer_recall_posthoc",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("outputs/checkpoints"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tables"))
    return parser.parse_args()


def run_dir(root: Path, target: str, seed: int) -> Path:
    if target == "ottawa" and seed == 42:
        return root / "ottawa_mechanism_guarded_inner_aggregator_seed42"
    return root / f"mechanism_guarded_inner_aggregator_target_{target}_seed{seed}"


def parse_baseline(report: Path) -> dict[str, object]:
    text = report.read_text(encoding="utf-8")
    match = re.search(r"^\| baseline:baseline@inf \| (?P<body>.+?) \|$", text, flags=re.MULTILINE)
    if not match:
        raise ValueError(f"Could not parse baseline row from {report}")
    cells = [cell.strip() for cell in match.group("body").split("|")]
    return {
        "source_macro_f1": float(cells[0]),
        "source_normal_recall": float(cells[1]),
        "source_inner_recall": float(cells[2]),
        "source_inner_precision": float(cells[3]),
        "source_outer_recall": float(cells[4]),
        "source_domain_min_outer_recall": float(cells[5]),
        "target_macro_f1_posthoc": float(cells[6]),
        "target_normal_recall_posthoc": float(cells[7]),
        "target_inner_recall_posthoc": float(cells[8]),
        "target_inner_precision_posthoc": float(cells[9]),
        "target_outer_recall_posthoc": float(cells[10]),
        "target_confusion_posthoc": cells[11].strip("`"),
    }


def load_rows(root: Path) -> list[dict[str, object]]:
    rows = []
    for target in TARGETS:
        for seed in SEEDS:
            directory = run_dir(root, target, seed)
            selected_path = directory / "mechanism_guarded_inner_selected.json"
            report_path = directory / "mechanism_guarded_inner_aggregator_audit.md"
            if not selected_path.exists() or not report_path.exists():
                rows.append({"target": target, "seed": seed, "status": "missing", "path": str(directory)})
                continue
            selected = json.loads(selected_path.read_text(encoding="utf-8"))
            baseline = parse_baseline(report_path)
            row = {
                "target": target,
                "seed": seed,
                "status": "ok",
                "selected_rule": selected["rule"],
                "selected_threshold": selected["threshold"],
                "source_macro_f1": selected["source_macro_f1"],
                "source_outer_recall": selected["source_outer_recall"],
                "source_domain_min_outer_recall": selected["source_domain_min_outer_recall"],
            }
            for metric in METRICS:
                row[f"baseline_{metric}"] = baseline[metric]
                row[f"selected_{metric}"] = selected[metric]
                row[f"delta_{metric}"] = float(selected[metric]) - float(baseline[metric])
            row["selected_confusion"] = selected["target_confusion_posthoc"]
            row["baseline_confusion"] = baseline["target_confusion_posthoc"]
            row["outer_harm_flag"] = float(row["delta_target_outer_recall_posthoc"]) < -1e-6
            rows.append(row)
    return rows


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values: list[float]) -> str:
    if not values:
        return "n/a"
    return f"{float(np.mean(values)):.6f} +/- {float(np.std(values)):.6f}"


def write_markdown(rows: list[dict[str, object]], path: Path) -> None:
    lines = [
        "# Mechanism-Guarded Recording Inner Aggregator Repeated-Seed Audit",
        "",
        "Scope: source-only mechanism guard over v5 recording decisions. Target labels are post-hoc only.",
        "",
        "| Target | Seed | Rule | Base F1 | Sel F1 | Delta F1 | Base Inner R | Sel Inner R | Delta Inner R | Base Outer R | Sel Outer R | Delta Outer R | Outer Harm |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        if row.get("status") != "ok":
            lines.append(f"| {row['target']} | {row['seed']} | missing |  |  |  |  |  |  |  |  |  | yes |")
            continue
        lines.append(
            f"| {row['target']} | {row['seed']} | {row['selected_rule']} | "
            f"{float(row['baseline_target_macro_f1_posthoc']):.6f} | {float(row['selected_target_macro_f1_posthoc']):.6f} | {float(row['delta_target_macro_f1_posthoc']):.6f} | "
            f"{float(row['baseline_target_inner_recall_posthoc']):.6f} | {float(row['selected_target_inner_recall_posthoc']):.6f} | {float(row['delta_target_inner_recall_posthoc']):.6f} | "
            f"{float(row['baseline_target_outer_recall_posthoc']):.6f} | {float(row['selected_target_outer_recall_posthoc']):.6f} | {float(row['delta_target_outer_recall_posthoc']):.6f} | "
            f"{row['outer_harm_flag']} |"
        )
    lines.extend(["", "## Mean +/- Std", ""])
    lines.append("| Target | Base F1 | Sel F1 | Delta F1 | Base Inner R | Sel Inner R | Delta Inner R | Base Outer R | Sel Outer R | Delta Outer R |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for target in TARGETS:
        target_rows = [row for row in rows if row.get("status") == "ok" and row["target"] == target]
        lines.append(
            f"| {target} | "
            f"{mean_std([float(row['baseline_target_macro_f1_posthoc']) for row in target_rows])} | "
            f"{mean_std([float(row['selected_target_macro_f1_posthoc']) for row in target_rows])} | "
            f"{mean_std([float(row['delta_target_macro_f1_posthoc']) for row in target_rows])} | "
            f"{mean_std([float(row['baseline_target_inner_recall_posthoc']) for row in target_rows])} | "
            f"{mean_std([float(row['selected_target_inner_recall_posthoc']) for row in target_rows])} | "
            f"{mean_std([float(row['delta_target_inner_recall_posthoc']) for row in target_rows])} | "
            f"{mean_std([float(row['baseline_target_outer_recall_posthoc']) for row in target_rows])} | "
            f"{mean_std([float(row['selected_target_outer_recall_posthoc']) for row in target_rows])} | "
            f"{mean_std([float(row['delta_target_outer_recall_posthoc']) for row in target_rows])} |"
        )
    lines.extend(
        [
            "",
            "## Adversarial Verdict",
            "",
            "- Ottawa shows consistent positive inner-recall improvement without target outer harm, but effect size is seed-dependent.",
            "- HUST shows inner-recall movement, but target outer remains collapsed; this guard should not be claimed as solving HUST.",
            "- Paderborn usually falls back to baseline, supporting the source-coverage limitation diagnosis.",
            "- This component is promising for CIC-MAN-vFinal as an optional recording-level intervention, but it needs final-selector integration rather than unconditional promotion.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = load_rows(args.checkpoint_root)
    write_csv(rows, args.output_dir / "mechanism_guarded_inner_repeated_seed_detail.csv")
    write_markdown(rows, args.output_dir / "mechanism_guarded_inner_repeated_seed_audit.md")
    print(f"Wrote {args.output_dir / 'mechanism_guarded_inner_repeated_seed_audit.md'}")


if __name__ == "__main__":
    main()
