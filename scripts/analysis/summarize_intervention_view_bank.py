#!/usr/bin/env python3
"""Summarize complete intervention view-bank diagnostics across target tasks."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_rows(input_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(input_dir.glob("intervention_view_bank_summary_*.csv")):
        if "smoke" in path.name:
            continue
        with path.open(newline="", encoding="utf-8") as f:
            rows.extend(dict(row) for row in csv.DictReader(f))
    rows.sort(key=lambda row: (str(row["target_name"]), -float(row["recommendation_score"])))
    return rows


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row["view"])].append(row)
    output = []
    for view, values in sorted(groups.items()):
        output.append(
            {
                "view": view,
                "mean_health_macro_f1": sum(float(row["health_macro_f1"]) for row in values) / len(values),
                "mean_domain_leakage_over_majority": sum(
                    float(row["domain_leakage_over_majority"]) for row in values
                )
                / len(values),
                "mean_fault_mechanism_fidelity_fisher": sum(
                    float(row["fault_mechanism_fidelity_fisher"]) for row in values
                )
                / len(values),
                "mean_fault_mechanism_fidelity": sum(
                    float(row["fault_mechanism_fidelity_mean"]) for row in values
                )
                / len(values),
                "mean_recommendation_score": sum(float(row["recommendation_score"]) for row in values)
                / len(values),
                "num_targets": len(values),
            }
        )
    output.sort(key=lambda row: float(row["mean_recommendation_score"]), reverse=True)
    return output


def best_by_target(rows: list[dict[str, object]], top_k: int = 3) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row["target_name"])].append(row)
    output = []
    for target, values in sorted(groups.items()):
        ordered = sorted(values, key=lambda row: float(row["recommendation_score"]), reverse=True)
        for rank, row in enumerate(ordered[:top_k], start=1):
            output.append(
                {
                    "target_name": target,
                    "rank": rank,
                    "view": row["view"],
                    "health_macro_f1": row["health_macro_f1"],
                    "domain_leakage_over_majority": row["domain_leakage_over_majority"],
                    "fault_mechanism_fidelity_fisher": row["fault_mechanism_fidelity_fisher"],
                    "recommendation_score": row["recommendation_score"],
                }
            )
    return output


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object) -> str:
    return f"{float(value):.6f}"


def write_markdown(rows: list[dict[str, object]], mean_rows: list[dict[str, object]], top_rows: list[dict[str, object]], path: Path) -> None:
    lines = [
        "# Complete Intervention View-Bank Diagnosis",
        "",
        "Source-only diagnosis over the target-free source-mixed cross-dataset protocols. Higher health Macro-F1 and recommendation score are better; lower domain leakage is better.",
        "",
        "## Top Views Per Target",
        "",
        "| Target | Rank | View | Health F1 | Domain Leakage | Fidelity Fisher | Recommendation |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in top_rows:
        lines.append(
            f"| {row['target_name']} | {row['rank']} | {row['view']} | "
            f"{fmt(row['health_macro_f1'])} | {fmt(row['domain_leakage_over_majority'])} | "
            f"{fmt(row['fault_mechanism_fidelity_fisher'])} | {fmt(row['recommendation_score'])} |"
        )
    lines.extend(
        [
            "",
            "## Mean Across Targets",
            "",
            "| View | Mean Health F1 | Mean Domain Leakage | Mean Fidelity Fisher | Mean Fidelity | Mean Recommendation |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in mean_rows:
        lines.append(
            f"| {row['view']} | {fmt(row['mean_health_macro_f1'])} | "
            f"{fmt(row['mean_domain_leakage_over_majority'])} | "
            f"{fmt(row['mean_fault_mechanism_fidelity_fisher'])} | "
            f"{fmt(row['mean_fault_mechanism_fidelity'])} | "
            f"{fmt(row['mean_recommendation_score'])} |"
        )
    lines.extend(
        [
            "",
            "## Full Per-Target Table",
            "",
            "| Target | View | Health F1 | Domain Acc. | Domain Majority | Leakage | Fidelity Fisher | Fidelity Mean | Recommendation |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['target_name']} | {row['view']} | {fmt(row['health_macro_f1'])} | "
            f"{fmt(row['domain_accuracy'])} | {fmt(row['domain_majority_baseline'])} | "
            f"{fmt(row['domain_leakage_over_majority'])} | "
            f"{fmt(row['fault_mechanism_fidelity_fisher'])} | "
            f"{fmt(row['fault_mechanism_fidelity_mean'])} | "
            f"{fmt(row['recommendation_score'])} |"
        )
    lines.extend(
        [
            "",
            "Interpretation: these are source-validation diagnostics. They should guide which intervention agents to implement in CIC-MAN-vFinal, but target test labels are not used for selecting views.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = read_rows(args.input_dir)
    mean_rows = aggregate(rows)
    top_rows = best_by_target(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.output_dir / "intervention_view_bank_detail.csv")
    write_csv(mean_rows, args.output_dir / "intervention_view_bank_mean.csv")
    write_csv(top_rows, args.output_dir / "intervention_view_bank_top_views.csv")
    write_markdown(rows, mean_rows, top_rows, args.output_dir / "intervention_view_bank_summary.md")
    print(f"Wrote {len(rows)} view rows to {args.output_dir / 'intervention_view_bank_summary.md'}")


if __name__ == "__main__":
    main()
