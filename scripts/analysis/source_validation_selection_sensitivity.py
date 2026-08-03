#!/usr/bin/env python3
"""Sensitivity audit for target-free source-validation intervention selection."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def add_script_dir_to_path() -> None:
    script_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(script_dir))


add_script_dir_to_path()

from source_validation_intervention_selection import (  # noqa: E402
    TARGETS,
    add_intervention_eligibility,
    add_selection_flags,
    collect_rows,
)


MARGINS = [0.0, 0.001, 0.0025, 0.005, 0.01, 0.02]
COST_WEIGHTS = [0.0, 0.01, 0.02, 0.05]
ROUTER_WEIGHT = 0.02
SELECTION_METRICS = ["worst_source_val_macro_f1", "mean_source_val_macro_f1", "source_val_macro_f1"]
BASELINE_METHOD = "CIC-MAN-v5-class-router-style"
CALIBRATED_METHOD = "CIC-MAN-gated-filterbank-calibrated"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def index_rows(rows: list[dict[str, object]]) -> dict[tuple[str, str], dict[str, object]]:
    return {(str(row["target"]), str(row["method"])): row for row in rows}


def collect_sensitivity_rows(project_dir: Path) -> list[dict[str, object]]:
    output_rows: list[dict[str, object]] = []
    for selection_metric in SELECTION_METRICS:
        for cost_weight in COST_WEIGHTS:
            base_rows = collect_rows(project_dir, selection_metric, cost_weight, ROUTER_WEIGHT)
            for margin in MARGINS:
                rows = [dict(row) for row in base_rows]
                add_intervention_eligibility(rows, selection_metric, margin)
                add_selection_flags(rows)
                row_index = index_rows(rows)
                for target in TARGETS:
                    selected = next(row for row in rows if row["target"] == target and row["selected_by_source_validation"])
                    baseline = row_index.get((target, BASELINE_METHOD), {})
                    calibrated = row_index.get((target, CALIBRATED_METHOD), {})
                    selected_target_f1 = float(selected.get("target_window_macro_f1_posthoc", 0.0))
                    baseline_target_f1 = float(baseline.get("target_window_macro_f1_posthoc", 0.0) or 0.0)
                    calibrated_target_f1 = float(calibrated.get("target_window_macro_f1_posthoc", 0.0) or 0.0)
                    output_rows.append(
                        {
                            "selection_metric": selection_metric,
                            "cost_weight": cost_weight,
                            "router_weight": ROUTER_WEIGHT,
                            "intervention_margin": margin,
                            "target": target,
                            "selected_method": selected["method"],
                            "selected_score": float(selected["selection_score"]),
                            "selected_target_window_macro_f1_posthoc": selected_target_f1,
                            "baseline_target_window_macro_f1_posthoc": baseline_target_f1,
                            "calibrated_target_window_macro_f1_posthoc": calibrated_target_f1,
                            "selected_minus_baseline_target_f1_posthoc": selected_target_f1 - baseline_target_f1,
                            "calibrated_minus_baseline_target_f1_posthoc": calibrated_target_f1 - baseline_target_f1,
                            "calibrated_passes_margin": bool(calibrated.get("passes_intervention_margin", False)),
                            "calibrated_selection_score": float(calibrated.get("selection_score", 0.0) or 0.0),
                            "calibrated_source_metric": float(calibrated.get(selection_metric, 0.0) or 0.0),
                            "baseline_source_metric": float(baseline.get(selection_metric, 0.0) or 0.0),
                        }
                    )
    return output_rows


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summaries = []
    keys = sorted(
        {
            (row["selection_metric"], row["cost_weight"], row["intervention_margin"])
            for row in rows
        }
    )
    for selection_metric, cost_weight, margin in keys:
        subset = [
            row
            for row in rows
            if row["selection_metric"] == selection_metric
            and row["cost_weight"] == cost_weight
            and row["intervention_margin"] == margin
        ]
        selected_methods = [str(row["selected_method"]) for row in subset]
        summaries.append(
            {
                "selection_metric": selection_metric,
                "cost_weight": cost_weight,
                "intervention_margin": margin,
                "selected_calibrated_count": sum(method == CALIBRATED_METHOD for method in selected_methods),
                "selected_v5_count": sum(method == BASELINE_METHOD for method in selected_methods),
                "mean_selected_minus_baseline_target_f1_posthoc": sum(
                    float(row["selected_minus_baseline_target_f1_posthoc"]) for row in subset
                )
                / max(len(subset), 1),
                "mean_calibrated_minus_baseline_target_f1_posthoc": sum(
                    float(row["calibrated_minus_baseline_target_f1_posthoc"]) for row in subset
                )
                / max(len(subset), 1),
                "selected_methods": ", ".join(f"{row['target']}={row['selected_method']}" for row in subset),
            }
        )
    return summaries


def collect_calibrated_vs_v5_rows(project_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    noninferiority_tolerances = [0.0, 0.001, 0.0025, 0.005, 0.01]
    min_target_rates = [0.0, 0.001, 0.005, 0.01, 0.02]
    base_rows = collect_rows(project_dir, "worst_source_val_macro_f1", 0.0, 0.0)
    row_index = index_rows(base_rows)
    gate_stats = load_calibrated_gate_stats(project_dir)
    for tolerance in noninferiority_tolerances:
        for min_target_rate in min_target_rates:
            for target in TARGETS:
                baseline = row_index[(target, BASELINE_METHOD)]
                calibrated = row_index[(target, CALIBRATED_METHOD)]
                stats = gate_stats.get(target, {})
                source_delta = float(calibrated["worst_source_val_macro_f1"]) - float(
                    baseline["worst_source_val_macro_f1"]
                )
                gate_target_rate = float(stats.get("gate_target_rate", 0.0))
                calibrated_eligible = source_delta >= -tolerance and gate_target_rate >= min_target_rate
                selected = calibrated if calibrated_eligible else baseline
                selected_name = str(selected["method"])
                selected_f1 = float(selected["target_window_macro_f1_posthoc"])
                baseline_f1 = float(baseline["target_window_macro_f1_posthoc"])
                calibrated_f1 = float(calibrated["target_window_macro_f1_posthoc"])
                rows.append(
                    {
                        "noninferiority_tolerance": tolerance,
                        "min_gate_target_rate": min_target_rate,
                        "target": target,
                        "source_worst_delta_calibrated_minus_v5": source_delta,
                        "gate_target_rate": gate_target_rate,
                        "selected_method": selected_name,
                        "selected_target_window_macro_f1_posthoc": selected_f1,
                        "selected_minus_v5_target_f1_posthoc": selected_f1 - baseline_f1,
                        "calibrated_minus_v5_target_f1_posthoc": calibrated_f1 - baseline_f1,
                    }
                )
    return rows


def load_calibrated_gate_stats(project_dir: Path) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for target in TARGETS:
        path = (
            project_dir
            / "outputs"
            / "checkpoints"
            / f"cic_man_gated_filterbank_calibrated_source_mixed_cross_dataset_task3_target_{target}"
            / "metrics.json"
        )
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        train = payload.get("history", [{}])[-1].get("train", {})
        stats[target] = {
            "gate_target_rate": float(train.get("gate_target_rate", 0.0)),
            "filterbank_gate_mean": float(train.get("filterbank_gate_mean", 0.0)),
            "gate_positive_margin": float(train.get("gate_positive_margin", 0.0)),
        }
    return stats


def summarize_calibrated_vs_v5(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summaries = []
    keys = sorted({(row["noninferiority_tolerance"], row["min_gate_target_rate"]) for row in rows})
    for tolerance, min_target_rate in keys:
        subset = [
            row
            for row in rows
            if row["noninferiority_tolerance"] == tolerance
            and row["min_gate_target_rate"] == min_target_rate
        ]
        summaries.append(
            {
                "noninferiority_tolerance": tolerance,
                "min_gate_target_rate": min_target_rate,
                "selected_calibrated_count": sum(
                    str(row["selected_method"]) == CALIBRATED_METHOD for row in subset
                ),
                "mean_selected_minus_v5_target_f1_posthoc": sum(
                    float(row["selected_minus_v5_target_f1_posthoc"]) for row in subset
                )
                / max(len(subset), 1),
                "selected_methods": ", ".join(f"{row['target']}={row['selected_method']}" for row in subset),
            }
        )
    return summaries


def write_markdown(
    summary_rows: list[dict[str, object]],
    detail_rows: list[dict[str, object]],
    calibrated_summary_rows: list[dict[str, object]],
    calibrated_detail_rows: list[dict[str, object]],
    path: Path,
) -> None:
    lines = [
        "# Source-Validation Selection Sensitivity",
        "",
        "This audit scans only source-side selection hyperparameters. Target metrics are post-hoc audit columns and are not used by the selector.",
        "",
        "## Summary",
        "",
        "| Metric | Cost W | Margin | Selected Calibrated | Selected v5 | Mean Selected-v5 Target F1 | Mean Calibrated-v5 Target F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['selection_metric']} | {float(row['cost_weight']):.3f} | "
            f"{float(row['intervention_margin']):.4f} | {int(row['selected_calibrated_count'])} | "
            f"{int(row['selected_v5_count'])} | "
            f"{float(row['mean_selected_minus_baseline_target_f1_posthoc']):.6f} | "
            f"{float(row['mean_calibrated_minus_baseline_target_f1_posthoc']):.6f} |"
        )

    best_rows = [
        row
        for row in summary_rows
        if row["selection_metric"] == "source_val_macro_f1"
        and float(row["cost_weight"]) == 0.0
        and float(row["intervention_margin"]) == 0.0
    ]
    strict_rows = [
        row
        for row in summary_rows
        if row["selection_metric"] == "worst_source_val_macro_f1"
        and float(row["cost_weight"]) == 0.02
        and float(row["intervention_margin"]) == 0.01
    ]

    lines.extend(["", "## Key Configurations", ""])
    for label, rows in [("Permissive source-F1 selector", best_rows), ("Strict worst-source selector", strict_rows)]:
        if not rows:
            continue
        row = rows[0]
        lines.append(f"- {label}: {row['selected_methods']}.")
        lines.append(
            f"  Mean selected-v5 post-hoc target F1: "
            f"{float(row['mean_selected_minus_baseline_target_f1_posthoc']):.6f}."
        )

    lines.extend(
        [
            "",
            "## Calibrated vs v5 Post-Hoc Target Delta",
            "",
            "| Target | Calibrated-v5 Window Macro-F1 |",
            "|---|---:|",
        ]
    )
    seen_targets = set()
    for row in detail_rows:
        key = (row["selection_metric"], row["cost_weight"], row["intervention_margin"])
        if key != ("worst_source_val_macro_f1", 0.02, 0.01):
            continue
        target = str(row["target"])
        if target in seen_targets:
            continue
        seen_targets.add(target)
        lines.append(f"| {target} | {float(row['calibrated_minus_baseline_target_f1_posthoc']):.6f} |")

    lines.extend(
        [
            "",
            "## Calibrated-Only Non-Inferiority Rule",
            "",
            "This candidate rule compares only v5 and calibrated gated-filterbank. It selects calibrated when source worst Macro-F1 is within a tolerance of v5 and the source gate target rate exceeds a minimum threshold.",
            "",
            "| Source Tolerance | Min Gate Target Rate | Selected Calibrated | Mean Selected-v5 Target F1 |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in calibrated_summary_rows:
        lines.append(
            f"| {float(row['noninferiority_tolerance']):.4f} | "
            f"{float(row['min_gate_target_rate']):.4f} | "
            f"{int(row['selected_calibrated_count'])} | "
            f"{float(row['mean_selected_minus_v5_target_f1_posthoc']):.6f} |"
        )

    preferred = [
        row
        for row in calibrated_summary_rows
        if float(row["noninferiority_tolerance"]) == 0.005
        and float(row["min_gate_target_rate"]) == 0.001
    ]
    if preferred:
        row = preferred[0]
        lines.extend(
            [
                "",
                "Candidate source-only rule: tolerance `0.005`, minimum gate-target rate `0.001`.",
                f"Selected methods: {row['selected_methods']}.",
                f"Mean selected-v5 post-hoc target F1: {float(row['mean_selected_minus_v5_target_f1_posthoc']):.6f}.",
            ]
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_dir = args.project_dir.expanduser().resolve()
    output_dir = (args.output_dir or project_dir / "outputs" / "tables").expanduser().resolve()
    rows = collect_sensitivity_rows(project_dir)
    summary_rows = summarize_rows(rows)
    calibrated_rows = collect_calibrated_vs_v5_rows(project_dir)
    calibrated_summary_rows = summarize_calibrated_vs_v5(calibrated_rows)

    detail_csv = output_dir / "source_validation_selection_sensitivity_detail.csv"
    summary_csv = output_dir / "source_validation_selection_sensitivity_summary.csv"
    calibrated_detail_csv = output_dir / "source_validation_calibrated_vs_v5_detail.csv"
    calibrated_summary_csv = output_dir / "source_validation_calibrated_vs_v5_summary.csv"
    md_path = output_dir / "source_validation_selection_sensitivity.md"
    json_path = output_dir / "source_validation_selection_sensitivity.json"
    write_csv(rows, detail_csv)
    write_csv(summary_rows, summary_csv)
    write_csv(calibrated_rows, calibrated_detail_csv)
    write_csv(calibrated_summary_rows, calibrated_summary_csv)
    write_markdown(summary_rows, rows, calibrated_summary_rows, calibrated_rows, md_path)
    json_path.write_text(
        json.dumps(
            {
                "summary": summary_rows,
                "detail": rows,
                "calibrated_vs_v5_summary": calibrated_summary_rows,
                "calibrated_vs_v5_detail": calibrated_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {detail_csv}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {calibrated_detail_csv}")
    print(f"Wrote {calibrated_summary_csv}")
    print(f"Wrote {md_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
