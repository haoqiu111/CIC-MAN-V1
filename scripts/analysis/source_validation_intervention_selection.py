#!/usr/bin/env python3
"""Target-free source-validation selection for intervention variants."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


CANDIDATES = [
    {
        "method": "CIC-MAN-v5-class-router-style",
        "prefix": "cic_man_v5_class_router_style_source_mixed_cross_dataset_task3_target_",
        "base": "minimal",
    },
    {
        "method": "CIC-MAN-heterogeneous-v3-physics",
        "prefix": "cic_man_heterogeneous_v3_physics_source_mixed_cross_dataset_task3_target_",
        "base": "heterogeneous",
    },
    {
        "method": "CIC-MAN-heterogeneous-v4-filterbank",
        "prefix": "cic_man_heterogeneous_v4_filterbank_source_mixed_cross_dataset_task3_target_",
        "base": "heterogeneous",
    },
    {
        "method": "CIC-MAN-gated-filterbank",
        "prefix": "cic_man_gated_filterbank_source_mixed_cross_dataset_task3_target_",
        "base": "gated_filterbank",
    },
    {
        "method": "CIC-MAN-gated-filterbank-frozen-core",
        "prefix": "cic_man_gated_filterbank_frozen_core_source_mixed_cross_dataset_task3_target_",
        "base": "gated_filterbank_frozen_core",
    },
    {
        "method": "CIC-MAN-gated-filterbank-calibrated",
        "prefix": "cic_man_gated_filterbank_calibrated_source_mixed_cross_dataset_task3_target_",
        "base": "gated_filterbank_calibrated",
    },
]

TARGETS = ["hust", "ottawa", "paderborn"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cost-weight", type=float, default=0.02)
    parser.add_argument("--router-weight", type=float, default=0.02)
    parser.add_argument(
        "--intervention-margin",
        type=float,
        default=0.01,
        help="Minimum source-worst-F1 gain over v5 required before a costlier intervention can be selected.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=["worst_source_val_macro_f1", "mean_source_val_macro_f1", "source_val_macro_f1"],
        default="worst_source_val_macro_f1",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_efficiency(project_dir: Path) -> dict[str, float]:
    path = project_dir / "outputs" / "tables" / "efficiency_summary.csv"
    if not path.exists():
        return {}
    rows = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[row["method"]] = float(row["sample_latency_ms"])
    return rows


def load_diagnostics(checkpoint_dir: Path) -> dict[str, float]:
    path = checkpoint_dir / "diagnostics" / "cic_man_diagnostics_test.json"
    if not path.exists():
        path = checkpoint_dir / "cic_man_diagnostics_test.json"
    if not path.exists():
        return {
            "effective_agents": 0.0,
            "normalized_router_entropy": 0.0,
        }
    payload = load_json(path)
    router = payload.get("router", {})
    return {
        "effective_agents": float(router.get("effective_agents_from_mean_weights", 0.0)),
        "normalized_router_entropy": float(router.get("normalized_mean_entropy", 0.0)),
    }


def load_model_comparison(project_dir: Path) -> dict[tuple[str, str], dict[str, str]]:
    path = project_dir / "outputs" / "tables" / "model_comparison.csv"
    rows = {}
    if not path.exists():
        return rows
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[(row["method"], row["target"])] = row
    return rows


def source_validation_metrics(metrics_payload: dict[str, object]) -> dict[str, float]:
    val = metrics_payload.get("best_val_metrics", {})
    return {
        "source_val_macro_f1": float(val.get("macro_f1", 0.0)),
        "mean_source_val_macro_f1": float(val.get("mean_dataset_macro_f1", val.get("macro_f1", 0.0))),
        "worst_source_val_macro_f1": float(val.get("worst_dataset_macro_f1", val.get("macro_f1", 0.0))),
        "source_val_accuracy": float(val.get("accuracy", 0.0)),
    }


def normalize_cost(rows: list[dict[str, object]]) -> None:
    costs = [float(row["sample_latency_ms"]) for row in rows]
    min_cost = min(costs) if costs else 0.0
    max_cost = max(costs) if costs else 0.0
    denom = max(max_cost - min_cost, 1e-12)
    for row in rows:
        row["normalized_cost"] = (float(row["sample_latency_ms"]) - min_cost) / denom


def collect_rows(project_dir: Path, selection_metric: str, cost_weight: float, router_weight: float) -> list[dict[str, object]]:
    checkpoint_root = project_dir / "outputs" / "checkpoints"
    efficiency = load_efficiency(project_dir)
    comparison = load_model_comparison(project_dir)
    rows: list[dict[str, object]] = []
    for target in TARGETS:
        target_rows = []
        for candidate in CANDIDATES:
            checkpoint_dir = checkpoint_root / f"{candidate['prefix']}{target}"
            metrics_path = checkpoint_dir / "metrics.json"
            if not metrics_path.exists():
                continue
            metrics_payload = load_json(metrics_path)
            val_metrics = source_validation_metrics(metrics_payload)
            diagnostics = load_diagnostics(checkpoint_dir)
            comparison_row = comparison.get((candidate["method"], f"target_dataset_{target}"), {})
            row = {
                "target": target,
                "method": candidate["method"],
                "checkpoint_dir": str(checkpoint_dir),
                "base": candidate["base"],
                "sample_latency_ms": efficiency.get(candidate["method"], 0.0),
                "target_window_macro_f1_posthoc": float(comparison_row.get("test_macro_f1", 0.0) or 0.0),
                "target_recording_macro_f1_posthoc": float(comparison_row.get("recording_macro_f1", 0.0) or 0.0),
                **val_metrics,
                **diagnostics,
            }
            target_rows.append(row)
        normalize_cost(target_rows)
        for row in target_rows:
            row["selection_metric"] = selection_metric
            row["selection_score"] = (
                float(row[selection_metric])
                + router_weight * float(row["normalized_router_entropy"])
                - cost_weight * float(row["normalized_cost"])
            )
            rows.append(row)
    return rows


def add_intervention_eligibility(rows: list[dict[str, object]], selection_metric: str, intervention_margin: float) -> None:
    for target in TARGETS:
        target_rows = [row for row in rows if row["target"] == target]
        baseline = next(
            (row for row in target_rows if row["method"] == "CIC-MAN-v5-class-router-style"),
            None,
        )
        if baseline is None:
            for row in target_rows:
                row["passes_intervention_margin"] = True
            continue
        baseline_score = float(baseline[selection_metric])
        for row in target_rows:
            if row["method"] == "CIC-MAN-v5-class-router-style":
                row["passes_intervention_margin"] = True
                continue
            row["passes_intervention_margin"] = float(row[selection_metric]) >= baseline_score + intervention_margin


def add_selection_flags(rows: list[dict[str, object]]) -> None:
    for target in TARGETS:
        target_rows = [row for row in rows if row["target"] == target]
        if not target_rows:
            continue
        eligible_rows = [row for row in target_rows if row.get("passes_intervention_margin", True)]
        selected_pool = eligible_rows or target_rows
        selected = max(selected_pool, key=lambda row: float(row["selection_score"]))
        oracle = max(target_rows, key=lambda row: float(row["target_window_macro_f1_posthoc"]))
        for row in target_rows:
            row["selected_by_source_validation"] = row is selected
            row["posthoc_oracle_best"] = row is oracle


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, object]], path: Path) -> None:
    lines = [
        "# Source-Validation Intervention Selection",
        "",
        "Selection uses only source validation metrics plus cost/router regularization. Target metrics are post-hoc audit columns and are not used by the selector.",
        "",
        "| Target | Method | Source Worst F1 | Cost Norm | Router Entropy | Selection Score | Selected | Target F1 Post-hoc |",
        "|---|---|---:|---:|---:|---:|---|---:|",
    ]
    for row in rows:
        selected = "yes" if row["selected_by_source_validation"] else ""
        eligible = "" if row.get("passes_intervention_margin", True) else " no-margin"
        lines.append(
            f"| {row['target']} | {row['method']}{eligible} | "
            f"{float(row['worst_source_val_macro_f1']):.6f} | "
            f"{float(row['normalized_cost']):.3f} | "
            f"{float(row['normalized_router_entropy']):.3f} | "
            f"{float(row['selection_score']):.6f} | {selected} | "
            f"{float(row['target_window_macro_f1_posthoc']):.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_dir = args.project_dir.expanduser().resolve()
    output_dir = (args.output_dir or project_dir / "outputs" / "tables").expanduser().resolve()
    rows = collect_rows(project_dir, args.selection_metric, args.cost_weight, args.router_weight)
    add_intervention_eligibility(rows, args.selection_metric, args.intervention_margin)
    add_selection_flags(rows)
    rows.sort(key=lambda row: (str(row["target"]), -float(row["selection_score"])))
    write_csv(rows, output_dir / "source_validation_intervention_selection.csv")
    write_markdown(rows, output_dir / "source_validation_intervention_selection.md")
    print(f"Wrote {len(rows)} rows to {output_dir}")


if __name__ == "__main__":
    main()
