#!/usr/bin/env python3
"""Summarize experiment metrics into CSV and Markdown tables."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def add_src_to_path() -> None:
    script_path = Path(__file__).resolve()
    project_dir = script_path.parents[2]
    sys.path.insert(0, str(project_dir / "src"))


add_src_to_path()


FIELDNAMES = [
    "experiment",
    "method",
    "dataset",
    "protocol",
    "target",
    "best_epoch",
    "test_accuracy",
    "test_macro_f1",
    "test_balanced_accuracy",
    "test_loss",
    "num_test_samples",
    "recording_accuracy",
    "recording_macro_f1",
    "recording_balanced_accuracy",
    "num_test_recordings",
    "checkpoint",
    "metrics_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Project directory containing outputs/checkpoints.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for summary tables. Defaults to <project-dir>/outputs/tables.",
    )
    parser.add_argument(
        "--include-smoke",
        action="store_true",
        help="Include smoke/probe runs in the summary.",
    )
    return parser.parse_args()


def infer_metadata(experiment: str) -> dict[str, str]:
    if experiment.startswith("cic_man_gated_filterbank_calibrated_source_mixed_cross_dataset_task3_target_"):
        target = experiment.removeprefix("cic_man_gated_filterbank_calibrated_source_mixed_cross_dataset_task3_target_")
        return {
            "method": "CIC-MAN-gated-filterbank-calibrated",
            "dataset": "cross_dataset",
            "protocol": "cross_dataset_task3_source_mixed",
            "target": f"target_dataset_{target}",
        }
    if experiment.startswith("cic_man_gated_filterbank_frozen_core_source_mixed_cross_dataset_task3_target_"):
        target = experiment.removeprefix("cic_man_gated_filterbank_frozen_core_source_mixed_cross_dataset_task3_target_")
        return {
            "method": "CIC-MAN-gated-filterbank-frozen-core",
            "dataset": "cross_dataset",
            "protocol": "cross_dataset_task3_source_mixed",
            "target": f"target_dataset_{target}",
        }
    if experiment.startswith("cic_man_gated_filterbank_source_mixed_cross_dataset_task3_target_"):
        target = experiment.removeprefix("cic_man_gated_filterbank_source_mixed_cross_dataset_task3_target_")
        return {
            "method": "CIC-MAN-gated-filterbank",
            "dataset": "cross_dataset",
            "protocol": "cross_dataset_task3_source_mixed",
            "target": f"target_dataset_{target}",
        }
    if experiment.startswith("cic_man_v8_prototype_coverage_style_source_mixed_cross_dataset_task3_target_"):
        target = experiment.removeprefix("cic_man_v8_prototype_coverage_style_source_mixed_cross_dataset_task3_target_")
        return {
            "method": "CIC-MAN-v8-prototype-coverage-style",
            "dataset": "cross_dataset",
            "protocol": "cross_dataset_task3_source_mixed",
            "target": f"target_dataset_{target}",
        }
    if experiment.startswith("cic_man_heterogeneous_source_mixed_cross_dataset_task3_target_"):
        target = experiment.removeprefix("cic_man_heterogeneous_source_mixed_cross_dataset_task3_target_")
        return {
            "method": "CIC-MAN-heterogeneous",
            "dataset": "cross_dataset",
            "protocol": "cross_dataset_task3_source_mixed",
            "target": f"target_dataset_{target}",
        }
    if experiment.startswith("cic_man_heterogeneous_v2_reliability_source_mixed_cross_dataset_task3_target_"):
        target = experiment.removeprefix("cic_man_heterogeneous_v2_reliability_source_mixed_cross_dataset_task3_target_")
        return {
            "method": "CIC-MAN-heterogeneous-v2-reliability",
            "dataset": "cross_dataset",
            "protocol": "cross_dataset_task3_source_mixed",
            "target": f"target_dataset_{target}",
        }
    if experiment.startswith("cic_man_heterogeneous_v3_physics_source_mixed_cross_dataset_task3_target_"):
        target = experiment.removeprefix("cic_man_heterogeneous_v3_physics_source_mixed_cross_dataset_task3_target_")
        return {
            "method": "CIC-MAN-heterogeneous-v3-physics",
            "dataset": "cross_dataset",
            "protocol": "cross_dataset_task3_source_mixed",
            "target": f"target_dataset_{target}",
        }
    if experiment.startswith("cic_man_heterogeneous_v4_filterbank_source_mixed_cross_dataset_task3_target_"):
        target = experiment.removeprefix("cic_man_heterogeneous_v4_filterbank_source_mixed_cross_dataset_task3_target_")
        return {
            "method": "CIC-MAN-heterogeneous-v4-filterbank",
            "dataset": "cross_dataset",
            "protocol": "cross_dataset_task3_source_mixed",
            "target": f"target_dataset_{target}",
        }
    if experiment.startswith("cic_man_v7_prototype_style_source_mixed_cross_dataset_task3_target_"):
        target = experiment.removeprefix("cic_man_v7_prototype_style_source_mixed_cross_dataset_task3_target_")
        return {
            "method": "CIC-MAN-v7-prototype-style",
            "dataset": "cross_dataset",
            "protocol": "cross_dataset_task3_source_mixed",
            "target": f"target_dataset_{target}",
        }
    if experiment.startswith("cic_man_v6b_nontarget_agent_style_source_mixed_cross_dataset_task3_target_"):
        target = experiment.removeprefix("cic_man_v6b_nontarget_agent_style_source_mixed_cross_dataset_task3_target_")
        return {
            "method": "CIC-MAN-v6b-nontarget-agent-style",
            "dataset": "cross_dataset",
            "protocol": "cross_dataset_task3_source_mixed",
            "target": f"target_dataset_{target}",
        }
    if experiment.startswith("cic_man_v6_class_agent_style_source_mixed_cross_dataset_task3_target_"):
        target = experiment.removeprefix("cic_man_v6_class_agent_style_source_mixed_cross_dataset_task3_target_")
        return {
            "method": "CIC-MAN-v6-class-agent-style",
            "dataset": "cross_dataset",
            "protocol": "cross_dataset_task3_source_mixed",
            "target": f"target_dataset_{target}",
        }
    if experiment.startswith("cic_man_v5_class_router_style_source_mixed_cross_dataset_task3_target_"):
        target = experiment.removeprefix("cic_man_v5_class_router_style_source_mixed_cross_dataset_task3_target_")
        return {
            "method": "CIC-MAN-v5-class-router-style",
            "dataset": "cross_dataset",
            "protocol": "cross_dataset_task3_source_mixed",
            "target": f"target_dataset_{target}",
        }
    if experiment.startswith("cic_man_v4_style_source_mixed_cross_dataset_task3_target_"):
        target = experiment.removeprefix("cic_man_v4_style_source_mixed_cross_dataset_task3_target_")
        return {
            "method": "CIC-MAN-v4-style-consistency",
            "dataset": "cross_dataset",
            "protocol": "cross_dataset_task3_source_mixed",
            "target": f"target_dataset_{target}",
        }
    if experiment.startswith("cic_man_v3_short_domain_mix_source_mixed_cross_dataset_task3_target_"):
        target = experiment.removeprefix("cic_man_v3_short_domain_mix_source_mixed_cross_dataset_task3_target_")
        return {
            "method": "CIC-MAN-v3-short-domain-mix",
            "dataset": "cross_dataset",
            "protocol": "cross_dataset_task3_source_mixed",
            "target": f"target_dataset_{target}",
        }
    if experiment.startswith("cic_man_v3_domain_mix_source_mixed_cross_dataset_task3_target_"):
        target = experiment.removeprefix("cic_man_v3_domain_mix_source_mixed_cross_dataset_task3_target_")
        return {
            "method": "CIC-MAN-v3-domain-mix",
            "dataset": "cross_dataset",
            "protocol": "cross_dataset_task3_source_mixed",
            "target": f"target_dataset_{target}",
        }
    if experiment.startswith("cic_man_v2_source_balanced_cross_dataset_task3_target_"):
        target = experiment.removeprefix("cic_man_v2_source_balanced_cross_dataset_task3_target_")
        return {
            "method": "CIC-MAN-v2-source-balanced",
            "dataset": "cross_dataset",
            "protocol": "cross_dataset_task3_source_mixed",
            "target": f"target_dataset_{target}",
        }
    if experiment.startswith("cic_man_v1_source_mixed_cross_dataset_task3_target_"):
        target = experiment.removeprefix("cic_man_v1_source_mixed_cross_dataset_task3_target_")
        return {
            "method": "CIC-MAN-v1-source-mixed",
            "dataset": "cross_dataset",
            "protocol": "cross_dataset_task3_source_mixed",
            "target": f"target_dataset_{target}",
        }
    if experiment.startswith("cic_man_v1_cross_dataset_task3_target_"):
        target = experiment.removeprefix("cic_man_v1_cross_dataset_task3_target_")
        return {
            "method": "CIC-MAN-v1",
            "dataset": "cross_dataset",
            "protocol": "cross_dataset_task3",
            "target": f"target_dataset_{target}",
        }
    if experiment.startswith("cic_man_v5_within_ottawa_speed_"):
        target = experiment.removeprefix("cic_man_v5_within_ottawa_speed_")
        return {
            "method": "CIC-MAN-v5-class-router-style",
            "dataset": "ottawa",
            "protocol": "ottawa_leave_speed",
            "target": f"target_speed_{target}",
        }
    if experiment.startswith("cic_man_v5_within_hust_bearing_type_"):
        target = experiment.removeprefix("cic_man_v5_within_hust_bearing_type_")
        return {
            "method": "CIC-MAN-v5-class-router-style",
            "dataset": "hust",
            "protocol": "hust_leave_bearing_type",
            "target": f"target_bearing_type_{target}",
        }
    if experiment.startswith("cic_man_v5_within_paderborn_condition_"):
        target = experiment.removeprefix("cic_man_v5_within_paderborn_condition_")
        return {
            "method": "CIC-MAN-v5-class-router-style",
            "dataset": "paderborn",
            "protocol": "paderborn_leave_condition",
            "target": f"target_{target}",
        }
    if experiment.startswith("cic_man_cross_dataset_task3_target_"):
        target = experiment.removeprefix("cic_man_cross_dataset_task3_target_")
        return {
            "method": "CIC-MAN-minimal",
            "dataset": "cross_dataset",
            "protocol": "cross_dataset_task3",
            "target": f"target_dataset_{target}",
        }
    if experiment.startswith("raw_cnn_cross_dataset_task3_target_"):
        target = experiment.removeprefix("raw_cnn_cross_dataset_task3_target_")
        return {
            "method": "RawCNN",
            "dataset": "cross_dataset",
            "protocol": "cross_dataset_task3",
            "target": f"target_dataset_{target}",
        }
    if experiment.startswith("raw_cnn_ottawa_speed_"):
        target = experiment.removeprefix("raw_cnn_ottawa_speed_")
        return {
            "method": "RawCNN",
            "dataset": "ottawa",
            "protocol": "ottawa_leave_speed",
            "target": f"target_speed_{target}",
        }
    if experiment.startswith("raw_cnn_hust_bearing_type_"):
        target = experiment.removeprefix("raw_cnn_hust_bearing_type_")
        return {
            "method": "RawCNN",
            "dataset": "hust",
            "protocol": "hust_leave_bearing_type",
            "target": f"target_bearing_type_{target}",
        }
    if experiment.startswith("raw_cnn_paderborn_condition_"):
        target = experiment.removeprefix("raw_cnn_paderborn_condition_")
        return {
            "method": "RawCNN",
            "dataset": "paderborn",
            "protocol": "paderborn_leave_condition",
            "target": f"target_{target}",
        }
    return {
        "method": "unknown",
        "dataset": "unknown",
        "protocol": "unknown",
        "target": "unknown",
    }


def should_skip(path: Path, include_smoke: bool) -> bool:
    if include_smoke:
        return False
    name = path.parent.name.lower()
    return "smoke" in name or "probe" in name


def load_row(metrics_path: Path) -> dict[str, object]:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    experiment = metrics_path.parent.name
    metadata = infer_metadata(experiment)
    metrics = payload["test_metrics"]
    recording_path = metrics_path.parent / "recording_metrics.json"
    recording_metrics = {}
    if recording_path.exists():
        recording_payload = json.loads(recording_path.read_text(encoding="utf-8"))
        recording_metrics = recording_payload.get("mean_logits", {})
    return {
        "experiment": experiment,
        **metadata,
        "best_epoch": payload.get("best_epoch", ""),
        "test_accuracy": float(metrics.get("accuracy", 0.0)),
        "test_macro_f1": float(metrics.get("macro_f1", 0.0)),
        "test_balanced_accuracy": float(metrics.get("balanced_accuracy", 0.0)),
        "test_loss": float(metrics.get("loss", 0.0)),
        "num_test_samples": int(metrics.get("num_samples", 0)),
        "recording_accuracy": float(recording_metrics.get("accuracy", 0.0)),
        "recording_macro_f1": float(recording_metrics.get("macro_f1", 0.0)),
        "recording_balanced_accuracy": float(recording_metrics.get("balanced_accuracy", 0.0)),
        "num_test_recordings": int(recording_metrics.get("num_recordings", 0)),
        "checkpoint": payload.get("best_checkpoint", ""),
        "metrics_path": str(metrics_path),
    }


def sort_key(row: dict[str, object]) -> tuple[str, str, str, str]:
    return (
        str(row["dataset"]),
        str(row["protocol"]),
        str(row["target"]),
        str(row["experiment"]),
    )


def write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_markdown(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Model Comparison Summary",
        "",
        "| Method | Dataset | Protocol | Target | Best epoch | Window Acc | Window Macro-F1 | Recording Acc | Recording Macro-F1 | Recordings |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["method"]),
                    str(row["dataset"]),
                    str(row["protocol"]),
                    str(row["target"]),
                    str(row["best_epoch"]),
                    fmt(row["test_accuracy"]),
                    fmt(row["test_macro_f1"]),
                    fmt(row["recording_accuracy"]),
                    fmt(row["recording_macro_f1"]),
                    str(row["num_test_recordings"]),
                ]
            )
            + " |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_dir = args.project_dir.expanduser().resolve()
    checkpoint_dir = project_dir / "outputs" / "checkpoints"
    output_dir = (args.output_dir or project_dir / "outputs" / "tables").expanduser().resolve()

    metrics_paths = sorted(checkpoint_dir.rglob("metrics.json"))
    rows = [load_row(path) for path in metrics_paths if not should_skip(path, args.include_smoke)]
    rows.sort(key=sort_key)

    csv_path = output_dir / "raw_cnn_baselines.csv"
    md_path = output_dir / "raw_cnn_baselines.md"
    write_csv(rows, csv_path)
    write_markdown(rows, md_path)
    print(f"Wrote {len(rows)} rows -> {csv_path}")
    print(f"Wrote {len(rows)} rows -> {md_path}")

    comparison_csv_path = output_dir / "model_comparison.csv"
    comparison_md_path = output_dir / "model_comparison.md"
    write_csv(rows, comparison_csv_path)
    write_markdown(rows, comparison_md_path)
    print(f"Wrote {len(rows)} rows -> {comparison_csv_path}")
    print(f"Wrote {len(rows)} rows -> {comparison_md_path}")


if __name__ == "__main__":
    main()
