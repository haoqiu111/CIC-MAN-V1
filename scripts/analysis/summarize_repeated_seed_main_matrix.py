#!/usr/bin/env python3
"""Summarize the minimal repeated-seed main experiment matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


TARGETS = ["hust", "ottawa", "paderborn"]
SEEDS = [42, 2025, 2026]

METHODS = {
    "RawCNN": {
        "kind": "raw",
        "seed42_prefix": "seed_42_raw_cnn_source_mixed_cross_dataset_task3_target_",
        "seed_prefix": "seed_{seed}_raw_cnn_source_mixed_cross_dataset_task3_target_",
    },
    "CIC-MAN-v5": {
        "kind": "cicman",
        "seed42_prefix": "cic_man_v5_class_router_style_source_mixed_cross_dataset_task3_target_",
        "seed_prefix": "seed_{seed}_cic_man_v5_class_router_style_source_mixed_cross_dataset_task3_target_",
    },
    "CIC-MAN-A10": {
        "kind": "cicman",
        "seed42_prefix": "cic_man_gated_filterbank_calibrated_source_mixed_cross_dataset_task3_target_",
        "seed_prefix": "seed_{seed}_cic_man_gated_filterbank_calibrated_source_mixed_cross_dataset_task3_target_",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--source-noninferiority-tolerance", type=float, default=0.005)
    parser.add_argument("--min-gate-target-rate", type=float, default=0.001)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def checkpoint_dir(root: Path, method: str, seed: int, target: str) -> Path:
    spec = METHODS[method]
    if seed == 42:
        prefix = str(spec["seed42_prefix"])
    else:
        prefix = str(spec["seed_prefix"]).format(seed=seed)
    return root / f"{prefix}{target}"


def metric(metrics: dict[str, object], group: str, key: str, default: float = 0.0) -> float:
    return float(metrics.get(group, {}).get(key, default))


def source_worst(metrics: dict[str, object]) -> float:
    val = metrics.get("best_val_metrics", {})
    return float(val.get("worst_dataset_macro_f1", val.get("macro_f1", 0.0)))


def source_mean(metrics: dict[str, object]) -> float:
    val = metrics.get("best_val_metrics", {})
    return float(val.get("mean_dataset_macro_f1", val.get("macro_f1", 0.0)))


def gate_stats(metrics: dict[str, object]) -> tuple[float, float, float]:
    history = metrics.get("history", [])
    train = history[-1].get("train", {}) if history else {}
    return (
        float(train.get("gate_target_rate", 0.0)),
        float(train.get("filterbank_gate_mean", 0.0)),
        float(train.get("gate_positive_margin", 0.0)),
    )


def recording_macro_f1(run_dir: Path) -> float:
    path = run_dir / "recording_metrics.json"
    if not path.exists():
        return math.nan
    payload = load_json(path)
    return float(payload.get("mean_logits", {}).get("macro_f1", math.nan))


def collect_method_row(root: Path, method: str, seed: int, target: str) -> dict[str, object] | None:
    run_dir = checkpoint_dir(root, method, seed, target)
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    metrics = load_json(metrics_path)
    gate_rate, gate_mean, gate_margin = gate_stats(metrics)
    return {
        "method": method,
        "seed": seed,
        "target": target,
        "selected_base_method": "",
        "checkpoint_dir": str(run_dir),
        "best_epoch": int(metrics.get("best_epoch", 0)),
        "source_mean_macro_f1": source_mean(metrics),
        "source_worst_macro_f1": source_worst(metrics),
        "gate_target_rate": gate_rate,
        "gate_mean": gate_mean,
        "gate_positive_margin": gate_margin,
        "window_macro_f1": metric(metrics, "test_metrics", "macro_f1", math.nan),
        "window_accuracy": metric(metrics, "test_metrics", "accuracy", math.nan),
        "recording_macro_f1": recording_macro_f1(run_dir),
    }


def final_selector_row(
    rows_by_key: dict[tuple[str, int, str], dict[str, object]],
    *,
    seed: int,
    target: str,
    source_noninferiority_tolerance: float,
    min_gate_target_rate: float,
) -> dict[str, object] | None:
    v5 = rows_by_key.get(("CIC-MAN-v5", seed, target))
    a10 = rows_by_key.get(("CIC-MAN-A10", seed, target))
    if v5 is None or a10 is None:
        return None
    source_delta = float(a10["source_worst_macro_f1"]) - float(v5["source_worst_macro_f1"])
    gate_rate = float(a10["gate_target_rate"])
    use_a10 = source_delta >= -source_noninferiority_tolerance and gate_rate >= min_gate_target_rate
    selected = a10 if use_a10 else v5
    reason = (
        f"A10 allowed: source_delta={source_delta:.6f}, gate_rate={gate_rate:.6f}"
        if use_a10
        else f"fallback v5: source_delta={source_delta:.6f}, gate_rate={gate_rate:.6f}"
    )
    row = dict(selected)
    row["method"] = "CIC-MAN-final-selector"
    row["selected_base_method"] = str(selected["method"])
    row["selection_reason"] = reason
    return row


def collect_rows(
    project_dir: Path,
    *,
    source_noninferiority_tolerance: float,
    min_gate_target_rate: float,
) -> list[dict[str, object]]:
    root = project_dir / "outputs" / "checkpoints"
    rows: list[dict[str, object]] = []
    for method in METHODS:
        for seed in SEEDS:
            for target in TARGETS:
                row = collect_method_row(root, method, seed, target)
                if row is not None:
                    row["selection_reason"] = ""
                    rows.append(row)

    rows_by_key = {(str(row["method"]), int(row["seed"]), str(row["target"])): row for row in rows}
    for seed in SEEDS:
        for target in TARGETS:
            row = final_selector_row(
                rows_by_key,
                seed=seed,
                target=target,
                source_noninferiority_tolerance=source_noninferiority_tolerance,
                min_gate_target_rate=min_gate_target_rate,
            )
            if row is not None:
                rows.append(row)
    return rows


def numeric_values(rows: list[dict[str, object]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = float(row[key])
        if not math.isnan(value):
            values.append(value)
    return values


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    overall: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), str(row["target"]))].append(row)
        overall[str(row["method"])].append(row)

    summary: list[dict[str, object]] = []
    for (method, target), group in sorted(grouped.items()):
        summary.append(summary_row(method, target, group))
    for method, group in sorted(overall.items()):
        summary.append(summary_row(method, "mean_over_targets", group))
    return summary


def summary_row(method: str, target: str, group: list[dict[str, object]]) -> dict[str, object]:
    window = numeric_values(group, "window_macro_f1")
    recording = numeric_values(group, "recording_macro_f1")
    return {
        "method": method,
        "target": target,
        "num_runs": len(group),
        "window_macro_f1_mean": mean(window) if window else math.nan,
        "window_macro_f1_std": stdev(window) if len(window) > 1 else 0.0,
        "recording_macro_f1_mean": mean(recording) if recording else math.nan,
        "recording_macro_f1_std": stdev(recording) if len(recording) > 1 else 0.0,
    }


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.6f}"
    return str(value)


def write_markdown(detail: list[dict[str, object]], summary: list[dict[str, object]], path: Path) -> None:
    lines = [
        "# Repeated-Seed Minimal Main Matrix",
        "",
        "Scope: source-mixed cross-dataset DG, 3 epochs, seeds 42/2025/2026. Target labels are used only for post-hoc reporting.",
        "",
        "## Mean Over Targets",
        "",
        "| Method | Runs | Window Macro-F1 Mean | Window Macro-F1 Std | Recording Macro-F1 Mean | Recording Macro-F1 Std |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        if row["target"] != "mean_over_targets":
            continue
        lines.append(
            f"| {row['method']} | {row['num_runs']} | {fmt(row['window_macro_f1_mean'])} | "
            f"{fmt(row['window_macro_f1_std'])} | {fmt(row['recording_macro_f1_mean'])} | "
            f"{fmt(row['recording_macro_f1_std'])} |"
        )

    lines.extend(
        [
            "",
            "## By Target",
            "",
            "| Method | Target | Runs | Window Macro-F1 Mean | Window Macro-F1 Std | Recording Macro-F1 Mean | Recording Macro-F1 Std |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary:
        if row["target"] == "mean_over_targets":
            continue
        lines.append(
            f"| {row['method']} | {row['target']} | {row['num_runs']} | {fmt(row['window_macro_f1_mean'])} | "
            f"{fmt(row['window_macro_f1_std'])} | {fmt(row['recording_macro_f1_mean'])} | "
            f"{fmt(row['recording_macro_f1_std'])} |"
        )

    lines.extend(
        [
            "",
            "## Final Selector Decisions",
            "",
            "| Seed | Target | Selected | Source Worst F1 | Gate Target Rate | Window Macro-F1 | Recording Macro-F1 | Reason |",
            "|---:|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in detail:
        if row["method"] != "CIC-MAN-final-selector":
            continue
        lines.append(
            f"| {row['seed']} | {row['target']} | {row['selected_base_method']} | "
            f"{fmt(row['source_worst_macro_f1'])} | {fmt(row['gate_target_rate'])} | "
            f"{fmt(row['window_macro_f1'])} | {fmt(row['recording_macro_f1'])} | {row['selection_reason']} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_dir = args.project_dir.expanduser().resolve()
    output_dir = (args.output_dir or project_dir / "outputs" / "tables").expanduser().resolve()
    rows = collect_rows(
        project_dir,
        source_noninferiority_tolerance=args.source_noninferiority_tolerance,
        min_gate_target_rate=args.min_gate_target_rate,
    )
    rows.sort(key=lambda row: (str(row["method"]), int(row["seed"]), str(row["target"])))
    summary = summarize(rows)
    write_csv(rows, output_dir / "repeated_seed_main_matrix_detail.csv")
    write_csv(summary, output_dir / "repeated_seed_main_matrix_summary.csv")
    write_markdown(rows, summary, output_dir / "repeated_seed_main_matrix.md")
    print(f"Wrote repeated-seed main matrix outputs to {output_dir}")


if __name__ == "__main__":
    main()
