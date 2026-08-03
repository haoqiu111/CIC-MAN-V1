#!/usr/bin/env python3
"""Repeated-seed counterfactual-boundary audit across targets.

Each run selects its boundary using only the corresponding source validation
split. Target labels are post-hoc evidence for Ottawa repeated validation and
HUST/Paderborn safety auditing.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from ottawa_counterfactual_inner_boundary import (  # noqa: E402
    aggregate_recordings,
    build_candidates,
    class_evidence_summary,
    collect_counterfactual_evidence,
    estimate_source_style_stats,
    evaluate_records,
    load_model,
    select_candidate,
    write_csv,
)


TARGETS = ["hust", "ottawa", "paderborn"]
SEEDS = [42, 2025, 2026]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--targets", default="hust,ottawa,paderborn")
    parser.add_argument("--seeds", default="42,2025,2026")
    parser.add_argument("--batch-size", type=int, default=768)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-style-items", type=int, default=12000)
    parser.add_argument("--source-drop-tolerance", type=float, default=0.01)
    parser.add_argument("--source-inner-drop-tolerance", type=float, default=0.02)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def checkpoint_dir(project_dir: Path, seed: int, target: str) -> Path:
    checkpoints = project_dir / "outputs" / "checkpoints"
    if seed == 42:
        name = f"cic_man_gated_filterbank_calibrated_source_mixed_cross_dataset_task3_target_{target}"
    else:
        name = f"seed_{seed}_cic_man_gated_filterbank_calibrated_source_mixed_cross_dataset_task3_target_{target}"
    return checkpoints / name


def metric_value(metrics: dict[str, object], key: str) -> float:
    value = metrics.get(key, math.nan)
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def selected_metrics(records: list[dict[str, object]], selected: dict[str, object], num_classes: int) -> dict[str, object]:
    threshold = selected["threshold"]
    threshold_value = None if threshold == "" else float(threshold)
    return evaluate_records(records, str(selected["rule"]), threshold_value, num_classes)


def confusion_json(metrics: dict[str, object]) -> str:
    return json.dumps(metrics.get("confusion_matrix", []), ensure_ascii=False)


def run_one(
    *,
    project_root: Path,
    project_dir: Path,
    output_dir: Path,
    target: str,
    seed: int,
    batch_size: int,
    device: str,
    num_workers: int,
    max_style_items: int,
    source_drop_tolerance: float,
    source_inner_drop_tolerance: float,
    force: bool,
) -> dict[str, object]:
    run_dir = output_dir / "counterfactual_boundary_runs" / f"seed_{seed}_target_{target}"
    summary_path = run_dir / "summary.json"
    if summary_path.exists() and not force:
        return json.loads(summary_path.read_text(encoding="utf-8"))

    import torch

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    base = (
        project_root
        / "data"
        / "paper1_cicman"
        / "cache"
        / "windows"
        / "cross_dataset_task3_source_mixed"
        / f"target_dataset_{target}"
    )
    ckpt_dir = checkpoint_dir(project_dir, seed, target)
    ckpt = ckpt_dir / "best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"missing checkpoint: {ckpt}")

    model, num_classes = load_model(ckpt, device)
    style_stats = estimate_source_style_stats(base / "train_windows.csv", batch_size, num_workers, max_style_items)
    source_window_rows = collect_counterfactual_evidence(
        model, base / "val_windows.csv", style_stats, batch_size, device, num_workers
    )
    target_window_rows = collect_counterfactual_evidence(
        model, base / "test_windows.csv", style_stats, batch_size, device, num_workers
    )
    source_records = aggregate_recordings(source_window_rows)
    target_records = aggregate_recordings(target_window_rows)
    candidates = build_candidates(source_records, target_records, num_classes)
    selected = select_candidate(candidates, source_drop_tolerance, source_inner_drop_tolerance)
    base_source = evaluate_records(source_records, "base_vote", None, num_classes)
    base_target = evaluate_records(target_records, "base_vote", None, num_classes)
    selected_source = selected_metrics(source_records, selected, num_classes)
    selected_target = selected_metrics(target_records, selected, num_classes)
    evidence = class_evidence_summary(source_records, "source_val") + class_evidence_summary(target_records, "target_test")

    run_dir.mkdir(parents=True, exist_ok=True)
    write_csv(candidates, run_dir / "counterfactual_boundary_candidates.csv")
    write_csv(evidence, run_dir / "counterfactual_evidence_summary.csv")
    write_csv(target_records, run_dir / "target_recording_evidence.csv")
    summary = {
        "seed": seed,
        "target": target,
        "checkpoint_dir": str(ckpt_dir),
        "style_stats": style_stats,
        "selected_rule": selected["rule"],
        "selected_threshold": selected["threshold"],
        "selection_reason": selected.get("selection_reason", ""),
        "source_base_macro_f1": metric_value(base_source, "macro_f1"),
        "source_selected_macro_f1": metric_value(selected_source, "macro_f1"),
        "source_base_inner_recall": metric_value(base_source, "inner_recall"),
        "source_selected_inner_recall": metric_value(selected_source, "inner_recall"),
        "target_base_macro_f1": metric_value(base_target, "macro_f1"),
        "target_selected_macro_f1": metric_value(selected_target, "macro_f1"),
        "target_macro_f1_delta": metric_value(selected_target, "macro_f1") - metric_value(base_target, "macro_f1"),
        "target_base_inner_recall": metric_value(base_target, "inner_recall"),
        "target_selected_inner_recall": metric_value(selected_target, "inner_recall"),
        "target_inner_recall_delta": metric_value(selected_target, "inner_recall") - metric_value(base_target, "inner_recall"),
        "target_base_outer_recall": metric_value(base_target, "outer_recall"),
        "target_selected_outer_recall": metric_value(selected_target, "outer_recall"),
        "target_outer_recall_delta": metric_value(selected_target, "outer_recall") - metric_value(base_target, "outer_recall"),
        "target_base_normal_recall": metric_value(base_target, "normal_recall"),
        "target_selected_normal_recall": metric_value(selected_target, "normal_recall"),
        "target_normal_recall_delta": metric_value(selected_target, "normal_recall") - metric_value(base_target, "normal_recall"),
        "target_base_confusion_matrix": confusion_json(base_target),
        "target_selected_confusion_matrix": confusion_json(selected_target),
    }
    safety_drop = summary["target_macro_f1_delta"] < -0.01
    class_drop = min(
        summary["target_inner_recall_delta"],
        summary["target_outer_recall_delta"],
        summary["target_normal_recall_delta"],
    ) < -0.05
    summary["safety_flag"] = bool(safety_drop or class_drop)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def mean_std(values: list[float]) -> tuple[float, float]:
    clean = [float(value) for value in values if not math.isnan(float(value))]
    if not clean:
        return math.nan, math.nan
    return float(np.mean(clean)), float(np.std(clean, ddof=1)) if len(clean) > 1 else 0.0


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    for target in sorted({str(row["target"]) for row in rows}):
        group = [row for row in rows if row["target"] == target]
        macro_mean, macro_std = mean_std([float(row["target_selected_macro_f1"]) for row in group])
        base_mean, base_std = mean_std([float(row["target_base_macro_f1"]) for row in group])
        delta_mean, delta_std = mean_std([float(row["target_macro_f1_delta"]) for row in group])
        inner_mean, inner_std = mean_std([float(row["target_selected_inner_recall"]) for row in group])
        inner_delta_mean, inner_delta_std = mean_std([float(row["target_inner_recall_delta"]) for row in group])
        out.append(
            {
                "target": target,
                "runs": len(group),
                "base_macro_f1_mean": base_mean,
                "base_macro_f1_std": base_std,
                "selected_macro_f1_mean": macro_mean,
                "selected_macro_f1_std": macro_std,
                "macro_f1_delta_mean": delta_mean,
                "macro_f1_delta_std": delta_std,
                "selected_inner_recall_mean": inner_mean,
                "selected_inner_recall_std": inner_std,
                "inner_recall_delta_mean": inner_delta_mean,
                "inner_recall_delta_std": inner_delta_std,
                "safety_flags": sum(1 for row in group if bool(row.get("safety_flag", False))),
            }
        )
    return out


def fmt(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return "nan"
    return f"{number:.6f}"


def write_markdown(rows: list[dict[str, object]], summary: list[dict[str, object]], path: Path) -> None:
    lines = [
        "# Repeated Counterfactual Boundary Audit",
        "",
        "Each run selects the counterfactual boundary from its own source validation split. Target labels are post-hoc repeated validation and safety evidence.",
        "",
        "## Per-Target Summary",
        "",
        "| Target | Runs | Base Macro-F1 | CF Macro-F1 | Delta | CF Inner Recall | Inner Recall Delta | Safety Flags |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['target']} | {row['runs']} | {fmt(row['base_macro_f1_mean'])} ± {fmt(row['base_macro_f1_std'])} | "
            f"{fmt(row['selected_macro_f1_mean'])} ± {fmt(row['selected_macro_f1_std'])} | "
            f"{fmt(row['macro_f1_delta_mean'])} ± {fmt(row['macro_f1_delta_std'])} | "
            f"{fmt(row['selected_inner_recall_mean'])} ± {fmt(row['selected_inner_recall_std'])} | "
            f"{fmt(row['inner_recall_delta_mean'])} ± {fmt(row['inner_recall_delta_std'])} | "
            f"{row['safety_flags']} |"
        )
    lines.extend(
        [
            "",
            "## Run Detail",
            "",
            "| Seed | Target | Rule | Threshold | Base Macro-F1 | CF Macro-F1 | Delta | Base Inner Recall | CF Inner Recall | Safety |",
            "|---:|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in sorted(rows, key=lambda item: (str(item["target"]), int(item["seed"]))):
        lines.append(
            f"| {row['seed']} | {row['target']} | {row['selected_rule']} | {fmt(row['selected_threshold'])} | "
            f"{fmt(row['target_base_macro_f1'])} | {fmt(row['target_selected_macro_f1'])} | "
            f"{fmt(row['target_macro_f1_delta'])} | {fmt(row['target_base_inner_recall'])} | "
            f"{fmt(row['target_selected_inner_recall'])} | {row['safety_flag']} |"
        )
    lines.extend(
        [
            "",
            "## Adversarial Interpretation",
            "",
            "- Ottawa validates whether the counterfactual inner boundary is seed-stable.",
            "- HUST and Paderborn are safety checks: a useful Ottawa selector should not cause systematic target Macro-F1 or class-recall collapse elsewhere.",
            "- Any target-favorable threshold that was not selected by source validation remains post-hoc evidence only.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    project_dir = project_root 
    output_dir = (args.output_dir or project_dir / "outputs" / "tables").resolve()
    targets = [item.strip() for item in args.targets.split(",") if item.strip()]
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    rows = []
    for target in targets:
        for seed in seeds:
            print(f"[counterfactual-boundary] seed={seed} target={target}", flush=True)
            rows.append(
                run_one(
                    project_root=project_root,
                    project_dir=project_dir,
                    output_dir=output_dir,
                    target=target,
                    seed=seed,
                    batch_size=args.batch_size,
                    device=args.device,
                    num_workers=args.num_workers,
                    max_style_items=args.max_style_items,
                    source_drop_tolerance=args.source_drop_tolerance,
                    source_inner_drop_tolerance=args.source_inner_drop_tolerance,
                    force=args.force,
                )
            )
    summary = summarize(rows)
    write_csv(rows, output_dir / "repeated_counterfactual_boundary_detail.csv")
    write_csv(summary, output_dir / "repeated_counterfactual_boundary_summary.csv")
    write_markdown(rows, summary, output_dir / "repeated_counterfactual_boundary_audit.md")
    print(f"Wrote repeated counterfactual boundary audit to {output_dir}")


if __name__ == "__main__":
    main()
