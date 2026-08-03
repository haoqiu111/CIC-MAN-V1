#!/usr/bin/env python3
"""Class-safe source-only counterfactual-boundary audit.

This is a stricter version of cf_margin_max. A threshold is eligible only when
source validation preserves normal/outer precision and recall while keeping
source Macro-F1 and inner recall non-inferior.
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
    collect_counterfactual_evidence,
    estimate_source_style_stats,
    evaluate_records,
    load_model,
    write_csv,
)


CLASS_NAMES = ["normal", "inner", "outer"]
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
    parser.add_argument("--source-macro-tolerance", type=float, default=0.01)
    parser.add_argument("--source-inner-recall-tolerance", type=float, default=0.02)
    parser.add_argument("--source-protected-class-tolerance", type=float, default=0.02)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def checkpoint_dir(project_dir: Path, seed: int, target: str) -> Path:
    checkpoints = project_dir / "outputs" / "checkpoints"
    if seed == 42:
        name = f"cic_man_gated_filterbank_calibrated_source_mixed_cross_dataset_task3_target_{target}"
    else:
        name = f"seed_{seed}_cic_man_gated_filterbank_calibrated_source_mixed_cross_dataset_task3_target_{target}"
    return checkpoints / name


def fmetric(metrics: dict[str, object], key: str) -> float:
    try:
        return float(metrics.get(key, math.nan))
    except (TypeError, ValueError):
        return math.nan


def candidate_thresholds(records: list[dict[str, object]]) -> list[float]:
    values = np.asarray([float(row["cf_margin_max"]) for row in records], dtype=np.float64)
    quantiles = np.quantile(values, np.linspace(0.05, 0.95, 37)).tolist()
    return sorted(set(float(value) for value in quantiles))


def evaluate_cf_margin_max(records: list[dict[str, object]], threshold: float | None, num_classes: int):
    if threshold is None:
        return evaluate_records(records, "base_vote", None, num_classes)
    return evaluate_records(records, "cf_margin_max", float(threshold), num_classes)


def build_class_safe_candidates(
    source_records: list[dict[str, object]],
    target_records: list[dict[str, object]],
    num_classes: int,
) -> list[dict[str, object]]:
    rows = []
    base_source = evaluate_cf_margin_max(source_records, None, num_classes)
    base_target = evaluate_cf_margin_max(target_records, None, num_classes)
    thresholds: list[float | None] = [None] + candidate_thresholds(source_records)
    for threshold in thresholds:
        source = evaluate_cf_margin_max(source_records, threshold, num_classes)
        target = evaluate_cf_margin_max(target_records, threshold, num_classes)
        row: dict[str, object] = {
            "rule": "base_vote" if threshold is None else "cf_margin_max",
            "threshold": "" if threshold is None else float(threshold),
            "source_macro_f1": fmetric(source, "macro_f1"),
            "source_macro_f1_delta": fmetric(source, "macro_f1") - fmetric(base_source, "macro_f1"),
            "target_macro_f1_posthoc": fmetric(target, "macro_f1"),
            "target_macro_f1_delta_posthoc": fmetric(target, "macro_f1") - fmetric(base_target, "macro_f1"),
            "target_confusion_matrix_posthoc": json.dumps(target.get("confusion_matrix", [])),
        }
        for class_name in CLASS_NAMES:
            for metric in ["recall", "precision", "predicted_support"]:
                key = f"{class_name}_{metric}"
                row[f"source_{key}"] = source.get(key, math.nan)
                row[f"source_{key}_delta"] = fmetric(source, key) - fmetric(base_source, key)
                row[f"target_{key}_posthoc"] = target.get(key, math.nan)
                row[f"target_{key}_delta_posthoc"] = fmetric(target, key) - fmetric(base_target, key)
        rows.append(row)
    return rows


def is_allowed(
    row: dict[str, object],
    baseline: dict[str, object],
    macro_tol: float,
    inner_tol: float,
    protected_tol: float,
) -> bool:
    if row["rule"] != "cf_margin_max":
        return False
    if float(row["source_macro_f1"]) < float(baseline["source_macro_f1"]) - macro_tol:
        return False
    if float(row["source_inner_recall"]) < float(baseline["source_inner_recall"]) - inner_tol:
        return False
    for class_name in ["normal", "outer"]:
        for metric in ["recall", "precision"]:
            key = f"source_{class_name}_{metric}"
            if float(row[key]) < float(baseline[key]) - protected_tol:
                return False
    return True


def select_class_safe(
    rows: list[dict[str, object]],
    macro_tol: float,
    inner_tol: float,
    protected_tol: float,
) -> dict[str, object]:
    baseline = next(row for row in rows if row["rule"] == "base_vote")
    allowed = [row for row in rows if is_allowed(row, baseline, macro_tol, inner_tol, protected_tol)]
    if not allowed:
        selected = dict(baseline)
        selected["selection_reason"] = "fallback base_vote: no class-safe cf_margin_max threshold"
        return selected
    selected = dict(
        max(
            allowed,
            key=lambda row: (
                float(row["source_inner_predicted_support"]),
                float(row["source_inner_recall"]),
                float(row["source_macro_f1"]),
                float(row["source_outer_recall"]),
                float(row["source_normal_recall"]),
            ),
        )
    )
    selected["selection_reason"] = (
        f"class-safe source-only cf_margin_max: source Macro-F1 within {macro_tol}, "
        f"inner recall within {inner_tol}, normal/outer precision+recall within {protected_tol}; "
        "maximize source inner support"
    )
    return selected


def row_from_selection(
    *,
    seed: int,
    target: str,
    selected: dict[str, object],
    baseline: dict[str, object],
) -> dict[str, object]:
    out = {
        "seed": seed,
        "target": target,
        "selected_rule": selected["rule"],
        "selected_threshold": selected["threshold"],
        "selection_reason": selected["selection_reason"],
        "source_base_macro_f1": baseline["source_macro_f1"],
        "source_selected_macro_f1": selected["source_macro_f1"],
        "source_selected_inner_recall": selected["source_inner_recall"],
        "source_selected_normal_recall": selected["source_normal_recall"],
        "source_selected_normal_precision": selected["source_normal_precision"],
        "source_selected_outer_recall": selected["source_outer_recall"],
        "source_selected_outer_precision": selected["source_outer_precision"],
        "target_base_macro_f1": baseline["target_macro_f1_posthoc"],
        "target_selected_macro_f1": selected["target_macro_f1_posthoc"],
        "target_macro_f1_delta": selected["target_macro_f1_delta_posthoc"],
        "target_base_inner_recall": baseline["target_inner_recall_posthoc"],
        "target_selected_inner_recall": selected["target_inner_recall_posthoc"],
        "target_inner_recall_delta": selected["target_inner_recall_delta_posthoc"],
        "target_base_normal_recall": baseline["target_normal_recall_posthoc"],
        "target_selected_normal_recall": selected["target_normal_recall_posthoc"],
        "target_normal_recall_delta": selected["target_normal_recall_delta_posthoc"],
        "target_base_outer_recall": baseline["target_outer_recall_posthoc"],
        "target_selected_outer_recall": selected["target_outer_recall_posthoc"],
        "target_outer_recall_delta": selected["target_outer_recall_delta_posthoc"],
        "target_selected_confusion_matrix": selected["target_confusion_matrix_posthoc"],
    }
    out["safety_flag"] = bool(
        float(out["target_macro_f1_delta"]) < -0.01
        or float(out["target_inner_recall_delta"]) < -0.05
        or float(out["target_normal_recall_delta"]) < -0.05
        or float(out["target_outer_recall_delta"]) < -0.05
    )
    return out


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
    macro_tol: float,
    inner_tol: float,
    protected_tol: float,
    force: bool,
) -> dict[str, object]:
    run_dir = output_dir / "class_safe_counterfactual_boundary_runs" / f"seed_{seed}_target_{target}"
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
    ckpt = checkpoint_dir(project_dir, seed, target) / "best.pt"
    model, num_classes = load_model(ckpt, device)
    style_stats = estimate_source_style_stats(base / "train_windows.csv", batch_size, num_workers, max_style_items)
    source_windows = collect_counterfactual_evidence(
        model, base / "val_windows.csv", style_stats, batch_size, device, num_workers
    )
    target_windows = collect_counterfactual_evidence(
        model, base / "test_windows.csv", style_stats, batch_size, device, num_workers
    )
    source_records = aggregate_recordings(source_windows)
    target_records = aggregate_recordings(target_windows)
    candidates = build_class_safe_candidates(source_records, target_records, num_classes)
    selected = select_class_safe(candidates, macro_tol, inner_tol, protected_tol)
    baseline = next(row for row in candidates if row["rule"] == "base_vote")
    summary = row_from_selection(seed=seed, target=target, selected=selected, baseline=baseline)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_csv(candidates, run_dir / "class_safe_candidates.csv")
    write_csv(source_records, run_dir / "source_recording_evidence.csv")
    write_csv(target_records, run_dir / "target_recording_evidence.csv")
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
        base_mean, base_std = mean_std([float(row["target_base_macro_f1"]) for row in group])
        selected_mean, selected_std = mean_std([float(row["target_selected_macro_f1"]) for row in group])
        delta_mean, delta_std = mean_std([float(row["target_macro_f1_delta"]) for row in group])
        inner_mean, inner_std = mean_std([float(row["target_selected_inner_recall"]) for row in group])
        inner_delta_mean, inner_delta_std = mean_std([float(row["target_inner_recall_delta"]) for row in group])
        out.append(
            {
                "target": target,
                "runs": len(group),
                "base_macro_f1_mean": base_mean,
                "base_macro_f1_std": base_std,
                "class_safe_macro_f1_mean": selected_mean,
                "class_safe_macro_f1_std": selected_std,
                "macro_f1_delta_mean": delta_mean,
                "macro_f1_delta_std": delta_std,
                "class_safe_inner_recall_mean": inner_mean,
                "class_safe_inner_recall_std": inner_std,
                "inner_recall_delta_mean": inner_delta_mean,
                "inner_recall_delta_std": inner_delta_std,
                "fallback_runs": sum(1 for row in group if row["selected_rule"] == "base_vote"),
                "safety_flags": sum(1 for row in group if bool(row["safety_flag"])),
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
        "# Class-Safe Counterfactual Boundary Audit",
        "",
        "Selection is source-only. `cf_margin_max` is allowed only if source validation preserves normal/outer precision and recall, while keeping Macro-F1 and inner recall non-inferior.",
        "",
        "## Summary",
        "",
        "| Target | Runs | Base Macro-F1 | Class-Safe Macro-F1 | Delta | Class-Safe Inner Recall | Inner Delta | Fallbacks | Safety Flags |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['target']} | {row['runs']} | {fmt(row['base_macro_f1_mean'])} ± {fmt(row['base_macro_f1_std'])} | "
            f"{fmt(row['class_safe_macro_f1_mean'])} ± {fmt(row['class_safe_macro_f1_std'])} | "
            f"{fmt(row['macro_f1_delta_mean'])} ± {fmt(row['macro_f1_delta_std'])} | "
            f"{fmt(row['class_safe_inner_recall_mean'])} ± {fmt(row['class_safe_inner_recall_std'])} | "
            f"{fmt(row['inner_recall_delta_mean'])} ± {fmt(row['inner_recall_delta_std'])} | "
            f"{row['fallback_runs']} | {row['safety_flags']} |"
        )
    lines.extend(
        [
            "",
            "## Detail",
            "",
            "| Seed | Target | Selected | Threshold | Base Macro-F1 | Selected Macro-F1 | Delta | Base Inner Recall | Selected Inner Recall | Safety |",
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
            "- If class-safe selection falls back often, source validation does not support activating the boundary.",
            "- If target safety flags remain, source normal/outer constraints are insufficient and this should stay diagnostic.",
            "- If Paderborn remains neutral, the source-only rule is still refusing to hallucinate missing mechanism coverage.",
        ]
    )
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
            print(f"[class-safe-cf] seed={seed} target={target}", flush=True)
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
                    macro_tol=args.source_macro_tolerance,
                    inner_tol=args.source_inner_recall_tolerance,
                    protected_tol=args.source_protected_class_tolerance,
                    force=args.force,
                )
            )
    summary = summarize(rows)
    write_csv(rows, output_dir / "class_safe_counterfactual_boundary_detail.csv")
    write_csv(summary, output_dir / "class_safe_counterfactual_boundary_summary.csv")
    write_markdown(rows, summary, output_dir / "class_safe_counterfactual_boundary_audit.md")
    print(f"Wrote class-safe counterfactual boundary audit to {output_dir}")


if __name__ == "__main__":
    main()
