#!/usr/bin/env python3
"""Domain-wise class-safe counterfactual-boundary audit from cached evidence."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


CLASS_NAMES = {0: "normal", 1: "inner", 2: "outer"}
TARGETS = ["hust", "ottawa", "paderborn"]
SEEDS = [42, 2025, 2026]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--source-macro-tolerance", type=float, default=0.01)
    parser.add_argument("--source-inner-recall-tolerance", type=float, default=0.02)
    parser.add_argument("--source-protected-class-tolerance", type=float, default=0.02)
    parser.add_argument("--outer-to-inner-tolerance", type=float, default=0.02)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    seen = set(fieldnames)
    for row in rows[1:]:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def predict(row: dict[str, str], threshold: float | None) -> int:
    base = int(row["base_vote_pred"])
    if threshold is None:
        return base
    if float(row["cf_margin_max"]) >= threshold:
        return 1
    return base


def confusion(y_true: list[int], y_pred: list[int], num_classes: int = 3) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true, pred in zip(y_true, y_pred):
        matrix[int(true), int(pred)] += 1
    return matrix


def class_stats(matrix: np.ndarray, class_id: int) -> dict[str, float]:
    tp = float(matrix[class_id, class_id])
    fn = float(matrix[class_id, :].sum() - tp)
    fp = float(matrix[:, class_id].sum() - tp)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def metrics(records: list[dict[str, str]], threshold: float | None) -> dict[str, object]:
    y_true = [int(row["label_id"]) for row in records]
    y_pred = [predict(row, threshold) for row in records]
    matrix = confusion(y_true, y_pred)
    per_class = {CLASS_NAMES[class_id]: class_stats(matrix, class_id) for class_id in CLASS_NAMES}
    out: dict[str, object] = {
        "macro_f1": float(np.mean([item["f1"] for item in per_class.values()])),
        "confusion_matrix": matrix.tolist(),
        "num_recordings": len(records),
    }
    for class_id, class_name in CLASS_NAMES.items():
        stats = per_class[class_name]
        out[f"{class_name}_precision"] = stats["precision"]
        out[f"{class_name}_recall"] = stats["recall"]
        out[f"{class_name}_f1"] = stats["f1"]
        out[f"{class_name}_predicted_support"] = int(matrix[:, class_id].sum())
    outer_total = float(matrix[2, :].sum())
    out["outer_to_inner_rate"] = float(matrix[2, 1] / outer_total) if outer_total else 0.0
    normal_total = float(matrix[0, :].sum())
    out["normal_to_inner_rate"] = float(matrix[0, 1] / normal_total) if normal_total else 0.0
    return out


def by_domain_metrics(records: list[dict[str, str]], threshold: float | None) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in records:
        grouped[row["dataset_id"]].append(row)
    return {domain: metrics(group, threshold) for domain, group in sorted(grouped.items())}


def candidate_thresholds(records: list[dict[str, str]]) -> list[float]:
    values = np.asarray([float(row["cf_margin_max"]) for row in records], dtype=np.float64)
    return sorted(set(float(value) for value in np.quantile(values, np.linspace(0.05, 0.95, 37)).tolist()))


def fget(payload: dict[str, object], key: str) -> float:
    return float(payload.get(key, math.nan))


def build_rows(source: list[dict[str, str]], target: list[dict[str, str]]) -> list[dict[str, object]]:
    base_source = metrics(source, None)
    base_target = metrics(target, None)
    base_domains = by_domain_metrics(source, None)
    rows: list[dict[str, object]] = []
    for threshold in [None] + candidate_thresholds(source):
        src = metrics(source, threshold)
        tgt = metrics(target, threshold)
        domains = by_domain_metrics(source, threshold)
        row: dict[str, object] = {
            "rule": "base_vote" if threshold is None else "cf_margin_max",
            "threshold": "" if threshold is None else threshold,
            "source_macro_f1": src["macro_f1"],
            "source_macro_f1_delta": fget(src, "macro_f1") - fget(base_source, "macro_f1"),
            "target_macro_f1_posthoc": tgt["macro_f1"],
            "target_macro_f1_delta_posthoc": fget(tgt, "macro_f1") - fget(base_target, "macro_f1"),
            "target_confusion_matrix_posthoc": json.dumps(tgt["confusion_matrix"]),
            "min_domain_normal_recall_delta": math.inf,
            "min_domain_normal_precision_delta": math.inf,
            "min_domain_outer_recall_delta": math.inf,
            "min_domain_outer_precision_delta": math.inf,
            "max_domain_outer_to_inner_increase": -math.inf,
        }
        for class_name in ["normal", "inner", "outer"]:
            for metric_name in ["recall", "precision", "predicted_support"]:
                key = f"{class_name}_{metric_name}"
                row[f"source_{key}"] = src[key]
                row[f"target_{key}_posthoc"] = tgt[key]
                row[f"target_{key}_delta_posthoc"] = fget(tgt, key) - fget(base_target, key)
        row["source_outer_to_inner_rate"] = src["outer_to_inner_rate"]
        row["target_outer_to_inner_rate_posthoc"] = tgt["outer_to_inner_rate"]
        row["target_outer_to_inner_rate_delta_posthoc"] = fget(tgt, "outer_to_inner_rate") - fget(
            base_target, "outer_to_inner_rate"
        )
        for domain, domain_metrics in domains.items():
            base_domain = base_domains[domain]
            for class_name in ["normal", "outer"]:
                for metric_name in ["recall", "precision"]:
                    key = f"{class_name}_{metric_name}"
                    delta = fget(domain_metrics, key) - fget(base_domain, key)
                    row[f"domain_{domain}_{key}_delta"] = delta
                    min_key = f"min_domain_{key}_delta"
                    row[min_key] = min(float(row[min_key]), delta)
            outer_flip_delta = fget(domain_metrics, "outer_to_inner_rate") - fget(base_domain, "outer_to_inner_rate")
            row[f"domain_{domain}_outer_to_inner_rate_delta"] = outer_flip_delta
            row["max_domain_outer_to_inner_increase"] = max(
                float(row["max_domain_outer_to_inner_increase"]), outer_flip_delta
            )
        rows.append(row)
    return rows


def is_allowed(
    row: dict[str, object],
    baseline: dict[str, object],
    macro_tol: float,
    inner_tol: float,
    protected_tol: float,
    outer_flip_tol: float,
) -> bool:
    if row["rule"] != "cf_margin_max":
        return False
    if float(row["source_macro_f1"]) < float(baseline["source_macro_f1"]) - macro_tol:
        return False
    if float(row["source_inner_recall"]) < float(baseline["source_inner_recall"]) - inner_tol:
        return False
    for key in [
        "min_domain_normal_recall_delta",
        "min_domain_normal_precision_delta",
        "min_domain_outer_recall_delta",
        "min_domain_outer_precision_delta",
    ]:
        if float(row[key]) < -protected_tol:
            return False
    if float(row["max_domain_outer_to_inner_increase"]) > outer_flip_tol:
        return False
    return True


def select_row(
    rows: list[dict[str, object]],
    macro_tol: float,
    inner_tol: float,
    protected_tol: float,
    outer_flip_tol: float,
) -> dict[str, object]:
    baseline = next(row for row in rows if row["rule"] == "base_vote")
    allowed = [row for row in rows if is_allowed(row, baseline, macro_tol, inner_tol, protected_tol, outer_flip_tol)]
    if not allowed:
        selected = dict(baseline)
        selected["selection_reason"] = "fallback base_vote: no domain-wise class-safe cf_margin_max threshold"
        return selected
    selected = dict(
        max(
            allowed,
            key=lambda row: (
                float(row["source_inner_predicted_support"]),
                float(row["source_inner_recall"]),
                -float(row["max_domain_outer_to_inner_increase"]),
                float(row["source_macro_f1"]),
            ),
        )
    )
    selected["selection_reason"] = (
        "domain-wise class-safe cf_margin_max: source Macro-F1/inner non-inferior, "
        "normal/outer precision+recall protected per source domain, outer-to-inner guarded"
    )
    return selected


def result_row(seed: int, target: str, selected: dict[str, object], baseline: dict[str, object]) -> dict[str, object]:
    row = {
        "seed": seed,
        "target": target,
        "selected_rule": selected["rule"],
        "selected_threshold": selected["threshold"],
        "selection_reason": selected["selection_reason"],
        "source_base_macro_f1": baseline["source_macro_f1"],
        "source_selected_macro_f1": selected["source_macro_f1"],
        "source_selected_inner_recall": selected["source_inner_recall"],
        "min_domain_normal_recall_delta": selected["min_domain_normal_recall_delta"],
        "min_domain_normal_precision_delta": selected["min_domain_normal_precision_delta"],
        "min_domain_outer_recall_delta": selected["min_domain_outer_recall_delta"],
        "min_domain_outer_precision_delta": selected["min_domain_outer_precision_delta"],
        "max_domain_outer_to_inner_increase": selected["max_domain_outer_to_inner_increase"],
        "target_base_macro_f1": baseline["target_macro_f1_posthoc"],
        "target_selected_macro_f1": selected["target_macro_f1_posthoc"],
        "target_macro_f1_delta": selected["target_macro_f1_delta_posthoc"],
        "target_base_inner_recall": baseline["target_inner_recall_posthoc"],
        "target_selected_inner_recall": selected["target_inner_recall_posthoc"],
        "target_inner_recall_delta": selected["target_inner_recall_delta_posthoc"],
        "target_base_outer_recall": baseline["target_outer_recall_posthoc"],
        "target_selected_outer_recall": selected["target_outer_recall_posthoc"],
        "target_outer_recall_delta": selected["target_outer_recall_delta_posthoc"],
        "target_outer_to_inner_delta": selected["target_outer_to_inner_rate_delta_posthoc"],
        "target_selected_confusion_matrix": selected["target_confusion_matrix_posthoc"],
    }
    row["safety_flag"] = bool(
        float(row["target_macro_f1_delta"]) < -0.01
        or float(row["target_inner_recall_delta"]) < -0.05
        or float(row["target_outer_recall_delta"]) < -0.05
        or float(row["target_outer_to_inner_delta"]) > 0.05
    )
    return row


def mean_std(values: list[float]) -> tuple[float, float]:
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    for target in sorted({str(row["target"]) for row in rows}):
        group = [row for row in rows if row["target"] == target]
        base_mean, base_std = mean_std([float(row["target_base_macro_f1"]) for row in group])
        selected_mean, selected_std = mean_std([float(row["target_selected_macro_f1"]) for row in group])
        delta_mean, delta_std = mean_std([float(row["target_macro_f1_delta"]) for row in group])
        inner_mean, inner_std = mean_std([float(row["target_selected_inner_recall"]) for row in group])
        out.append(
            {
                "target": target,
                "runs": len(group),
                "base_macro_f1_mean": base_mean,
                "base_macro_f1_std": base_std,
                "domain_safe_macro_f1_mean": selected_mean,
                "domain_safe_macro_f1_std": selected_std,
                "macro_f1_delta_mean": delta_mean,
                "macro_f1_delta_std": delta_std,
                "domain_safe_inner_recall_mean": inner_mean,
                "domain_safe_inner_recall_std": inner_std,
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
        "# Domain-Wise Class-Safe Counterfactual Boundary Audit",
        "",
        "Selection is source-only. `cf_margin_max` is allowed only when normal/outer recall and precision are preserved within each source domain, and source outer-to-inner flips are explicitly guarded.",
        "",
        "## Summary",
        "",
        "| Target | Runs | Base Macro-F1 | Domain-Safe Macro-F1 | Delta | Domain-Safe Inner Recall | Fallbacks | Safety Flags |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['target']} | {row['runs']} | {fmt(row['base_macro_f1_mean'])} +/- {fmt(row['base_macro_f1_std'])} | "
            f"{fmt(row['domain_safe_macro_f1_mean'])} +/- {fmt(row['domain_safe_macro_f1_std'])} | "
            f"{fmt(row['macro_f1_delta_mean'])} +/- {fmt(row['macro_f1_delta_std'])} | "
            f"{fmt(row['domain_safe_inner_recall_mean'])} +/- {fmt(row['domain_safe_inner_recall_std'])} | "
            f"{row['fallback_runs']} | {row['safety_flags']} |"
        )
    lines.extend(
        [
            "",
            "## Detail",
            "",
            "| Seed | Target | Selected | Threshold | Base Macro-F1 | Selected Macro-F1 | Delta | Inner Recall | Outer Recall Delta | Outer-to-Inner Delta | Safety |",
            "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in sorted(rows, key=lambda item: (str(item["target"]), int(item["seed"]))):
        lines.append(
            f"| {row['seed']} | {row['target']} | {row['selected_rule']} | {fmt(row['selected_threshold'])} | "
            f"{fmt(row['target_base_macro_f1'])} | {fmt(row['target_selected_macro_f1'])} | "
            f"{fmt(row['target_macro_f1_delta'])} | {fmt(row['target_selected_inner_recall'])} | "
            f"{fmt(row['target_outer_recall_delta'])} | {fmt(row['target_outer_to_inner_delta'])} | {row['safety_flag']} |"
        )
    lines.extend(
        [
            "",
            "## Adversarial Interpretation",
            "",
            "- Domain-wise source protection is stricter than aggregate class-safe selection.",
            "- If Ottawa safety flags remain, source domains still do not expose the target outer/inner reversal risk.",
            "- If the selector falls back, it is behaving conservatively under target-free rules.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.project_dir / "outputs" / "tables"
    rows = []
    for target in TARGETS:
        for seed in SEEDS:
            run_dir = output_dir / "class_safe_counterfactual_boundary_runs" / f"seed_{seed}_target_{target}"
            source = read_csv(run_dir / "source_recording_evidence.csv")
            target_records = read_csv(run_dir / "target_recording_evidence.csv")
            candidates = build_rows(source, target_records)
            selected = select_row(
                candidates,
                args.source_macro_tolerance,
                args.source_inner_recall_tolerance,
                args.source_protected_class_tolerance,
                args.outer_to_inner_tolerance,
            )
            baseline = next(row for row in candidates if row["rule"] == "base_vote")
            rows.append(result_row(seed, target, selected, baseline))
            write_csv(candidates, run_dir / "domain_wise_class_safe_candidates.csv")
    summary = summarize(rows)
    write_csv(rows, output_dir / "domain_wise_class_safe_boundary_detail.csv")
    write_csv(summary, output_dir / "domain_wise_class_safe_boundary_summary.csv")
    write_markdown(rows, summary, output_dir / "domain_wise_class_safe_boundary_audit.md")
    print(f"Wrote domain-wise class-safe boundary audit to {output_dir}")


if __name__ == "__main__":
    main()
