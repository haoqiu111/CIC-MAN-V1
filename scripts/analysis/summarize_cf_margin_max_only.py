#!/usr/bin/env python3
"""Summarize strict cf_margin_max-only selection from cached boundary candidates."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


TARGETS = ["hust", "ottawa", "paderborn"]
SEEDS = [42, 2025, 2026]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--source-drop-tolerance", type=float, default=0.01)
    parser.add_argument("--source-inner-drop-tolerance", type=float, default=0.02)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        return math.nan
    return float(value)


def select_cf_margin_max(rows: list[dict[str, str]], macro_tol: float, inner_tol: float) -> dict[str, object]:
    baseline = next(row for row in rows if row["rule"] == "base_vote")
    base_macro = fnum(baseline, "source_macro_f1")
    base_inner = fnum(baseline, "source_inner_recall")
    allowed = [
        row
        for row in rows
        if row["rule"] == "cf_margin_max"
        and fnum(row, "source_macro_f1") >= base_macro - macro_tol
        and fnum(row, "source_inner_recall") >= base_inner - inner_tol
    ]
    if not allowed:
        selected = dict(baseline)
        selected["selection_reason"] = "fallback base_vote: no source-noninferior cf_margin_max threshold"
    else:
        selected = dict(
            max(
                allowed,
                key=lambda row: (
                    fnum(row, "source_inner_predicted_support"),
                    fnum(row, "source_inner_recall"),
                    fnum(row, "source_macro_f1"),
                ),
            )
        )
        selected["selection_reason"] = (
            f"cf_margin_max-only source selection: source Macro-F1 >= {base_macro - macro_tol:.6f}, "
            f"source inner recall >= {base_inner - inner_tol:.6f}; maximize source inner support"
        )
    return {
        "selected_rule": selected["rule"],
        "selected_threshold": selected["threshold"],
        "selection_reason": selected["selection_reason"],
        "source_base_macro_f1": fnum(baseline, "source_macro_f1"),
        "source_selected_macro_f1": fnum(selected, "source_macro_f1"),
        "source_base_inner_recall": fnum(baseline, "source_inner_recall"),
        "source_selected_inner_recall": fnum(selected, "source_inner_recall"),
        "target_base_macro_f1": fnum(baseline, "target_macro_f1_posthoc"),
        "target_selected_macro_f1": fnum(selected, "target_macro_f1_posthoc"),
        "target_macro_f1_delta": fnum(selected, "target_macro_f1_posthoc") - fnum(baseline, "target_macro_f1_posthoc"),
        "target_base_inner_recall": fnum(baseline, "target_inner_recall_posthoc"),
        "target_selected_inner_recall": fnum(selected, "target_inner_recall_posthoc"),
        "target_inner_recall_delta": fnum(selected, "target_inner_recall_posthoc") - fnum(baseline, "target_inner_recall_posthoc"),
        "target_base_inner_support": int(float(baseline["target_inner_predicted_support_posthoc"])),
        "target_selected_inner_support": int(float(selected["target_inner_predicted_support_posthoc"])),
        "target_selected_confusion_matrix": selected.get("target_confusion_matrix_posthoc", ""),
    }


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    for target in sorted({str(row["target"]) for row in rows}):
        group = [row for row in rows if row["target"] == target]
        base_mean, base_std = mean_std([float(row["target_base_macro_f1"]) for row in group])
        sel_mean, sel_std = mean_std([float(row["target_selected_macro_f1"]) for row in group])
        delta_mean, delta_std = mean_std([float(row["target_macro_f1_delta"]) for row in group])
        inner_mean, inner_std = mean_std([float(row["target_selected_inner_recall"]) for row in group])
        inner_delta_mean, inner_delta_std = mean_std([float(row["target_inner_recall_delta"]) for row in group])
        out.append(
            {
                "target": target,
                "runs": len(group),
                "base_macro_f1_mean": base_mean,
                "base_macro_f1_std": base_std,
                "cf_margin_max_macro_f1_mean": sel_mean,
                "cf_margin_max_macro_f1_std": sel_std,
                "macro_f1_delta_mean": delta_mean,
                "macro_f1_delta_std": delta_std,
                "cf_margin_max_inner_recall_mean": inner_mean,
                "cf_margin_max_inner_recall_std": inner_std,
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
        "# Strict cf_margin_max Repeated-Seed Audit",
        "",
        "This table reuses cached counterfactual candidate sweeps but restricts selection to `cf_margin_max` only. Each threshold is selected from source validation only.",
        "",
        "## Summary",
        "",
        "| Target | Runs | Base Macro-F1 | cf_margin_max Macro-F1 | Delta | cf_margin_max Inner Recall | Inner Delta | Fallbacks | Safety Flags |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['target']} | {row['runs']} | {fmt(row['base_macro_f1_mean'])} ± {fmt(row['base_macro_f1_std'])} | "
            f"{fmt(row['cf_margin_max_macro_f1_mean'])} ± {fmt(row['cf_margin_max_macro_f1_std'])} | "
            f"{fmt(row['macro_f1_delta_mean'])} ± {fmt(row['macro_f1_delta_std'])} | "
            f"{fmt(row['cf_margin_max_inner_recall_mean'])} ± {fmt(row['cf_margin_max_inner_recall_std'])} | "
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
            "## Adversarial Conclusion",
            "",
            "- If cf_margin_max helps only seed 42 on Ottawa, it is not yet a stable final method.",
            "- If HUST has safety flags, cf_margin_max must remain a target-specific diagnostic selector rather than a universal recording rule.",
            "- Paderborn fallback behavior means the source-only rule correctly refuses to intervene, consistent with the mechanism-coverage limitation.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.project_dir / "outputs" / "tables"
    rows = []
    for target in TARGETS:
        for seed in SEEDS:
            candidate_path = (
                output_dir
                / "counterfactual_boundary_runs"
                / f"seed_{seed}_target_{target}"
                / "counterfactual_boundary_candidates.csv"
            )
            selected = select_cf_margin_max(
                read_csv(candidate_path),
                args.source_drop_tolerance,
                args.source_inner_drop_tolerance,
            )
            selected["seed"] = seed
            selected["target"] = target
            selected["safety_flag"] = bool(
                float(selected["target_macro_f1_delta"]) < -0.01
                or float(selected["target_inner_recall_delta"]) < -0.05
            )
            rows.append(selected)
    summary = summarize(rows)
    write_csv(rows, output_dir / "repeated_cf_margin_max_only_detail.csv")
    write_csv(summary, output_dir / "repeated_cf_margin_max_only_summary.csv")
    write_markdown(rows, summary, output_dir / "repeated_cf_margin_max_only_audit.md")
    print(f"Wrote strict cf_margin_max audit to {output_dir}")


if __name__ == "__main__":
    main()
