#!/usr/bin/env python3
"""Final target-free source-validation selector for paper-1 CIC-MAN."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


TARGETS = ["hust", "ottawa", "paderborn"]
FINAL_BASELINE = "CIC-MAN-v5-class-router-style"
FINAL_INTERVENTION = "CIC-MAN-A10-calibrated-gated-filterbank"

CANDIDATES = [
    {
        "method": FINAL_BASELINE,
        "prefix": "cic_man_v5_class_router_style_source_mixed_cross_dataset_task3_target_",
        "family": "stable_core",
        "final_eligible": True,
    },
    {
        "method": FINAL_INTERVENTION,
        "prefix": "cic_man_gated_filterbank_calibrated_source_mixed_cross_dataset_task3_target_",
        "family": "source_calibrated_optional_filterbank",
        "final_eligible": True,
    },
    {
        "method": "CIC-MAN-viewbank-disentangle",
        "prefix": "cic_man_gated_viewbank_disentangle_source_mixed_cross_dataset_task3_target_",
        "family": "full_viewbank_diagnostic",
        "final_eligible": False,
    },
    {
        "method": "CIC-MAN-viewbank-health-gate",
        "prefix": "cic_man_gated_viewbank_health_gate_source_mixed_cross_dataset_task3_target_",
        "family": "full_viewbank_diagnostic",
        "final_eligible": False,
    },
    {
        "method": "CIC-MAN-viewbank-orthogonal-aux",
        "prefix": "cic_man_gated_viewbank_orthogonal_aux_source_mixed_cross_dataset_task3_target_",
        "family": "full_viewbank_diagnostic",
        "final_eligible": False,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--source-noninferiority-tolerance", type=float, default=0.005)
    parser.add_argument("--min-gate-target-rate", type=float, default=0.001)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def source_val(metrics: dict[str, object]) -> dict[str, float]:
    val = metrics.get("best_val_metrics", {})
    return {
        "source_val_macro_f1": float(val.get("macro_f1", 0.0)),
        "mean_source_val_macro_f1": float(val.get("mean_dataset_macro_f1", val.get("macro_f1", 0.0))),
        "worst_source_val_macro_f1": float(val.get("worst_dataset_macro_f1", val.get("macro_f1", 0.0))),
        "source_val_accuracy": float(val.get("accuracy", 0.0)),
    }


def target_recording(checkpoint_dir: Path) -> float:
    path = checkpoint_dir / "recording_metrics.json"
    if not path.exists():
        return 0.0
    payload = load_json(path)
    return float(payload.get("mean_logits", {}).get("macro_f1", 0.0))


def gate_stats(metrics: dict[str, object]) -> dict[str, float]:
    history = metrics.get("history", [])
    train = history[-1].get("train", {}) if history else {}
    return {
        "gate_target_rate": float(train.get("gate_target_rate", 0.0)),
        "gate_mean": float(train.get("filterbank_gate_mean", 0.0)),
        "gate_positive_margin": float(train.get("gate_positive_margin", 0.0)),
    }


def diagnostics(checkpoint_dir: Path) -> dict[str, float]:
    path = checkpoint_dir / "diagnostics" / "cic_man_diagnostics_test.json"
    if not path.exists():
        path = checkpoint_dir / "cic_man_diagnostics_test.json"
    if not path.exists():
        return {"effective_agents": 0.0, "router_entropy": 0.0}
    router = load_json(path).get("router", {})
    return {
        "effective_agents": float(router.get("effective_agents_from_mean_weights", 0.0)),
        "router_entropy": float(router.get("normalized_mean_entropy", 0.0)),
    }


def collect_rows(project_dir: Path) -> list[dict[str, object]]:
    root = project_dir / "outputs" / "checkpoints"
    rows: list[dict[str, object]] = []
    for target in TARGETS:
        for candidate in CANDIDATES:
            checkpoint_dir = root / f"{candidate['prefix']}{target}"
            metrics_path = checkpoint_dir / "metrics.json"
            if not metrics_path.exists():
                continue
            metrics = load_json(metrics_path)
            rows.append(
                {
                    "target": target,
                    "method": candidate["method"],
                    "family": candidate["family"],
                    "final_eligible": bool(candidate["final_eligible"]),
                    "checkpoint_dir": str(checkpoint_dir),
                    "target_window_macro_f1_posthoc": float(metrics.get("test_metrics", {}).get("macro_f1", 0.0)),
                    "target_recording_macro_f1_posthoc": target_recording(checkpoint_dir),
                    "best_epoch": int(metrics.get("best_epoch", 0)),
                    **source_val(metrics),
                    **gate_stats(metrics),
                    **diagnostics(checkpoint_dir),
                }
            )
    return rows


def index_rows(rows: list[dict[str, object]]) -> dict[tuple[str, str], dict[str, object]]:
    return {(str(row["target"]), str(row["method"])): row for row in rows}


def add_final_selection(
    rows: list[dict[str, object]],
    *,
    source_noninferiority_tolerance: float,
    min_gate_target_rate: float,
) -> list[dict[str, object]]:
    row_index = index_rows(rows)
    selected_rows: list[dict[str, object]] = []
    for row in rows:
        row["selected_by_final_source_rule"] = False
        row["final_rule_reason"] = ""

    for target in TARGETS:
        baseline = row_index[(target, FINAL_BASELINE)]
        intervention = row_index[(target, FINAL_INTERVENTION)]
        source_delta = float(intervention["worst_source_val_macro_f1"]) - float(baseline["worst_source_val_macro_f1"])
        gate_rate = float(intervention["gate_target_rate"])
        intervention_allowed = source_delta >= -source_noninferiority_tolerance and gate_rate >= min_gate_target_rate
        selected = intervention if intervention_allowed else baseline
        selected["selected_by_final_source_rule"] = True
        if intervention_allowed:
            selected["final_rule_reason"] = (
                f"A10 allowed: source worst-F1 delta {source_delta:.6f} >= "
                f"-{source_noninferiority_tolerance:.6f}, gate target rate {gate_rate:.6f} >= {min_gate_target_rate:.6f}."
            )
        else:
            baseline["final_rule_reason"] = (
                f"Fallback to v5: A10 source worst-F1 delta {source_delta:.6f}, "
                f"gate target rate {gate_rate:.6f}."
            )
        selected_rows.append(selected)

        for row in rows:
            if row["target"] == target and not bool(row["final_eligible"]):
                row["final_rule_reason"] = "Diagnostic-only: full-viewbank/health-style variants are audited but not final-eligible."

    return selected_rows


def mean_value(rows: list[dict[str, object]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / max(len(rows), 1)


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(
    rows: list[dict[str, object]],
    selected: list[dict[str, object]],
    path: Path,
    *,
    source_noninferiority_tolerance: float,
    min_gate_target_rate: float,
) -> None:
    lines = [
        "# Final Source-Validation Selector",
        "",
        "Target data are used only in post-hoc audit columns. The final rule uses source validation and source-side gate evidence.",
        "",
        f"Rule: select A10 over v5 when source worst Macro-F1 is within `{source_noninferiority_tolerance:.4f}` of v5 and source gate target rate is at least `{min_gate_target_rate:.4f}`; otherwise fall back to v5.",
        "",
        "## Final Selection",
        "",
        "| Target | Selected Method | Source Worst F1 | Gate Target Rate | Post-hoc Window F1 | Post-hoc Recording F1 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in selected:
        lines.append(
            f"| {row['target']} | {row['method']} | {float(row['worst_source_val_macro_f1']):.6f} | "
            f"{float(row['gate_target_rate']):.6f} | {float(row['target_window_macro_f1_posthoc']):.6f} | "
            f"{float(row['target_recording_macro_f1_posthoc']):.6f} |"
        )
    lines.extend(
        [
            "",
            f"Final selector mean window Macro-F1: `{mean_value(selected, 'target_window_macro_f1_posthoc'):.6f}`",
            f"Final selector mean recording Macro-F1: `{mean_value(selected, 'target_recording_macro_f1_posthoc'):.6f}`",
            "",
            "## Candidate Audit",
            "",
            "| Target | Method | Eligible | Source Worst F1 | Gate Target Rate | Window F1 Post-hoc | Recording F1 Post-hoc | Selected |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in sorted(rows, key=lambda item: (str(item["target"]), not bool(item["selected_by_final_source_rule"]), str(item["method"]))):
        selected_flag = "yes" if row["selected_by_final_source_rule"] else ""
        eligible = "yes" if row["final_eligible"] else "no"
        lines.append(
            f"| {row['target']} | {row['method']} | {eligible} | "
            f"{float(row['worst_source_val_macro_f1']):.6f} | {float(row['gate_target_rate']):.6f} | "
            f"{float(row['target_window_macro_f1_posthoc']):.6f} | "
            f"{float(row['target_recording_macro_f1_posthoc']):.6f} | {selected_flag} |"
        )
    lines.extend(
        [
            "",
            "Interpretation: full-viewbank and health/style variants remain diagnostic evidence. They are not final-eligible because their target-free source validation and prior audits did not produce a stable improvement over v5/A10.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_dir = args.project_dir.expanduser().resolve()
    output_dir = (args.output_dir or project_dir / "outputs" / "tables").expanduser().resolve()
    rows = collect_rows(project_dir)
    selected = add_final_selection(
        rows,
        source_noninferiority_tolerance=args.source_noninferiority_tolerance,
        min_gate_target_rate=args.min_gate_target_rate,
    )
    rows.sort(key=lambda item: (str(item["target"]), str(item["method"])))
    selected.sort(key=lambda item: str(item["target"]))
    write_csv(rows, output_dir / "source_validation_final_selector_detail.csv")
    write_csv(selected, output_dir / "source_validation_final_selector_selected.csv")
    write_markdown(
        rows,
        selected,
        output_dir / "source_validation_final_selector.md",
        source_noninferiority_tolerance=args.source_noninferiority_tolerance,
        min_gate_target_rate=args.min_gate_target_rate,
    )
    print(f"Wrote final selector outputs to {output_dir}")


if __name__ == "__main__":
    main()
