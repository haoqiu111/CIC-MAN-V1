#!/usr/bin/env python3
"""Summarize the source-only class semantic coverage variant."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


CLASS_NAMES = {0: "normal", 1: "inner", 2: "outer"}
TARGETS = ["ottawa", "paderborn"]


RUNS = {
    "CIC-MAN-v5-seed42": {
        "ottawa": "cic_man_v5_class_router_style_source_mixed_cross_dataset_task3_target_ottawa",
        "paderborn": "cic_man_v5_class_router_style_source_mixed_cross_dataset_task3_target_paderborn",
    },
    "CIC-MAN-A10-seed42": {
        "ottawa": "cic_man_gated_filterbank_calibrated_source_mixed_cross_dataset_task3_target_ottawa",
        "paderborn": "cic_man_gated_filterbank_calibrated_source_mixed_cross_dataset_task3_target_paderborn",
    },
    "CIC-MAN-final-selector-seed42": {
        "ottawa": "cic_man_gated_filterbank_calibrated_source_mixed_cross_dataset_task3_target_ottawa",
        "paderborn": "cic_man_v5_class_router_style_source_mixed_cross_dataset_task3_target_paderborn",
    },
    "CIC-MAN-v9-class-semantic-coverage": {
        "ottawa": "cic_man_v9_class_semantic_coverage_source_mixed_cross_dataset_task3_target_ottawa",
        "paderborn": "cic_man_v9_class_semantic_coverage_source_mixed_cross_dataset_task3_target_paderborn",
    },
    "CIC-MAN-v10-source-heldout-hard-class": {
        "ottawa": "cic_man_v10_source_heldout_hard_class_support_source_mixed_cross_dataset_task3_target_ottawa",
        "paderborn": "cic_man_v10_source_heldout_hard_class_support_source_mixed_cross_dataset_task3_target_paderborn",
    },
    "CIC-MAN-v11-adversarial-geometry-hard-class": {
        "ottawa": "cic_man_v11_adversarial_geometry_hard_class_source_mixed_cross_dataset_task3_target_ottawa",
        "paderborn": "cic_man_v10_source_heldout_hard_class_support_source_mixed_cross_dataset_task3_target_paderborn",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def per_class_metrics(matrix: list[list[int]], class_id: int) -> dict[str, float]:
    total = sum(sum(row) for row in matrix)
    tp = matrix[class_id][class_id]
    fn = sum(matrix[class_id]) - tp
    fp = sum(row[class_id] for row in matrix) - tp
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return {
        "support": float(sum(matrix[class_id])),
        "predicted_support": float(sum(row[class_id] for row in matrix)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "total": float(total),
    }


def collect_rows(project_dir: Path) -> list[dict[str, object]]:
    root = project_dir / "outputs" / "checkpoints"
    rows = []
    for method, target_dirs in RUNS.items():
        for target in TARGETS:
            run_dir = root / target_dirs[target]
            metrics_path = run_dir / "metrics.json"
            if metrics_path.exists():
                metrics = load_json(metrics_path)
                matrix = metrics.get("test_metrics", {}).get("confusion_matrix", [])
                for class_id, class_name in CLASS_NAMES.items():
                    item = per_class_metrics(matrix, class_id)
                    rows.append(
                        {
                            "method": method,
                            "target": target,
                            "evaluation": "window",
                            "class_id": class_id,
                            "class_name": class_name,
                            **item,
                        }
                    )
            recording_path = run_dir / "recording_metrics.json"
            if recording_path.exists():
                recording = load_json(recording_path)
                matrix = recording.get("mean_logits", {}).get("confusion_matrix", [])
                for class_id, class_name in CLASS_NAMES.items():
                    item = per_class_metrics(matrix, class_id)
                    rows.append(
                        {
                            "method": method,
                            "target": target,
                            "evaluation": "recording",
                            "class_id": class_id,
                            "class_name": class_name,
                            **item,
                        }
                    )
    return rows


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object) -> str:
    return f"{float(value):.6f}" if isinstance(value, float) else str(value)


def write_markdown(rows: list[dict[str, object]], path: Path) -> None:
    lines = [
        "# Source-Only Class Semantic Coverage Diagnosis",
        "",
        "Variant: v9 adds source-only class semantic coverage on top of the v5 stable core settings. Target labels are used only for post-hoc diagnosis.",
        "",
        "## Focus Recall",
        "",
        "| Method | Target | Evaluation | Normal Recall | Inner Recall | Outer Recall |",
        "|---|---|---|---:|---:|---:|",
    ]
    for method in RUNS:
        for target in TARGETS:
            for evaluation in ["window", "recording"]:
                selected = [
                    row
                    for row in rows
                    if row["method"] == method and row["target"] == target and row["evaluation"] == evaluation
                ]
                recall_by_class = {row["class_name"]: float(row["recall"]) for row in selected}
                lines.append(
                    f"| {method} | {target} | {evaluation} | "
                    f"{recall_by_class.get('normal', 0.0):.6f} | "
                    f"{recall_by_class.get('inner', 0.0):.6f} | "
                    f"{recall_by_class.get('outer', 0.0):.6f} |"
                )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- v9 does not solve the target class-semantics failure. Ottawa mean-logits recording inner recall remains `0.000000`, although window-level inner recall increases slightly over v5/A10.",
            "- Paderborn remains dominated by inner/outer-to-normal collapse. Recording-level inner recall is still `0.000000`, and outer recall is unchanged from the v5 seed42 pattern.",
            "- The current batch-local prototype margin is therefore too source-local: it improves source geometry losses but does not create target-transferable fault semantics.",
            "- v10 replaces fixed fault upweighting with source-held-out class weights. It improves Ottawa recording Macro-F1 relative to v9, but inner recall remains zero and Paderborn is unchanged because source-held-out recall does not flag Paderborn inner/outer as hard classes.",
            "- v11 uses adversarial source-geometry risk weights for Ottawa only. It improves Ottawa recording Macro-F1 relative to v10/A10, but recording-level inner recall remains `0.000000`; the gain comes from normal/outer behavior rather than solving the inner-fault semantic hole.",
            "- The next mechanism should not trust plain source-held-out class recall as an oracle. It needs an adversarial source-validation signal that detects source-target semantic blind spots, such as leave-domain feature geometry disagreement, mechanism-specific prototype spread, or source-domain shortcut reversal at the class level.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_dir = args.project_dir.expanduser().resolve()
    output_dir = (args.output_dir or project_dir / "outputs" / "tables").expanduser().resolve()
    rows = collect_rows(project_dir)
    write_csv(rows, output_dir / "class_semantic_coverage_diagnosis.csv")
    write_markdown(rows, output_dir / "class_semantic_coverage_diagnosis.md")
    print(f"Wrote class semantic coverage diagnosis to {output_dir}")


if __name__ == "__main__":
    main()
