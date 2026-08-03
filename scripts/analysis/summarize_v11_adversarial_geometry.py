#!/usr/bin/env python3
"""Summarize the Ottawa-focused v11 adversarial-geometry hard-class run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METHOD_DIRS = {
    "A10-final-seed42": "cic_man_gated_filterbank_calibrated_source_mixed_cross_dataset_task3_target_ottawa",
    "v10-source-heldout-hard-class": "cic_man_v10_source_heldout_hard_class_support_source_mixed_cross_dataset_task3_target_ottawa",
    "v11-adversarial-geometry-hard-class": "cic_man_v11_adversarial_geometry_hard_class_source_mixed_cross_dataset_task3_target_ottawa",
}
CLASS_NAMES = {0: "normal", 1: "inner", 2: "outer"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def class_rows(method: str, evaluation: str, matrix: list[list[int]]) -> list[dict[str, object]]:
    rows = []
    for class_id, class_name in CLASS_NAMES.items():
        tp = matrix[class_id][class_id]
        fn = sum(matrix[class_id]) - tp
        fp = sum(row[class_id] for row in matrix) - tp
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        rows.append(
            {
                "method": method,
                "evaluation": evaluation,
                "class_id": class_id,
                "class_name": class_name,
                "recall": recall,
                "precision": precision,
                "f1": f1,
                "predicted_support": sum(row[class_id] for row in matrix),
                "support": sum(matrix[class_id]),
            }
        )
    return rows


def collect(project_dir: Path) -> list[dict[str, object]]:
    root = project_dir / "outputs" / "checkpoints"
    rows = []
    for method, dirname in METHOD_DIRS.items():
        run_dir = root / dirname
        metrics = load_json(run_dir / "metrics.json")
        rows.extend(class_rows(method, "window", metrics["test_metrics"]["confusion_matrix"]))
        recording = load_json(run_dir / "recording_metrics.json")
        rows.extend(class_rows(method, "recording", recording["mean_logits"]["confusion_matrix"]))
    return rows


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def row_lookup(rows: list[dict[str, object]]) -> dict[tuple[str, str, str], dict[str, object]]:
    return {(str(row["method"]), str(row["evaluation"]), str(row["class_name"])): row for row in rows}


def write_markdown(rows: list[dict[str, object]], path: Path) -> None:
    lookup = row_lookup(rows)
    lines = [
        "# v11 Adversarial-Geometry Hard-Class Audit",
        "",
        "Scope: Ottawa only. v11 uses source-only adversarial geometry risk weights. Target labels are used only for post-hoc evaluation.",
        "",
        "| Method | Evaluation | Normal Recall | Inner Recall | Outer Recall | Inner Predicted Support | Macro-F1 Proxy |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHOD_DIRS:
        for evaluation in ["window", "recording"]:
            items = [lookup[(method, evaluation, name)] for name in ["normal", "inner", "outer"]]
            macro_f1 = sum(float(item["f1"]) for item in items) / 3.0
            lines.append(
                f"| {method} | {evaluation} | {float(items[0]['recall']):.6f} | "
                f"{float(items[1]['recall']):.6f} | {float(items[2]['recall']):.6f} | "
                f"{int(items[1]['predicted_support'])} | {macro_f1:.6f} |"
            )
    lines.extend(
        [
            "",
            "## Adversarial Conclusion",
            "",
            "- v11 improves Ottawa recording Macro-F1 relative to v10/A10, but it does not improve the target failure mode: recording-level inner recall remains zero.",
            "- Window-level inner recall is lower than v9, so adversarial geometry weighting alone is not a valid class-semantic fix.",
            "- The next Ottawa-specific step should change the decision boundary or calibration for inner, not merely increase source semantic-margin weight.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_dir = args.project_dir.expanduser().resolve()
    output_dir = (args.output_dir or project_dir / "outputs" / "tables").expanduser().resolve()
    rows = collect(project_dir)
    write_csv(rows, output_dir / "v11_adversarial_geometry_audit.csv")
    write_markdown(rows, output_dir / "v11_adversarial_geometry_audit.md")
    print(f"Wrote v11 adversarial geometry audit to {output_dir}")


if __name__ == "__main__":
    main()
