#!/usr/bin/env python3
"""Per-class confusion and error diagnosis for the repeated-seed main matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev


CLASS_NAMES = {0: "normal", 1: "inner", 2: "outer"}
FOCUS_TARGETS = {"ottawa", "paderborn"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_repeated_detail(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def per_class_rows(
    *,
    method: str,
    selected_base_method: str,
    seed: int,
    target: str,
    evaluation: str,
    matrix: list[list[int]],
) -> list[dict[str, object]]:
    total = sum(sum(row) for row in matrix)
    rows = []
    for class_id, class_name in CLASS_NAMES.items():
        tp = matrix[class_id][class_id]
        fn = sum(matrix[class_id]) - tp
        fp = sum(row[class_id] for row in matrix) - tp
        tn = total - tp - fn - fp
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        rows.append(
            {
                "method": method,
                "selected_base_method": selected_base_method,
                "seed": seed,
                "target": target,
                "evaluation": evaluation,
                "class_id": class_id,
                "class_name": class_name,
                "support": sum(matrix[class_id]),
                "predicted_support": sum(row[class_id] for row in matrix),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return rows


def confusion_flow_rows(
    *,
    method: str,
    selected_base_method: str,
    seed: int,
    target: str,
    evaluation: str,
    matrix: list[list[int]],
) -> list[dict[str, object]]:
    rows = []
    for true_id, row in enumerate(matrix):
        support = sum(row)
        for pred_id, count in enumerate(row):
            rows.append(
                {
                    "method": method,
                    "selected_base_method": selected_base_method,
                    "seed": seed,
                    "target": target,
                    "evaluation": evaluation,
                    "true_class_id": true_id,
                    "true_class_name": CLASS_NAMES[true_id],
                    "pred_class_id": pred_id,
                    "pred_class_name": CLASS_NAMES[pred_id],
                    "count": count,
                    "rate_within_true_class": safe_div(count, support),
                    "is_error": true_id != pred_id,
                }
            )
    return rows


def collect_rows(project_dir: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    detail_path = project_dir / "outputs" / "tables" / "repeated_seed_main_matrix_detail.csv"
    repeated = read_repeated_detail(detail_path)
    per_class: list[dict[str, object]] = []
    flows: list[dict[str, object]] = []
    seen: set[tuple[str, int, str, str]] = set()

    for row in repeated:
        method = row["method"]
        seed = int(row["seed"])
        target = row["target"]
        run_dir = Path(row["checkpoint_dir"])
        selected_base = row.get("selected_base_method", "")
        key = (method, seed, target, str(run_dir))
        if key in seen:
            continue
        seen.add(key)

        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists():
            metrics = load_json(metrics_path)
            matrix = metrics.get("test_metrics", {}).get("confusion_matrix", [])
            if matrix:
                per_class.extend(
                    per_class_rows(
                        method=method,
                        selected_base_method=selected_base,
                        seed=seed,
                        target=target,
                        evaluation="window",
                        matrix=matrix,
                    )
                )
                flows.extend(
                    confusion_flow_rows(
                        method=method,
                        selected_base_method=selected_base,
                        seed=seed,
                        target=target,
                        evaluation="window",
                        matrix=matrix,
                    )
                )

        recording_path = run_dir / "recording_metrics.json"
        if recording_path.exists():
            recording = load_json(recording_path)
            matrix = recording.get("mean_logits", {}).get("confusion_matrix", [])
            if matrix:
                per_class.extend(
                    per_class_rows(
                        method=method,
                        selected_base_method=selected_base,
                        seed=seed,
                        target=target,
                        evaluation="recording",
                        matrix=matrix,
                    )
                )
                flows.extend(
                    confusion_flow_rows(
                        method=method,
                        selected_base_method=selected_base,
                        seed=seed,
                        target=target,
                        evaluation="recording",
                        matrix=matrix,
                    )
                )
    return per_class, flows


def summarize_per_class(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["method"]), str(row["target"]), str(row["evaluation"]), int(row["class_id"]))].append(row)

    summary = []
    for (method, target, evaluation, class_id), group in sorted(groups.items()):
        recalls = [float(item["recall"]) for item in group]
        precisions = [float(item["precision"]) for item in group]
        f1s = [float(item["f1"]) for item in group]
        supports = [float(item["support"]) for item in group]
        pred_supports = [float(item["predicted_support"]) for item in group]
        summary.append(
            {
                "method": method,
                "target": target,
                "evaluation": evaluation,
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "num_runs": len(group),
                "support_mean": mean(supports),
                "predicted_support_mean": mean(pred_supports),
                "precision_mean": mean(precisions),
                "precision_std": stdev(precisions) if len(precisions) > 1 else 0.0,
                "recall_mean": mean(recalls),
                "recall_std": stdev(recalls) if len(recalls) > 1 else 0.0,
                "f1_mean": mean(f1s),
                "f1_std": stdev(f1s) if len(f1s) > 1 else 0.0,
            }
        )
    return summary


def summarize_flows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str, int, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                str(row["method"]),
                str(row["target"]),
                str(row["evaluation"]),
                int(row["true_class_id"]),
                int(row["pred_class_id"]),
            )
        ].append(row)

    summary = []
    for (method, target, evaluation, true_id, pred_id), group in sorted(groups.items()):
        counts = [float(item["count"]) for item in group]
        rates = [float(item["rate_within_true_class"]) for item in group]
        summary.append(
            {
                "method": method,
                "target": target,
                "evaluation": evaluation,
                "true_class_id": true_id,
                "true_class_name": CLASS_NAMES[true_id],
                "pred_class_id": pred_id,
                "pred_class_name": CLASS_NAMES[pred_id],
                "is_error": true_id != pred_id,
                "num_runs": len(group),
                "count_mean": mean(counts),
                "count_std": stdev(counts) if len(counts) > 1 else 0.0,
                "rate_mean": mean(rates),
                "rate_std": stdev(rates) if len(rates) > 1 else 0.0,
            }
        )
    return summary


def recording_error_breakdown(project_dir: Path) -> list[dict[str, object]]:
    repeated = read_repeated_detail(project_dir / "outputs" / "tables" / "repeated_seed_main_matrix_detail.csv")
    rows = []
    for item in repeated:
        target = item["target"]
        if target not in FOCUS_TARGETS:
            continue
        pred_path = Path(item["checkpoint_dir"]) / "recording_predictions.csv"
        if not pred_path.exists():
            continue
        with pred_path.open("r", newline="", encoding="utf-8") as f:
            predictions = list(csv.DictReader(f))
        total_by_class = Counter(row["label"] for row in predictions)
        error_by_class = Counter(row["label"] for row in predictions if row["true_label_id"] != row["mean_logit_pred"])
        error_by_pair = Counter(
            (row["label"], CLASS_NAMES.get(int(row["mean_logit_pred"]), str(row["mean_logit_pred"])))
            for row in predictions
            if row["true_label_id"] != row["mean_logit_pred"]
        )
        for label, total in sorted(total_by_class.items()):
            errors = error_by_class[label]
            rows.append(
                {
                    "method": item["method"],
                    "selected_base_method": item.get("selected_base_method", ""),
                    "seed": int(item["seed"]),
                    "target": target,
                    "label": label,
                    "num_recordings": total,
                    "num_errors": errors,
                    "error_rate": safe_div(errors, total),
                    "top_wrong_prediction": top_wrong_prediction(error_by_pair, label),
                }
            )
    return rows


def top_wrong_prediction(counter: Counter[tuple[str, str]], label: str) -> str:
    candidates = [(pred, count) for (true_label, pred), count in counter.items() if true_label == label]
    if not candidates:
        return ""
    pred, count = max(candidates, key=lambda item: item[1])
    return f"{pred}:{count}"


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.3f}"
    return str(value)


def focus_rows(summary: list[dict[str, object]], *, method: str, evaluation: str) -> list[dict[str, object]]:
    return [
        row
        for row in summary
        if row["method"] == method and row["evaluation"] == evaluation and row["target"] in FOCUS_TARGETS
    ]


def write_markdown(
    per_class_summary: list[dict[str, object]],
    flow_summary: list[dict[str, object]],
    error_rows: list[dict[str, object]],
    path: Path,
) -> None:
    lines = [
        "# Repeated-Seed Per-Class Error Diagnosis",
        "",
        "Scope: seeds 42/2025/2026, source-mixed cross-dataset DG. Target labels are used only for post-hoc diagnosis.",
        "",
        "## Final Selector Per-Class Recall",
        "",
        "| Target | Evaluation | Class | Recall Mean | Recall Std | Precision Mean | F1 Mean | Predicted Support Mean |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in focus_rows(per_class_summary, method="CIC-MAN-final-selector", evaluation="window"):
        lines.append(per_class_line(row))
    for row in focus_rows(per_class_summary, method="CIC-MAN-final-selector", evaluation="recording"):
        lines.append(per_class_line(row))

    lines.extend(
        [
            "",
            "## Main Error Flows For Final Selector",
            "",
            "| Target | Evaluation | True Class | Predicted Class | Error Rate Within True Class | Count Mean |",
            "|---|---|---|---|---:|---:|",
        ]
    )
    important_flows = [
        row
        for row in flow_summary
        if row["method"] == "CIC-MAN-final-selector"
        and row["target"] in FOCUS_TARGETS
        and row["is_error"] is True
        and float(row["rate_mean"]) >= 0.10
    ]
    important_flows.sort(key=lambda row: (str(row["target"]), str(row["evaluation"]), -float(row["rate_mean"])))
    for row in important_flows:
        lines.append(
            f"| {row['target']} | {row['evaluation']} | {row['true_class_name']} | {row['pred_class_name']} | "
            f"{fmt(float(row['rate_mean']))} | {fmt(float(row['count_mean']))} |"
        )

    lines.extend(
        [
            "",
            "## Method Comparison On Focus Targets",
            "",
            "| Method | Target | Evaluation | Class | Recall Mean | F1 Mean |",
            "|---|---|---|---|---:|---:|",
        ]
    )
    for row in per_class_summary:
        if row["target"] not in FOCUS_TARGETS or row["evaluation"] != "recording":
            continue
        lines.append(
            f"| {row['method']} | {row['target']} | {row['evaluation']} | {row['class_name']} | "
            f"{fmt(float(row['recall_mean']))} | {fmt(float(row['f1_mean']))} |"
        )

    lines.extend(
        [
            "",
            "## Recording Error Rate By Label",
            "",
            "| Method | Seed | Target | Label | Error Rate | Top Wrong Prediction |",
            "|---|---:|---|---|---:|---|",
        ]
    )
    for row in error_rows:
        if row["method"] != "CIC-MAN-final-selector":
            continue
        lines.append(
            f"| {row['method']} | {row['seed']} | {row['target']} | {row['label']} | "
            f"{fmt(float(row['error_rate']))} | {row['top_wrong_prediction']} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Ottawa instability is dominated by the inner class: recording-level inner recall is near zero for the final selector, while outer is often predicted correctly.",
            "- Paderborn instability is dominated by severe inner-to-normal and outer-to-normal collapse. This confirms the earlier feature-geometry finding that Paderborn fault classes, especially inner, are not semantically covered by the source class geometry.",
            "- A10/final selector helps slightly on average, but it does not solve the class-semantics problem. The next method change should target class prototype coverage or mechanism-aware class semantics, not more global router balance.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def per_class_line(row: dict[str, object]) -> str:
    return (
        f"| {row['target']} | {row['evaluation']} | {row['class_name']} | "
        f"{fmt(float(row['recall_mean']))} | {fmt(float(row['recall_std']))} | "
        f"{fmt(float(row['precision_mean']))} | {fmt(float(row['f1_mean']))} | "
        f"{fmt(float(row['predicted_support_mean']))} |"
    )


def main() -> None:
    args = parse_args()
    project_dir = args.project_dir.expanduser().resolve()
    output_dir = (args.output_dir or project_dir / "outputs" / "tables").expanduser().resolve()

    per_class, flows = collect_rows(project_dir)
    per_class_summary = summarize_per_class(per_class)
    flow_summary = summarize_flows(flows)
    error_rows = recording_error_breakdown(project_dir)

    write_csv(per_class, output_dir / "repeated_seed_per_class_detail.csv")
    write_csv(per_class_summary, output_dir / "repeated_seed_per_class_summary.csv")
    write_csv(flows, output_dir / "repeated_seed_confusion_flow_detail.csv")
    write_csv(flow_summary, output_dir / "repeated_seed_confusion_flow_summary.csv")
    write_csv(error_rows, output_dir / "repeated_seed_recording_error_breakdown.csv")
    write_markdown(
        per_class_summary,
        flow_summary,
        error_rows,
        output_dir / "repeated_seed_class_error_diagnosis.md",
    )
    print(f"Wrote repeated-seed class-error diagnosis outputs to {output_dir}")


if __name__ == "__main__":
    main()
