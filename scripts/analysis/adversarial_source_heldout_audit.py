#!/usr/bin/env python3
"""Adversarial audit of source-held-out signals versus target class failures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


CLASS_NAMES = {0: "normal", 1: "inner", 2: "outer"}
TARGETS = ["hust", "ottawa", "paderborn"]
SEEDS = [42, 2025, 2026]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_dir(root: Path, seed: int, target: str, method: str) -> Path:
    if method == "v5":
        if seed == 42:
            return root / f"cic_man_v5_class_router_style_source_mixed_cross_dataset_task3_target_{target}"
        return root / f"seed_{seed}_cic_man_v5_class_router_style_source_mixed_cross_dataset_task3_target_{target}"
    if method == "a10":
        if seed == 42:
            return root / f"cic_man_gated_filterbank_calibrated_source_mixed_cross_dataset_task3_target_{target}"
        return root / f"seed_{seed}_cic_man_gated_filterbank_calibrated_source_mixed_cross_dataset_task3_target_{target}"
    raise ValueError(method)


def recall(matrix: list[list[int]], class_id: int) -> float:
    denom = sum(matrix[class_id])
    return matrix[class_id][class_id] / denom if denom else 0.0


def precision(matrix: list[list[int]], class_id: int) -> float:
    denom = sum(row[class_id] for row in matrix)
    return matrix[class_id][class_id] / denom if denom else 0.0


def collect_rows(project_dir: Path) -> list[dict[str, object]]:
    root = project_dir / "outputs" / "checkpoints"
    rows = []
    for method in ["v5", "a10"]:
        for seed in SEEDS:
            for target in TARGETS:
                metrics_path = run_dir(root, seed, target, method) / "metrics.json"
                if not metrics_path.exists():
                    continue
                metrics = load_json(metrics_path)
                source_matrix = metrics.get("best_val_metrics", {}).get("confusion_matrix", [])
                target_matrix = metrics.get("test_metrics", {}).get("confusion_matrix", [])
                for class_id, class_name in CLASS_NAMES.items():
                    source_recall = recall(source_matrix, class_id)
                    target_recall = recall(target_matrix, class_id)
                    rows.append(
                        {
                            "method": method,
                            "seed": seed,
                            "target": target,
                            "class_id": class_id,
                            "class_name": class_name,
                            "source_val_recall": source_recall,
                            "source_val_precision": precision(source_matrix, class_id),
                            "target_window_recall": target_recall,
                            "target_window_precision": precision(target_matrix, class_id),
                            "source_target_recall_gap": source_recall - target_recall,
                        }
                    )
    return rows


def source_class_weights(rows: list[dict[str, object]], *, method: str, target: str, seed: int) -> list[float]:
    selected = [
        row
        for row in rows
        if row["method"] == method and row["target"] == target and int(row["seed"]) == seed
    ]
    weights = [1.0, 1.0, 1.0]
    for row in selected:
        class_id = int(row["class_id"])
        source_recall = float(row["source_val_recall"])
        weights[class_id] = 1.0 + 2.0 * max(0.0, 1.0 - source_recall)
    return weights


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_weights(rows: list[dict[str, object]], output_dir: Path) -> None:
    payload = {}
    for target in TARGETS:
        payload[target] = source_class_weights(rows, method="v5", target=target, seed=42)
    (output_dir / "source_heldout_class_weights_seed42_v5.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_markdown(rows: list[dict[str, object]], path: Path) -> None:
    focus = [
        row
        for row in rows
        if row["method"] == "v5"
        and int(row["seed"]) == 42
        and row["target"] in {"ottawa", "paderborn"}
        and row["class_name"] in {"inner", "outer"}
    ]
    lines = [
        "# Adversarial Source-Held-Out Audit",
        "",
        "Question: can source-held-out class metrics identify the classes that fail on target?",
        "",
        "## Seed42 v5 Focus Cases",
        "",
        "| Target | Class | Source-Val Recall | Target Window Recall | Gap | Source-Val Precision | Target Precision |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in focus:
        lines.append(
            f"| {row['target']} | {row['class_name']} | {float(row['source_val_recall']):.6f} | "
            f"{float(row['target_window_recall']):.6f} | {float(row['source_target_recall_gap']):.6f} | "
            f"{float(row['source_val_precision']):.6f} | {float(row['target_window_precision']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Adversarial Finding",
            "",
            "- Source-held-out class recall is often near perfect exactly when target recall collapses.",
            "- Therefore a naive source-validation hard-class rule will under-weight the real target failures, especially Paderborn inner/outer.",
            "- The generated source-held-out class weights are intentionally target-free, but they should be treated as a falsifiable baseline rather than a trusted oracle.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_dir = args.project_dir.expanduser().resolve()
    output_dir = (args.output_dir or project_dir / "outputs" / "tables").expanduser().resolve()
    rows = collect_rows(project_dir)
    write_csv(rows, output_dir / "adversarial_source_heldout_audit.csv")
    write_weights(rows, output_dir)
    write_markdown(rows, output_dir / "adversarial_source_heldout_audit.md")
    print(f"Wrote adversarial source-held-out audit to {output_dir}")


if __name__ == "__main__":
    main()
