#!/usr/bin/env python3
"""Summarize frozen-feature health/domain leakage probe outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def fmt(value: object) -> str:
    return f"{float(value):.6f}"


def load_rows(input_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(input_dir.glob("domain_leakage_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "model_name": payload["model_name"],
                "target_name": payload["target_name"],
                "feature_key": payload["feature_key"],
                "health_macro_f1": payload["health_probe"]["macro_f1"],
                "health_accuracy": payload["health_probe"]["accuracy"],
                "domain_accuracy": payload["domain_probe"]["accuracy"],
                "domain_macro_f1": payload["domain_probe"]["macro_f1"],
                "domain_chance_accuracy": payload["domain_chance_accuracy"],
                "domain_majority_baseline_accuracy": payload.get("domain_majority_baseline_accuracy", 0.0),
                "domain_leakage_over_chance": payload["domain_leakage_over_chance"],
                "domain_leakage_over_majority": payload.get(
                    "domain_leakage_over_majority", payload["domain_leakage_over_chance"]
                ),
                "health_minus_domain_leakage": payload["health_minus_domain_leakage"],
                "class_conditional_domain_centroid_distance": payload[
                    "class_conditional_domain_centroid_distance"
                ],
                "global_domain_centroid_distance": payload["global_domain_centroid_distance"],
                "num_train_samples": payload["health_probe"]["num_train_samples"],
                "num_eval_samples": payload["health_probe"]["num_eval_samples"],
                "source_domains": ";".join(payload["source_domains"]),
                "source_file": str(path),
            }
        )
    rows.sort(key=lambda row: (str(row["model_name"]), str(row["feature_key"]), str(row["target_name"])))
    return rows


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["model_name"]), str(row["feature_key"]))].append(row)
    out: list[dict[str, object]] = []
    for (model_name, feature_key), values in sorted(groups.items()):
        out.append(
            {
                "model_name": model_name,
                "feature_key": feature_key,
                "mean_health_macro_f1": sum(float(row["health_macro_f1"]) for row in values) / len(values),
                "mean_domain_accuracy": sum(float(row["domain_accuracy"]) for row in values) / len(values),
                "mean_domain_leakage_over_chance": sum(float(row["domain_leakage_over_chance"]) for row in values)
                / len(values),
                "mean_domain_leakage_over_majority": sum(
                    float(row["domain_leakage_over_majority"]) for row in values
                )
                / len(values),
                "mean_health_minus_domain_leakage": sum(float(row["health_minus_domain_leakage"]) for row in values)
                / len(values),
                "mean_class_conditional_domain_centroid_distance": sum(
                    float(row["class_conditional_domain_centroid_distance"]) for row in values
                )
                / len(values),
                "num_targets": len(values),
            }
        )
    return out


def write_markdown(rows: list[dict[str, object]], aggregate: list[dict[str, object]], path: Path) -> None:
    lines = [
        "# Health/Domain Disentanglement Probe",
        "",
        "Frozen CIC-MAN features are evaluated with source-only linear probes. Higher health macro-F1 is better; lower domain leakage over the majority-domain baseline is better.",
        "",
        "## Per Target",
        "",
        "| Model | Feature | Target | Health F1 | Domain Acc. | Majority | Leakage | Health - Leakage | Class-cond. Domain Dist. |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model_name']} | {row['feature_key']} | {row['target_name']} | "
            f"{fmt(row['health_macro_f1'])} | {fmt(row['domain_accuracy'])} | "
            f"{fmt(row['domain_majority_baseline_accuracy'])} | {fmt(row['domain_leakage_over_majority'])} | "
            f"{fmt(row['health_minus_domain_leakage'])} | "
            f"{fmt(row['class_conditional_domain_centroid_distance'])} |"
        )
    lines.extend(
        [
            "",
            "## Mean Over Targets",
            "",
            "| Model | Feature | Mean Health F1 | Mean Domain Acc. | Mean Leakage | Mean Health - Leakage | Mean Class-cond. Domain Dist. |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in aggregate:
        lines.append(
            f"| {row['model_name']} | {row['feature_key']} | {fmt(row['mean_health_macro_f1'])} | "
            f"{fmt(row['mean_domain_accuracy'])} | {fmt(row['mean_domain_leakage_over_majority'])} | "
            f"{fmt(row['mean_health_minus_domain_leakage'])} | "
            f"{fmt(row['mean_class_conditional_domain_centroid_distance'])} |"
        )
    lines.extend(
        [
            "",
            "Paper-use note: this is a target-free diagnostic. The probe uses source train features for fitting and source-held-out validation features for reporting.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input_dir)
    aggregate = aggregate_rows(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.output_dir / "domain_leakage_summary.csv")
    write_csv(aggregate, args.output_dir / "domain_leakage_summary_mean.csv")
    write_markdown(rows, aggregate, args.output_dir / "domain_leakage_summary.md")
    print(f"Loaded {len(rows)} probe outputs.")
    print(f"Wrote {args.output_dir / 'domain_leakage_summary.md'}")


if __name__ == "__main__":
    main()
