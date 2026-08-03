#!/usr/bin/env python3
"""Summarize view-fidelity Fisher diagnostics across targets."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


METRICS = [
    "kurtosis_fisher",
    "spectral_concentration_fisher",
    "high_freq_ratio_fisher",
    "envelope_peak_ratio_fisher",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def infer_target(path: Path) -> str:
    name = path.parent.name
    return name.removeprefix("view_fidelity_source_mixed_target_")


def read_rows(project_dir: Path) -> list[dict[str, object]]:
    rows = []
    pattern = "analysis/view_fidelity_source_mixed_target_*/view_fidelity_fisher_test.csv"
    for path in sorted((project_dir / "outputs").glob(pattern)):
        target = infer_target(path)
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                item = {"target": target, **row}
                scores = [float(item[metric]) for metric in METRICS]
                item["mean_fault_fidelity_fisher"] = sum(scores) / len(scores)
                rows.append(item)
    return rows


def best_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    targets = sorted({str(row["target"]) for row in rows})
    for target in targets:
        target_rows = [row for row in rows if row["target"] == target]
        for metric in [*METRICS, "mean_fault_fidelity_fisher"]:
            best = max(target_rows, key=lambda row: float(row[metric]))
            output.append(
                {
                    "target": target,
                    "criterion": metric,
                    "best_view": best["view"],
                    "best_score": float(best[metric]),
                    "dataset_id": best["dataset_id"],
                }
            )
    return output


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, object]], path: Path) -> None:
    lines = [
        "# View Fidelity Summary",
        "",
        "| Target | Criterion | Best View | Score |",
        "|---|---|---|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['target']} | {row['criterion']} | {row['best_view']} | {float(row['best_score']):.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_dir = args.project_dir.expanduser().resolve()
    output_dir = (args.output_dir or project_dir / "outputs" / "tables").expanduser().resolve()
    rows = read_rows(project_dir)
    rows.sort(key=lambda row: (str(row["target"]), str(row["view"])))
    best = best_rows(rows)
    write_csv(rows, output_dir / "view_fidelity_detail.csv")
    write_csv(best, output_dir / "view_fidelity_best_views.csv")
    write_markdown(best, output_dir / "view_fidelity_best_views.md")
    print(f"Wrote {len(rows)} detail rows and {len(best)} best-view rows to {output_dir}")


if __name__ == "__main__":
    main()
