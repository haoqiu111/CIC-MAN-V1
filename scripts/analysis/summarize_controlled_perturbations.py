#!/usr/bin/env python3
"""Summarize controlled perturbation robustness results."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


DETAIL_FIELDS = [
    "experiment",
    "method",
    "model",
    "target",
    "perturbation",
    "window_accuracy",
    "window_macro_f1",
    "window_balanced_accuracy",
    "recording_accuracy",
    "recording_macro_f1",
    "recording_balanced_accuracy",
    "num_windows",
    "num_recordings",
]

SUMMARY_FIELDS = [
    "method",
    "target",
    "clean_window_macro_f1",
    "mean_perturbed_window_macro_f1",
    "window_f1_retention",
    "clean_recording_macro_f1",
    "mean_perturbed_recording_macro_f1",
    "recording_f1_retention",
    "num_perturbations",
]

FAMILY_FIELDS = [
    "method",
    "mean_clean_window_macro_f1",
    "mean_perturbed_window_macro_f1",
    "mean_window_f1_retention",
    "mean_clean_recording_macro_f1",
    "mean_perturbed_recording_macro_f1",
    "mean_recording_f1_retention",
    "num_targets",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Project directory containing outputs/checkpoints.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def infer_target(experiment: str) -> str:
    for target in ("hust", "ottawa", "paderborn"):
        if experiment.endswith(f"target_{target}"):
            return target
    return "unknown"


def read_rows(checkpoint_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(checkpoint_dir.glob("*/controlled_perturbation/controlled_perturbation_metrics.csv")):
        experiment = path.parents[1].name
        target = infer_target(experiment)
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(
                    {
                        "experiment": experiment,
                        "target": target,
                        **row,
                    }
                )
    return rows


def fnum(value: str) -> float:
    return float(value) if value not in {"", None} else 0.0


def summarize(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["target"])].append(row)

    summaries: list[dict[str, object]] = []
    for (method, target), group in sorted(grouped.items()):
        clean = [row for row in group if row["perturbation"] == "clean"]
        perturbed = [row for row in group if row["perturbation"] != "clean"]
        if not clean or not perturbed:
            continue
        clean_row = clean[0]
        clean_win = fnum(clean_row["window_macro_f1"])
        clean_rec = fnum(clean_row["recording_macro_f1"])
        mean_win = sum(fnum(row["window_macro_f1"]) for row in perturbed) / len(perturbed)
        mean_rec = sum(fnum(row["recording_macro_f1"]) for row in perturbed) / len(perturbed)
        summaries.append(
            {
                "method": method,
                "target": target,
                "clean_window_macro_f1": clean_win,
                "mean_perturbed_window_macro_f1": mean_win,
                "window_f1_retention": mean_win / clean_win if clean_win else 0.0,
                "clean_recording_macro_f1": clean_rec,
                "mean_perturbed_recording_macro_f1": mean_rec,
                "recording_f1_retention": mean_rec / clean_rec if clean_rec else 0.0,
                "num_perturbations": len(perturbed),
            }
        )
    return summaries


def write_csv(rows: list[dict[str, object]], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_detail_md(rows: list[dict[str, str]], path: Path) -> None:
    lines = [
        "# Controlled Perturbation Robustness Detail",
        "",
        "| Method | Target | Perturbation | Window Macro-F1 | Recording Macro-F1 |",
        "|---|---|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['target']} | {row['perturbation']} | "
            f"{float(row['window_macro_f1']):.6f} | {float(row['recording_macro_f1']):.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_md(rows: list[dict[str, object]], path: Path) -> None:
    lines = [
        "# Controlled Perturbation Robustness Summary",
        "",
        "| Method | Target | Clean Window F1 | Perturbed Window F1 | Window Retention | Clean Rec F1 | Perturbed Rec F1 | Rec Retention |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["method"]),
                    str(row["target"]),
                    fmt(row["clean_window_macro_f1"]),
                    fmt(row["mean_perturbed_window_macro_f1"]),
                    fmt(row["window_f1_retention"]),
                    fmt(row["clean_recording_macro_f1"]),
                    fmt(row["mean_perturbed_recording_macro_f1"]),
                    fmt(row["recording_f1_retention"]),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_family(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(row)
    family_rows = []
    for method, group in sorted(grouped.items()):
        family_rows.append(
            {
                "method": method,
                "mean_clean_window_macro_f1": sum(float(row["clean_window_macro_f1"]) for row in group)
                / max(len(group), 1),
                "mean_perturbed_window_macro_f1": sum(float(row["mean_perturbed_window_macro_f1"]) for row in group)
                / max(len(group), 1),
                "mean_window_f1_retention": sum(float(row["window_f1_retention"]) for row in group)
                / max(len(group), 1),
                "mean_clean_recording_macro_f1": sum(float(row["clean_recording_macro_f1"]) for row in group)
                / max(len(group), 1),
                "mean_perturbed_recording_macro_f1": sum(
                    float(row["mean_perturbed_recording_macro_f1"]) for row in group
                )
                / max(len(group), 1),
                "mean_recording_f1_retention": sum(float(row["recording_f1_retention"]) for row in group)
                / max(len(group), 1),
                "num_targets": len(group),
            }
        )
    return family_rows


def write_family_md(rows: list[dict[str, object]], path: Path) -> None:
    lines = [
        "# Controlled Perturbation Family Summary",
        "",
        "| Method | Clean Window F1 | Perturbed Window F1 | Window Retention | Clean Rec F1 | Perturbed Rec F1 | Rec Retention | Targets |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["method"]),
                    fmt(row["mean_clean_window_macro_f1"]),
                    fmt(row["mean_perturbed_window_macro_f1"]),
                    fmt(row["mean_window_f1_retention"]),
                    fmt(row["mean_clean_recording_macro_f1"]),
                    fmt(row["mean_perturbed_recording_macro_f1"]),
                    fmt(row["mean_recording_f1_retention"]),
                    str(row["num_targets"]),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_dir = args.project_dir.expanduser().resolve()
    output_dir = (args.output_dir or project_dir / "outputs" / "tables").expanduser().resolve()
    rows = read_rows(project_dir / "outputs" / "checkpoints")
    rows.sort(key=lambda row: (row["target"], row["method"], row["perturbation"]))
    summaries = summarize(rows)
    family_rows = summarize_family(summaries)

    write_csv(rows, output_dir / "controlled_perturbation_detail.csv", DETAIL_FIELDS)
    write_csv(summaries, output_dir / "controlled_perturbation_summary.csv", SUMMARY_FIELDS)
    write_csv(family_rows, output_dir / "controlled_perturbation_family_summary.csv", FAMILY_FIELDS)
    write_detail_md(rows, output_dir / "controlled_perturbation_detail.md")
    write_summary_md(summaries, output_dir / "controlled_perturbation_summary.md")
    write_family_md(family_rows, output_dir / "controlled_perturbation_family_summary.md")
    print(
        f"Wrote {len(rows)} detail rows, {len(summaries)} summary rows, "
        f"and {len(family_rows)} family rows to {output_dir}"
    )


if __name__ == "__main__":
    main()
