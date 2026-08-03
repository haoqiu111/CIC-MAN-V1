#!/usr/bin/env python3
"""Build paper-ready CIC-MAN A1-A10 ablation matrix."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


ABLATIONS = [
    {
        "id": "A1",
        "method": "CIC-MAN-v1-source-mixed",
        "label": "Source-mixed minimal CIC-MAN",
        "module": "source-mixed multi-agent routing",
        "role": "baseline CIC-MAN",
    },
    {
        "id": "A2",
        "method": "CIC-MAN-v4-style-consistency",
        "label": "+ domain style consistency",
        "module": "domain-intervention consistency",
        "role": "positive module",
    },
    {
        "id": "A3",
        "method": "CIC-MAN-v5-class-router-style",
        "label": "+ class-conditional router balance",
        "module": "class-wise router balance",
        "role": "stable core",
    },
    {
        "id": "A4",
        "method": "CIC-MAN-heterogeneous",
        "label": "heterogeneous time-domain agents",
        "module": "raw/smoothed/high-pass/envelope agents",
        "role": "negative diagnostic",
    },
    {
        "id": "A5",
        "method": "CIC-MAN-heterogeneous-v2-reliability",
        "label": "+ source view-reliability routing",
        "module": "source-supervised view reliability",
        "role": "negative diagnostic",
    },
    {
        "id": "A6",
        "method": "CIC-MAN-heterogeneous-v3-physics",
        "label": "+ lightweight physics-fidelity routing",
        "module": "time-domain physics fidelity",
        "role": "mechanism diagnostic",
    },
    {
        "id": "A7",
        "method": "CIC-MAN-heterogeneous-v4-filterbank",
        "label": "+ fixed filterbank intervention agent",
        "module": "multi-scale filterbank envelope",
        "role": "useful but unstable intervention",
    },
    {
        "id": "A8",
        "method": "CIC-MAN-gated-filterbank",
        "label": "joint gated filterbank",
        "module": "optional filterbank, joint training",
        "role": "negative diagnostic",
    },
    {
        "id": "A9",
        "method": "CIC-MAN-gated-filterbank-frozen-core",
        "label": "frozen-core gated filterbank",
        "module": "frozen v5 core + trainable gate",
        "role": "core-preservation diagnostic",
    },
    {
        "id": "A10",
        "method": "CIC-MAN-gated-filterbank-calibrated",
        "label": "source-calibrated gated filterbank",
        "module": "source loss margin + physics-gated intervention",
        "role": "final candidate",
    },
]

TARGETS = ["target_dataset_hust", "target_dataset_ottawa", "target_dataset_paderborn"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def load_model_rows(project_dir: Path) -> dict[tuple[str, str], dict[str, str]]:
    path = project_dir / "outputs" / "tables" / "model_comparison.csv"
    rows = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["protocol"] != "cross_dataset_task3_source_mixed":
                continue
            rows[(row["method"], row["target"])] = row
    return rows


def fnum(value: str) -> float:
    return float(value) if value not in {"", None} else 0.0


def build_rows(model_rows: dict[tuple[str, str], dict[str, str]]) -> list[dict[str, object]]:
    rows = []
    for ablation in ABLATIONS:
        target_values = {}
        rec_values = {}
        for target in TARGETS:
            source = model_rows.get((ablation["method"], target))
            target_key = target.removeprefix("target_dataset_")
            if source is None:
                target_values[target_key] = None
                rec_values[target_key] = None
                continue
            target_values[target_key] = fnum(source["test_macro_f1"])
            rec_values[target_key] = fnum(source["recording_macro_f1"])

        present_window = [value for value in target_values.values() if value is not None]
        present_recording = [value for value in rec_values.values() if value is not None]
        rows.append(
            {
                **ablation,
                "hust_window_macro_f1": target_values["hust"],
                "ottawa_window_macro_f1": target_values["ottawa"],
                "paderborn_window_macro_f1": target_values["paderborn"],
                "mean_window_macro_f1": sum(present_window) / len(present_window) if present_window else None,
                "hust_recording_macro_f1": rec_values["hust"],
                "ottawa_recording_macro_f1": rec_values["ottawa"],
                "paderborn_recording_macro_f1": rec_values["paderborn"],
                "mean_recording_macro_f1": (
                    sum(present_recording) / len(present_recording) if present_recording else None
                ),
                "num_targets": len(present_window),
            }
        )

    reference_v5 = next(row for row in rows if row["id"] == "A3")
    reference_final = next(row for row in rows if row["id"] == "A10")
    for row in rows:
        row["delta_mean_window_vs_v5"] = nullable_delta(row["mean_window_macro_f1"], reference_v5["mean_window_macro_f1"])
        row["delta_mean_window_vs_final"] = nullable_delta(
            row["mean_window_macro_f1"],
            reference_final["mean_window_macro_f1"],
        )
    return rows


def nullable_delta(value: object, reference: object) -> float | None:
    if value is None or reference is None:
        return None
    return float(value) - float(reference)


def fmt(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    fieldnames = [
        "id",
        "label",
        "module",
        "role",
        "method",
        "hust_window_macro_f1",
        "ottawa_window_macro_f1",
        "paderborn_window_macro_f1",
        "mean_window_macro_f1",
        "hust_recording_macro_f1",
        "ottawa_recording_macro_f1",
        "paderborn_recording_macro_f1",
        "mean_recording_macro_f1",
        "delta_mean_window_vs_v5",
        "delta_mean_window_vs_final",
        "num_targets",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, object]], path: Path) -> None:
    lines = [
        "# CIC-MAN A1-A10 Ablation Matrix",
        "",
        "Scope: source-mixed target-free cross-dataset DG. Metrics are target test Macro-F1; target data are not used for model selection.",
        "",
        "| ID | Variant | Role | HUST | Ottawa | Paderborn | Mean | Delta vs A3 |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['id']} | {row['label']} | {row['role']} | "
            f"{fmt(row['hust_window_macro_f1'])} | {fmt(row['ottawa_window_macro_f1'])} | "
            f"{fmt(row['paderborn_window_macro_f1'])} | {fmt(row['mean_window_macro_f1'])} | "
            f"{fmt(row['delta_mean_window_vs_v5'])} |"
        )

    lines.extend(
        [
            "",
            "## Recording-Level Macro-F1",
            "",
            "| ID | Variant | HUST | Ottawa | Paderborn | Mean |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['id']} | {row['label']} | "
            f"{fmt(row['hust_recording_macro_f1'])} | {fmt(row['ottawa_recording_macro_f1'])} | "
            f"{fmt(row['paderborn_recording_macro_f1'])} | {fmt(row['mean_recording_macro_f1'])} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- A3 is the stable v5 core: class-conditional router balance plus style consistency.",
            "- A4-A7 show that simply adding heterogeneous/filterbank intervention views is not sufficient; filterbank helps HUST/Paderborn but collapses Ottawa when used as a fixed architecture.",
            "- A8 shows joint gated filterbank training damages the stable core.",
            "- A9 shows freezing the v5 core prevents that damage but does not add consistent gains.",
            "- A10 is the best optional-intervention candidate: source-calibrated gating preserves Paderborn, improves HUST/Ottawa window-level F1, and is compatible with the target-free source-selection rule.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_dir = args.project_dir.expanduser().resolve()
    output_dir = (args.output_dir or project_dir / "outputs" / "tables").expanduser().resolve()
    rows = build_rows(load_model_rows(project_dir))
    write_csv(rows, output_dir / "ablation_matrix_a1_a10.csv")
    write_markdown(rows, output_dir / "ablation_matrix_a1_a10.md")
    print(f"Wrote {len(rows)} ablation rows to {output_dir}")


if __name__ == "__main__":
    main()
