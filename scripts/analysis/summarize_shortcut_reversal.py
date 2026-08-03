#!/usr/bin/env python3
"""Summarize shortcut reversal experiments."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def load_rows(project_dir: Path) -> list[dict[str, str]]:
    rows = []
    checkpoint_dir = project_dir / "outputs" / "checkpoints"
    for path in sorted(checkpoint_dir.glob("shortcut_*/shortcut_reversal_amp_*/shortcut_reversal_metrics.csv")):
        with path.open(newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    return rows


def fnum(row: dict[str, str], key: str) -> float:
    return float(row.get(key, 0.0) or 0.0)


def summarize(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        grouped[(row["method"], row.get("shortcut_amplitude", ""))][row["mode"]] = row

    summaries = []
    for (method, amplitude), by_mode in sorted(grouped.items()):
        clean = by_mode.get("clean", {})
        correlated = by_mode.get("correlated", {})
        reversed_row = by_mode.get("reversed", {})
        neutral = by_mode.get("neutral", {})
        summaries.append(
            {
                "method": method,
                "shortcut_amplitude": amplitude,
                "clean_window_macro_f1": fnum(clean, "window_macro_f1"),
                "correlated_window_macro_f1": fnum(correlated, "window_macro_f1"),
                "reversed_window_macro_f1": fnum(reversed_row, "window_macro_f1"),
                "neutral_window_macro_f1": fnum(neutral, "window_macro_f1"),
                "shortcut_gain_window": fnum(correlated, "window_macro_f1") - fnum(clean, "window_macro_f1"),
                "reversal_drop_window": fnum(correlated, "window_macro_f1") - fnum(reversed_row, "window_macro_f1"),
                "clean_recording_macro_f1": fnum(clean, "recording_macro_f1"),
                "correlated_recording_macro_f1": fnum(correlated, "recording_macro_f1"),
                "reversed_recording_macro_f1": fnum(reversed_row, "recording_macro_f1"),
                "neutral_recording_macro_f1": fnum(neutral, "recording_macro_f1"),
                "shortcut_gain_recording": fnum(correlated, "recording_macro_f1") - fnum(clean, "recording_macro_f1"),
                "reversal_drop_recording": fnum(correlated, "recording_macro_f1") - fnum(reversed_row, "recording_macro_f1"),
            }
        )
    return summaries


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(detail_rows: list[dict[str, str]], summary_rows: list[dict[str, object]], path: Path) -> None:
    lines = [
        "# Shortcut Reversal Summary",
        "",
        "Training uses source-correlated label-conditioned harmonic shortcuts. Evaluation compares clean, correlated, reversed, and neutral shortcut modes on the held-out target dataset. Target metrics are post-hoc evaluation only.",
        "",
        "## Summary",
        "",
        "| Method | Clean F1 | Correlated F1 | Reversed F1 | Neutral F1 | Shortcut Gain | Reversal Drop |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        label = str(row["method"])
        if row.get("shortcut_amplitude", "") not in {"", None}:
            label = f"{label} amp={float(row['shortcut_amplitude']):.2f}"
        lines.append(
            f"| {label} | {float(row['clean_window_macro_f1']):.6f} | "
            f"{float(row['correlated_window_macro_f1']):.6f} | "
            f"{float(row['reversed_window_macro_f1']):.6f} | "
            f"{float(row['neutral_window_macro_f1']):.6f} | "
            f"{float(row['shortcut_gain_window']):.6f} | "
            f"{float(row['reversal_drop_window']):.6f} |"
        )

    lines.extend(
        [
            "",
            "## Detail",
            "",
            "| Method | Amp | Mode | Window Macro-F1 | Recording Macro-F1 |",
            "|---|---:|---|---:|---:|",
        ]
    )
    for row in detail_rows:
        amplitude = row.get("shortcut_amplitude", "")
        amplitude_text = f"{float(amplitude):.2f}" if amplitude not in {"", None} else ""
        lines.append(
            f"| {row['method']} | {amplitude_text} | {row['mode']} | "
            f"{float(row['window_macro_f1']):.6f} | {float(row['recording_macro_f1']):.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def family_name(method: str) -> str:
    for suffix in ("-hust", "-ottawa", "-paderborn"):
        if method.endswith(suffix):
            return method[: -len(suffix)]
    return method


def summarize_by_family(summary_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in summary_rows:
        groups[(family_name(str(row["method"])), str(row.get("shortcut_amplitude", "")))].append(row)
    rows = []
    for (method, amplitude), values in sorted(groups.items()):
        rows.append(
            {
                "method": method,
                "shortcut_amplitude": amplitude,
                "mean_clean_window_macro_f1": sum(float(row["clean_window_macro_f1"]) for row in values)
                / max(len(values), 1),
                "mean_correlated_window_macro_f1": sum(float(row["correlated_window_macro_f1"]) for row in values)
                / max(len(values), 1),
                "mean_reversed_window_macro_f1": sum(float(row["reversed_window_macro_f1"]) for row in values)
                / max(len(values), 1),
                "mean_neutral_window_macro_f1": sum(float(row["neutral_window_macro_f1"]) for row in values)
                / max(len(values), 1),
                "mean_shortcut_gain_window": sum(float(row["shortcut_gain_window"]) for row in values)
                / max(len(values), 1),
                "mean_reversal_drop_window": sum(float(row["reversal_drop_window"]) for row in values)
                / max(len(values), 1),
            }
        )
    return rows


def write_family_markdown(rows: list[dict[str, object]], path: Path) -> None:
    lines = [
        "# Shortcut Reversal Family Summary",
        "",
        "| Method | Amp | Mean Clean F1 | Mean Correlated F1 | Mean Reversed F1 | Mean Shortcut Gain | Mean Reversal Drop |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {float(row['shortcut_amplitude']):.2f} | "
            f"{float(row['mean_clean_window_macro_f1']):.6f} | "
            f"{float(row['mean_correlated_window_macro_f1']):.6f} | "
            f"{float(row['mean_reversed_window_macro_f1']):.6f} | "
            f"{float(row['mean_shortcut_gain_window']):.6f} | "
            f"{float(row['mean_reversal_drop_window']):.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_dir = args.project_dir.expanduser().resolve()
    output_dir = (args.output_dir or project_dir / "outputs" / "tables").expanduser().resolve()
    detail_rows = load_rows(project_dir)
    summary_rows = summarize(detail_rows)
    family_rows = summarize_by_family(summary_rows)
    write_csv(detail_rows, output_dir / "shortcut_reversal_detail.csv")
    write_csv(summary_rows, output_dir / "shortcut_reversal_summary.csv")
    write_csv(family_rows, output_dir / "shortcut_reversal_family_summary.csv")
    write_markdown(detail_rows, summary_rows, output_dir / "shortcut_reversal_summary.md")
    write_family_markdown(family_rows, output_dir / "shortcut_reversal_family_summary.md")
    print(
        f"Wrote {len(detail_rows)} detail rows, {len(summary_rows)} summary rows, "
        f"and {len(family_rows)} family rows to {output_dir}"
    )


if __name__ == "__main__":
    main()
