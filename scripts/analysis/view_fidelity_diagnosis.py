#!/usr/bin/env python3
"""Diagnose signal-view fault-mechanism fidelity for heterogeneous CIC-MAN views."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def add_src_to_path() -> None:
    script_path = Path(__file__).resolve()
    project_dir = script_path.parents[2]
    sys.path.insert(0, str(project_dir / "src"))


add_src_to_path()

from cicman.data.dataset import WindowIndexDataset  # noqa: E402


VIEW_NAMES = ["raw", "smoothed", "highpass", "envelope_like"]
METRIC_NAMES = [
    "rms",
    "kurtosis",
    "crest_factor",
    "spectral_entropy",
    "spectral_concentration",
    "high_freq_ratio",
    "envelope_peak_ratio",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-name", default="test")
    parser.add_argument("--max-items", type=int, default=20000)
    parser.add_argument("--stride", type=int, default=1)
    return parser.parse_args()


def moving_average(x: np.ndarray, kernel: int = 33) -> np.ndarray:
    kernel = min(kernel, len(x))
    if kernel <= 1:
        return x.copy()
    pad = kernel // 2
    padded = np.pad(x, (pad, pad), mode="edge")
    weights = np.ones(kernel, dtype=np.float32) / float(kernel)
    return np.convolve(padded, weights, mode="valid")[: len(x)].astype(np.float32)


def make_views(x: np.ndarray) -> dict[str, np.ndarray]:
    raw = x.astype(np.float32)
    smoothed = moving_average(raw)
    highpass = raw - smoothed
    envelope_like = np.sqrt(highpass * highpass + 1e-6).astype(np.float32)
    return {
        "raw": raw,
        "smoothed": smoothed,
        "highpass": highpass.astype(np.float32),
        "envelope_like": envelope_like,
    }


def spectral_entropy(magnitude: np.ndarray) -> float:
    power = magnitude.astype(np.float64) ** 2
    total = float(power.sum())
    if total <= 1e-12:
        return 0.0
    prob = power / total
    entropy = -float(np.sum(prob * np.log(prob + 1e-12)))
    return entropy / math.log(len(prob)) if len(prob) > 1 else 0.0


def view_metrics(x: np.ndarray) -> dict[str, float]:
    centered = x.astype(np.float64) - float(np.mean(x))
    rms = float(np.sqrt(np.mean(centered**2) + 1e-12))
    std = float(np.std(centered) + 1e-12)
    kurtosis = float(np.mean((centered / std) ** 4))
    crest = float(np.max(np.abs(centered)) / (rms + 1e-12))
    spectrum = np.abs(np.fft.rfft(centered))
    if len(spectrum) <= 2:
        high_ratio = 0.0
        peak_ratio = 0.0
        entropy = 0.0
    else:
        entropy = spectral_entropy(spectrum[1:])
        power = spectrum[1:] ** 2
        split = max(1, int(len(power) * 0.35))
        high_ratio = float(power[split:].sum() / (power.sum() + 1e-12))
        peak_ratio = float(power.max() / (power.mean() + 1e-12))
    return {
        "rms": rms,
        "kurtosis": kurtosis,
        "crest_factor": crest,
        "spectral_entropy": entropy,
        "spectral_concentration": 1.0 - entropy,
        "high_freq_ratio": high_ratio,
        "envelope_peak_ratio": peak_ratio,
    }


def sample_indices(n: int, max_items: int, stride: int) -> list[int]:
    indices = list(range(0, n, max(1, stride)))
    if max_items > 0 and len(indices) > max_items:
        positions = np.linspace(0, len(indices) - 1, num=max_items).round().astype(int)
        indices = [indices[int(pos)] for pos in positions]
    return indices


def scalar(value) -> str:
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value)


def collect_rows(dataset: WindowIndexDataset, indices: list[int]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in indices:
        item = dataset[index]
        signal = item["x"].squeeze(0).numpy()
        metadata = item["metadata"]
        views = make_views(signal)
        for view_name, view_signal in views.items():
            metrics = view_metrics(view_signal)
            rows.append(
                {
                    "dataset_id": scalar(metadata["dataset_id"]),
                    "recording_id": scalar(metadata["recording_id"]),
                    "label": scalar(metadata["label"]),
                    "label_id": int(metadata["label_id"]),
                    "view": view_name,
                    **metrics,
                }
            )
    return rows


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std())


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, int, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (str(row["dataset_id"]), str(row["label"]), int(row["label_id"]), str(row["view"]))
        grouped[key].append(row)

    output = []
    for (dataset_id, label, label_id, view), group in sorted(grouped.items()):
        item: dict[str, object] = {
            "dataset_id": dataset_id,
            "label": label,
            "label_id": label_id,
            "view": view,
            "num_windows": len(group),
        }
        for metric in METRIC_NAMES:
            mean, std = mean_std([float(row[metric]) for row in group])
            item[f"{metric}_mean"] = mean
            item[f"{metric}_std"] = std
        output.append(item)
    return output


def fisher_scores(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset_id"]), str(row["view"]))].append(row)

    output = []
    for (dataset_id, view), group in sorted(grouped.items()):
        labels = sorted({int(row["label_id"]) for row in group})
        item: dict[str, object] = {
            "dataset_id": dataset_id,
            "view": view,
            "num_labels": len(labels),
            "num_windows": len(group),
        }
        for metric in METRIC_NAMES:
            all_values = np.asarray([float(row[metric]) for row in group], dtype=np.float64)
            overall_mean = float(all_values.mean()) if len(all_values) else 0.0
            between = 0.0
            within = 0.0
            for label_id in labels:
                values = np.asarray(
                    [float(row[metric]) for row in group if int(row["label_id"]) == label_id],
                    dtype=np.float64,
                )
                if len(values) == 0:
                    continue
                between += len(values) * (float(values.mean()) - overall_mean) ** 2
                within += float(((values - float(values.mean())) ** 2).sum())
            item[f"{metric}_fisher"] = float(between / (within + 1e-12))
        output.append(item)
    return output


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(fisher_rows: list[dict[str, object]], path: Path) -> None:
    lines = [
        "# View Fidelity Diagnosis",
        "",
        "| Dataset | View | Windows | Kurtosis Fisher | Spectral Concentration Fisher | High-Freq Fisher | Envelope Peak Fisher |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in fisher_rows:
        lines.append(
            f"| {row['dataset_id']} | {row['view']} | {row['num_windows']} | "
            f"{float(row['kurtosis_fisher']):.6f} | "
            f"{float(row['spectral_concentration_fisher']):.6f} | "
            f"{float(row['high_freq_ratio_fisher']):.6f} | "
            f"{float(row['envelope_peak_ratio_fisher']):.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset = WindowIndexDataset(args.index)
    indices = sample_indices(len(dataset), args.max_items, args.stride)
    rows = collect_rows(dataset, indices)
    aggregate_rows = aggregate(rows)
    fisher_rows = fisher_scores(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.output_dir / f"view_fidelity_windows_{args.split_name}.csv")
    write_csv(aggregate_rows, args.output_dir / f"view_fidelity_aggregate_{args.split_name}.csv")
    write_csv(fisher_rows, args.output_dir / f"view_fidelity_fisher_{args.split_name}.csv")
    write_markdown(fisher_rows, args.output_dir / f"view_fidelity_fisher_{args.split_name}.md")
    summary = {
        "index": str(args.index),
        "split_name": args.split_name,
        "num_input_windows": len(indices),
        "num_view_rows": len(rows),
        "views": VIEW_NAMES,
        "metrics": METRIC_NAMES,
    }
    (args.output_dir / f"view_fidelity_summary_{args.split_name}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
