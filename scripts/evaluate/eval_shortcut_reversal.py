#!/usr/bin/env python3
"""Evaluate checkpoints under label-conditioned shortcut reversal."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def add_src_to_path() -> None:
    script_path = Path(__file__).resolve()
    project_dir = script_path.parents[2]
    sys.path.insert(0, str(project_dir / "src"))
    sys.path.insert(0, str(script_path.parents[3]))


add_src_to_path()

from cicman.data.dataset import WindowIndexDataset  # noqa: E402
from cicman.evaluation.metrics import accuracy, balanced_accuracy, confusion_matrix, macro_f1  # noqa: E402
from cicman.evaluation.shortcuts import apply_label_shortcut  # noqa: E402
from cicman.models.cic_man import build_cic_man  # noqa: E402
from cicman.models.cic_man_gated_filterbank import build_cic_man_gated_filterbank  # noqa: E402
from cicman.models.cic_man_gated_viewbank import build_cic_man_gated_viewbank  # noqa: E402
from cicman.models.cic_man_heterogeneous import build_cic_man_heterogeneous  # noqa: E402
from cicman.models.raw_cnn import build_raw_cnn  # noqa: E402
from recording_protocol import OFFICIAL_AGGREGATION, resolve_majority_vote  # noqa: E402


DEFAULT_MODES = ["clean", "correlated", "reversed", "neutral"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", choices=["raw_cnn", "cic_man"], required=True)
    parser.add_argument("--method-name", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--shortcut-amplitude", type=float, default=1.0)
    parser.add_argument("--modes", nargs="+", default=DEFAULT_MODES)
    return parser.parse_args()


def make_loader(index_csv: Path, batch_size: int, num_workers: int):
    import torch

    dataset = WindowIndexDataset(index_csv)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def metadata_column(metadata: dict[str, object], key: str) -> list[str]:
    values = metadata[key]
    if isinstance(values, list):
        return [str(value) for value in values]
    return [str(value) for value in list(values)]


def metric_dict(y_true: list[int], y_pred: list[int], num_classes: int) -> dict[str, object]:
    return {
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred, num_classes),
        "balanced_accuracy": balanced_accuracy(y_true, y_pred, num_classes),
        "num_samples": len(y_true),
        "confusion_matrix": confusion_matrix(y_true, y_pred, num_classes).tolist(),
    }


def build_model(model_name: str, checkpoint: dict[str, object], device: str):
    num_classes = int(checkpoint["num_classes"])
    if model_name == "raw_cnn":
        model = build_raw_cnn(num_classes=num_classes)
    else:
        config = checkpoint.get("config", {})
        num_agents = int(checkpoint.get("num_agents", config.get("num_agents", 4)))
        architecture = str(config.get("architecture", "minimal"))
        if architecture == "gated_filterbank":
            model = build_cic_man_gated_filterbank(num_classes=num_classes, core_agents=max(1, num_agents - 1))
        elif architecture == "gated_viewbank":
            model = build_cic_man_gated_viewbank(
                num_classes=num_classes,
                view_names=config.get("view_bank_views", ["envelope", "order", "denoise"]),
                max_total_gate=float(config.get("max_total_gate", 0.35)),
                use_health_style_split=bool(config.get("use_health_style_split", False)),
                health_logit_weight=float(config.get("health_logit_weight", 0.0)),
            )
        elif architecture == "heterogeneous":
            model = build_cic_man_heterogeneous(num_classes=num_classes, num_agents=num_agents)
        else:
            model = build_cic_man(num_classes=num_classes, num_agents=num_agents)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model


def evaluate_mode(
    *,
    model,
    loader,
    mode: str,
    num_classes: int,
    device: str,
    shortcut_amplitude: float,
) -> dict[str, object]:
    import torch

    y_true: list[int] = []
    y_pred: list[int] = []
    logit_sums: dict[str, np.ndarray] = {}
    probability_sums: dict[str, np.ndarray] = {}
    vote_counts: dict[str, Counter[int]] = defaultdict(Counter)
    recording_labels: dict[str, int] = {}
    eval_mode = "none" if mode == "clean" else mode

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y_tensor = batch["y"].to(device)
            x = apply_label_shortcut(
                x,
                y_tensor,
                mode=eval_mode,
                num_classes=num_classes,
                amplitude=shortcut_amplitude,
            )
            y = y_tensor.detach().cpu().numpy().astype(int)
            logits = model(x).detach().cpu().numpy()
            shifted = logits - logits.max(axis=1, keepdims=True)
            probabilities = np.exp(shifted)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            pred = probabilities.argmax(axis=1).astype(int)
            metadata = batch["metadata"]
            recording_ids = metadata_column(metadata, "recording_id")

            y_true.extend(int(v) for v in y)
            y_pred.extend(int(v) for v in pred)
            for i, recording_id in enumerate(recording_ids):
                if recording_id not in logit_sums:
                    logit_sums[recording_id] = np.zeros(num_classes, dtype=np.float64)
                    probability_sums[recording_id] = np.zeros(num_classes, dtype=np.float64)
                    recording_labels[recording_id] = int(y[i])
                logit_sums[recording_id] += logits[i]
                probability_sums[recording_id] += probabilities[i]
                vote_counts[recording_id][int(pred[i])] += 1

    rec_true: list[int] = []
    rec_mean_pred: list[int] = []
    rec_probability_pred: list[int] = []
    rec_vote_pred: list[int] = []
    for recording_id in sorted(logit_sums):
        rec_true.append(recording_labels[recording_id])
        rec_mean_pred.append(int(logit_sums[recording_id].argmax()))
        rec_probability_pred.append(int(probability_sums[recording_id].argmax()))
        votes = np.array([vote_counts[recording_id][c] for c in range(num_classes)], dtype=np.int64)
        rec_vote_pred.append(resolve_majority_vote(votes, probability_sums[recording_id])[0])

    return {
        "mode": mode,
        "window": metric_dict(y_true, y_pred, num_classes),
        "recording_mean_logits": metric_dict(rec_true, rec_mean_pred, num_classes),
        "recording_probability_sum": metric_dict(rec_true, rec_probability_pred, num_classes),
        "recording_majority_vote": metric_dict(rec_true, rec_vote_pred, num_classes),
        "recording_aggregation": OFFICIAL_AGGREGATION,
    }


def write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "model",
        "shortcut_amplitude",
        "mode",
        "window_accuracy",
        "window_macro_f1",
        "window_balanced_accuracy",
        "recording_accuracy",
        "recording_macro_f1",
        "recording_balanced_accuracy",
        "num_windows",
        "num_recordings",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, object]], output_path: Path) -> None:
    lines = [
        "# Shortcut Reversal Evaluation",
        "",
        "| Method | Mode | Window Acc | Window Macro-F1 | Recording Acc | Recording Macro-F1 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["method"]),
                    str(row["mode"]),
                    f"{float(row['window_accuracy']):.6f}",
                    f"{float(row['window_macro_f1']):.6f}",
                    f"{float(row['recording_accuracy']):.6f}",
                    f"{float(row['recording_macro_f1']):.6f}",
                ]
            )
            + " |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()

    import torch

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(args.checkpoint, map_location=device)
    num_classes = int(checkpoint["num_classes"])
    model = build_model(args.model, checkpoint, device)
    loader = make_loader(args.index, args.batch_size, args.num_workers)
    method = args.method_name or args.checkpoint.parent.name

    results = []
    rows = []
    for mode in args.modes:
        result = evaluate_mode(
            model=model,
            loader=loader,
            mode=mode,
            num_classes=num_classes,
            device=device,
            shortcut_amplitude=args.shortcut_amplitude,
        )
        results.append(result)
        window = result["window"]
        recording = result["recording_majority_vote"]
        rows.append(
            {
                "method": method,
                "model": args.model,
                "shortcut_amplitude": args.shortcut_amplitude,
                "mode": mode,
                "window_accuracy": window["accuracy"],
                "window_macro_f1": window["macro_f1"],
                "window_balanced_accuracy": window["balanced_accuracy"],
                "recording_accuracy": recording["accuracy"],
                "recording_macro_f1": recording["macro_f1"],
                "recording_balanced_accuracy": recording["balanced_accuracy"],
                "num_windows": window["num_samples"],
                "num_recordings": recording["num_samples"],
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "index": str(args.index),
        "checkpoint": str(args.checkpoint),
        "model": args.model,
        "method": method,
        "shortcut_amplitude": args.shortcut_amplitude,
        "results": results,
    }
    json_path = args.output_dir / "shortcut_reversal_metrics.json"
    csv_path = args.output_dir / "shortcut_reversal_metrics.csv"
    md_path = args.output_dir / "shortcut_reversal_metrics.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(rows, csv_path)
    write_markdown(rows, md_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
