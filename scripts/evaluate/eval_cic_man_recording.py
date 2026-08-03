#!/usr/bin/env python3
"""Evaluate a minimal CIC-MAN checkpoint at recording level."""

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
from cicman.models.cic_man import build_cic_man  # noqa: E402
from cicman.models.cic_man_gated_filterbank import build_cic_man_gated_filterbank  # noqa: E402
from cicman.models.cic_man_gated_viewbank import build_cic_man_gated_viewbank  # noqa: E402
from cicman.models.cic_man_heterogeneous import build_cic_man_heterogeneous  # noqa: E402
from cicman.models.cic_man_vfinal import build_cic_man_vfinal  # noqa: E402
from recording_protocol import OFFICIAL_AGGREGATION, resolve_majority_vote  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True, help="Window-index CSV to evaluate.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="CIC-MAN checkpoint .pt file.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
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


def metric_dict(y_true: list[int], y_pred: list[int], num_classes: int) -> dict[str, object]:
    return {
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred, num_classes),
        "balanced_accuracy": balanced_accuracy(y_true, y_pred, num_classes),
        "num_recordings": len(y_true),
        "confusion_matrix": confusion_matrix(y_true, y_pred, num_classes).tolist(),
    }


def metadata_column(metadata: dict[str, object], key: str) -> list[str]:
    values = metadata[key]
    if isinstance(values, list):
        return [str(value) for value in values]
    return [str(value) for value in list(values)]


def evaluate_recordings(
    *,
    index_csv: Path,
    checkpoint_path: Path,
    batch_size: int,
    device: str,
    num_workers: int,
) -> dict[str, object]:
    import torch

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(checkpoint_path, map_location=device)
    num_classes = int(checkpoint["num_classes"])
    config = checkpoint.get("config", {})
    num_agents = int(checkpoint.get("num_agents", config.get("num_agents", 4)))
    architecture = str(config.get("architecture", "minimal"))
    if architecture == "gated_filterbank":
        model = build_cic_man_gated_filterbank(num_classes=num_classes, core_agents=max(1, num_agents - 1)).to(device)
    elif architecture == "gated_viewbank":
        model = build_cic_man_gated_viewbank(
            num_classes=num_classes,
            view_names=config.get("view_bank_views", ["envelope", "order", "denoise"]),
            max_total_gate=float(config.get("max_total_gate", 0.35)),
            use_health_style_split=bool(config.get("use_health_style_split", False)),
            health_logit_weight=float(config.get("health_logit_weight", 0.0)),
        ).to(device)
    elif architecture == "vfinal":
        model = build_cic_man_vfinal(
            num_classes=num_classes,
            view_names=config.get(
                "view_bank_views",
                ["envelope", "stft", "wavelet", "order", "denoise", "filterbank"],
            ),
            max_total_gate=float(config.get("max_total_gate", 0.30)),
            use_health_style_split=bool(config.get("use_health_style_split", True)),
            health_logit_weight=float(config.get("health_logit_weight", 0.0)),
        ).to(device)
    elif architecture == "heterogeneous":
        model = build_cic_man_heterogeneous(num_classes=num_classes, num_agents=num_agents).to(device)
    else:
        model = build_cic_man(num_classes=num_classes, num_agents=num_agents).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    loader = make_loader(index_csv, batch_size, num_workers)
    logit_sums: dict[str, np.ndarray] = {}
    probability_sums: dict[str, np.ndarray] = {}
    vote_counts: dict[str, Counter[int]] = defaultdict(Counter)
    labels: dict[str, int] = {}
    metadata_rows: dict[str, dict[str, str]] = {}

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].detach().cpu().numpy().astype(int)
            logits = model(x).detach().cpu().numpy()
            shifted = logits - logits.max(axis=1, keepdims=True)
            probabilities = np.exp(shifted)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            pred = probabilities.argmax(axis=1).astype(int)
            metadata = batch["metadata"]
            recording_ids = metadata_column(metadata, "recording_id")

            for i, recording_id in enumerate(recording_ids):
                if recording_id not in logit_sums:
                    logit_sums[recording_id] = np.zeros(num_classes, dtype=np.float64)
                    probability_sums[recording_id] = np.zeros(num_classes, dtype=np.float64)
                    labels[recording_id] = int(y[i])
                    metadata_rows[recording_id] = {
                        "recording_id": recording_id,
                        "dataset_id": str(metadata["dataset_id"][i]),
                        "label": str(metadata["label"][i]),
                        "label_id": str(metadata["label_id"][i]),
                        "condition_id": str(metadata["condition_id"][i]),
                        "speed_profile_id": str(metadata["speed_profile_id"][i]),
                        "bearing_type_id": str(metadata["bearing_type_id"][i]),
                        "bearing_id": str(metadata["bearing_id"][i]),
                    }
                logit_sums[recording_id] += logits[i]
                probability_sums[recording_id] += probabilities[i]
                vote_counts[recording_id][int(pred[i])] += 1

    y_true: list[int] = []
    mean_logit_pred: list[int] = []
    majority_vote_pred: list[int] = []
    per_recording: list[dict[str, object]] = []

    for recording_id in sorted(logit_sums):
        true_label = labels[recording_id]
        mean_pred = int(logit_sums[recording_id].argmax())
        votes = np.array([vote_counts[recording_id][c] for c in range(num_classes)], dtype=np.int64)
        vote_pred = resolve_majority_vote(votes, probability_sums[recording_id])[0]
        y_true.append(true_label)
        mean_logit_pred.append(mean_pred)
        majority_vote_pred.append(vote_pred)
        per_recording.append(
            {
                **metadata_rows[recording_id],
                "true_label_id": true_label,
                "mean_logit_pred": mean_pred,
                "majority_vote_pred": vote_pred,
                "num_windows": int(sum(vote_counts[recording_id].values())),
                "vote_counts": dict(sorted(vote_counts[recording_id].items())),
            }
        )

    return {
        "index": str(index_csv),
        "checkpoint": str(checkpoint_path),
        "num_classes": num_classes,
        "num_agents": num_agents,
        "mean_logits": metric_dict(y_true, mean_logit_pred, num_classes),
        "majority_vote": metric_dict(y_true, majority_vote_pred, num_classes),
        "recording_aggregation": OFFICIAL_AGGREGATION,
        "per_recording": per_recording,
    }


def write_per_recording(rows: list[dict[str, object]], path: Path) -> None:
    fieldnames = [
        "recording_id",
        "dataset_id",
        "label",
        "label_id",
        "condition_id",
        "speed_profile_id",
        "bearing_type_id",
        "bearing_id",
        "true_label_id",
        "mean_logit_pred",
        "majority_vote_pred",
        "num_windows",
        "vote_counts",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            item = dict(row)
            item["vote_counts"] = json.dumps(item["vote_counts"], ensure_ascii=False, sort_keys=True)
            writer.writerow(item)


def main() -> None:
    args = parse_args()
    result = evaluate_recordings(
        index_csv=args.index,
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        device=args.device,
        num_workers=args.num_workers,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {key: value for key, value in result.items() if key != "per_recording"}
    summary_path = args.output_dir / "recording_metrics.json"
    detail_path = args.output_dir / "recording_predictions.csv"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_per_recording(result["per_recording"], detail_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {summary_path}")
    print(f"Wrote {detail_path}")


if __name__ == "__main__":
    main()
