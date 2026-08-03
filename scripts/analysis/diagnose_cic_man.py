#!/usr/bin/env python3
"""Diagnose router collapse and agent specialization in CIC-MAN checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
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
from cicman.evaluation.metrics import accuracy, balanced_accuracy, confusion_matrix, macro_f1  # noqa: E402
from cicman.models.cic_man import build_cic_man  # noqa: E402
from cicman.models.cic_man_gated_filterbank import build_cic_man_gated_filterbank  # noqa: E402
from cicman.models.cic_man_gated_viewbank import build_cic_man_gated_viewbank  # noqa: E402
from cicman.models.cic_man_heterogeneous import build_cic_man_heterogeneous  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-name", default="test")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-items", type=int, default=None)
    return parser.parse_args()


def limit_dataset(dataset, max_items: int | None):
    if max_items is None or max_items <= 0 or max_items >= len(dataset):
        return dataset
    import torch

    return torch.utils.data.Subset(dataset, list(range(max_items)))


def make_loader(index_csv: Path, batch_size: int, num_workers: int, max_items: int | None):
    import torch

    dataset = limit_dataset(WindowIndexDataset(index_csv), max_items)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def as_list(metadata: dict[str, object], key: str) -> list[str]:
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


def grouped_metrics(rows: list[dict[str, object]], key: str, num_classes: int) -> dict[str, dict[str, object]]:
    groups: dict[str, tuple[list[int], list[int]]] = defaultdict(lambda: ([], []))
    for row in rows:
        true_values, pred_values = groups[str(row[key])]
        true_values.append(int(row["true_label_id"]))
        pred_values.append(int(row["pred_label_id"]))
    return {
        group: metric_dict(true_values, pred_values, num_classes)
        for group, (true_values, pred_values) in sorted(groups.items())
    }


def entropy(weights: np.ndarray) -> np.ndarray:
    eps = 1e-12
    return -(weights * np.log(weights + eps)).sum(axis=1)


def effective_agents(mean_weights: np.ndarray) -> float:
    eps = 1e-12
    return float(np.exp(-(mean_weights * np.log(mean_weights + eps)).sum()))


def diagnose(
    *,
    index_csv: Path,
    checkpoint_path: Path,
    batch_size: int,
    device: str,
    num_workers: int,
    max_items: int | None,
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
    elif architecture == "heterogeneous":
        model = build_cic_man_heterogeneous(num_classes=num_classes, num_agents=num_agents).to(device)
    else:
        model = build_cic_man(num_classes=num_classes, num_agents=num_agents).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    loader = make_loader(index_csv, batch_size, num_workers, max_items)
    y_true: list[int] = []
    y_pred: list[int] = []
    agent_preds: list[list[int]] = [[] for _ in range(num_agents)]
    router_weights: list[np.ndarray] = []
    top_agents: list[int] = []
    rows: list[dict[str, object]] = []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].detach().cpu().numpy().astype(int)
            details = model(x, return_details=True)
            logits = details["logits"].detach().cpu().numpy()
            weights = details["router_weights"].detach().cpu().numpy()
            agent_logits = details["agent_logits"].detach().cpu().numpy()
            pred = logits.argmax(axis=1).astype(int)
            metadata = batch["metadata"]
            dataset_ids = as_list(metadata, "dataset_id")
            labels = as_list(metadata, "label")
            recording_ids = as_list(metadata, "recording_id")
            top = weights.argmax(axis=1).astype(int)

            y_true.extend(y.tolist())
            y_pred.extend(pred.tolist())
            router_weights.append(weights)
            top_agents.extend(top.tolist())
            for agent_idx in range(num_agents):
                agent_preds[agent_idx].extend(agent_logits[:, agent_idx, :].argmax(axis=1).astype(int).tolist())
            for i in range(len(y)):
                rows.append(
                    {
                        "dataset_id": dataset_ids[i],
                        "label": labels[i],
                        "recording_id": recording_ids[i],
                        "true_label_id": int(y[i]),
                        "pred_label_id": int(pred[i]),
                        "top_agent": int(top[i]),
                        **{f"router_weight_{j}": float(weights[i, j]) for j in range(num_agents)},
                    }
                )

    weight_array = np.concatenate(router_weights, axis=0) if router_weights else np.zeros((0, num_agents))
    top_counts = np.bincount(np.asarray(top_agents, dtype=np.int64), minlength=num_agents)
    top_usage = top_counts / max(1, top_counts.sum())
    mean_weights = weight_array.mean(axis=0) if len(weight_array) else np.zeros(num_agents)
    entropy_values = entropy(weight_array) if len(weight_array) else np.zeros(0)

    agent_metrics = {
        f"agent_{idx}": metric_dict(y_true, agent_preds[idx], num_classes)
        for idx in range(num_agents)
    }

    return {
        "index": str(index_csv),
        "checkpoint": str(checkpoint_path),
        "num_classes": num_classes,
        "num_agents": num_agents,
        "overall": metric_dict(y_true, y_pred, num_classes),
        "router": {
            "mean_weights": mean_weights.tolist(),
            "top_agent_counts": top_counts.tolist(),
            "top_agent_usage": top_usage.tolist(),
            "mean_entropy": float(entropy_values.mean()) if len(entropy_values) else 0.0,
            "max_entropy": float(np.log(num_agents)) if num_agents > 0 else 0.0,
            "normalized_mean_entropy": float(entropy_values.mean() / np.log(num_agents)) if num_agents > 1 and len(entropy_values) else 0.0,
            "effective_agents_from_mean_weights": effective_agents(mean_weights) if len(weight_array) else 0.0,
        },
        "agent_metrics": agent_metrics,
        "by_dataset": grouped_metrics(rows, "dataset_id", num_classes),
        "by_label": grouped_metrics(rows, "label", num_classes),
        "rows": rows,
    }


def write_rows(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    result = diagnose(
        index_csv=args.index,
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        device=args.device,
        num_workers=args.num_workers,
        max_items=args.max_items,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {key: value for key, value in result.items() if key != "rows"}
    summary_path = args.output_dir / f"cic_man_diagnostics_{args.split_name}.json"
    rows_path = args.output_dir / f"cic_man_diagnostics_{args.split_name}.csv"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_rows(result["rows"], rows_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {summary_path}")
    print(f"Wrote {rows_path}")


if __name__ == "__main__":
    main()
