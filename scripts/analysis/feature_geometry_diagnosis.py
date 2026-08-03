#!/usr/bin/env python3
"""Diagnose source/target feature geometry for CIC-MAN checkpoints."""

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
from cicman.models.cic_man import build_cic_man  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-index", type=Path, action="append", required=True)
    parser.add_argument("--target-index", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-source-items", type=int, default=None)
    parser.add_argument("--max-target-items", type=int, default=None)
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


def extract_features(
    *,
    model,
    index_csvs: list[Path],
    batch_size: int,
    device: str,
    num_workers: int,
    max_items: int | None,
) -> list[dict[str, object]]:
    import torch
    import torch.nn.functional as F

    rows: list[dict[str, object]] = []
    model.eval()
    with torch.no_grad():
        for index_csv in index_csvs:
            loader = make_loader(index_csv, batch_size, num_workers, max_items)
            for batch in loader:
                x = batch["x"].to(device)
                y = batch["y"].detach().cpu().numpy().astype(int)
                details = model(x, return_details=True)
                features = F.normalize(details["features"], dim=1).detach().cpu().numpy().astype(np.float32)
                pred = details["logits"].argmax(dim=1).detach().cpu().numpy().astype(int)
                metadata = batch["metadata"]
                dataset_ids = as_list(metadata, "dataset_id")
                labels = as_list(metadata, "label")
                recording_ids = as_list(metadata, "recording_id")
                for i in range(len(y)):
                    rows.append(
                        {
                            "feature": features[i],
                            "dataset_id": dataset_ids[i],
                            "label": labels[i],
                            "label_id": int(y[i]),
                            "pred_label_id": int(pred[i]),
                            "recording_id": recording_ids[i],
                        }
                    )
    return rows


def make_prototypes(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    prototypes: dict[str, dict[str, object]] = {}
    groups: dict[tuple[str, str, int], list[np.ndarray]] = defaultdict(list)
    label_groups: dict[tuple[str, int], list[np.ndarray]] = defaultdict(list)
    for row in rows:
        key = (str(row["dataset_id"]), str(row["label"]), int(row["label_id"]))
        groups[key].append(row["feature"])
        label_key = (str(row["label"]), int(row["label_id"]))
        label_groups[label_key].append(row["feature"])

    for (dataset_id, label, label_id), features in sorted(groups.items()):
        vector = normalize(np.mean(np.stack(features, axis=0), axis=0))
        name = f"{dataset_id}:{label}"
        prototypes[name] = {
            "name": name,
            "dataset_id": dataset_id,
            "label": label,
            "label_id": label_id,
            "num_samples": len(features),
            "feature": vector,
        }
    for (label, label_id), features in sorted(label_groups.items()):
        vector = normalize(np.mean(np.stack(features, axis=0), axis=0))
        name = f"source_all:{label}"
        prototypes[name] = {
            "name": name,
            "dataset_id": "source_all",
            "label": label,
            "label_id": label_id,
            "num_samples": len(features),
            "feature": vector,
        }
    return prototypes


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


def aggregate_target_geometry(
    target_rows: list[dict[str, object]],
    prototypes: dict[str, dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    prototype_items = list(prototypes.values())
    sample_rows: list[dict[str, object]] = []
    grouped: dict[tuple[str, int], dict[str, object]] = {}

    for row in target_rows:
        feature = row["feature"]
        similarities = [
            (proto["name"], float(np.dot(feature, proto["feature"])))
            for proto in prototype_items
        ]
        nearest_name, nearest_similarity = max(similarities, key=lambda item: item[1])
        nearest_proto = prototypes[nearest_name]
        out = {
            "target_dataset_id": row["dataset_id"],
            "target_label": row["label"],
            "target_label_id": row["label_id"],
            "pred_label_id": row["pred_label_id"],
            "recording_id": row["recording_id"],
            "nearest_prototype": nearest_name,
            "nearest_prototype_label": nearest_proto["label"],
            "nearest_prototype_dataset": nearest_proto["dataset_id"],
            "nearest_similarity": nearest_similarity,
        }
        for proto_name, similarity in similarities:
            out[f"sim::{proto_name}"] = similarity
        sample_rows.append(out)

        group_key = (str(row["label"]), int(row["label_id"]))
        group = grouped.setdefault(
            group_key,
            {
                "target_label": row["label"],
                "target_label_id": row["label_id"],
                "num_samples": 0,
                "pred_counts": defaultdict(int),
                "nearest_counts": defaultdict(int),
                "similarity_sums": defaultdict(float),
            },
        )
        group["num_samples"] += 1
        group["pred_counts"][str(row["pred_label_id"])] += 1
        group["nearest_counts"][nearest_name] += 1
        for proto_name, similarity in similarities:
            group["similarity_sums"][proto_name] += similarity

    summary_rows: list[dict[str, object]] = []
    for group in grouped.values():
        num_samples = int(group["num_samples"])
        summary = {
            "target_label": group["target_label"],
            "target_label_id": group["target_label_id"],
            "num_samples": num_samples,
        }
        for proto in prototype_items:
            proto_name = proto["name"]
            summary[f"mean_sim::{proto_name}"] = group["similarity_sums"][proto_name] / max(1, num_samples)
            summary[f"nearest_frac::{proto_name}"] = group["nearest_counts"][proto_name] / max(1, num_samples)
        for pred_label_id, count in sorted(group["pred_counts"].items()):
            summary[f"pred_frac::{pred_label_id}"] = count / max(1, num_samples)
        summary_rows.append(summary)
    summary_rows.sort(key=lambda item: int(item["target_label_id"]))
    return summary_rows, sample_rows


def aggregate_recording_geometry(sample_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[str, dict[str, object]] = {}
    similarity_keys = [key for key in sample_rows[0] if key.startswith("sim::")] if sample_rows else []
    for row in sample_rows:
        group = groups.setdefault(
            str(row["recording_id"]),
            {
                "recording_id": row["recording_id"],
                "target_label": row["target_label"],
                "target_label_id": row["target_label_id"],
                "num_windows": 0,
                "nearest_counts": defaultdict(int),
                "similarity_sums": defaultdict(float),
            },
        )
        group["num_windows"] += 1
        group["nearest_counts"][row["nearest_prototype"]] += 1
        for key in similarity_keys:
            group["similarity_sums"][key] += float(row[key])

    rows: list[dict[str, object]] = []
    for group in groups.values():
        num_windows = int(group["num_windows"])
        mean_sims = {
            key.removeprefix("sim::"): group["similarity_sums"][key] / max(1, num_windows)
            for key in similarity_keys
        }
        nearest_proto, nearest_similarity = max(mean_sims.items(), key=lambda item: item[1])
        row = {
            "recording_id": group["recording_id"],
            "target_label": group["target_label"],
            "target_label_id": group["target_label_id"],
            "num_windows": num_windows,
            "nearest_mean_prototype": nearest_proto,
            "nearest_mean_similarity": nearest_similarity,
        }
        for proto_name, similarity in mean_sims.items():
            row[f"mean_sim::{proto_name}"] = similarity
            row[f"nearest_window_frac::{proto_name}"] = group["nearest_counts"][proto_name] / max(1, num_windows)
        rows.append(row)
    rows.sort(key=lambda item: (int(item["target_label_id"]), str(item["recording_id"])))
    return rows


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    seen = set(fieldnames)
    for row in rows[1:]:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    import torch

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(args.checkpoint, map_location=device)
    num_classes = int(checkpoint["num_classes"])
    num_agents = int(checkpoint.get("num_agents", checkpoint.get("config", {}).get("num_agents", 4)))
    model = build_cic_man(num_classes=num_classes, num_agents=num_agents).to(device)
    model.load_state_dict(checkpoint["model_state"])

    source_rows = extract_features(
        model=model,
        index_csvs=args.source_index,
        batch_size=args.batch_size,
        device=device,
        num_workers=args.num_workers,
        max_items=args.max_source_items,
    )
    target_rows = extract_features(
        model=model,
        index_csvs=[args.target_index],
        batch_size=args.batch_size,
        device=device,
        num_workers=args.num_workers,
        max_items=args.max_target_items,
    )
    prototypes = make_prototypes(source_rows)
    summary_rows, sample_rows = aggregate_target_geometry(target_rows, prototypes)
    recording_rows = aggregate_recording_geometry(sample_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prototype_rows = [
        {key: value for key, value in proto.items() if key != "feature"}
        for proto in prototypes.values()
    ]
    write_csv(prototype_rows, args.output_dir / "feature_geometry_prototypes.csv")
    write_csv(summary_rows, args.output_dir / "feature_geometry_target_label_summary.csv")
    write_csv(recording_rows, args.output_dir / "feature_geometry_recording_summary.csv")

    json_payload = {
        "checkpoint": str(args.checkpoint),
        "source_indices": [str(path) for path in args.source_index],
        "target_index": str(args.target_index),
        "num_source_samples": len(source_rows),
        "num_target_samples": len(target_rows),
        "target_label_summary": summary_rows,
    }
    json_path = args.output_dir / "feature_geometry_summary.json"
    json_path.write_text(json.dumps(json_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(json_payload, ensure_ascii=False, indent=2))
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
