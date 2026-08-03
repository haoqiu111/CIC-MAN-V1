#!/usr/bin/env python3
"""Paderborn fault-mechanism/source-coverage limitation audit."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


FEATURE_NAMES = ["kurtosis", "crest", "roughness", "high_energy", "envelope_ratio", "zero_crossing"]
CLASS_NAMES = {0: "normal", 1: "inner", 2: "outer"}


def add_src_to_path() -> None:
    script_path = Path(__file__).resolve()
    project_dir = script_path.parents[2]
    sys.path.insert(0, str(project_dir / "src"))


add_src_to_path()

from cicman.data.dataset import WindowIndexDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def metadata_column(metadata: dict[str, object], key: str) -> list[str]:
    values = metadata[key]
    if isinstance(values, list):
        return [str(value) for value in values]
    return [str(value) for value in list(values)]


def make_loader(index_csv: Path, batch_size: int, num_workers: int):
    import torch

    dataset = WindowIndexDataset(index_csv)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )


def mechanism_features(x):
    import torch
    import torch.nn.functional as F

    centered = x - x.mean(dim=-1, keepdim=True)
    std = centered.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
    normalized = centered / std
    rms = centered.pow(2).mean(dim=-1).sqrt().clamp_min(1e-6)
    kurtosis = torch.log1p(normalized.pow(4).mean(dim=-1))
    crest = torch.log1p(centered.abs().amax(dim=-1) / rms)
    roughness = (x[:, :, 1:] - x[:, :, :-1]).abs().mean(dim=-1) / x.abs().mean(dim=-1).clamp_min(1e-6)
    smooth = F.avg_pool1d(x, kernel_size=33, stride=1, padding=16)
    high_energy = (x - smooth).pow(2).mean(dim=-1) / x.pow(2).mean(dim=-1).clamp_min(1e-6)
    envelope = torch.sqrt((x - smooth).pow(2) + 1e-6)
    envelope_ratio = envelope.mean(dim=-1) / x.abs().mean(dim=-1).clamp_min(1e-6)
    signs = torch.sign(centered[:, :, 1:] * centered[:, :, :-1])
    zero_crossing = (signs < 0).float().mean(dim=-1)
    return torch.cat([kurtosis, crest, roughness, high_energy, envelope_ratio, zero_crossing], dim=1)


def collect(index_csv: Path, split_name: str, batch_size: int, num_workers: int) -> list[dict[str, object]]:
    import torch

    rows = []
    loader = make_loader(index_csv, batch_size, num_workers)
    with torch.no_grad():
        for batch in loader:
            x = batch["x"]
            y = batch["y"].detach().cpu().numpy().astype(int)
            features = mechanism_features(x).detach().cpu().numpy().astype(np.float64)
            metadata = batch["metadata"]
            dataset_ids = metadata_column(metadata, "dataset_id")
            labels = metadata_column(metadata, "label")
            for i in range(len(y)):
                rows.append(
                    {
                        "split_name": split_name,
                        "dataset_id": dataset_ids[i],
                        "label": labels[i],
                        "label_id": int(y[i]),
                        **{name: float(features[i, idx]) for idx, name in enumerate(FEATURE_NAMES)},
                    }
                )
    return rows


def group_stats(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, int], list[np.ndarray]] = defaultdict(list)
    for row in rows:
        key = (str(row["dataset_id"]), str(row["label"]), int(row["label_id"]))
        groups[key].append(np.asarray([float(row[name]) for name in FEATURE_NAMES], dtype=np.float64))
    out = []
    for (dataset_id, label, label_id), values in sorted(groups.items()):
        array = np.stack(values, axis=0)
        item = {
            "dataset_id": dataset_id,
            "label": label,
            "label_id": label_id,
            "num_windows": len(values),
        }
        for idx, name in enumerate(FEATURE_NAMES):
            item[f"{name}_mean"] = float(array[:, idx].mean())
            item[f"{name}_std"] = float(array[:, idx].std())
        out.append(item)
    return out


def class_centroids(rows: list[dict[str, object]], split_name: str) -> dict[int, np.ndarray]:
    groups: dict[int, list[np.ndarray]] = defaultdict(list)
    for row in rows:
        if row["split_name"] != split_name:
            continue
        groups[int(row["label_id"])].append(np.asarray([float(row[name]) for name in FEATURE_NAMES], dtype=np.float64))
    return {label_id: np.stack(values, axis=0).mean(axis=0) for label_id, values in groups.items()}


def coverage_rows(source_rows: list[dict[str, object]], target_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    source_centroids = class_centroids(source_rows, "source_train")
    target_centroids = class_centroids(target_rows, "paderborn_test")
    all_source = np.stack(list(source_centroids.values()), axis=0)
    scale = all_source.std(axis=0) + 1e-6
    rows = []
    for target_label, target_vector in sorted(target_centroids.items()):
        distances = []
        for source_label, source_vector in sorted(source_centroids.items()):
            distance = float(np.linalg.norm((target_vector - source_vector) / scale))
            distances.append((source_label, distance))
        nearest_label, nearest_distance = min(distances, key=lambda item: item[1])
        correct_distance = dict(distances).get(target_label, float("nan"))
        rows.append(
            {
                "target_label_id": target_label,
                "target_label": CLASS_NAMES[target_label],
                "nearest_source_label_id": nearest_label,
                "nearest_source_label": CLASS_NAMES[nearest_label],
                "nearest_distance": nearest_distance,
                "same_class_distance": correct_distance,
                "same_minus_nearest_distance": correct_distance - nearest_distance,
                **{f"distance_to_source_{CLASS_NAMES[label]}": distance for label, distance in distances},
            }
        )
    return rows


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
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


def write_markdown(coverage: list[dict[str, object]], path: Path) -> None:
    lines = [
        "# Paderborn Mechanism Coverage Audit",
        "",
        "Scope: source train = HUST + Ottawa under `target_dataset_paderborn`; target = Paderborn test. Target labels are used only for limitation analysis, not model selection.",
        "",
        "| Paderborn Class | Nearest Source Class | Nearest Distance | Same-Class Distance | Same - Nearest |",
        "|---|---|---:|---:|---:|",
    ]
    for row in coverage:
        lines.append(
            f"| {row['target_label']} | {row['nearest_source_label']} | "
            f"{float(row['nearest_distance']):.6f} | {float(row['same_class_distance']):.6f} | "
            f"{float(row['same_minus_nearest_distance']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Adversarial Conclusion",
            "",
            "- If Paderborn inner/outer are closer to the wrong source class in mechanism space, source weighting is the wrong remedy.",
            "- This audit is post-hoc limitation evidence: it can justify fault-mechanism priors or explicit source coverage expansion, but it must not be used as a target-free selector.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    output_dir = (args.output_dir or project_root  / "outputs" / "tables").resolve()
    base = project_root / "data" / "paper1_cicman" / "cache" / "windows" / "cross_dataset_task3_source_mixed" / "target_dataset_paderborn"
    source_rows = collect(base / "train_windows.csv", "source_train", args.batch_size, args.num_workers)
    target_rows = collect(base / "test_windows.csv", "paderborn_test", args.batch_size, args.num_workers)
    stats = group_stats(source_rows + target_rows)
    coverage = coverage_rows(source_rows, target_rows)
    write_csv(stats, output_dir / "paderborn_mechanism_coverage_stats.csv")
    write_csv(coverage, output_dir / "paderborn_mechanism_coverage_distances.csv")
    write_markdown(coverage, output_dir / "paderborn_mechanism_coverage_audit.md")
    print(f"Wrote Paderborn mechanism coverage audit to {output_dir}")


if __name__ == "__main__":
    main()
