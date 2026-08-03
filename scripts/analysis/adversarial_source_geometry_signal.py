#!/usr/bin/env python3
"""Source-only adversarial geometry signals for fragile class semantics.

The audit intentionally avoids target labels when computing risk signals. Target
recall is appended only as a post-hoc falsification column.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np


TARGETS = ["hust", "ottawa", "paderborn"]
CLASS_NAMES = {0: "normal", 1: "inner", 2: "outer"}


def add_src_to_path() -> None:
    script_path = Path(__file__).resolve()
    project_dir = script_path.parents[2]
    sys.path.insert(0, str(project_dir / "src"))


add_src_to_path()

from cicman.data.dataset import WindowIndexDataset  # noqa: E402
from cicman.models.cic_man import build_cic_man  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-items", type=int, default=None)
    return parser.parse_args()


def metadata_column(metadata: dict[str, object], key: str) -> list[str]:
    values = metadata[key]
    if isinstance(values, list):
        return [str(value) for value in values]
    return [str(value) for value in list(values)]


def make_loader(index_csv: Path, batch_size: int, num_workers: int, max_items: int | None):
    import torch

    dataset = WindowIndexDataset(index_csv)
    if max_items is not None and 0 < max_items < len(dataset):
        dataset = torch.utils.data.Subset(dataset, list(range(max_items)))
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def load_v5_model(project_root: Path, target: str, device: str):
    import torch

    checkpoint_root = project_root  / "outputs" / "checkpoints"
    checkpoint_path = checkpoint_root / f"cic_man_v5_class_router_style_source_mixed_cross_dataset_task3_target_{target}" / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    num_classes = int(checkpoint["num_classes"])
    num_agents = int(checkpoint.get("num_agents", checkpoint.get("config", {}).get("num_agents", 4)))
    model = build_cic_man(num_classes=num_classes, num_agents=num_agents).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint_path


def normalize(array: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(array, axis=-1, keepdims=True)
    return array / np.clip(norm, 1e-12, None)


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


def extract_source_rows(
    *,
    model,
    index_csv: Path,
    batch_size: int,
    device: str,
    num_workers: int,
    max_items: int | None,
) -> list[dict[str, object]]:
    import torch
    import torch.nn.functional as F

    rows: list[dict[str, object]] = []
    loader = make_loader(index_csv, batch_size, num_workers, max_items)
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].detach().cpu().numpy().astype(int)
            details = model(x, return_details=True)
            features = F.normalize(details["features"], dim=1).detach().cpu().numpy().astype(np.float32)
            mech = mechanism_features(x).detach().cpu().numpy().astype(np.float32)
            metadata = batch["metadata"]
            dataset_ids = metadata_column(metadata, "dataset_id")
            labels = metadata_column(metadata, "label")
            for i in range(len(y)):
                rows.append(
                    {
                        "dataset_id": dataset_ids[i],
                        "label": labels[i],
                        "label_id": int(y[i]),
                        "feature": features[i],
                        "mechanism": mech[i],
                    }
                )
    return rows


def centroid(items: list[np.ndarray]) -> np.ndarray:
    return normalize(np.mean(np.stack(items, axis=0), axis=0, keepdims=True))[0].astype(np.float32)


def source_geometry(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_domain_class: dict[tuple[str, int], list[np.ndarray]] = defaultdict(list)
    by_class: dict[int, list[np.ndarray]] = defaultdict(list)
    mech_by_domain_class: dict[tuple[str, int], list[np.ndarray]] = defaultdict(list)
    mech_by_class: dict[int, list[np.ndarray]] = defaultdict(list)
    for row in rows:
        key = (str(row["dataset_id"]), int(row["label_id"]))
        by_domain_class[key].append(row["feature"])
        by_class[int(row["label_id"])].append(row["feature"])
        mech_by_domain_class[key].append(row["mechanism"])
        mech_by_class[int(row["label_id"])].append(row["mechanism"])

    domain_feature_centroids = {key: centroid(values) for key, values in by_domain_class.items()}
    class_feature_centroids = {label_id: centroid(values) for label_id, values in by_class.items()}
    domain_mech_centroids = {
        key: np.mean(np.stack(values, axis=0), axis=0).astype(np.float32)
        for key, values in mech_by_domain_class.items()
    }
    class_mech_centroids = {
        label_id: np.mean(np.stack(values, axis=0), axis=0).astype(np.float32)
        for label_id, values in mech_by_class.items()
    }

    rows_out = []
    for label_id, class_name in CLASS_NAMES.items():
        feature_protos = [
            proto for (domain_id, cls), proto in domain_feature_centroids.items() if cls == label_id
        ]
        mech_protos = [
            proto for (domain_id, cls), proto in domain_mech_centroids.items() if cls == label_id
        ]
        if not feature_protos:
            continue
        pair_sims = []
        for i, left in enumerate(feature_protos):
            for right in feature_protos[i + 1 :]:
                pair_sims.append(float(np.dot(left, right)))
        min_pair_sim = min(pair_sims) if pair_sims else 1.0
        mean_pair_sim = mean(pair_sims) if pair_sims else 1.0
        class_proto = class_feature_centroids[label_id]
        negative_sims = [
            float(np.dot(class_proto, proto))
            for cls, proto in class_feature_centroids.items()
            if cls != label_id
        ]
        nearest_negative_sim = max(negative_sims) if negative_sims else 0.0
        class_self_sims = [float(np.dot(class_proto, proto)) for proto in feature_protos]
        worst_self_sim = min(class_self_sims) if class_self_sims else 1.0
        boundary_margin = worst_self_sim - nearest_negative_sim
        domain_feature_spread = 1.0 - min_pair_sim

        mech_array = np.stack(mech_protos, axis=0)
        class_mech = class_mech_centroids[label_id]
        mech_scale = np.std(np.stack(list(class_mech_centroids.values()), axis=0), axis=0) + 1e-6
        mechanism_spread = float(np.mean(np.std(mech_array, axis=0) / mech_scale))
        mechanism_distances = np.linalg.norm((mech_array - class_mech) / mech_scale, axis=1)
        worst_mechanism_distance = float(np.max(mechanism_distances))
        adversarial_risk = (
            0.45 * domain_feature_spread
            + 0.35 * max(0.0, -boundary_margin)
            + 0.20 * min(3.0, mechanism_spread) / 3.0
        )
        rows_out.append(
            {
                "class_id": label_id,
                "class_name": class_name,
                "num_source_domains": len(feature_protos),
                "min_domain_feature_similarity": min_pair_sim,
                "mean_domain_feature_similarity": mean_pair_sim,
                "domain_feature_spread": domain_feature_spread,
                "worst_self_similarity_to_class_centroid": worst_self_sim,
                "nearest_negative_class_similarity": nearest_negative_sim,
                "source_boundary_margin": boundary_margin,
                "mechanism_spread": mechanism_spread,
                "worst_mechanism_distance": worst_mechanism_distance,
                "adversarial_source_risk": adversarial_risk,
            }
        )
    return rows_out


def target_recall_lookup(project_root: Path) -> dict[tuple[str, int], float]:
    path = project_root  / "outputs" / "tables" / "class_semantic_coverage_diagnosis.csv"
    lookup = {}
    if not path.exists():
        return lookup
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (
                row["method"] == "CIC-MAN-final-selector-seed42"
                and row["evaluation"] == "recording"
            ):
                lookup[(row["target"], int(row["class_id"]))] = float(row["recall"])
    return lookup


def source_val_recall_lookup(project_root: Path) -> dict[tuple[str, int], float]:
    path = project_root  / "outputs" / "tables" / "adversarial_source_heldout_audit.csv"
    lookup = {}
    if not path.exists():
        return lookup
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["method"] == "v5" and int(row["seed"]) == 42:
                lookup[(row["target"], int(row["class_id"]))] = float(row["source_val_recall"])
    return lookup


def collect(project_root: Path, args: argparse.Namespace) -> list[dict[str, object]]:
    import torch

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    base = project_root / "data" / "paper1_cicman" / "cache" / "windows" / "cross_dataset_task3_source_mixed"
    target_recalls = target_recall_lookup(project_root)
    source_recalls = source_val_recall_lookup(project_root)
    all_rows = []
    for target in TARGETS:
        model, checkpoint_path = load_v5_model(project_root, target, device)
        source_index = base / f"target_dataset_{target}" / "train_windows.csv"
        rows = extract_source_rows(
            model=model,
            index_csv=source_index,
            batch_size=args.batch_size,
            device=device,
            num_workers=args.num_workers,
            max_items=args.max_items,
        )
        for item in source_geometry(rows):
            class_id = int(item["class_id"])
            target_recall = target_recalls.get((target, class_id), None)
            source_recall = source_recalls.get((target, class_id), None)
            item.update(
                {
                    "target": target,
                    "checkpoint": str(checkpoint_path),
                    "source_index": str(source_index),
                    "source_val_recall_posthoc": source_recall,
                    "target_recording_recall_posthoc": target_recall,
                    "source_target_recall_gap_posthoc": (
                        float(source_recall) - float(target_recall)
                        if source_recall is not None and target_recall is not None
                        else ""
                    ),
                }
            )
            all_rows.append(item)
    return all_rows


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object) -> str:
    if value == "" or value is None:
        return ""
    return f"{float(value):.6f}"


def write_markdown(rows: list[dict[str, object]], path: Path) -> None:
    sorted_rows = sorted(rows, key=lambda row: (str(row["target"]), -float(row["adversarial_source_risk"])))
    lines = [
        "# Adversarial Source-Only Geometry Signal",
        "",
        "Signals are computed only from source train windows and v5 features. Target recall columns are post-hoc audit only.",
        "",
        "| Target | Class | Risk | Feature Spread | Boundary Margin | Mechanism Spread | Source-Val Recall | Target Recording Recall |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted_rows:
        lines.append(
            f"| {row['target']} | {row['class_name']} | {fmt(row['adversarial_source_risk'])} | "
            f"{fmt(row['domain_feature_spread'])} | {fmt(row['source_boundary_margin'])} | "
            f"{fmt(row['mechanism_spread'])} | {fmt(row['source_val_recall_posthoc'])} | "
            f"{fmt(row['target_recording_recall_posthoc'])} |"
        )
    lines.extend(
        [
            "",
            "## Adversarial Reading",
            "",
            "- A useful source-only signal should assign high risk to classes whose target recall later collapses, without using target labels to compute the risk.",
            "- If source-val recall is high but this geometry/mechanism risk is also high, the class is source-validated but semantically fragile.",
            "- If both source-val recall and this risk are benign while target recall collapses, the current source-only evidence is insufficient and the protocol needs stronger mechanism priors or data coverage.",
            "",
            "## Findings",
            "",
            "- Ottawa is detectable by source-only adversarial geometry: inner has high risk despite high source-val recall, and outer has the highest risk because same-class source prototypes disagree and the boundary margin turns negative.",
            "- Paderborn is not detectable by this source-only geometry signal: inner and outer look stable in HUST/Ottawa source features, yet target recording recall collapses. This is a stronger failure mode than a hard-class weighting issue.",
            "- The next trainable mechanism should use this signal only where it is informative. For Paderborn, the result argues for explicit fault-mechanism priors or source coverage expansion rather than another source-validation weighting rule.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_class_weights(rows: list[dict[str, object]], path: Path) -> None:
    weights: dict[str, list[float]] = {}
    for target in TARGETS:
        target_rows = [row for row in rows if row["target"] == target]
        by_class = {int(row["class_id"]): float(row["adversarial_source_risk"]) for row in target_rows}
        values = [1.0 + 2.0 * by_class.get(class_id, 0.0) for class_id in sorted(CLASS_NAMES)]
        weights[target] = values
    path.write_text(json.dumps(weights, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    output_dir = (args.output_dir or project_root  / "outputs" / "tables").resolve()
    rows = collect(project_root, args)
    write_csv(rows, output_dir / "adversarial_source_geometry_signal.csv")
    write_markdown(rows, output_dir / "adversarial_source_geometry_signal.md")
    write_class_weights(rows, output_dir / "adversarial_source_geometry_class_weights_seed42_v5.json")
    print(f"Wrote adversarial source-only geometry signal to {output_dir}")


if __name__ == "__main__":
    main()
