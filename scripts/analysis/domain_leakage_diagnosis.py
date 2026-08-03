#!/usr/bin/env python3
"""Probe health separability and source-domain leakage in frozen CIC-MAN features."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np


def add_src_to_path() -> None:
    script_path = Path(__file__).resolve()
    project_dir = script_path.parents[2]
    sys.path.insert(0, str(project_dir / "src"))


add_src_to_path()

from cicman.data.dataset import WindowIndexDataset  # noqa: E402
from cicman.evaluation.metrics import accuracy, macro_f1  # noqa: E402
from cicman.models.cic_man import build_cic_man  # noqa: E402
from cicman.models.cic_man_gated_filterbank import build_cic_man_gated_filterbank  # noqa: E402
from cicman.models.cic_man_gated_viewbank import build_cic_man_gated_viewbank  # noqa: E402
from cicman.models.cic_man_heterogeneous import build_cic_man_heterogeneous  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-index", type=Path, required=True, help="Source training window-index CSV.")
    parser.add_argument("--eval-index", type=Path, required=True, help="Source held-out validation window-index CSV.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--target-name", required=True)
    parser.add_argument(
        "--feature-key",
        default="features",
        choices=["features", "core_features", "filterbank_features", "health_features", "style_features"],
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--probe-batch-size", type=int, default=512)
    parser.add_argument("--probe-epochs", type=int, default=80)
    parser.add_argument("--probe-lr", type=float, default=0.02)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-items", type=int, default=None)
    parser.add_argument("--max-eval-items", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def limit_dataset(dataset, max_items: int | None, seed: int):
    if max_items is None or max_items <= 0 or max_items >= len(dataset):
        return dataset
    import torch

    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    indices = sorted(indices[:max_items])
    return torch.utils.data.Subset(dataset, indices)


def make_loader(index_csv: Path, batch_size: int, num_workers: int, max_items: int | None, seed: int):
    import torch

    dataset = limit_dataset(WindowIndexDataset(index_csv), max_items, seed)
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


def build_model_from_checkpoint(checkpoint: dict[str, object], device: str):
    num_classes = int(checkpoint["num_classes"])
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
    return model.to(device).eval()


def extract_features(
    *,
    model,
    index_csv: Path,
    feature_key: str,
    batch_size: int,
    device: str,
    num_workers: int,
    max_items: int | None,
    seed: int,
) -> dict[str, object]:
    import torch
    import torch.nn.functional as F

    loader = make_loader(index_csv, batch_size, num_workers, max_items, seed)
    features: list[np.ndarray] = []
    health_labels: list[int] = []
    domain_names: list[str] = []
    label_names: list[str] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            details = model(x, return_details=True)
            if feature_key not in details:
                available = ", ".join(sorted(details.keys()))
                raise KeyError(f"Feature key '{feature_key}' not found. Available keys: {available}")
            feature_tensor = F.normalize(details[feature_key], dim=1)
            features.append(feature_tensor.detach().cpu().numpy().astype(np.float32))
            health_labels.extend(batch["y"].detach().cpu().numpy().astype(int).tolist())
            metadata = batch["metadata"]
            domain_names.extend(metadata_column(metadata, "dataset_id"))
            label_names.extend(metadata_column(metadata, "label"))

    feature_array = np.concatenate(features, axis=0) if features else np.zeros((0, 0), dtype=np.float32)
    return {
        "features": feature_array,
        "health_labels": np.asarray(health_labels, dtype=np.int64),
        "domain_names": domain_names,
        "label_names": label_names,
    }


def encode_domains(train_names: list[str], eval_names: list[str]) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    domains = {name: idx for idx, name in enumerate(sorted(set(train_names) | set(eval_names)))}
    train_y = np.asarray([domains[name] for name in train_names], dtype=np.int64)
    eval_y = np.asarray([domains[name] for name in eval_names], dtype=np.int64)
    return train_y, eval_y, domains


def standardize(train_x: np.ndarray, eval_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.maximum(std, 1e-6)
    return (train_x - mean) / std, (eval_x - mean) / std


def class_distribution(values: np.ndarray, num_classes: int) -> list[int]:
    return np.bincount(values.astype(np.int64), minlength=num_classes).astype(int).tolist()


def majority_baseline(values: np.ndarray, num_classes: int) -> float:
    if len(values) == 0:
        return 0.0
    counts = np.bincount(values.astype(np.int64), minlength=num_classes)
    return float(counts.max() / max(1, counts.sum()))


def train_linear_probe(
    *,
    train_x: np.ndarray,
    train_y: np.ndarray,
    eval_x: np.ndarray,
    eval_y: np.ndarray,
    num_classes: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: str,
    seed: int,
) -> dict[str, object]:
    import torch
    import torch.nn as nn

    if len(train_x) == 0 or len(eval_x) == 0 or num_classes <= 1:
        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "num_classes": num_classes,
            "num_train_samples": int(len(train_x)),
            "num_eval_samples": int(len(eval_x)),
            "class_distribution_train": class_distribution(train_y, num_classes),
            "class_distribution_eval": class_distribution(eval_y, num_classes),
        }

    torch.manual_seed(seed)
    train_tensor = torch.from_numpy(train_x.astype(np.float32))
    train_label_tensor = torch.from_numpy(train_y.astype(np.int64))
    dataset = torch.utils.data.TensorDataset(train_tensor, train_label_tensor)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)

    probe = nn.Linear(train_x.shape[1], num_classes).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    probe.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(probe(batch_x), batch_y)
            loss.backward()
            optimizer.step()

    probe.eval()
    with torch.no_grad():
        eval_tensor = torch.from_numpy(eval_x.astype(np.float32)).to(device)
        pred = probe(eval_tensor).argmax(dim=1).detach().cpu().numpy().astype(int)
    return {
        "accuracy": accuracy(eval_y, pred),
        "macro_f1": macro_f1(eval_y, pred, num_classes),
        "num_classes": int(num_classes),
        "num_train_samples": int(len(train_x)),
        "num_eval_samples": int(len(eval_x)),
        "class_distribution_train": class_distribution(train_y, num_classes),
        "class_distribution_eval": class_distribution(eval_y, num_classes),
    }


def summarize_domain_feature_means(
    features: np.ndarray,
    health_labels: np.ndarray,
    domain_labels: np.ndarray,
    num_health_classes: int,
    num_domains: int,
) -> dict[str, float]:
    if len(features) == 0:
        return {"class_conditional_domain_centroid_distance": 0.0, "global_domain_centroid_distance": 0.0}
    domain_centroids = []
    for domain_id in range(num_domains):
        mask = domain_labels == domain_id
        if mask.any():
            domain_centroids.append(features[mask].mean(axis=0))
    global_distance = mean_pairwise_distance(domain_centroids)

    class_distances = []
    for cls in range(num_health_classes):
        cls_centroids = []
        for domain_id in range(num_domains):
            mask = (health_labels == cls) & (domain_labels == domain_id)
            if mask.any():
                cls_centroids.append(features[mask].mean(axis=0))
        if len(cls_centroids) > 1:
            class_distances.append(mean_pairwise_distance(cls_centroids))
    return {
        "class_conditional_domain_centroid_distance": float(np.mean(class_distances)) if class_distances else 0.0,
        "global_domain_centroid_distance": float(global_distance),
    }


def mean_pairwise_distance(vectors: list[np.ndarray]) -> float:
    if len(vectors) < 2:
        return 0.0
    distances = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            distances.append(float(np.linalg.norm(vectors[i] - vectors[j])))
    return float(np.mean(distances)) if distances else 0.0


def write_markdown(summary: dict[str, object], path: Path) -> None:
    health = summary["health_probe"]
    domain = summary["domain_probe"]
    rows = [
        "# Health/Domain Disentanglement Diagnosis",
        "",
        f"- Model: `{summary['model_name']}`",
        f"- Target-free task: `{summary['target_name']}`",
        f"- Feature key: `{summary['feature_key']}`",
        f"- Source domains: `{', '.join(summary['source_domains'])}`",
        f"- Health probe macro-F1: `{health['macro_f1']:.6f}`",
        f"- Domain probe accuracy: `{domain['accuracy']:.6f}`",
        f"- Domain chance accuracy: `{summary['domain_chance_accuracy']:.6f}`",
        f"- Domain majority baseline accuracy: `{summary['domain_majority_baseline_accuracy']:.6f}`",
        f"- Domain leakage over chance: `{summary['domain_leakage_over_chance']:.6f}`",
        f"- Domain leakage over majority baseline: `{summary['domain_leakage_over_majority']:.6f}`",
        f"- Health-minus-domain-leakage score: `{summary['health_minus_domain_leakage']:.6f}`",
        f"- Class-conditional domain centroid distance: `{summary['class_conditional_domain_centroid_distance']:.6f}`",
        "",
        "Interpretation: lower domain leakage is better only when health separability is retained. This probe is source-only and does not use target labels for selection.",
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    import torch

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = build_model_from_checkpoint(checkpoint, device)
    num_health_classes = int(checkpoint["num_classes"])

    train_payload = extract_features(
        model=model,
        index_csv=args.train_index,
        feature_key=args.feature_key,
        batch_size=args.batch_size,
        device=device,
        num_workers=args.num_workers,
        max_items=args.max_train_items,
        seed=args.seed,
    )
    eval_payload = extract_features(
        model=model,
        index_csv=args.eval_index,
        feature_key=args.feature_key,
        batch_size=args.batch_size,
        device=device,
        num_workers=args.num_workers,
        max_items=args.max_eval_items,
        seed=args.seed + 1,
    )

    train_x = train_payload["features"]
    eval_x = eval_payload["features"]
    train_x, eval_x = standardize(train_x, eval_x)
    train_health = train_payload["health_labels"]
    eval_health = eval_payload["health_labels"]
    train_domain, eval_domain, domain_map = encode_domains(train_payload["domain_names"], eval_payload["domain_names"])
    num_domains = len(domain_map)

    health_probe = train_linear_probe(
        train_x=train_x,
        train_y=train_health,
        eval_x=eval_x,
        eval_y=eval_health,
        num_classes=num_health_classes,
        epochs=args.probe_epochs,
        batch_size=args.probe_batch_size,
        lr=args.probe_lr,
        weight_decay=args.weight_decay,
        device=device,
        seed=args.seed,
    )
    domain_probe = train_linear_probe(
        train_x=train_x,
        train_y=train_domain,
        eval_x=eval_x,
        eval_y=eval_domain,
        num_classes=num_domains,
        epochs=args.probe_epochs,
        batch_size=args.probe_batch_size,
        lr=args.probe_lr,
        weight_decay=args.weight_decay,
        device=device,
        seed=args.seed + 13,
    )
    centroid_metrics = summarize_domain_feature_means(
        features=eval_x,
        health_labels=eval_health,
        domain_labels=eval_domain,
        num_health_classes=num_health_classes,
        num_domains=num_domains,
    )
    chance = 1.0 / max(1, num_domains)
    majority = majority_baseline(eval_domain, num_domains)
    leakage_over_chance = max(0.0, float(domain_probe["accuracy"]) - chance)
    leakage_over_majority = max(0.0, float(domain_probe["accuracy"]) - majority)
    summary = {
        "model_name": args.model_name,
        "target_name": args.target_name,
        "feature_key": args.feature_key,
        "checkpoint": str(args.checkpoint),
        "train_index": str(args.train_index),
        "eval_index": str(args.eval_index),
        "source_domains": [name for name, _ in sorted(domain_map.items(), key=lambda item: item[1])],
        "num_health_classes": num_health_classes,
        "health_probe": health_probe,
        "domain_probe": domain_probe,
        "domain_chance_accuracy": chance,
        "domain_majority_baseline_accuracy": majority,
        "domain_leakage_over_chance": leakage_over_chance,
        "domain_leakage_over_majority": leakage_over_majority,
        "health_minus_domain_leakage": float(health_probe["macro_f1"]) - leakage_over_majority,
        **centroid_metrics,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"domain_leakage_{args.model_name}_{args.target_name}_{args.feature_key}.json"
    md_path = args.output_dir / f"domain_leakage_{args.model_name}_{args.target_name}_{args.feature_key}.md"
    csv_path = args.output_dir / f"domain_leakage_{args.model_name}_{args.target_name}_{args.feature_key}.csv"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(summary, md_path)
    flat_row = {
        "model_name": args.model_name,
        "target_name": args.target_name,
        "feature_key": args.feature_key,
        "health_macro_f1": health_probe["macro_f1"],
        "health_accuracy": health_probe["accuracy"],
        "domain_accuracy": domain_probe["accuracy"],
        "domain_macro_f1": domain_probe["macro_f1"],
        "domain_chance_accuracy": chance,
        "domain_majority_baseline_accuracy": majority,
        "domain_leakage_over_chance": leakage_over_chance,
        "domain_leakage_over_majority": leakage_over_majority,
        "health_minus_domain_leakage": summary["health_minus_domain_leakage"],
        "class_conditional_domain_centroid_distance": summary["class_conditional_domain_centroid_distance"],
        "global_domain_centroid_distance": summary["global_domain_centroid_distance"],
        "num_train_samples": health_probe["num_train_samples"],
        "num_eval_samples": health_probe["num_eval_samples"],
        "source_domains": ";".join(summary["source_domains"]),
    }
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_row.keys()))
        writer.writeheader()
        writer.writerow(flat_row)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
