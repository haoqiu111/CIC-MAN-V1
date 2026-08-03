#!/usr/bin/env python3
"""Pretrain and validate CIC-MAN vFinal intervention view agents.

This stage intentionally does not train the router/gate. It tests whether each
intervention view can learn source-domain health semantics before it is allowed
to participate in the final CIC-MAN router.
"""

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


add_src_to_path()

from cicman.data.dataset import WindowIndexDataset  # noqa: E402
from cicman.evaluation.metrics import balanced_accuracy, confusion_matrix, macro_f1  # noqa: E402
from cicman.models.cic_man_vfinal import build_cic_man_vfinal  # noqa: E402
from cicman.training.cic_man import limit_dataset_stratified, rows_for_dataset  # noqa: E402
from cicman.training.raw_cnn import set_seed  # noqa: E402


CLASS_NAMES = {0: "normal", 1: "inner", 2: "outer", 3: "ball"}
DEFAULT_VIEWS = ["envelope", "stft", "wavelet", "order", "denoise", "filterbank"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-index", type=Path, required=True)
    parser.add_argument("--val-index", type=Path, required=True)
    parser.add_argument("--test-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--views", default=",".join(DEFAULT_VIEWS))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-items", type=int, default=6000)
    parser.add_argument("--max-val-items", type=int, default=6000)
    parser.add_argument("--max-test-items", type=int, default=0)
    parser.add_argument("--selection-metric", choices=["mean_source_macro_f1", "worst_source_macro_f1"], default="mean_source_macro_f1")
    return parser.parse_args()


def make_loader(index_csv: Path, batch_size: int, num_workers: int, max_items: int | None, *, stratified: bool, shuffle: bool):
    import torch

    dataset = WindowIndexDataset(index_csv)
    if stratified:
        dataset = limit_dataset_stratified(dataset, max_items, ("dataset_id", "label_id"))
    elif max_items is not None and max_items > 0 and max_items < len(dataset):
        dataset = torch.utils.data.Subset(dataset, list(range(max_items)))
    sampler = None
    if shuffle:
        rows = rows_for_dataset(dataset)
        counts = Counter((row["dataset_id"], row["label_id"]) for row in rows)
        weights = torch.tensor([1.0 / counts[(row["dataset_id"], row["label_id"])] for row in rows], dtype=torch.double)
        sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        shuffle = False
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def metadata_column(metadata: dict[str, object], key: str) -> list[str]:
    values = metadata[key]
    if isinstance(values, list):
        return [str(value) for value in values]
    return [str(value) for value in list(values)]


def per_class_payload(y_true: list[int], y_pred: list[int], num_classes: int) -> dict[str, object]:
    cm = confusion_matrix(y_true, y_pred, num_classes)
    payload: dict[str, object] = {
        "macro_f1": macro_f1(y_true, y_pred, num_classes),
        "balanced_accuracy": balanced_accuracy(y_true, y_pred, num_classes),
        "num_samples": len(y_true),
        "confusion_matrix": cm.tolist(),
    }
    for class_id in range(num_classes):
        denom = int(cm[class_id, :].sum())
        pred_support = int(cm[:, class_id].sum())
        recall = float(cm[class_id, class_id] / denom) if denom else 0.0
        name = CLASS_NAMES.get(class_id, f"class{class_id}")
        payload[f"{name}_recall"] = recall
        payload[f"{name}_predicted_support"] = pred_support
    return payload


def evaluate(model, loader, device: str, views: list[str], num_classes: int) -> dict[str, object]:
    import torch

    model.eval()
    true_by_view: dict[str, list[int]] = {view: [] for view in views}
    pred_by_view: dict[str, list[int]] = {view: [] for view in views}
    margin_by_view: dict[str, list[float]] = {view: [] for view in views}
    fidelity_by_view: dict[str, list[float]] = {view: [] for view in views}
    dataset_true: dict[str, dict[str, list[int]]] = {view: defaultdict(list) for view in views}
    dataset_pred: dict[str, dict[str, list[int]]] = {view: defaultdict(list) for view in views}

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            datasets = metadata_column(batch["metadata"], "dataset_id")
            for view in views:
                view_x = model.make_view(x, view)
                features = model.view_encoders[view](view_x)
                logits = model.view_agents[view](features)
                _, fidelity = model.mechanism_features(view_x)
                pred = logits.argmax(dim=1)
                if num_classes > 2:
                    inner_margin = logits[:, 1] - torch.maximum(logits[:, 0], logits[:, 2])
                else:
                    inner_margin = logits[:, 1] - logits[:, 0]
                true_values = y.detach().cpu().numpy().astype(int).tolist()
                pred_values = pred.detach().cpu().numpy().astype(int).tolist()
                true_by_view[view].extend(true_values)
                pred_by_view[view].extend(pred_values)
                margin_by_view[view].extend(inner_margin.detach().cpu().numpy().astype(float).tolist())
                fidelity_by_view[view].extend(fidelity.detach().cpu().numpy().astype(float).tolist())
                for domain_id, true_label, pred_label in zip(datasets, true_values, pred_values):
                    dataset_true[view][domain_id].append(true_label)
                    dataset_pred[view][domain_id].append(pred_label)

    result: dict[str, object] = {}
    for view in views:
        metrics = per_class_payload(true_by_view[view], pred_by_view[view], num_classes)
        metrics["inner_margin_mean"] = float(np.mean(margin_by_view[view])) if margin_by_view[view] else 0.0
        metrics["mechanism_fidelity_mean"] = float(np.mean(fidelity_by_view[view])) if fidelity_by_view[view] else 0.0
        by_dataset = {}
        for domain_id in sorted(dataset_true[view]):
            by_dataset[domain_id] = per_class_payload(dataset_true[view][domain_id], dataset_pred[view][domain_id], num_classes)
        metrics["by_dataset"] = by_dataset
        if by_dataset:
            metrics["mean_dataset_macro_f1"] = float(np.mean([item["macro_f1"] for item in by_dataset.values()]))
            metrics["worst_dataset_macro_f1"] = float(np.min([item["macro_f1"] for item in by_dataset.values()]))
        else:
            metrics["mean_dataset_macro_f1"] = metrics["macro_f1"]
            metrics["worst_dataset_macro_f1"] = metrics["macro_f1"]
        result[view] = metrics
    return result


def write_summary(summary_rows: list[dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "view_agent_summary.csv"
    if summary_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
    lines = [
        "# vFinal View-Agent Pretraining Summary",
        "",
        "| View | Source Mean F1 | Source Worst F1 | Source Inner Recall | Source Outer Recall | Target F1 | Target Normal Recall | Target Inner Recall | Target Outer Recall | Fidelity | Recommendation |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['view']} | {float(row['source_mean_macro_f1']):.6f} | "
            f"{float(row['source_worst_macro_f1']):.6f} | {float(row['source_inner_recall']):.6f} | "
            f"{float(row['source_outer_recall']):.6f} | {float(row['target_macro_f1']):.6f} | "
            f"{float(row['target_normal_recall']):.6f} | {float(row['target_inner_recall']):.6f} | "
            f"{float(row['target_outer_recall']):.6f} | {float(row['source_mechanism_fidelity_mean']):.6f} | "
            f"{row['recommendation']} |"
        )
    lines.extend(
        [
            "",
            "Recommendation is source-first. Target columns are post-hoc diagnostics only and must not be used for source selection.",
        ]
    )
    (output_dir / "view_agent_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    import torch
    import torch.nn.functional as F

    args = parse_args()
    set_seed(args.seed)
    views = [item.strip() for item in args.views.split(",") if item.strip()]
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_loader = make_loader(
        args.train_index,
        args.batch_size,
        args.num_workers,
        args.max_train_items,
        stratified=True,
        shuffle=True,
    )
    val_loader = make_loader(
        args.val_index,
        args.batch_size,
        args.num_workers,
        args.max_val_items,
        stratified=True,
        shuffle=False,
    )
    test_loader = make_loader(
        args.test_index,
        args.batch_size,
        args.num_workers,
        args.max_test_items,
        stratified=False,
        shuffle=False,
    )

    model = build_cic_man_vfinal(num_classes=args.num_classes, view_names=views, use_health_style_split=False).to(device)
    trainable = list(model.view_encoders.parameters()) + list(model.view_agents.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)

    best_score = -1.0
    best_path = args.output_dir / "best.pt"
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            view_losses = []
            for view in views:
                view_x = model.make_view(x, view)
                features = model.view_encoders[view](view_x)
                logits = model.view_agents[view](features)
                view_losses.append(F.cross_entropy(logits, y))
            loss = sum(view_losses) / len(view_losses)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))

        val = evaluate(model, val_loader, device, views, args.num_classes)
        source_scores = [float(val[view]["mean_dataset_macro_f1"]) for view in views]
        worst_scores = [float(val[view]["worst_dataset_macro_f1"]) for view in views]
        selection_score = float(np.mean(source_scores)) if args.selection_metric == "mean_source_macro_f1" else float(np.min(worst_scores))
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else 0.0,
            "selection_score": selection_score,
            "source_val": val,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))
        if selection_score > best_score:
            best_score = selection_score
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "num_classes": args.num_classes,
                    "views": views,
                    "config": vars(args),
                    "selection_score": selection_score,
                },
                best_path,
            )

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    source_val = evaluate(model, val_loader, device, views, args.num_classes)
    target_test = evaluate(model, test_loader, device, views, args.num_classes)
    rows = []
    for view in views:
        source = source_val[view]
        target = target_test[view]
        recommendation = "reject"
        if (
            float(source["mean_dataset_macro_f1"]) >= 0.60
            and float(source.get("inner_recall", 0.0)) >= 0.50
            and float(source.get("outer_recall", 0.0)) >= 0.50
        ):
            recommendation = "candidate"
        rows.append(
            {
                "view": view,
                "source_macro_f1": source["macro_f1"],
                "source_mean_macro_f1": source["mean_dataset_macro_f1"],
                "source_worst_macro_f1": source["worst_dataset_macro_f1"],
                "source_normal_recall": source.get("normal_recall", 0.0),
                "source_inner_recall": source.get("inner_recall", 0.0),
                "source_outer_recall": source.get("outer_recall", 0.0),
                "source_inner_margin_mean": source["inner_margin_mean"],
                "source_mechanism_fidelity_mean": source["mechanism_fidelity_mean"],
                "target_macro_f1": target["macro_f1"],
                "target_normal_recall": target.get("normal_recall", 0.0),
                "target_inner_recall": target.get("inner_recall", 0.0),
                "target_outer_recall": target.get("outer_recall", 0.0),
                "target_inner_margin_mean": target["inner_margin_mean"],
                "target_mechanism_fidelity_mean": target["mechanism_fidelity_mean"],
                "recommendation": recommendation,
            }
        )
    rows.sort(key=lambda row: (row["recommendation"] != "candidate", -float(row["source_mean_macro_f1"])))
    write_summary(rows, args.output_dir)
    payload = {
        "checkpoint": str(best_path),
        "best_epoch": checkpoint["epoch"],
        "history": history,
        "source_val": source_val,
        "target_test": target_test,
        "summary": rows,
    }
    (args.output_dir / "view_agent_results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"checkpoint": str(best_path), "summary": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
