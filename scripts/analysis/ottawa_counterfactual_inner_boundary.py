#!/usr/bin/env python3
"""Source-domain counterfactual inner-boundary audit for Ottawa.

The boundary is selected with source validation only. Ottawa target labels are
used only for post-hoc adversarial auditing.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


CLASS_NAMES = {0: "normal", 1: "inner", 2: "outer"}


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
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=384)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-style-items", type=int, default=12000)
    parser.add_argument("--source-drop-tolerance", type=float, default=0.01)
    parser.add_argument("--source-inner-drop-tolerance", type=float, default=0.02)
    return parser.parse_args()


def metadata_column(metadata: dict[str, object], key: str) -> list[str]:
    values = metadata[key]
    if isinstance(values, list):
        return [str(value) for value in values]
    return [str(value) for value in list(values)]


def make_loader(index_csv: Path, batch_size: int, num_workers: int, indices: list[int] | None = None):
    import torch

    dataset = WindowIndexDataset(index_csv)
    if indices is not None:
        dataset = torch.utils.data.Subset(dataset, indices)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def load_model(checkpoint_path: Path, device: str):
    import torch

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
    return model, num_classes


def sample_style_indices(index_csv: Path, max_items: int) -> list[int] | None:
    if max_items <= 0:
        return None
    with index_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) <= max_items:
        return None
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        groups[(row["dataset_id"], row["label_id"])].append(idx)
    rng = np.random.default_rng(20260702)
    per_group = max(1, max_items // max(1, len(groups)))
    selected: list[int] = []
    leftovers: list[int] = []
    for values in groups.values():
        shuffled = np.asarray(values)
        rng.shuffle(shuffled)
        selected.extend(shuffled[:per_group].tolist())
        leftovers.extend(shuffled[per_group:].tolist())
    if len(selected) < max_items:
        shuffled = np.asarray(leftovers)
        rng.shuffle(shuffled)
        selected.extend(shuffled[: max_items - len(selected)].tolist())
    if len(selected) > max_items:
        selected = rng.choice(np.asarray(selected), size=max_items, replace=False).tolist()
    return sorted(int(item) for item in selected)


def estimate_source_style_stats(index_csv: Path, batch_size: int, num_workers: int, max_items: int):
    import torch

    loader = make_loader(index_csv, batch_size, num_workers, sample_style_indices(index_csv, max_items))
    accum: dict[str, dict[str, float]] = defaultdict(lambda: {"sum_mean": 0.0, "sum_std": 0.0, "count": 0.0})
    with torch.no_grad():
        for batch in loader:
            x = batch["x"]
            means = x.mean(dim=-1).squeeze(1).detach().cpu().numpy()
            stds = x.std(dim=-1, unbiased=False).squeeze(1).detach().cpu().numpy()
            dataset_ids = metadata_column(batch["metadata"], "dataset_id")
            for dataset_id, mean, std in zip(dataset_ids, means, stds):
                item = accum[dataset_id]
                item["sum_mean"] += float(mean)
                item["sum_std"] += float(std)
                item["count"] += 1.0
    stats = {}
    for dataset_id, item in sorted(accum.items()):
        count = max(1.0, item["count"])
        stats[dataset_id] = {
            "mean": item["sum_mean"] / count,
            "std": max(1e-4, item["sum_std"] / count),
            "count": int(item["count"]),
        }
    return stats


def smooth(x, kernel_size: int):
    import torch.nn.functional as F

    return F.avg_pool1d(x, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)


def restyle_to_source_domain(x, target_mean: float, target_std: float):
    centered = x - x.mean(dim=-1, keepdim=True)
    std = centered.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
    return centered / std * float(target_std) + float(target_mean)


def counterfactual_variants(x, style_stats: dict[str, dict[str, float]]):
    variants = {"identity": x}
    for dataset_id, stats in style_stats.items():
        variants[f"style_{dataset_id}"] = restyle_to_source_domain(x, stats["mean"], stats["std"])
    smooth_short = smooth(x, 9)
    smooth_mid = smooth(x, 33)
    smooth_long = smooth(x, 129)
    high = x - smooth_short
    mid = smooth_short - smooth_mid
    low = smooth_mid - smooth_long
    variants["denoise"] = x - smooth_mid
    variants["order"] = x - smooth_long
    variants["filterbank"] = (high.pow(2) + 0.5 * mid.pow(2) + 0.25 * low.pow(2) + 1e-6).sqrt()
    return variants


def model_logits(model, x):
    output = model(x)
    return output["logits"] if isinstance(output, dict) else output


def collect_counterfactual_evidence(
    model,
    index_csv: Path,
    style_stats: dict[str, dict[str, float]],
    batch_size: int,
    device: str,
    num_workers: int,
):
    import torch

    loader = make_loader(index_csv, batch_size, num_workers)
    rows = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            labels = batch["y"].detach().cpu().numpy().astype(int)
            metadata = batch["metadata"]
            recording_ids = metadata_column(metadata, "recording_id")
            dataset_ids = metadata_column(metadata, "dataset_id")
            base_logits = model_logits(model, x).detach().cpu().numpy().astype(np.float64)
            variant_margins = []
            variant_inner_top1 = []
            variant_names = []
            for variant_name, variant_x in counterfactual_variants(x, style_stats).items():
                logits = model_logits(model, variant_x).detach().cpu().numpy().astype(np.float64)
                margin = logits[:, 1] - np.maximum(logits[:, 0], logits[:, 2])
                variant_margins.append(margin)
                variant_inner_top1.append((logits.argmax(axis=1) == 1).astype(np.float64))
                variant_names.append(variant_name)
            margins = np.stack(variant_margins, axis=1)
            inner_top1 = np.stack(variant_inner_top1, axis=1)
            base_pred = base_logits.argmax(axis=1).astype(int)
            base_margin = base_logits[:, 1] - np.maximum(base_logits[:, 0], base_logits[:, 2])
            for i in range(len(labels)):
                rows.append(
                    {
                        "recording_id": recording_ids[i],
                        "dataset_id": dataset_ids[i],
                        "label_id": int(labels[i]),
                        "base_pred": int(base_pred[i]),
                        "base_inner_margin": float(base_margin[i]),
                        "cf_max_inner_margin": float(margins[i].max()),
                        "cf_mean_inner_margin": float(margins[i].mean()),
                        "cf_inner_top1_fraction": float(inner_top1[i].mean()),
                        "cf_argmax_view": variant_names[int(margins[i].argmax())],
                    }
                )
    return rows


def per_class(matrix: np.ndarray, class_id: int) -> dict[str, float]:
    tp = float(matrix[class_id, class_id])
    fn = float(matrix[class_id, :].sum() - tp)
    fp = float(matrix[:, class_id].sum() - tp)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def metric_payload(y_true: list[int], y_pred: list[int], num_classes: int) -> dict[str, object]:
    matrix = confusion_matrix(y_true, y_pred, num_classes)
    out: dict[str, object] = {
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred, num_classes),
        "balanced_accuracy": balanced_accuracy(y_true, y_pred, num_classes),
        "num_recordings": len(y_true),
        "confusion_matrix": matrix.tolist(),
    }
    for class_id, class_name in CLASS_NAMES.items():
        stats = per_class(matrix, class_id)
        out[f"{class_name}_precision"] = stats["precision"]
        out[f"{class_name}_recall"] = stats["recall"]
        out[f"{class_name}_f1"] = stats["f1"]
        out[f"{class_name}_predicted_support"] = int(matrix[:, class_id].sum())
    return out


def aggregate_recordings(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["recording_id"])].append(row)
    out = []
    for recording_id, group in sorted(grouped.items()):
        label = int(group[0]["label_id"])
        base_votes = Counter(int(row["base_pred"]) for row in group)
        max_margins = np.asarray([float(row["cf_max_inner_margin"]) for row in group], dtype=np.float64)
        mean_margins = np.asarray([float(row["cf_mean_inner_margin"]) for row in group], dtype=np.float64)
        top1_fracs = np.asarray([float(row["cf_inner_top1_fraction"]) for row in group], dtype=np.float64)
        argmax_views = Counter(str(row["cf_argmax_view"]) for row in group)
        out.append(
            {
                "recording_id": recording_id,
                "dataset_id": str(group[0]["dataset_id"]),
                "label_id": label,
                "label": CLASS_NAMES[label],
                "num_windows": len(group),
                "base_vote_pred": int(base_votes.most_common(1)[0][0]),
                "cf_margin_q90": float(np.quantile(max_margins, 0.90)),
                "cf_margin_q95": float(np.quantile(max_margins, 0.95)),
                "cf_margin_max": float(max_margins.max()),
                "cf_mean_margin_q90": float(np.quantile(mean_margins, 0.90)),
                "cf_top1_fraction_mean": float(top1_fracs.mean()),
                "cf_top1_fraction_q90": float(np.quantile(top1_fracs, 0.90)),
                "dominant_cf_view": argmax_views.most_common(1)[0][0],
                "cf_view_counts": json.dumps(dict(sorted(argmax_views.items())), sort_keys=True),
            }
        )
    return out


def predict_recording(row: dict[str, object], rule: str, threshold: float | None) -> int:
    if rule == "base_vote":
        return int(row["base_vote_pred"])
    assert threshold is not None
    if float(row[rule]) >= threshold:
        return 1
    return int(row["base_vote_pred"])


def evaluate_records(records: list[dict[str, object]], rule: str, threshold: float | None, num_classes: int):
    y_true = [int(row["label_id"]) for row in records]
    y_pred = [predict_recording(row, rule, threshold) for row in records]
    return metric_payload(y_true, y_pred, num_classes)


def candidate_thresholds(records: list[dict[str, object]], key: str) -> list[float]:
    values = np.asarray([float(row[key]) for row in records], dtype=np.float64)
    quantiles = np.quantile(values, np.linspace(0.05, 0.95, 37)).tolist()
    return sorted(set(float(value) for value in quantiles))


def build_candidates(source_records: list[dict[str, object]], target_records: list[dict[str, object]], num_classes: int):
    candidates = [("base_vote", None)]
    for key in [
        "cf_margin_q90",
        "cf_margin_q95",
        "cf_margin_max",
        "cf_mean_margin_q90",
        "cf_top1_fraction_mean",
        "cf_top1_fraction_q90",
    ]:
        for threshold in candidate_thresholds(source_records, key):
            candidates.append((key, threshold))
    rows = []
    for rule, threshold in candidates:
        source = evaluate_records(source_records, rule, threshold, num_classes)
        target = evaluate_records(target_records, rule, threshold, num_classes)
        rows.append(
            {
                "rule": rule,
                "threshold": "" if threshold is None else float(threshold),
                "source_macro_f1": source["macro_f1"],
                "source_inner_recall": source["inner_recall"],
                "source_inner_precision": source["inner_precision"],
                "source_inner_predicted_support": source["inner_predicted_support"],
                "target_macro_f1_posthoc": target["macro_f1"],
                "target_inner_recall_posthoc": target["inner_recall"],
                "target_inner_precision_posthoc": target["inner_precision"],
                "target_inner_predicted_support_posthoc": target["inner_predicted_support"],
                "target_confusion_matrix_posthoc": json.dumps(target["confusion_matrix"]),
            }
        )
    return rows


def select_candidate(rows: list[dict[str, object]], macro_tol: float, inner_tol: float) -> dict[str, object]:
    baseline = next(row for row in rows if row["rule"] == "base_vote")
    base_macro = float(baseline["source_macro_f1"])
    base_inner = float(baseline["source_inner_recall"])
    allowed = [
        row
        for row in rows
        if float(row["source_macro_f1"]) >= base_macro - macro_tol
        and float(row["source_inner_recall"]) >= base_inner - inner_tol
    ]
    selected = max(
        allowed or [baseline],
        key=lambda row: (
            float(row["source_inner_predicted_support"]),
            float(row["source_inner_recall"]),
            float(row["source_macro_f1"]),
        ),
    )
    selected["selection_reason"] = (
        f"source-only: source Macro-F1 >= {base_macro - macro_tol:.6f}, "
        f"source inner recall >= {base_inner - inner_tol:.6f}; maximize source inner support"
    )
    return selected


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


def class_evidence_summary(records: list[dict[str, object]], split: str) -> list[dict[str, object]]:
    out = []
    for class_id, class_name in CLASS_NAMES.items():
        group = [row for row in records if int(row["label_id"]) == class_id]
        if not group:
            continue
        out.append(
            {
                "split": split,
                "class": class_name,
                "recordings": len(group),
                "cf_margin_q90_mean": float(np.mean([float(row["cf_margin_q90"]) for row in group])),
                "cf_margin_q90_max": float(np.max([float(row["cf_margin_q90"]) for row in group])),
                "cf_top1_fraction_mean": float(np.mean([float(row["cf_top1_fraction_mean"]) for row in group])),
                "dominant_views": json.dumps(Counter(str(row["dominant_cf_view"]) for row in group), sort_keys=True),
            }
        )
    return out


def write_markdown(
    style_stats: dict[str, dict[str, float]],
    evidence_rows: list[dict[str, object]],
    candidates: list[dict[str, object]],
    selected: dict[str, object],
    path: Path,
) -> None:
    top_posthoc = sorted(candidates, key=lambda row: float(row["target_inner_recall_posthoc"]), reverse=True)[:5]
    lines = [
        "# Ottawa Counterfactual Inner Boundary Audit",
        "",
        "Boundary selection is source-validation only. Ottawa target labels are post-hoc audit evidence.",
        "",
        "## Source Style Bank",
        "",
        "| Source Domain | Windows | Mean | Std |",
        "|---|---:|---:|---:|",
    ]
    for dataset_id, stats in style_stats.items():
        lines.append(f"| {dataset_id} | {stats['count']} | {stats['mean']:.6f} | {stats['std']:.6f} |")
    lines.extend(
        [
            "",
            "## Evidence Summary",
            "",
            "| Split | Class | Recordings | CF Q90 Margin Mean | CF Q90 Margin Max | CF Inner Top1 Mean | Dominant Views |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in evidence_rows:
        lines.append(
            f"| {row['split']} | {row['class']} | {row['recordings']} | "
            f"{float(row['cf_margin_q90_mean']):.6f} | {float(row['cf_margin_q90_max']):.6f} | "
            f"{float(row['cf_top1_fraction_mean']):.6f} | `{row['dominant_views']}` |"
        )
    lines.extend(
        [
            "",
            "## Source-Selected Boundary",
            "",
            f"Selected rule: `{selected['rule']}` threshold `{selected['threshold']}`.",
            "",
            f"Selection reason: {selected['selection_reason']}",
            "",
            "| Rule | Threshold | Source Macro-F1 | Source Inner Recall | Target Macro-F1 | Target Inner Recall | Target Inner Pred Support |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    interesting = [next(row for row in candidates if row["rule"] == "base_vote"), selected]
    for row in top_posthoc:
        if row not in interesting:
            interesting.append(row)
    seen = set()
    for row in interesting:
        key = (row["rule"], str(row["threshold"]))
        if key in seen:
            continue
        seen.add(key)
        lines.append(
            f"| {row['rule']} | {row['threshold']} | {float(row['source_macro_f1']):.6f} | "
            f"{float(row['source_inner_recall']):.6f} | {float(row['target_macro_f1_posthoc']):.6f} | "
            f"{float(row['target_inner_recall_posthoc']):.6f} | {int(row['target_inner_predicted_support_posthoc'])} |"
        )
    lines.extend(
        [
            "",
            "## Adversarial Conclusion",
            "",
            "- This is a target-free boundary construction audit, not a final trained CIC-MAN variant.",
            "- If the source-selected counterfactual boundary improves only marginally, Ottawa needs a stronger inner mechanism prior or explicit source support synthesis.",
            "- If the best post-hoc target boundary is much better than the selected boundary, that gap is not publishable as a target-free method without a source-only selector.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    import torch

    project_root = args.project_root.expanduser().resolve()
    output_dir = (args.output_dir or project_root  / "outputs" / "tables").resolve()
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    base = project_root / "data" / "paper1_cicman" / "cache" / "windows" / "cross_dataset_task3_source_mixed" / "target_dataset_ottawa"
    checkpoint = (
        project_root
        
        / "outputs"
        / "checkpoints"
        / "cic_man_gated_filterbank_calibrated_source_mixed_cross_dataset_task3_target_ottawa"
        / "best.pt"
    )
    model, num_classes = load_model(checkpoint, device)
    style_stats = estimate_source_style_stats(base / "train_windows.csv", args.batch_size, args.num_workers, args.max_style_items)
    source_window_rows = collect_counterfactual_evidence(
        model, base / "val_windows.csv", style_stats, args.batch_size, device, args.num_workers
    )
    target_window_rows = collect_counterfactual_evidence(
        model, base / "test_windows.csv", style_stats, args.batch_size, device, args.num_workers
    )
    source_records = aggregate_recordings(source_window_rows)
    target_records = aggregate_recordings(target_window_rows)
    candidates = build_candidates(source_records, target_records, num_classes)
    selected = select_candidate(candidates, args.source_drop_tolerance, args.source_inner_drop_tolerance)
    evidence_rows = class_evidence_summary(source_records, "source_val") + class_evidence_summary(
        target_records, "ottawa_test"
    )
    write_csv(source_records + target_records, output_dir / "ottawa_counterfactual_recording_evidence.csv")
    write_csv(candidates, output_dir / "ottawa_counterfactual_boundary_candidates.csv")
    write_csv(evidence_rows, output_dir / "ottawa_counterfactual_evidence_summary.csv")
    (output_dir / "ottawa_counterfactual_boundary_selected.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown(
        style_stats,
        evidence_rows,
        candidates,
        selected,
        output_dir / "ottawa_counterfactual_inner_boundary.md",
    )
    print(f"Wrote Ottawa counterfactual inner-boundary audit to {output_dir}")


if __name__ == "__main__":
    main()
