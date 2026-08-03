#!/usr/bin/env python3
"""Ottawa inner-class evidence and recording-aggregation audit.

All aggregation/boundary selections are made on source validation only. Target
labels are used only for post-hoc failure analysis.
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
from cicman.models.cic_man_vfinal import build_cic_man_vfinal  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--source-drop-tolerance", type=float, default=0.005)
    parser.add_argument("--source-inner-drop-tolerance", type=float, default=0.02)
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
    return model, num_classes


def collect_logits(model, index_csv: Path, batch_size: int, device: str, num_workers: int) -> dict[str, object]:
    import torch

    loader = make_loader(index_csv, batch_size, num_workers)
    logits_all = []
    labels_all = []
    recording_ids_all = []
    metadata_rows = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            output = model(x)
            logits = output["logits"] if isinstance(output, dict) else output
            logits_all.append(logits.detach().cpu().numpy().astype(np.float64))
            labels = batch["y"].detach().cpu().numpy().astype(int)
            labels_all.append(labels)
            metadata = batch["metadata"]
            recording_ids = metadata_column(metadata, "recording_id")
            dataset_ids = metadata_column(metadata, "dataset_id")
            labels_text = metadata_column(metadata, "label")
            for i, recording_id in enumerate(recording_ids):
                recording_ids_all.append(recording_id)
                metadata_rows.append(
                    {
                        "recording_id": recording_id,
                        "dataset_id": dataset_ids[i],
                        "label": labels_text[i],
                        "label_id": int(labels[i]),
                    }
                )
    return {
        "logits": np.concatenate(logits_all, axis=0),
        "labels": np.concatenate(labels_all, axis=0).astype(int),
        "recording_ids": recording_ids_all,
        "metadata_rows": metadata_rows,
    }


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
    payload: dict[str, object] = {
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred, num_classes),
        "balanced_accuracy": balanced_accuracy(y_true, y_pred, num_classes),
        "num_samples": len(y_true),
        "confusion_matrix": matrix.tolist(),
    }
    for class_id, class_name in CLASS_NAMES.items():
        stats = per_class(matrix, class_id)
        payload[f"{class_name}_precision"] = stats["precision"]
        payload[f"{class_name}_recall"] = stats["recall"]
        payload[f"{class_name}_f1"] = stats["f1"]
        payload[f"{class_name}_predicted_support"] = int(matrix[:, class_id].sum())
    return payload


def grouped(payload: dict[str, object]) -> dict[str, dict[str, object]]:
    logits = np.asarray(payload["logits"], dtype=np.float64)
    labels = np.asarray(payload["labels"], dtype=int)
    recording_ids = list(payload["recording_ids"])
    groups: dict[str, dict[str, object]] = {}
    for i, recording_id in enumerate(recording_ids):
        if recording_id not in groups:
            groups[recording_id] = {"label": int(labels[i]), "logits": [], "indices": []}
        groups[recording_id]["logits"].append(logits[i])
        groups[recording_id]["indices"].append(i)
    for item in groups.values():
        item["logits"] = np.stack(item["logits"], axis=0)
    return groups


def window_evidence(payload: dict[str, object], split_name: str) -> list[dict[str, object]]:
    logits = np.asarray(payload["logits"], dtype=np.float64)
    labels = np.asarray(payload["labels"], dtype=int)
    pred = logits.argmax(axis=1)
    rank = (-logits).argsort(axis=1)
    inner_rank = np.where(rank == 1)[1] + 1
    inner_margin = logits[:, 1] - np.maximum(logits[:, 0], logits[:, 2])
    inner_outer_margin = logits[:, 1] - logits[:, 2]
    inner_normal_margin = logits[:, 1] - logits[:, 0]
    rows = []
    for class_id, class_name in CLASS_NAMES.items():
        mask = labels == class_id
        if not mask.any():
            continue
        rows.append(
            {
                "split": split_name,
                "true_class": class_name,
                "num_windows": int(mask.sum()),
                "top1_inner_fraction": float((pred[mask] == 1).mean()),
                "top2_inner_fraction": float((inner_rank[mask] <= 2).mean()),
                "inner_rank_mean": float(inner_rank[mask].mean()),
                "inner_margin_mean": float(inner_margin[mask].mean()),
                "inner_margin_q10": float(np.quantile(inner_margin[mask], 0.10)),
                "inner_margin_q50": float(np.quantile(inner_margin[mask], 0.50)),
                "inner_margin_q90": float(np.quantile(inner_margin[mask], 0.90)),
                "inner_normal_margin_mean": float(inner_normal_margin[mask].mean()),
                "inner_outer_margin_mean": float(inner_outer_margin[mask].mean()),
            }
        )
    return rows


def record_features(logits: np.ndarray) -> dict[str, object]:
    pred = logits.argmax(axis=1)
    rank = (-logits).argsort(axis=1)
    inner_rank = np.where(rank == 1)[1] + 1
    inner_margin = logits[:, 1] - np.maximum(logits[:, 0], logits[:, 2])
    topk_count = max(1, int(math.ceil(len(logits) * 0.10)))
    topk_scores = np.sort(logits, axis=0)[-topk_count:, :].mean(axis=0)
    return {
        "num_windows": int(len(logits)),
        "mean_logits": logits.mean(axis=0),
        "median_logits": np.median(logits, axis=0),
        "max_logits": logits.max(axis=0),
        "q90_logits": np.quantile(logits, 0.90, axis=0),
        "top10_mean_logits": topk_scores,
        "vote_counts": Counter(pred.tolist()),
        "top1_inner_fraction": float((pred == 1).mean()),
        "top2_inner_fraction": float((inner_rank <= 2).mean()),
        "inner_margin_mean": float(inner_margin.mean()),
        "inner_margin_q90": float(np.quantile(inner_margin, 0.90)),
        "inner_margin_max": float(inner_margin.max()),
    }


def logsumexp_scores(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = logits / temperature
    peak = scaled.max(axis=0)
    return peak + np.log(np.exp(scaled - peak).mean(axis=0))


def predict_recording(features: dict[str, object], rule: str, threshold: float | None = None) -> int:
    if rule == "mean_logits":
        return int(np.asarray(features["mean_logits"]).argmax())
    if rule == "median_logits":
        return int(np.asarray(features["median_logits"]).argmax())
    if rule == "max_logits":
        return int(np.asarray(features["max_logits"]).argmax())
    if rule == "q90_logits":
        return int(np.asarray(features["q90_logits"]).argmax())
    if rule == "top10_mean_logits":
        return int(np.asarray(features["top10_mean_logits"]).argmax())
    if rule == "logsumexp":
        return int(np.asarray(features["logsumexp_logits"]).argmax())
    if rule == "majority_vote":
        return int(features["vote_counts"].most_common(1)[0][0])
    if rule == "inner_q90_boundary":
        assert threshold is not None
        if float(features["inner_margin_q90"]) >= threshold:
            return 1
        return int(np.asarray(features["mean_logits"]).argmax())
    if rule == "inner_top2_boundary":
        assert threshold is not None
        if float(features["top2_inner_fraction"]) >= threshold:
            return 1
        return int(np.asarray(features["mean_logits"]).argmax())
    raise ValueError(f"unknown rule: {rule}")


def build_record_rows(payload: dict[str, object], split_name: str) -> list[dict[str, object]]:
    rows = []
    for recording_id, item in sorted(grouped(payload).items()):
        logits = np.asarray(item["logits"], dtype=np.float64)
        features = record_features(logits)
        features["logsumexp_logits"] = logsumexp_scores(logits)
        mean_logits = np.asarray(features["mean_logits"])
        q90_logits = np.asarray(features["q90_logits"])
        rows.append(
            {
                "split": split_name,
                "recording_id": recording_id,
                "true_label_id": int(item["label"]),
                "true_label": CLASS_NAMES[int(item["label"])],
                "num_windows": features["num_windows"],
                "mean_pred": predict_recording(features, "mean_logits"),
                "vote_pred": predict_recording(features, "majority_vote"),
                "q90_pred": predict_recording(features, "q90_logits"),
                "max_pred": predict_recording(features, "max_logits"),
                "mean_inner_logit": float(mean_logits[1]),
                "mean_normal_logit": float(mean_logits[0]),
                "mean_outer_logit": float(mean_logits[2]),
                "q90_inner_logit": float(q90_logits[1]),
                "q90_normal_logit": float(q90_logits[0]),
                "q90_outer_logit": float(q90_logits[2]),
                "top1_inner_fraction": features["top1_inner_fraction"],
                "top2_inner_fraction": features["top2_inner_fraction"],
                "inner_margin_mean": features["inner_margin_mean"],
                "inner_margin_q90": features["inner_margin_q90"],
                "inner_margin_max": features["inner_margin_max"],
                "vote_counts": json.dumps(dict(sorted(features["vote_counts"].items())), sort_keys=True),
            }
        )
    return rows


def evaluate_rule(payload: dict[str, object], rule: str, threshold: float | None, num_classes: int) -> dict[str, object]:
    y_true = []
    y_pred = []
    for item in grouped(payload).values():
        logits = np.asarray(item["logits"], dtype=np.float64)
        features = record_features(logits)
        features["logsumexp_logits"] = logsumexp_scores(logits)
        y_true.append(int(item["label"]))
        y_pred.append(predict_recording(features, rule, threshold))
    return metric_payload(y_true, y_pred, num_classes)


def candidate_rows(source_payload: dict[str, object], target_payload: dict[str, object], num_classes: int) -> list[dict[str, object]]:
    candidates: list[tuple[str, float | None]] = [
        ("mean_logits", None),
        ("majority_vote", None),
        ("median_logits", None),
        ("q90_logits", None),
        ("max_logits", None),
        ("top10_mean_logits", None),
        ("logsumexp", None),
    ]
    for threshold in np.arange(-4.0, 4.0001, 0.25):
        candidates.append(("inner_q90_boundary", float(threshold)))
    for threshold in np.arange(0.05, 1.0001, 0.05):
        candidates.append(("inner_top2_boundary", float(threshold)))
    rows = []
    for rule, threshold in candidates:
        source = evaluate_rule(source_payload, rule, threshold, num_classes)
        target = evaluate_rule(target_payload, rule, threshold, num_classes)
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
    baseline = next(row for row in rows if row["rule"] == "mean_logits" and row["threshold"] == "")
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
        f"source inner recall >= {base_inner - inner_tol:.6f}; ranked by source inner support/recall, then Macro-F1"
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


def write_markdown(
    *,
    window_rows: list[dict[str, object]],
    candidate_rows_: list[dict[str, object]],
    selected: dict[str, object],
    target_record_rows: list[dict[str, object]],
    path: Path,
) -> None:
    source_inner = next(row for row in window_rows if row["split"] == "source_val" and row["true_class"] == "inner")
    target_inner = next(row for row in window_rows if row["split"] == "ottawa_test" and row["true_class"] == "inner")
    target_inner_records = [row for row in target_record_rows if row["true_label"] == "inner"]
    mean_inner_margin = (
        sum(float(row["inner_margin_mean"]) for row in target_inner_records) / len(target_inner_records)
        if target_inner_records
        else float("nan")
    )
    max_inner_fraction = max((float(row["top1_inner_fraction"]) for row in target_inner_records), default=0.0)
    lines = [
        "# Ottawa Inner Evidence Audit",
        "",
        "Scope: source validation selects aggregation/boundary rules; Ottawa target labels are post-hoc audit only.",
        "",
        "## Window Evidence",
        "",
        "| Split | True Class | Windows | Top-1 Inner | Top-2 Inner | Inner Margin Mean | Margin Q90 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in window_rows:
        lines.append(
            f"| {row['split']} | {row['true_class']} | {int(row['num_windows'])} | "
            f"{float(row['top1_inner_fraction']):.6f} | {float(row['top2_inner_fraction']):.6f} | "
            f"{float(row['inner_margin_mean']):.6f} | {float(row['inner_margin_q90']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Source-Selected Recording Rule",
            "",
            f"Selected rule: `{selected['rule']}` threshold `{selected['threshold']}`.",
            "",
            f"Selection reason: {selected['selection_reason']}",
            "",
            "| Rule | Threshold | Source Macro-F1 | Source Inner Recall | Target Macro-F1 | Target Inner Recall | Target Inner Pred Support |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    interesting = [row for row in candidate_rows_ if row["rule"] in {"mean_logits", "majority_vote", "q90_logits", "max_logits", "top10_mean_logits", "logsumexp"}]
    if selected not in interesting:
        interesting.append(selected)
    for row in interesting:
        lines.append(
            f"| {row['rule']} | {row['threshold']} | {float(row['source_macro_f1']):.6f} | "
            f"{float(row['source_inner_recall']):.6f} | {float(row['target_macro_f1_posthoc']):.6f} | "
            f"{float(row['target_inner_recall_posthoc']):.6f} | {int(row['target_inner_predicted_support_posthoc'])} |"
        )
    lines.extend(
        [
            "",
            "## Adversarial Findings",
            "",
            f"- Source inner windows are easy: top-1 inner `{float(source_inner['top1_inner_fraction']):.6f}`, top-2 inner `{float(source_inner['top2_inner_fraction']):.6f}`.",
            f"- Ottawa target inner windows are weakly represented: top-1 inner `{float(target_inner['top1_inner_fraction']):.6f}`, top-2 inner `{float(target_inner['top2_inner_fraction']):.6f}`.",
            f"- Across target inner recordings, mean inner-vs-other margin is `{mean_inner_margin:.6f}` and best top-1 inner window fraction is `{max_inner_fraction:.6f}`.",
            "- If no source-selected aggregation recovers target inner recordings, the failure is not just recording-level voting; the learned inner boundary lacks target evidence.",
            "- This supports an Ottawa-specific next step: source-domain counterfactual inner boundary construction. It does not justify target-tuned thresholds.",
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
    source_payload = collect_logits(model, base / "val_windows.csv", args.batch_size, device, args.num_workers)
    target_payload = collect_logits(model, base / "test_windows.csv", args.batch_size, device, args.num_workers)
    window_rows = window_evidence(source_payload, "source_val") + window_evidence(target_payload, "ottawa_test")
    source_record_rows = build_record_rows(source_payload, "source_val")
    target_record_rows = build_record_rows(target_payload, "ottawa_test")
    candidates = candidate_rows(source_payload, target_payload, num_classes)
    selected = select_candidate(candidates, args.source_drop_tolerance, args.source_inner_drop_tolerance)
    write_csv(window_rows, output_dir / "ottawa_inner_window_evidence.csv")
    write_csv(source_record_rows + target_record_rows, output_dir / "ottawa_inner_recording_evidence.csv")
    write_csv(candidates, output_dir / "ottawa_recording_aggregation_candidates.csv")
    (output_dir / "ottawa_recording_aggregation_selected.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown(
        window_rows=window_rows,
        candidate_rows_=candidates,
        selected=selected,
        target_record_rows=target_record_rows,
        path=output_dir / "ottawa_inner_evidence_audit.md",
    )
    print(f"Wrote Ottawa inner evidence audit to {output_dir}")


if __name__ == "__main__":
    main()
