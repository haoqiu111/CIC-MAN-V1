#!/usr/bin/env python3
"""Source-only Ottawa inner-class decision calibration audit."""

from __future__ import annotations

import argparse
import csv
import json
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
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--source-drop-tolerance", type=float, default=0.005)
    parser.add_argument("--inner-drop-tolerance", type=float, default=0.02)
    parser.add_argument("--bias-max", type=float, default=5.0)
    parser.add_argument("--bias-step", type=float, default=0.1)
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
    elif architecture == "heterogeneous":
        model = build_cic_man_heterogeneous(num_classes=num_classes, num_agents=num_agents).to(device)
    else:
        model = build_cic_man(num_classes=num_classes, num_agents=num_agents).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, num_classes


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
    payload = {
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


def collect_logits(model, index_csv: Path, batch_size: int, device: str, num_workers: int):
    import torch

    loader = make_loader(index_csv, batch_size, num_workers)
    logits_all = []
    labels_all = []
    recording_ids_all = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            logits = model(x).detach().cpu().numpy().astype(np.float64)
            labels = batch["y"].detach().cpu().numpy().astype(int)
            recording_ids = metadata_column(batch["metadata"], "recording_id")
            logits_all.append(logits)
            labels_all.append(labels)
            recording_ids_all.extend(recording_ids)
    return {
        "logits": np.concatenate(logits_all, axis=0),
        "labels": np.concatenate(labels_all, axis=0).astype(int),
        "recording_ids": recording_ids_all,
    }


def biased_predictions(payload: dict[str, object], bias: float) -> tuple[list[int], list[int]]:
    logits = np.asarray(payload["logits"], dtype=np.float64).copy()
    labels = np.asarray(payload["labels"], dtype=int)
    logits[:, 1] += float(bias)
    pred = logits.argmax(axis=1).astype(int)
    return labels.tolist(), pred.tolist()


def evaluate_windows(payload: dict[str, object], bias: float, num_classes: int):
    y_true, y_pred = biased_predictions(payload, bias)
    return metric_payload(y_true, y_pred, num_classes)


def evaluate_recordings(payload: dict[str, object], bias: float, num_classes: int):
    logits = np.asarray(payload["logits"], dtype=np.float64).copy()
    true_labels = np.asarray(payload["labels"], dtype=int)
    recording_ids = list(payload["recording_ids"])
    logits[:, 1] += float(bias)
    pred = logits.argmax(axis=1).astype(int)
    logit_sums: dict[str, np.ndarray] = {}
    recording_labels: dict[str, int] = {}
    vote_counts: dict[str, Counter[int]] = defaultdict(Counter)
    for i, recording_id in enumerate(recording_ids):
        if recording_id not in logit_sums:
            logit_sums[recording_id] = np.zeros(num_classes, dtype=np.float64)
            recording_labels[recording_id] = int(true_labels[i])
        logit_sums[recording_id] += logits[i]
        vote_counts[recording_id][int(pred[i])] += 1
    y_true = []
    mean_pred = []
    vote_pred = []
    for recording_id in sorted(logit_sums):
        y_true.append(recording_labels[recording_id])
        mean_pred.append(int(logit_sums[recording_id].argmax()))
        vote_pred.append(int(vote_counts[recording_id].most_common(1)[0][0]))
    return {
        "mean_logits": metric_payload(y_true, mean_pred, num_classes),
        "majority_vote": metric_payload(y_true, vote_pred, num_classes),
    }


def select_bias(rows: list[dict[str, object]], source_drop_tolerance: float, inner_drop_tolerance: float) -> dict[str, object]:
    baseline = rows[0]
    base_macro = float(baseline["source_macro_f1"])
    base_inner_recall = float(baseline["source_inner_recall"])
    allowed = [
        row
        for row in rows
        if float(row["source_macro_f1"]) >= base_macro - source_drop_tolerance
        and float(row["source_inner_recall"]) >= base_inner_recall - inner_drop_tolerance
    ]
    if not allowed:
        return baseline | {"selection_reason": "fallback: no source-noninferior bias found"}
    selected = max(
        allowed,
        key=lambda row: (
            float(row["source_inner_predicted_support"]),
            float(row["source_inner_recall"]),
            float(row["source_macro_f1"]),
        ),
    )
    selected["selection_reason"] = (
        f"source-noninferior: source_macro_f1 >= {base_macro - source_drop_tolerance:.6f}, "
        f"source_inner_recall >= {base_inner_recall - inner_drop_tolerance:.6f}; "
        "maximized source inner predicted support"
    )
    return selected


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
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


def write_markdown(rows: list[dict[str, object]], selected: dict[str, object], path: Path) -> None:
    lines = [
        "# Ottawa Inner Decision Calibration Audit",
        "",
        "Calibration is selected using source validation only. Target columns are post-hoc audit evidence.",
        "",
        f"Selected inner logit bias: `{float(selected['bias']):.3f}`",
        "",
        f"Selection reason: {selected['selection_reason']}",
        "",
        "| Bias | Source Macro-F1 | Source Inner Recall | Source Inner Pred Support | Target Window Inner Recall | Target Rec Inner Recall | Target Rec Macro-F1 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    interesting = [rows[0], selected]
    for row in rows:
        if float(row["target_recording_inner_recall"]) > 0 and row not in interesting:
            interesting.append(row)
    seen = set()
    for row in interesting:
        key = float(row["bias"])
        if key in seen:
            continue
        seen.add(key)
        lines.append(
            f"| {float(row['bias']):.3f} | {float(row['source_macro_f1']):.6f} | "
            f"{float(row['source_inner_recall']):.6f} | {int(row['source_inner_predicted_support'])} | "
            f"{float(row['target_window_inner_recall']):.6f} | "
            f"{float(row['target_recording_inner_recall']):.6f} | "
            f"{float(row['target_recording_macro_f1']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Adversarial Conclusion",
            "",
            "- If selected source-only calibration still gives zero target recording inner recall, logit bias is not enough.",
            "- If a larger unselected bias gives target inner recall but violates source non-inferiority, it is not target-free defensible as a main method.",
        ]
    )
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
    biases = np.arange(0.0, args.bias_max + args.bias_step / 2.0, args.bias_step)
    rows = []
    for bias in biases:
        source = evaluate_windows(source_payload, float(bias), num_classes)
        target_window = evaluate_windows(target_payload, float(bias), num_classes)
        target_recording = evaluate_recordings(target_payload, float(bias), num_classes)["mean_logits"]
        rows.append(
            {
                "bias": float(bias),
                "source_macro_f1": source["macro_f1"],
                "source_accuracy": source["accuracy"],
                "source_inner_recall": source["inner_recall"],
                "source_inner_precision": source["inner_precision"],
                "source_inner_predicted_support": source["inner_predicted_support"],
                "target_window_macro_f1": target_window["macro_f1"],
                "target_window_inner_recall": target_window["inner_recall"],
                "target_window_inner_precision": target_window["inner_precision"],
                "target_window_inner_predicted_support": target_window["inner_predicted_support"],
                "target_recording_macro_f1": target_recording["macro_f1"],
                "target_recording_inner_recall": target_recording["inner_recall"],
                "target_recording_inner_precision": target_recording["inner_precision"],
                "target_recording_inner_predicted_support": target_recording["inner_predicted_support"],
            }
        )
    selected = select_bias(rows, args.source_drop_tolerance, args.inner_drop_tolerance)
    selected_path = output_dir / "ottawa_inner_decision_calibration_selected.json"
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    selected_path.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(rows, output_dir / "ottawa_inner_decision_calibration_sweep.csv")
    write_markdown(rows, selected, output_dir / "ottawa_inner_decision_calibration.md")
    print(f"Wrote Ottawa inner decision calibration audit to {output_dir}")


if __name__ == "__main__":
    main()
