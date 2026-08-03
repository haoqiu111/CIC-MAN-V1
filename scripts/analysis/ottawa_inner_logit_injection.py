#!/usr/bin/env python3
"""Source-calibrated inner-logit injection using pretrained vFinal view agents.

The stable core keeps normal/outer decisions. STFT/filterbank view agents may
add calibrated evidence only to the inner logit. All rule selection is based on
source validation; Ottawa target labels are used only for post-hoc auditing.
"""

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
from cicman.evaluation.metrics import balanced_accuracy, confusion_matrix, macro_f1  # noqa: E402
from cicman.models.cic_man import build_cic_man  # noqa: E402
from cicman.models.cic_man_vfinal import build_cic_man_vfinal  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-index", type=Path, required=True)
    parser.add_argument("--val-index", type=Path, required=True)
    parser.add_argument("--test-index", type=Path, required=True)
    parser.add_argument("--core-checkpoint", type=Path, required=True)
    parser.add_argument("--view-agent-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--views", default="stft,filterbank")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--source-macro-drop-tol", type=float, default=0.01)
    parser.add_argument("--source-normal-drop-tol", type=float, default=0.01)
    parser.add_argument("--source-outer-drop-tol", type=float, default=0.01)
    parser.add_argument("--source-inner-precision-drop-tol", type=float, default=0.05)
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


def load_core(checkpoint_path: Path, device: str):
    import torch

    checkpoint = torch.load(checkpoint_path, map_location=device)
    num_classes = int(checkpoint["num_classes"])
    num_agents = int(checkpoint.get("num_agents", checkpoint.get("config", {}).get("num_agents", 4)))
    model = build_cic_man(num_classes=num_classes, num_agents=num_agents).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, num_classes


def load_view_model(checkpoint_path: Path, views: list[str], num_classes: int, device: str):
    import torch

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_cic_man_vfinal(num_classes=num_classes, view_names=views, use_health_style_split=False).to(device)
    source = checkpoint["model_state"]
    target = model.state_dict()
    copied = {}
    for key, value in target.items():
        if key.startswith(("view_encoders.", "view_agents.")) and key in source and tuple(source[key].shape) == tuple(value.shape):
            copied[key] = source[key]
    target.update(copied)
    model.load_state_dict(target)
    model.eval()
    if len(copied) == 0:
        raise ValueError("No view-agent parameters were copied from checkpoint.")
    return model


def per_class(metrics_cm: np.ndarray, class_id: int) -> dict[str, float]:
    tp = float(metrics_cm[class_id, class_id])
    fp = float(metrics_cm[:, class_id].sum() - tp)
    fn = float(metrics_cm[class_id, :].sum() - tp)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "support": int(metrics_cm[class_id, :].sum()), "predicted": int(metrics_cm[:, class_id].sum())}


def metric_payload(y_true: list[int], y_pred: list[int], num_classes: int) -> dict[str, object]:
    cm = confusion_matrix(y_true, y_pred, num_classes)
    payload: dict[str, object] = {
        "macro_f1": macro_f1(y_true, y_pred, num_classes),
        "balanced_accuracy": balanced_accuracy(y_true, y_pred, num_classes),
        "num_samples": len(y_true),
        "confusion_matrix": cm.tolist(),
    }
    for class_id, name in CLASS_NAMES.items():
        stats = per_class(cm, class_id)
        payload[f"{name}_precision"] = stats["precision"]
        payload[f"{name}_recall"] = stats["recall"]
        payload[f"{name}_f1"] = stats["f1"]
        payload[f"{name}_support"] = stats["support"]
        payload[f"{name}_predicted_support"] = stats["predicted"]
    return payload


def collect_evidence(core_model, view_model, views: list[str], index_csv: Path, batch_size: int, device: str, num_workers: int) -> dict[str, object]:
    import torch

    loader = make_loader(index_csv, batch_size, num_workers)
    core_logits = []
    view_inner_logits = {view: [] for view in views}
    view_inner_margins = {view: [] for view in views}
    labels = []
    recording_ids = []
    metadata_rows = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].detach().cpu().numpy().astype(int)
            core = core_model(x).detach().cpu().numpy().astype(np.float64)
            core_logits.append(core)
            for view in views:
                view_x = view_model.make_view(x, view)
                features = view_model.view_encoders[view](view_x)
                logits = view_model.view_agents[view](features).detach().cpu().numpy().astype(np.float64)
                view_inner_logits[view].append(logits[:, 1])
                view_inner_margins[view].append(logits[:, 1] - np.maximum(logits[:, 0], logits[:, 2]))
            metadata = batch["metadata"]
            ids = metadata_column(metadata, "recording_id")
            datasets = metadata_column(metadata, "dataset_id")
            label_names = metadata_column(metadata, "label")
            for idx, recording_id in enumerate(ids):
                labels.append(int(y[idx]))
                recording_ids.append(recording_id)
                metadata_rows.append(
                    {
                        "recording_id": recording_id,
                        "dataset_id": datasets[idx],
                        "label": label_names[idx],
                        "label_id": int(y[idx]),
                    }
                )
    return {
        "core_logits": np.concatenate(core_logits, axis=0),
        "view_inner_logits": {view: np.concatenate(parts, axis=0) for view, parts in view_inner_logits.items()},
        "view_inner_margins": {view: np.concatenate(parts, axis=0) for view, parts in view_inner_margins.items()},
        "labels": np.asarray(labels, dtype=int),
        "recording_ids": recording_ids,
        "metadata_rows": metadata_rows,
        "views": views,
    }


def injected_logits(payload: dict[str, object], alpha: float, margin_threshold: float, support_views: int, cap: float) -> np.ndarray:
    logits = np.asarray(payload["core_logits"], dtype=np.float64).copy()
    margins = np.stack([payload["view_inner_margins"][view] for view in payload["views"]], axis=1)
    positive = margins >= float(margin_threshold)
    support = positive.sum(axis=1)
    evidence = np.where(positive, margins - float(margin_threshold), 0.0).sum(axis=1)
    injection = np.minimum(float(cap), float(alpha) * evidence)
    injection = np.where(support >= int(support_views), injection, 0.0)
    logits[:, 1] += injection
    return logits


def evaluate_window(payload: dict[str, object], alpha: float, margin_threshold: float, support_views: int, cap: float, num_classes: int) -> dict[str, object]:
    logits = injected_logits(payload, alpha, margin_threshold, support_views, cap)
    pred = logits.argmax(axis=1).astype(int)
    return metric_payload(payload["labels"].tolist(), pred.tolist(), num_classes)


def recording_groups(payload: dict[str, object], logits: np.ndarray) -> dict[str, dict[str, object]]:
    groups: dict[str, dict[str, object]] = {}
    for idx, recording_id in enumerate(payload["recording_ids"]):
        if recording_id not in groups:
            groups[recording_id] = {"label": int(payload["labels"][idx]), "logits": []}
        groups[recording_id]["logits"].append(logits[idx])
    for item in groups.values():
        item["logits"] = np.stack(item["logits"], axis=0)
    return groups


def evaluate_recording(payload: dict[str, object], alpha: float, margin_threshold: float, support_views: int, cap: float, num_classes: int) -> dict[str, object]:
    logits = injected_logits(payload, alpha, margin_threshold, support_views, cap)
    y_true = []
    y_pred = []
    for item in recording_groups(payload, logits).values():
        y_true.append(int(item["label"]))
        y_pred.append(int(np.asarray(item["logits"]).mean(axis=0).argmax()))
    return metric_payload(y_true, y_pred, num_classes)


def candidate_grid():
    for alpha in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
        for threshold in [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0]:
            for support_views in [1, 2]:
                for cap in [0.5, 1.0, 2.0, 3.0, 5.0]:
                    yield alpha, threshold, support_views, cap


def select_candidate(rows: list[dict[str, object]], baseline: dict[str, object], args: argparse.Namespace) -> dict[str, object]:
    allowed = []
    for row in rows:
        if float(row["source_window_macro_f1"]) < float(baseline["source_window_macro_f1"]) - args.source_macro_drop_tol:
            continue
        if float(row["source_window_normal_recall"]) < float(baseline["source_window_normal_recall"]) - args.source_normal_drop_tol:
            continue
        if float(row["source_window_outer_recall"]) < float(baseline["source_window_outer_recall"]) - args.source_outer_drop_tol:
            continue
        if float(row["source_window_inner_precision"]) < float(baseline["source_window_inner_precision"]) - args.source_inner_precision_drop_tol:
            continue
        allowed.append(row)
    if not allowed:
        selected = baseline
        selected["selection_reason"] = "fallback baseline: no injection candidate satisfied source class-safety constraints"
        return selected
    selected = max(
        allowed,
        key=lambda row: (
            float(row["source_window_inner_recall"]),
            float(row["source_recording_inner_recall"]),
            float(row["source_window_macro_f1"]),
            -float(row["alpha"]),
        ),
    )
    selected["selection_reason"] = "source-only class-safe: noninferior source macro/normal/outer and inner precision; ranked by source inner recall"
    return selected


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    import torch

    args = parse_args()
    views = [item.strip() for item in args.views.split(",") if item.strip()]
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    core_model, num_classes = load_core(args.core_checkpoint, device)
    view_model = load_view_model(args.view_agent_checkpoint, views, num_classes, device)
    source_payload = collect_evidence(core_model, view_model, views, args.val_index, args.batch_size, device, args.num_workers)
    target_payload = collect_evidence(core_model, view_model, views, args.test_index, args.batch_size, device, args.num_workers)

    base_window_source = evaluate_window(source_payload, 0.0, 0.0, 1, 0.0, num_classes)
    base_record_source = evaluate_recording(source_payload, 0.0, 0.0, 1, 0.0, num_classes)
    base_window_target = evaluate_window(target_payload, 0.0, 0.0, 1, 0.0, num_classes)
    base_record_target = evaluate_recording(target_payload, 0.0, 0.0, 1, 0.0, num_classes)
    rows = []
    baseline_row = {
        "alpha": 0.0,
        "margin_threshold": 0.0,
        "support_views": 1,
        "cap": 0.0,
        "source_window_macro_f1": base_window_source["macro_f1"],
        "source_window_normal_recall": base_window_source["normal_recall"],
        "source_window_inner_recall": base_window_source["inner_recall"],
        "source_window_inner_precision": base_window_source["inner_precision"],
        "source_window_outer_recall": base_window_source["outer_recall"],
        "source_recording_macro_f1": base_record_source["macro_f1"],
        "source_recording_inner_recall": base_record_source["inner_recall"],
        "target_window_macro_f1_posthoc": base_window_target["macro_f1"],
        "target_window_normal_recall_posthoc": base_window_target["normal_recall"],
        "target_window_inner_recall_posthoc": base_window_target["inner_recall"],
        "target_window_outer_recall_posthoc": base_window_target["outer_recall"],
        "target_recording_macro_f1_posthoc": base_record_target["macro_f1"],
        "target_recording_normal_recall_posthoc": base_record_target["normal_recall"],
        "target_recording_inner_recall_posthoc": base_record_target["inner_recall"],
        "target_recording_outer_recall_posthoc": base_record_target["outer_recall"],
        "target_recording_confusion_posthoc": json.dumps(base_record_target["confusion_matrix"]),
    }
    rows.append(baseline_row)
    for alpha, threshold, support_views, cap in candidate_grid():
        source_w = evaluate_window(source_payload, alpha, threshold, support_views, cap, num_classes)
        source_r = evaluate_recording(source_payload, alpha, threshold, support_views, cap, num_classes)
        target_w = evaluate_window(target_payload, alpha, threshold, support_views, cap, num_classes)
        target_r = evaluate_recording(target_payload, alpha, threshold, support_views, cap, num_classes)
        rows.append(
            {
                "alpha": alpha,
                "margin_threshold": threshold,
                "support_views": support_views,
                "cap": cap,
                "source_window_macro_f1": source_w["macro_f1"],
                "source_window_normal_recall": source_w["normal_recall"],
                "source_window_inner_recall": source_w["inner_recall"],
                "source_window_inner_precision": source_w["inner_precision"],
                "source_window_outer_recall": source_w["outer_recall"],
                "source_recording_macro_f1": source_r["macro_f1"],
                "source_recording_inner_recall": source_r["inner_recall"],
                "target_window_macro_f1_posthoc": target_w["macro_f1"],
                "target_window_normal_recall_posthoc": target_w["normal_recall"],
                "target_window_inner_recall_posthoc": target_w["inner_recall"],
                "target_window_outer_recall_posthoc": target_w["outer_recall"],
                "target_recording_macro_f1_posthoc": target_r["macro_f1"],
                "target_recording_normal_recall_posthoc": target_r["normal_recall"],
                "target_recording_inner_recall_posthoc": target_r["inner_recall"],
                "target_recording_outer_recall_posthoc": target_r["outer_recall"],
                "target_recording_confusion_posthoc": json.dumps(target_r["confusion_matrix"]),
            }
        )
    selected = select_candidate(rows, baseline_row, args)
    write_csv(rows, args.output_dir / "inner_logit_injection_candidates.csv")
    (args.output_dir / "inner_logit_injection_selected.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Ottawa Inner Logit Injection Audit",
        "",
        "Scope: v5 core normal/outer path plus pretrained STFT/filterbank inner evidence. Source validation selects injection parameters; target labels are post-hoc only.",
        "",
        f"Selected: alpha `{selected['alpha']}`, margin_threshold `{selected['margin_threshold']}`, support_views `{selected['support_views']}`, cap `{selected['cap']}`.",
        "",
        f"Selection reason: {selected['selection_reason']}",
        "",
        "| Rule | Source F1 | Source Normal R | Source Inner R | Source Inner P | Source Outer R | Target Rec F1 | Target Rec Normal R | Target Rec Inner R | Target Rec Outer R | Target Rec CM |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, row in [("baseline", baseline_row), ("selected", selected)]:
        lines.append(
            f"| {name} | {float(row['source_window_macro_f1']):.6f} | "
            f"{float(row['source_window_normal_recall']):.6f} | {float(row['source_window_inner_recall']):.6f} | "
            f"{float(row['source_window_inner_precision']):.6f} | {float(row['source_window_outer_recall']):.6f} | "
            f"{float(row['target_recording_macro_f1_posthoc']):.6f} | {float(row['target_recording_normal_recall_posthoc']):.6f} | "
            f"{float(row['target_recording_inner_recall_posthoc']):.6f} | {float(row['target_recording_outer_recall_posthoc']):.6f} | "
            f"`{row['target_recording_confusion_posthoc']}` |"
        )
    lines.extend(
        [
            "",
            "## Adversarial Verdict",
            "",
            "- The selector is source-only and protects source normal/outer recall and source inner precision.",
            "- If selected falls back to baseline, the pretrained view evidence is not source-safe under this injection rule.",
            "- If target inner improves post-hoc while source constraints hold, promote this as a candidate recording-level CIC-MAN evidence aggregator, not as target-tuned adaptation.",
        ]
    )
    (args.output_dir / "inner_logit_injection_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"selected": selected, "report": str(args.output_dir / "inner_logit_injection_audit.md")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
