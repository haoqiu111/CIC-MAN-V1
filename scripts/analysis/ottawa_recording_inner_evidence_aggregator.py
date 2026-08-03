#!/usr/bin/env python3
"""Source-selected recording-level inner evidence aggregator for Ottawa.

The stable v5 core keeps the default recording decision. Pretrained STFT and
filterbank view agents contribute only recording-level inner evidence. Rule
selection uses source validation only; target labels are post-hoc audit only.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
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
    parser.add_argument("--val-index", type=Path, required=True)
    parser.add_argument("--test-index", type=Path, required=True)
    parser.add_argument("--core-checkpoint", type=Path, required=True)
    parser.add_argument("--view-agent-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--views", default="stft,filterbank")
    parser.add_argument("--batch-size", type=int, default=512)
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


def load_core(path: Path, device: str):
    import torch

    checkpoint = torch.load(path, map_location=device)
    num_classes = int(checkpoint["num_classes"])
    num_agents = int(checkpoint.get("num_agents", checkpoint.get("config", {}).get("num_agents", 4)))
    model = build_cic_man(num_classes=num_classes, num_agents=num_agents).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, num_classes


def load_view_model(path: Path, views: list[str], num_classes: int, device: str):
    import torch

    checkpoint = torch.load(path, map_location=device)
    model = build_cic_man_vfinal(num_classes=num_classes, view_names=views, use_health_style_split=False).to(device)
    target = model.state_dict()
    copied = {}
    for key, value in target.items():
        if key.startswith(("view_encoders.", "view_agents.")) and key in checkpoint["model_state"]:
            source = checkpoint["model_state"][key]
            if tuple(source.shape) == tuple(value.shape):
                copied[key] = source
    if not copied:
        raise ValueError("No view-agent parameters copied.")
    target.update(copied)
    model.load_state_dict(target)
    model.eval()
    return model


def per_class(cm: np.ndarray, class_id: int) -> dict[str, float]:
    tp = float(cm[class_id, class_id])
    fp = float(cm[:, class_id].sum() - tp)
    fn = float(cm[class_id, :].sum() - tp)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": precision, "recall": recall, "predicted": int(cm[:, class_id].sum())}


def metric_payload(y_true: list[int], y_pred: list[int], num_classes: int) -> dict[str, object]:
    cm = confusion_matrix(y_true, y_pred, num_classes)
    payload: dict[str, object] = {
        "macro_f1": macro_f1(y_true, y_pred, num_classes),
        "balanced_accuracy": balanced_accuracy(y_true, y_pred, num_classes),
        "num_recordings": len(y_true),
        "confusion_matrix": cm.tolist(),
    }
    for class_id, name in CLASS_NAMES.items():
        stats = per_class(cm, class_id)
        payload[f"{name}_recall"] = stats["recall"]
        payload[f"{name}_precision"] = stats["precision"]
        payload[f"{name}_predicted_support"] = stats["predicted"]
    return payload


def collect_recordings(core_model, view_model, views: list[str], index_csv: Path, batch_size: int, device: str, num_workers: int):
    import torch

    groups: dict[str, dict[str, object]] = {}
    loader = make_loader(index_csv, batch_size, num_workers)
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].detach().cpu().numpy().astype(int)
            core_logits = core_model(x).detach().cpu().numpy().astype(np.float64)
            view_logits = {}
            for view in views:
                view_x = view_model.make_view(x, view)
                features = view_model.view_encoders[view](view_x)
                view_logits[view] = view_model.view_agents[view](features).detach().cpu().numpy().astype(np.float64)
            metadata = batch["metadata"]
            recording_ids = metadata_column(metadata, "recording_id")
            for idx, recording_id in enumerate(recording_ids):
                if recording_id not in groups:
                    groups[recording_id] = {
                        "label": int(y[idx]),
                        "recording_id": recording_id,
                        "core_logits": [],
                        "view_logits": {view: [] for view in views},
                    }
                groups[recording_id]["core_logits"].append(core_logits[idx])
                for view in views:
                    groups[recording_id]["view_logits"][view].append(view_logits[view][idx])
    rows = []
    for recording_id, item in sorted(groups.items()):
        core = np.stack(item["core_logits"], axis=0)
        core_mean = core.mean(axis=0)
        row = {
            "recording_id": recording_id,
            "true_label": int(item["label"]),
            "core_pred": int(core_mean.argmax()),
            "core_inner_margin_q90": float(np.quantile(core[:, 1] - np.maximum(core[:, 0], core[:, 2]), 0.90)),
            "num_windows": int(len(core)),
        }
        top1_fracs = []
        top2_fracs = []
        q90_margins = []
        mean_margins = []
        for view in views:
            logits = np.stack(item["view_logits"][view], axis=0)
            pred = logits.argmax(axis=1)
            rank = (-logits).argsort(axis=1)
            inner_rank = np.where(rank == 1)[1] + 1
            margin = logits[:, 1] - np.maximum(logits[:, 0], logits[:, 2])
            row[f"{view}_top1_inner_fraction"] = float((pred == 1).mean())
            row[f"{view}_top2_inner_fraction"] = float((inner_rank <= 2).mean())
            row[f"{view}_inner_margin_q90"] = float(np.quantile(margin, 0.90))
            row[f"{view}_inner_margin_mean"] = float(margin.mean())
            top1_fracs.append(row[f"{view}_top1_inner_fraction"])
            top2_fracs.append(row[f"{view}_top2_inner_fraction"])
            q90_margins.append(row[f"{view}_inner_margin_q90"])
            mean_margins.append(row[f"{view}_inner_margin_mean"])
        row["mean_top1_inner_fraction"] = float(np.mean(top1_fracs))
        row["max_top1_inner_fraction"] = float(np.max(top1_fracs))
        row["mean_top2_inner_fraction"] = float(np.mean(top2_fracs))
        row["min_top2_inner_fraction"] = float(np.min(top2_fracs))
        row["mean_q90_inner_margin"] = float(np.mean(q90_margins))
        row["max_q90_inner_margin"] = float(np.max(q90_margins))
        row["mean_inner_margin"] = float(np.mean(mean_margins))
        row["view_agreement_top1_inner"] = float(np.mean([value >= 0.5 for value in top1_fracs]))
        rows.append(row)
    return rows


def apply_rule(rows: list[dict[str, object]], rule: str, threshold: float) -> list[int]:
    preds = []
    for row in rows:
        pred = int(row["core_pred"])
        if rule == "baseline":
            preds.append(pred)
            continue
        score = float(row[rule])
        if score >= float(threshold):
            pred = 1
        preds.append(pred)
    return preds


def evaluate(rows: list[dict[str, object]], rule: str, threshold: float, num_classes: int) -> dict[str, object]:
    true = [int(row["true_label"]) for row in rows]
    pred = apply_rule(rows, rule, threshold)
    return metric_payload(true, pred, num_classes)


def candidate_grid(rows: list[dict[str, object]]):
    rules = [
        "mean_top1_inner_fraction",
        "max_top1_inner_fraction",
        "mean_top2_inner_fraction",
        "min_top2_inner_fraction",
        "mean_q90_inner_margin",
        "max_q90_inner_margin",
        "mean_inner_margin",
        "view_agreement_top1_inner",
    ]
    yield "baseline", float("inf")
    for rule in rules:
        values = np.asarray([float(row[rule]) for row in rows], dtype=np.float64)
        quantiles = sorted(set(float(np.quantile(values, q)) for q in np.linspace(0.05, 0.95, 19)))
        for threshold in quantiles:
            yield rule, threshold


def select_candidate(candidate_rows: list[dict[str, object]], baseline: dict[str, object], args: argparse.Namespace) -> dict[str, object]:
    allowed = []
    for row in candidate_rows:
        if float(row["source_macro_f1"]) < float(baseline["source_macro_f1"]) - args.source_macro_drop_tol:
            continue
        if float(row["source_normal_recall"]) < float(baseline["source_normal_recall"]) - args.source_normal_drop_tol:
            continue
        if float(row["source_outer_recall"]) < float(baseline["source_outer_recall"]) - args.source_outer_drop_tol:
            continue
        if float(row["source_inner_precision"]) < float(baseline["source_inner_precision"]) - args.source_inner_precision_drop_tol:
            continue
        allowed.append(row)
    if not allowed:
        baseline["selection_reason"] = "fallback baseline: no recording evidence rule satisfied source class-safety constraints"
        return baseline
    selected = max(
        allowed,
        key=lambda row: (
            float(row["source_inner_recall"]),
            float(row["source_inner_predicted_support"]),
            float(row["source_macro_f1"]),
        ),
    )
    selected["selection_reason"] = "source-only recording evidence: noninferior source macro/normal/outer and inner precision; ranked by source inner recall/support"
    return selected


def candidate_label(row: dict[str, object]) -> str:
    return f"{row['rule']}@{row['threshold']}"


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
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
    source_rows = collect_recordings(core_model, view_model, views, args.val_index, args.batch_size, device, args.num_workers)
    target_rows = collect_recordings(core_model, view_model, views, args.test_index, args.batch_size, device, args.num_workers)

    candidate_rows = []
    baseline = None
    for rule, threshold in candidate_grid(source_rows):
        source = evaluate(source_rows, rule, threshold, num_classes)
        target = evaluate(target_rows, rule, threshold, num_classes)
        row = {
            "rule": rule,
            "threshold": threshold,
            "source_macro_f1": source["macro_f1"],
            "source_normal_recall": source["normal_recall"],
            "source_inner_recall": source["inner_recall"],
            "source_inner_precision": source["inner_precision"],
            "source_inner_predicted_support": source["inner_predicted_support"],
            "source_outer_recall": source["outer_recall"],
            "target_macro_f1_posthoc": target["macro_f1"],
            "target_normal_recall_posthoc": target["normal_recall"],
            "target_inner_recall_posthoc": target["inner_recall"],
            "target_inner_precision_posthoc": target["inner_precision"],
            "target_outer_recall_posthoc": target["outer_recall"],
            "target_confusion_posthoc": json.dumps(target["confusion_matrix"]),
        }
        candidate_rows.append(row)
        if rule == "baseline":
            baseline = row
    assert baseline is not None
    selected = select_candidate(candidate_rows, baseline, args)

    write_csv(source_rows, args.output_dir / "source_recording_evidence.csv")
    write_csv(target_rows, args.output_dir / "target_recording_evidence.csv")
    write_csv(candidate_rows, args.output_dir / "recording_inner_aggregator_candidates.csv")
    (args.output_dir / "recording_inner_aggregator_selected.json").write_text(json.dumps(selected, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Ottawa Recording-Level Inner Evidence Aggregator",
        "",
        "Scope: v5 core keeps default normal/outer recording decisions; pretrained STFT/filterbank provide recording-level inner evidence. Source validation selects the rule; target labels are post-hoc only.",
        "",
        f"Selected rule: `{selected['rule']}` threshold `{selected['threshold']}`.",
        "",
        f"Selection reason: {selected['selection_reason']}",
        "",
        "| Rule | Source F1 | Source Normal R | Source Inner R | Source Inner P | Source Outer R | Target F1 | Target Normal R | Target Inner R | Target Inner P | Target Outer R | Target CM |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, row in [("baseline", baseline), ("selected", selected)]:
        lines.append(
            f"| {name}:{row['rule']} | {float(row['source_macro_f1']):.6f} | "
            f"{float(row['source_normal_recall']):.6f} | {float(row['source_inner_recall']):.6f} | "
            f"{float(row['source_inner_precision']):.6f} | {float(row['source_outer_recall']):.6f} | "
            f"{float(row['target_macro_f1_posthoc']):.6f} | {float(row['target_normal_recall_posthoc']):.6f} | "
            f"{float(row['target_inner_recall_posthoc']):.6f} | {float(row['target_inner_precision_posthoc']):.6f} | "
            f"{float(row['target_outer_recall_posthoc']):.6f} | `{row['target_confusion_posthoc']}` |"
        )
    best_target_inner = max(
        candidate_rows,
        key=lambda row: (float(row["target_inner_recall_posthoc"]), float(row["target_macro_f1_posthoc"])),
    )
    best_target_macro = max(candidate_rows, key=lambda row: float(row["target_macro_f1_posthoc"]))
    lines.extend(
        [
            "",
            "## Post-Hoc Upper Check",
            "",
            f"Best searched target inner recall post-hoc: `{float(best_target_inner['target_inner_recall_posthoc']):.6f}` from `{candidate_label(best_target_inner)}`.",
            f"Best searched target macro-F1 post-hoc: `{float(best_target_macro['target_macro_f1_posthoc']):.6f}` from `{candidate_label(best_target_macro)}`.",
            "",
            "| Candidate | Source F1 | Source Normal R | Source Inner R | Source Inner P | Source Outer R | Target F1 | Target Normal R | Target Inner R | Target Inner P | Target Outer R | Target CM |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for name, row in [("best_inner", best_target_inner), ("best_macro", best_target_macro)]:
        lines.append(
            f"| {name}:{candidate_label(row)} | {float(row['source_macro_f1']):.6f} | "
            f"{float(row['source_normal_recall']):.6f} | {float(row['source_inner_recall']):.6f} | "
            f"{float(row['source_inner_precision']):.6f} | {float(row['source_outer_recall']):.6f} | "
            f"{float(row['target_macro_f1_posthoc']):.6f} | {float(row['target_normal_recall_posthoc']):.6f} | "
            f"{float(row['target_inner_recall_posthoc']):.6f} | {float(row['target_inner_precision_posthoc']):.6f} | "
            f"{float(row['target_outer_recall_posthoc']):.6f} | `{row['target_confusion_posthoc']}` |"
        )
    lines.extend(
        [
            "",
            "## Adversarial Verdict",
            "",
            "- The selected rule is source-only and class-safe by source validation constraints.",
            "- If selected falls back to baseline, the pretrained view evidence is not safe enough for recording-level override.",
            "- If post-hoc upper check is high but selected remains baseline, source validation cannot certify the target inner override under current constraints.",
            "- A high-yield target-posthoc rule that sacrifices source outer recall is diagnostic evidence, not a publishable target-free selector.",
        ]
    )
    report = args.output_dir / "recording_inner_aggregator_audit.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"selected": selected, "report": str(report)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
