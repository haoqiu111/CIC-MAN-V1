#!/usr/bin/env python3
"""Source-only outer-vs-inner mechanism guard for Ottawa inner aggregation.

The v5 core remains the default recording-level classifier. Pretrained STFT and
filterbank view agents can override a recording to inner only when source-built
mechanism prototypes say the recording is closer to inner than outer. Target
labels are used only for post-hoc audit.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


CLASS_NAMES = {0: "normal", 1: "inner", 2: "outer"}
FEATURE_NAMES = ["kurtosis", "crest", "roughness", "high_energy", "envelope_ratio", "zero_crossing"]


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
            mech = mechanism_features(x).detach().cpu().numpy().astype(np.float64)
            core_logits = core_model(x).detach().cpu().numpy().astype(np.float64)
            view_logits = {}
            for view in views:
                view_x = view_model.make_view(x, view)
                features = view_model.view_encoders[view](view_x)
                view_logits[view] = view_model.view_agents[view](features).detach().cpu().numpy().astype(np.float64)
            metadata = batch["metadata"]
            recording_ids = metadata_column(metadata, "recording_id")
            dataset_ids = metadata_column(metadata, "dataset_id")
            for idx, recording_id in enumerate(recording_ids):
                if recording_id not in groups:
                    groups[recording_id] = {
                        "label": int(y[idx]),
                        "recording_id": recording_id,
                        "dataset_id": dataset_ids[idx],
                        "core_logits": [],
                        "view_logits": {view: [] for view in views},
                        "mechanism_features": [],
                    }
                groups[recording_id]["core_logits"].append(core_logits[idx])
                groups[recording_id]["mechanism_features"].append(mech[idx])
                for view in views:
                    groups[recording_id]["view_logits"][view].append(view_logits[view][idx])
    rows = []
    for recording_id, item in sorted(groups.items()):
        core = np.stack(item["core_logits"], axis=0)
        core_mean = core.mean(axis=0)
        mech_mean = np.stack(item["mechanism_features"], axis=0).mean(axis=0)
        row = {
            "recording_id": recording_id,
            "dataset_id": str(item["dataset_id"]),
            "true_label": int(item["label"]),
            "core_pred": int(core_mean.argmax()),
            "core_inner_margin_q90": float(np.quantile(core[:, 1] - np.maximum(core[:, 0], core[:, 2]), 0.90)),
            "num_windows": int(len(core)),
            **{name: float(mech_mean[idx]) for idx, name in enumerate(FEATURE_NAMES)},
        }
        top1_fracs = []
        q90_margins = []
        for view in views:
            logits = np.stack(item["view_logits"][view], axis=0)
            pred = logits.argmax(axis=1)
            margin = logits[:, 1] - np.maximum(logits[:, 0], logits[:, 2])
            row[f"{view}_top1_inner_fraction"] = float((pred == 1).mean())
            row[f"{view}_inner_margin_q90"] = float(np.quantile(margin, 0.90))
            top1_fracs.append(row[f"{view}_top1_inner_fraction"])
            q90_margins.append(row[f"{view}_inner_margin_q90"])
        row["mean_top1_inner_fraction"] = float(np.mean(top1_fracs))
        row["view_agreement_top1_inner"] = float(np.mean([value >= 0.5 for value in top1_fracs]))
        row["mean_q90_inner_margin"] = float(np.mean(q90_margins))
        rows.append(row)
    return rows


def fit_mechanism_guard(source_rows: list[dict[str, object]]) -> dict[str, object]:
    vectors = np.stack([[float(row[name]) for name in FEATURE_NAMES] for row in source_rows], axis=0)
    scale = vectors.std(axis=0) + 1e-6
    centroids = {}
    for class_id in CLASS_NAMES:
        class_vectors = np.stack(
            [[float(row[name]) for name in FEATURE_NAMES] for row in source_rows if int(row["true_label"]) == class_id],
            axis=0,
        )
        centroids[class_id] = class_vectors.mean(axis=0)
    return {"scale": scale, "centroids": centroids}


def add_guard_scores(rows: list[dict[str, object]], guard: dict[str, object]) -> None:
    scale = guard["scale"]
    centroids = guard["centroids"]
    for row in rows:
        vector = np.asarray([float(row[name]) for name in FEATURE_NAMES], dtype=np.float64)
        distances = {}
        for class_id in CLASS_NAMES:
            distances[class_id] = float(np.linalg.norm((vector - centroids[class_id]) / scale))
            row[f"mechanism_distance_to_{CLASS_NAMES[class_id]}"] = distances[class_id]
        row["inner_vs_outer_guard_score"] = distances[2] - distances[1]
        row["inner_vs_normal_guard_score"] = distances[0] - distances[1]
        row["mechanism_nearest_class"] = CLASS_NAMES[min(distances, key=distances.get)]


def apply_rule(rows: list[dict[str, object]], rule: str, threshold: float) -> list[int]:
    preds = []
    for row in rows:
        pred = int(row["core_pred"])
        if rule == "baseline":
            preds.append(pred)
            continue
        if rule == "agreement_only":
            if float(row["view_agreement_top1_inner"]) >= 1.0:
                pred = 1
            preds.append(pred)
            continue
        if rule == "mechanism_guarded_agreement":
            enough_inner_evidence = float(row["view_agreement_top1_inner"]) >= 1.0
            unlike_outer = float(row["inner_vs_outer_guard_score"]) >= threshold
            not_normal_like = float(row["inner_vs_normal_guard_score"]) >= threshold
            if enough_inner_evidence and unlike_outer and not_normal_like:
                pred = 1
            preds.append(pred)
            continue
        raise ValueError(f"Unknown rule: {rule}")
    return preds


def evaluate(rows: list[dict[str, object]], rule: str, threshold: float, num_classes: int) -> dict[str, object]:
    true = [int(row["true_label"]) for row in rows]
    pred = apply_rule(rows, rule, threshold)
    return metric_payload(true, pred, num_classes)


def candidate_grid(source_rows: list[dict[str, object]]):
    yield "baseline", float("inf")
    yield "agreement_only", float("-inf")
    values = np.asarray(
        [
            min(float(row["inner_vs_outer_guard_score"]), float(row["inner_vs_normal_guard_score"]))
            for row in source_rows
            if float(row["view_agreement_top1_inner"]) >= 1.0
        ],
        dtype=np.float64,
    )
    if len(values) == 0:
        return
    for threshold in sorted(set(float(np.quantile(values, q)) for q in np.linspace(0.05, 0.95, 19))):
        yield "mechanism_guarded_agreement", threshold


def source_domain_metrics(rows: list[dict[str, object]], rule: str, threshold: float, num_classes: int) -> list[dict[str, object]]:
    preds = apply_rule(rows, rule, threshold)
    by_domain: dict[str, dict[str, list[int]]] = defaultdict(lambda: {"true": [], "pred": []})
    for row, pred in zip(rows, preds):
        domain = str(row["dataset_id"])
        by_domain[domain]["true"].append(int(row["true_label"]))
        by_domain[domain]["pred"].append(int(pred))
    out = []
    for domain, payload in sorted(by_domain.items()):
        metrics = metric_payload(payload["true"], payload["pred"], num_classes)
        out.append(
            {
                "dataset_id": domain,
                "macro_f1": metrics["macro_f1"],
                "normal_recall": metrics["normal_recall"],
                "inner_recall": metrics["inner_recall"],
                "inner_precision": metrics["inner_precision"],
                "outer_recall": metrics["outer_recall"],
                "confusion_matrix": json.dumps(metrics["confusion_matrix"]),
            }
        )
    return out


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
        if float(row["source_domain_min_outer_recall"]) < float(row["baseline_domain_min_outer_recall"]) - args.source_outer_drop_tol:
            continue
        allowed.append(row)
    if not allowed:
        baseline["selection_reason"] = "fallback baseline: no mechanism-guarded rule satisfied source and domain-wise outer safety"
        return baseline
    selected = max(
        allowed,
        key=lambda row: (
            float(row["source_inner_recall"]),
            float(row["source_inner_predicted_support"]),
            float(row["source_macro_f1"]),
        ),
    )
    selected["selection_reason"] = "source-only mechanism guard: double-view inner agreement plus inner-vs-outer/normal prototype separation"
    return selected


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


def row_summary(name: str, row: dict[str, object]) -> str:
    return (
        f"| {name}:{row['rule']}@{row['threshold']} | {float(row['source_macro_f1']):.6f} | "
        f"{float(row['source_normal_recall']):.6f} | {float(row['source_inner_recall']):.6f} | "
        f"{float(row['source_inner_precision']):.6f} | {float(row['source_outer_recall']):.6f} | "
        f"{float(row['source_domain_min_outer_recall']):.6f} | {float(row['target_macro_f1_posthoc']):.6f} | "
        f"{float(row['target_normal_recall_posthoc']):.6f} | {float(row['target_inner_recall_posthoc']):.6f} | "
        f"{float(row['target_inner_precision_posthoc']):.6f} | {float(row['target_outer_recall_posthoc']):.6f} | "
        f"`{row['target_confusion_posthoc']}` |"
    )


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

    guard = fit_mechanism_guard(source_rows)
    add_guard_scores(source_rows, guard)
    add_guard_scores(target_rows, guard)

    candidate_rows = []
    baseline = None
    baseline_domains = source_domain_metrics(source_rows, "baseline", float("inf"), num_classes)
    baseline_domain_min_outer = min(float(row["outer_recall"]) for row in baseline_domains)
    for rule, threshold in candidate_grid(source_rows):
        source = evaluate(source_rows, rule, threshold, num_classes)
        target = evaluate(target_rows, rule, threshold, num_classes)
        domains = source_domain_metrics(source_rows, rule, threshold, num_classes)
        row = {
            "rule": rule,
            "threshold": threshold,
            "source_macro_f1": source["macro_f1"],
            "source_normal_recall": source["normal_recall"],
            "source_inner_recall": source["inner_recall"],
            "source_inner_precision": source["inner_precision"],
            "source_inner_predicted_support": source["inner_predicted_support"],
            "source_outer_recall": source["outer_recall"],
            "source_domain_min_outer_recall": min(float(item["outer_recall"]) for item in domains),
            "baseline_domain_min_outer_recall": baseline_domain_min_outer,
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
    selected_domains = source_domain_metrics(source_rows, str(selected["rule"]), float(selected["threshold"]), num_classes)

    write_csv(source_rows, args.output_dir / "source_mechanism_guard_recordings.csv")
    write_csv(target_rows, args.output_dir / "target_mechanism_guard_recordings.csv")
    write_csv(candidate_rows, args.output_dir / "mechanism_guarded_inner_candidates.csv")
    write_csv(baseline_domains, args.output_dir / "mechanism_guarded_baseline_domain_metrics.csv")
    write_csv(selected_domains, args.output_dir / "mechanism_guarded_selected_domain_metrics.csv")
    (args.output_dir / "mechanism_guarded_inner_selected.json").write_text(
        json.dumps(selected, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    best_target_macro = max(candidate_rows, key=lambda row: float(row["target_macro_f1_posthoc"]))
    best_target_inner = max(
        candidate_rows,
        key=lambda row: (float(row["target_inner_recall_posthoc"]), float(row["target_macro_f1_posthoc"])),
    )
    lines = [
        "# Ottawa Source-Only Mechanism-Guarded Inner Aggregator",
        "",
        "Scope: v5 core remains the default recording classifier. STFT/filterbank may override to inner only under double-view inner agreement and source-built mechanism guards. Target labels are post-hoc only.",
        "",
        f"Selected rule: `{selected['rule']}` threshold `{selected['threshold']}`.",
        "",
        f"Selection reason: {selected['selection_reason']}",
        "",
        "| Rule | Source F1 | Source Normal R | Source Inner R | Source Inner P | Source Outer R | Source Domain-Min Outer R | Target F1 | Target Normal R | Target Inner R | Target Inner P | Target Outer R | Target CM |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        row_summary("baseline", baseline),
        row_summary("selected", selected),
        row_summary("best_target_macro", best_target_macro),
        row_summary("best_target_inner", best_target_inner),
        "",
        "## Source Domain Check",
        "",
        "| Dataset | F1 | Normal R | Inner R | Inner P | Outer R | CM |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in selected_domains:
        lines.append(
            f"| {row['dataset_id']} | {float(row['macro_f1']):.6f} | {float(row['normal_recall']):.6f} | "
            f"{float(row['inner_recall']):.6f} | {float(row['inner_precision']):.6f} | "
            f"{float(row['outer_recall']):.6f} | `{row['confusion_matrix']}` |"
        )
    lines.extend(
        [
            "",
            "## Adversarial Verdict",
            "",
            "- This selector is target-free: source validation builds both mechanism prototypes and safety constraints.",
            "- If the selected rule remains baseline, the current mechanism features cannot certify safe inner override.",
            "- If post-hoc target gains require source outer degradation, the rule is diagnostic rather than publishable.",
            "- A publishable CIC-MAN component must improve Ottawa inner while preserving source domain-wise outer safety.",
        ]
    )
    report = args.output_dir / "mechanism_guarded_inner_aggregator_audit.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"selected": selected, "report": str(report)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
