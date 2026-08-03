#!/usr/bin/env python3
"""Reconcile legacy soft metrics that do not reproduce from frozen checkpoints.

The historical ``metrics.json`` files are kept immutable.  This script repeats
the original DataLoader(batch=256) evaluation path in the canonical current
environment, records hashes and both aggregation results, and writes a
machine-checkable supersession record for every audit discrepancy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add_paths(root: Path) -> None:
    sys.path[:0] = [
        str(root),
        str(root / "src"),
        str(root / "scripts/analysis"),
    ]


def evaluate_both(model, loader, dataset, device, num_classes: int, views: list[str]):
    import torch

    from recording_protocol import OFFICIAL_AGGREGATION, RecordingAccumulator

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    recording = RecordingAccumulator(num_classes)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            view_batch = {view: batch["views"][view].to(device) for view in views}
            feats = batch["feats"].to(device)
            labels = batch["y"].cpu().numpy()
            probs = torch.softmax(model(view_batch, feats)["logits"], dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            for true, pred in zip(labels, preds):
                cm[int(true), int(pred)] += 1
            for offset, dataset_index in enumerate(batch["index"].cpu().numpy()):
                row = dataset.rows[int(dataset_index)]
                recording.add_many(row["recording_id"], int(labels[offset]), probs[offset : offset + 1])
    soft = recording.metrics("probability_sum")
    official = recording.metrics(OFFICIAL_AGGREGATION)
    return {
        "window_confusion_matrix": cm.tolist(),
        "probability_sum_recording_macro_f1": soft["recording_macro_f1"],
        "probability_sum_confusion_matrix": soft["recording_confusion_matrix"],
        "official_recording_macro_f1": official["recording_macro_f1"],
        "official_confusion_matrix": official["recording_confusion_matrix"],
        "official_vote_ties": official["vote_ties"],
        "official_probability_tiebreaks": official["probability_tiebreaks"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--tolerance", type=float, default=0.001)
    parser.add_argument("--repeat", type=int, default=2)
    args = parser.parse_args()
    root = args.project_root.resolve()
    add_paths(root)

    import torch
    from torch.utils.data import DataLoader

    from audit_recording_aggregation import load_checkpoint_model
    from cicman.v2.data import CachedWindowDataset, ViewCache
    from recording_protocol import OFFICIAL_AGGREGATION, PROTOCOL_VERSION

    tables = root / "outputs/tables"
    audit = pd.read_csv(tables / "recording_aggregation_audit.csv")
    cases = audit[audit["soft_reproduction_abs_error"].fillna(0.0) > args.tolerance].copy()
    entries = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = root / "data/paper1_cicman/cache/views_v2"

    for row in cases.itertuples(index=False):
        run_dir = (
            root
            / "outputs/checkpoints"
            / f"v2_{row.model}_target_{row.target}_seed{row.seed}"
        )
        checkpoint_path = run_dir / f"{row.epoch_rule}.pt"
        index_path = (
            root
            / "data/paper1_cicman/cache/windows/cross_dataset_task3_source_mixed"
            / f"target_dataset_{row.target}/test_windows.csv"
        )
        metrics_path = run_dir / "metrics.json"
        model, views, num_classes = load_checkpoint_model(checkpoint_path, device, row.model)
        dataset = CachedWindowDataset(index_path, ViewCache(cache_dir, views=views))
        loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0, pin_memory=False)
        repeats = [evaluate_both(model, loader, dataset, device, num_classes, views) for _ in range(args.repeat)]
        soft_values = [item["probability_sum_recording_macro_f1"] for item in repeats]
        repeat_stable = max(soft_values) - min(soft_values) <= 1e-12
        canonical_soft = soft_values[0]
        audit_matches = abs(canonical_soft - float(row.soft_recording_macro_f1)) <= 1e-12
        status = (
            "historical_metric_not_reproduced_in_canonical_environment"
            if repeat_stable and audit_matches
            else "unresolved_inference_path_mismatch"
        )
        entries.append(
            {
                "model": row.model,
                "target": row.target,
                "seed": int(row.seed),
                "epoch_rule": row.epoch_rule,
                "status": status,
                "accepted_exception": status == "historical_metric_not_reproduced_in_canonical_environment",
                "historical_metrics_path": str(metrics_path),
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": sha256(checkpoint_path),
                "test_index_sha256": sha256(index_path),
                "view_cache_master_sha256": sha256(cache_dir / "master.csv"),
                "historical_probability_sum_macro_f1": float(row.stored_soft_recording_macro_f1),
                "audit_direct_probability_sum_macro_f1": float(row.soft_recording_macro_f1),
                "canonical_dataloader_probability_sum_macro_f1": canonical_soft,
                "canonical_official_macro_f1": repeats[0]["official_recording_macro_f1"],
                "repeat_count": args.repeat,
                "repeat_stable": repeat_stable,
                "audit_matches_canonical_dataloader": audit_matches,
                "canonical_result": repeats[0],
            }
        )
        print(f"{row.model}/{row.target}/seed{row.seed}/{row.epoch_rule}: {status}", flush=True)

    payload = {
        "recording_protocol_version": PROTOCOL_VERSION,
        "official_aggregation": OFFICIAL_AGGREGATION,
        "canonical_inference_path": "DataLoader(batch_size=256, shuffle=false, num_workers=0)",
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
        "entries": entries,
    }
    output = tables / "metric_reconciliation.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"-> {output}")
    if any(not entry["accepted_exception"] for entry in entries):
        raise SystemExit("one or more metric discrepancies remain unresolved")


if __name__ == "__main__":
    main()
