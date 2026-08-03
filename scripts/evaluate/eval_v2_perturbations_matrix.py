#!/usr/bin/env python3
"""Re-evaluate the four paper perturbation models with one shared view pass.

For a target dataset, every perturbed recording is transformed into the six
views once.  The resulting tensors are then scored by all requested existing
checkpoints.  This is numerically equivalent to four invocations of
``eval_v2_perturbations.py`` but avoids repeating the expensive CWT/STFT work.
No training or checkpoint mutation is performed.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def add_src_to_path() -> None:
    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(root))


add_src_to_path()

from cicman.v2 import views as V  # noqa: E402
from cicman.v2.model import CICMANv2  # noqa: E402
from eval_v2_perturbations import (  # noqa: E402
    macro_f1,
    perturbations,
    read_recording_signal,
    shaft_for_window,
)
from recording_protocol import (  # noqa: E402
    OFFICIAL_AGGREGATION,
    PROTOCOL_VERSION,
    RecordingAccumulator,
)

DEFAULT_MODELS = ("cicman_v6ic", "cicman_v4", "single_env_order", "single_raw")


def load_model(checkpoint_dir: Path, num_classes: int, device):
    import torch

    ckpt = torch.load(checkpoint_dir / "last.pt", map_location="cpu", weights_only=True)
    views = list(ckpt["views"])
    state = ckpt["model"]
    has_prior = "router_log_prior" in state
    model = CICMANv2(
        num_classes=num_classes,
        views=views,
        num_domains=int(state["domain_adv_head.2.weight"].shape[0]),
        router_mode="causal" if has_prior else "uniform",
        router_prior=[1.0 / len(views)] * len(views) if has_prior else None,
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"state_dict mismatch for {checkpoint_dir}: missing={missing}, unexpected={unexpected}")
    return model.to(device).eval(), views


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--target", choices=["hust", "ottawa", "paderborn"], required=True)
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--max-recordings", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import torch

    task_dir = (
        args.project_root
        / "data/paper1_cicman/cache/windows/cross_dataset_task3_source_mixed"
        / f"target_dataset_{args.target}"
    )
    checkpoint_root = args.project_root / "outputs/checkpoints"
    output_dir = args.project_root / "outputs/tables/v2_perturbations"
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted_root = args.project_root / "data/paper1_cicman/extracted/paderborn"

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    models = {}
    for name in [x.strip() for x in args.models.split(",") if x.strip()]:
        checkpoint_dir = checkpoint_root / f"v2_{name}_target_{args.target}_seed{args.seed}"
        model, views = load_model(checkpoint_dir, args.num_classes, device)
        models[name] = {"model": model, "views": views, "checkpoint": checkpoint_dir}

    by_rec: dict[str, list[dict]] = defaultdict(list)
    with (task_dir / "test_windows.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_rec[row["recording_id"]].append(row)
    rec_ids = sorted(by_rec)
    rng = np.random.default_rng(args.seed)
    if len(rec_ids) > args.max_recordings:
        rec_ids = sorted(rng.choice(rec_ids, size=args.max_recordings, replace=False).tolist())
    print(f"target={args.target} recordings={len(rec_ids)} models={list(models)}", flush=True)

    all_results = {name: {} for name in models}
    all_views = list(V.VIEW_SPECS)
    perts, _ = perturbations()
    for perturbation_index, (pert_name, pert_fn) in enumerate(perts.items()):
        cms = {name: np.zeros((args.num_classes, args.num_classes), dtype=np.int64) for name in models}
        recordings = {name: RecordingAccumulator(args.num_classes) for name in models}
        pert_rng = np.random.default_rng(args.seed + 1009 * perturbation_index)

        for rid in rec_ids:
            rows = by_rec[rid]
            sig, src_rate, shaft_const, enc = read_recording_signal(rows, extracted_root)
            sig_rs = V.robust_normalize(V.resample(sig, src_rate, V.TARGET_RATE))
            sig_p = pert_fn(sig_rs, pert_rng)
            label = int(rows[0]["label_id"])

            batch_views = {view: [] for view in all_views}
            batch_feats = []
            for row in rows:
                start = int(row["target_start"])
                if start + V.WINDOW_LEN > len(sig_p):
                    continue
                shaft, slope = shaft_for_window(start, src_rate, shaft_const, enc, len(sig_p))
                views, feats = V.compute_window_views(sig_p, start, shaft, slope)
                for view in all_views:
                    batch_views[view].append(views[view])
                batch_feats.append(feats)
            if not batch_feats:
                continue

            tensors = {
                view: torch.from_numpy(np.stack(values)).to(device)
                for view, values in batch_views.items()
            }
            feats_tensor = torch.from_numpy(np.stack(batch_feats)).to(device)
            with torch.no_grad():
                for name, spec in models.items():
                    out = spec["model"]({v: tensors[v] for v in spec["views"]}, feats_tensor)
                    probs = torch.softmax(out["logits"], dim=1).cpu().numpy()
                    preds = probs.argmax(axis=1)
                    for pred in preds:
                        cms[name][label, int(pred)] += 1
                    recordings[name].add_many(rid, label, probs)

        for name in models:
            rec_metrics = recordings[name].metrics(OFFICIAL_AGGREGATION)
            rec_cm = np.asarray(rec_metrics["recording_confusion_matrix"], dtype=np.int64)
            all_results[name][pert_name] = {
                "window_macro_f1": macro_f1(cms[name]),
                "window_accuracy": float(np.trace(cms[name]) / max(cms[name].sum(), 1)),
                "recording_macro_f1": macro_f1(rec_cm),
                "recording_accuracy": float(np.trace(rec_cm) / max(rec_cm.sum(), 1)),
                "num_recordings": int(rec_cm.sum()),
                "recording_aggregation": OFFICIAL_AGGREGATION,
                "recording_protocol_version": PROTOCOL_VERSION,
                "recording_tie_break": "cumulative_probability_then_lowest_class_id",
                "recording_vote_ties": int(rec_metrics["vote_ties"]),
                "recording_probability_tiebreaks": int(rec_metrics["probability_tiebreaks"]),
            }
        summary = ", ".join(f"{name}={all_results[name][pert_name]['recording_macro_f1']:.4f}" for name in models)
        print(f"{pert_name:18s} {summary}", flush=True)

    for name, results in all_results.items():
        output = output_dir / f"{name}_{args.target}_seed{args.seed}.json"
        output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"-> {output}", flush=True)


if __name__ == "__main__":
    main()
