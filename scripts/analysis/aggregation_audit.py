#!/usr/bin/env python3
"""Epoch x aggregation sensitivity audit (metric-caliber unification).

For every cross-dataset checkpoint v2_{model}_target_{t}_seed{s}, re-runs test
inference ONCE per checkpoint file and scores recording-level macro-F1 under
all four calibers:

    {best.pt, last.pt} x {probability-sum, majority-vote}

Majority vote uses per-window argmax counts with a probability-sum tiebreak
(deterministic). Existing metrics.json files are never touched.

Outputs
  outputs/tables/aggregation_audit.csv    raw grid
  outputs/tables/aggregation_audit.md     mean-over-targets and strict
                                          worst-target (per-seed min) summary
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

P1 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(P1 / "src"))
from cicman.v2.data import CachedWindowDataset, ViewCache  # noqa: E402
from cicman.v2.model import CICMANv2  # noqa: E402
from cicman.v2.train import macro_f1_from_confusion  # noqa: E402

CKPT = P1 / "outputs" / "checkpoints"
TABLES = P1 / "outputs" / "tables"
WINDOWS = Path(__file__).resolve().parents[2] / "data/paper1_cicman/cache/windows/cross_dataset_task3_source_mixed"
CACHE = Path(__file__).resolve().parents[2] / "data/paper1_cicman/cache/views_v2"

MODELS = ["cicman_v6ic", "cicman_v4", "single_env_order", "single_raw", "ensemble",
          "moe", "dann", "dg_coral", "dg_mmd", "dg_irm", "dg_groupdro"]
TARGETS = ["hust", "ottawa", "paderborn"]
SEEDS = [42, 2025, 2026, 7, 123]
NUM_CLASSES = 3


def load_model(run_dir: Path, which: str, device):
    ckpt = torch.load(run_dir / f"{which}.pt", map_location="cpu", weights_only=True)
    views = ckpt["views"]
    state = ckpt["model"]
    has_prior = "router_log_prior" in state
    num_domains = state["domain_adv_head.2.weight"].shape[0]
    # router_mode affects forward only via prior/evidence branches; recover it
    router_mode = "causal" if has_prior else (
        "uniform" if len(views) == 1 or "router.0.weight" not in state or True else "causal")
    # uniform-router models (ensemble/dg/dann/single) still carry router params;
    # replicate training-time behavior: prior => causal, else uniform for
    # single/ensemble/dg/dann, feats_only for moe
    m = CICMANv2(num_classes=NUM_CLASSES, views=views, num_domains=num_domains,
                 router_mode="causal" if has_prior else "uniform",
                 router_prior=[1.0 / len(views)] * len(views) if has_prior else None)
    missing, unexpected = m.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"    state_dict diff: missing={missing} unexpected={unexpected}")
    return m.to(device).eval(), views


def load_model_moe(run_dir: Path, which: str, device):
    ckpt = torch.load(run_dir / f"{which}.pt", map_location="cpu", weights_only=True)
    views, state = ckpt["views"], ckpt["model"]
    num_domains = state["domain_adv_head.2.weight"].shape[0]
    m = CICMANv2(num_classes=NUM_CLASSES, views=views, num_domains=num_domains,
                 router_mode="feats_only")
    m.load_state_dict(state, strict=False)
    return m.to(device).eval(), views


@torch.no_grad()
def infer(model, views, target, device, batch=512):
    cache = ViewCache(CACHE, views=views)
    ds = CachedWindowDataset(WINDOWS / f"target_dataset_{target}" / "test_windows.csv", cache)
    rec_probs: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(NUM_CLASSES))
    rec_votes: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(NUM_CLASSES))
    rec_label: dict[str, int] = {}
    for s in range(0, len(ds), batch):
        items = [ds[i] for i in range(s, min(s + batch, len(ds)))]
        x = {v: torch.stack([it["views"][v] for it in items]).to(device) for v in views}
        f = torch.stack([it["feats"] for it in items]).to(device)
        probs = torch.softmax(model(x, f)["logits"], dim=1).cpu().numpy()
        preds = probs.argmax(1)
        for j, i in enumerate(range(s, min(s + batch, len(ds)))):
            row = ds.rows[i]
            rid = f"{row['dataset_id']}::{row['recording_id']}"
            rec_probs[rid] += probs[j]
            rec_votes[rid][preds[j]] += 1
            rec_label[rid] = row["label"]

    def score(agg: str) -> float:
        cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        for rid in rec_label:
            if agg == "prob_sum":
                pred = int(np.argmax(rec_probs[rid]))
            else:  # majority with prob-sum tiebreak
                votes = rec_votes[rid]
                top = np.flatnonzero(votes == votes.max())
                pred = int(top[np.argmax(rec_probs[rid][top])]) if len(top) > 1 else int(top[0])
            cm[rec_label[rid], pred] += 1
        return macro_f1_from_confusion(cm)

    return score("prob_sum"), score("majority")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    out_csv = TABLES / "aggregation_audit.csv"
    done = set()
    rows = []
    if out_csv.exists() and not args.force:
        with out_csv.open(newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                rows.append(r)
                done.add((r["model"], r["target"], r["seed"], r["which"]))

    for model_name in args.models.split(","):
        for target in TARGETS:
            for seed in SEEDS:
                run_dir = CKPT / f"v2_{model_name}_target_{target}_seed{seed}"
                if not (run_dir / "last.pt").exists():
                    print(f"missing: {run_dir.name}")
                    continue
                for which in ("best", "last"):
                    key = (model_name, target, str(seed), which)
                    if key in done:
                        continue
                    loader = load_model_moe if model_name == "moe" else load_model
                    m, views = loader(run_dir, which, device)
                    ps, mv = infer(m, views, target, device)
                    rows.append({"model": model_name, "target": target, "seed": seed,
                                 "which": which, "prob_sum": round(ps, 5),
                                 "majority": round(mv, 5)})
                    print(f"{model_name} {target} s{seed} {which}: "
                          f"prob={ps:.4f} maj={mv:.4f}", flush=True)
                    del m
                    torch.cuda.empty_cache()
                    with out_csv.open("w", newline="", encoding="utf-8") as fh:
                        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                        w.writeheader(); w.writerows(rows)

    # ---- summary ----
    import pandas as pd
    df = pd.read_csv(out_csv)
    md = TABLES / "aggregation_audit.md"
    with md.open("w", encoding="utf-8") as fh:
        fh.write("# Epoch x aggregation sensitivity audit\n\n")
        fh.write("Recording macro-F1; mean-over-targets and strict worst-target\n")
        fh.write("(per-seed min over targets), aggregated over 5 seeds.\n\n")
        for which in ("last", "best"):
            for agg in ("majority", "prob_sum"):
                sub = df[df.which == which]
                per_seed_mean = sub.groupby(["model", "seed"])[agg].mean()
                per_seed_min = sub.groupby(["model", "seed"])[agg].min()
                mean_tbl = per_seed_mean.groupby("model").agg(["mean", "std"]).round(4)
                worst_tbl = per_seed_min.groupby("model").agg(["mean", "std"]).round(4)
                fh.write(f"## {which}-epoch x {agg}\n\n")
                fh.write("| Model | Mean | Worst-target |\n|---|---|---|\n")
                for mname in mean_tbl.sort_values("mean", ascending=False).index:
                    fh.write(f"| {mname} | {mean_tbl.loc[mname,'mean']:.4f} +- "
                             f"{mean_tbl.loc[mname,'std']:.4f} | "
                             f"{worst_tbl.loc[mname,'mean']:.4f} +- "
                             f"{worst_tbl.loc[mname,'std']:.4f} |\n")
                fh.write("\n")
    print("->", md)


if __name__ == "__main__":
    main()
