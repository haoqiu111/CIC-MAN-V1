#!/usr/bin/env python3
"""DEPRECATED (2026-07-10, metric-protocol unification).

This script reads legacy metrics.json files whose recording aggregation is
probability-sum — NOT the official majority_probability_tiebreak protocol.
The ablation table is now produced from the official re-inference audit
(outputs/tables/recording_aggregation_audit.csv). Running this script would
silently regenerate a soft-aggregation table and desynchronize the paper.

If you believe you need it, use recording_aggregation_audit.csv instead.
"""

from __future__ import annotations

import sys

sys.exit(
    "DEPRECATED: summarize_v2_ablation.py produces legacy probability-sum "
    "numbers. Use outputs/tables/recording_aggregation_audit.csv "
    "(official majority protocol) instead."
)

import argparse  # noqa: E402  (unreachable, kept for history)
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402

MODELS = [
    "cicman_v4",
    "v4_no_prior",
    "v4_no_consensus",
    "v4_no_disentangle",
    "v4_no_view_dropout",
    "v4_router_no_evidence",
]
TARGETS = ["hust", "ottawa", "paderborn"]
SEEDS = [42, 2025, 2026]

LABELS = {
    "cicman_v4": "A10 Full CIC-MAN",
    "v4_no_prior": "A4 w/o reliability prior",
    "v4_no_consensus": "A5 w/o consensus loss",
    "v4_no_disentangle": "A6 w/o disentanglement",
    "v4_no_view_dropout": "A7 w/o view dropout",
    "v4_router_no_evidence": "A8 w/o sample evidence",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()

    audit_path = args.project_root / "outputs/tables/recording_aggregation_audit.csv"
    audit = pd.read_csv(audit_path)
    df = audit[
        audit["model"].isin(MODELS)
        & audit["target"].isin(TARGETS)
        & audit["seed"].isin(SEEDS)
        & audit["epoch_rule"].eq("last")
    ][["model", "target", "seed", "majority_recording_macro_f1"]].rename(
        columns={"majority_recording_macro_f1": "rec_f1"}
    )
    expected = len(MODELS) * len(TARGETS) * len(SEEDS)
    if len(df) != expected:
        raise RuntimeError(f"official ablation audit incomplete: expected {expected} rows, found {len(df)}")
    per_target = df.pivot_table(index="model", columns="target", values="rec_f1", aggfunc="mean")
    per_seed_mean = df.groupby(["model", "seed"])["rec_f1"].mean().groupby("model").agg(["mean", "std"])

    out = args.project_root / "outputs/tables/v2_ablation_matrix.md"
    with out.open("w", encoding="utf-8") as f:
        f.write("# CIC-MAN v4 Ablation Matrix (3 seeds, official last-epoch majority vote, rec Macro-F1)\n\n")
        f.write("A1 single-view / A2 uniform ensemble / A3 plain MoE are covered by the main matrix baselines.\n\n")
        f.write("| Ablation | HUST | Ottawa | Paderborn | Mean +/- Std |\n|---|---:|---:|---:|---|\n")
        for model_name in MODELS:
            if model_name not in per_target.index:
                continue
            row = per_target.loc[model_name]
            summary = per_seed_mean.loc[model_name]
            f.write(
                f"| {LABELS[model_name]} | {row.get('hust', float('nan')):.3f} | "
                f"{row.get('ottawa', float('nan')):.3f} | {row.get('paderborn', float('nan')):.3f} | "
                f"{summary['mean']:.3f} +/- {summary['std']:.3f} |\n"
            )
    print(f"-> {out}")
    print(per_seed_mean.round(3).to_string())


if __name__ == "__main__":
    main()
