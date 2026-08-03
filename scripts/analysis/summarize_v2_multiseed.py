#!/usr/bin/env python3
"""Aggregate v2 cross-dataset results under the frozen recording protocol.

When ``recording_aggregation_audit.csv`` is present, recording metrics come
from fresh checkpoint inference and are selected explicitly by
``--epoch`` x ``--aggregation``.  The default is the paper-wide official
protocol: last epoch plus true recording majority vote.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

MAIN_MODELS = [
    "single_raw", "dann", "ensemble", "moe", "dg_irm", "dg_coral", "dg_mmd",
    "dg_groupdro", "single_env_order", "cicman_v4", "cicman_v6ic",
]


def _pilot_frames(tables_dir: Path, seeds: list[str], prefix: str) -> pd.DataFrame:
    frames = []
    for seed in seeds:
        path = tables_dir / f"{prefix}_pilot_summary_seed{seed}.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError(f"No {prefix}_pilot_summary_seed*.csv files found in {tables_dir}")
    return pd.concat(frames, ignore_index=True)


def _official_frame(tables_dir: Path, seeds: list[str], prefix: str, epoch: str, aggregation: str) -> pd.DataFrame:
    audit_path = tables_dir / "recording_aggregation_audit.csv"
    if not audit_path.exists():
        raise FileNotFoundError(f"Missing fresh re-inference audit: {audit_path}")
    audit = pd.read_csv(audit_path)
    audit = audit[(audit["epoch_rule"] == epoch)].copy()
    value_col = f"{aggregation}_recording_macro_f1"
    if value_col not in audit.columns:
        raise ValueError(f"Unsupported aggregation {aggregation!r}; expected soft or majority")

    pilot = _pilot_frames(tables_dir, seeds, prefix)
    pilot["seed"] = pilot["seed"].astype(int)
    window_col = "window_macro_f1_last" if epoch == "last" else "window_macro_f1"
    window = pilot[["model", "target", "seed", window_col]].rename(columns={window_col: "window_macro_f1"})
    out = audit[["model", "target", "seed", value_col]].rename(columns={value_col: "recording_macro_f1"})
    out = out.merge(window, on=["model", "target", "seed"], how="left")
    out["recording_macro_f1"] = pd.to_numeric(out["recording_macro_f1"], errors="coerce")
    return out.dropna(subset=["recording_macro_f1"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-dir", type=Path, default=Path(__file__).resolve().parents[2] / "outputs/tables")
    parser.add_argument("--seeds", default="42,2025,2026,7,123")
    parser.add_argument("--prefix", default="v2")
    parser.add_argument("--epoch", choices=["best", "last"], default="last")
    parser.add_argument("--aggregation", choices=["majority", "soft"], default="majority")
    parser.add_argument("--models", default=",".join(MAIN_MODELS))
    args = parser.parse_args()
    seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]
    df = _official_frame(args.tables_dir, seeds, args.prefix, args.epoch, args.aggregation)
    allowed_models = {value.strip() for value in args.models.split(",") if value.strip()}
    df = df[df["model"].isin(allowed_models)].copy()

    grouped = df.groupby(["model", "target"])[["recording_macro_f1", "window_macro_f1"]].agg(["mean", "std", "count"])
    per_seed_mean = df.groupby(["model", "seed"])["recording_macro_f1"].mean()
    per_seed_worst = df.groupby(["model", "seed"])["recording_macro_f1"].min()
    per_model = per_seed_mean.groupby("model").agg(["mean", "std", "count"])
    worst = per_seed_worst.groupby("model").agg(["mean", "std", "count"])

    out = args.tables_dir / f"{args.prefix}_multiseed_main_matrix.md"
    with out.open("w", encoding="utf-8") as f:
        f.write(f"# {args.prefix} Cross-Dataset Main Matrix (official: {args.epoch}-epoch, {args.aggregation} vote)\n\n")
        f.write("3-class leave-one-dataset-out, strict target-free, recording-level macro-F1.\n")
        f.write("The recording metric is freshly re-inferred from checkpoints; mean and worst-target aggregate over seeds.\n\n")
        f.write("## Per target (mean +- sample std over available seeds)\n\n")
        f.write("| Model | Target | Rec Macro-F1 | Win Macro-F1 | Seeds |\n|---|---|---|---|---:|\n")
        for (model, target), row in grouped.sort_index().iterrows():
            rec = row[("recording_macro_f1", "mean")]
            rec_sd = row[("recording_macro_f1", "std")]
            win = row[("window_macro_f1", "mean")]
            win_sd = row[("window_macro_f1", "std")]
            win_txt = f"{win:.3f} +- {win_sd:.3f}" if pd.notna(win) else "-"
            count = int(row[("recording_macro_f1", "count")])
            f.write(f"| {model} | {target} | {rec:.3f} +- {rec_sd:.3f} | {win_txt} | {count} |\n")
        f.write("\n## Mean over targets and strict worst target (per seed, then aggregated)\n\n")
        f.write("| Model | Mean Rec Macro-F1 | Strict worst-target Rec Macro-F1 | Seeds |\n|---|---|---|---:|\n")
        for model in per_model.index:
            m, w = per_model.loc[model], worst.loc[model]
            f.write(f"| {model} | {m['mean']:.3f} +- {m['std']:.3f} | {w['mean']:.3f} +- {w['std']:.3f} | {int(m['count'])} |\n")
    print(f"-> {out}")
    print(per_model.to_string())


if __name__ == "__main__":
    main()
