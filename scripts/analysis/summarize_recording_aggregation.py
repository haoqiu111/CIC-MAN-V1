#!/usr/bin/env python3
"""Write the four-caliber aggregation audit and metric-protocol report."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd


def fmt(mean: float, sd: float) -> str:
    return f"{mean:.3f} +- {sd:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables-dir", type=Path, default=Path(__file__).resolve().parents[2] / "outputs/tables")
    args = ap.parse_args()
    tables = args.tables_dir
    df = pd.read_csv(tables / "recording_aggregation_audit.csv")
    key = ["model", "target", "seed", "epoch_rule"]
    if df.duplicated(key).any():
        raise ValueError("duplicate audit keys found")
    main_models = {
        "cicman_v6ic", "cicman_v4", "single_env_order", "single_raw", "ensemble",
        "moe", "dann", "dg_coral", "dg_mmd", "dg_irm", "dg_groupdro",
    }
    for model in main_models:
        if len(df[df.model == model]) != 30:
            raise ValueError(f"official five-seed grid incomplete for {model}")

    lines = [
        "# Epoch x recording-aggregation sensitivity audit",
        "",
        "All values are recording-level macro-F1 from fresh checkpoint inference.",
        "Majority vote counts per-window argmax labels; vote ties are resolved by cumulative probability among tied classes, then by the smallest class id.",
        "Probability-sum sensitivity sums per-window class probabilities before argmax.",
        "Mean and strict worst-target values are computed per seed, then averaged over seeds.",
        "",
    ]
    for epoch in ["best", "last"]:
        for agg, col in [("probability-sum", "soft_recording_macro_f1"), ("majority-vote", "majority_recording_macro_f1")]:
            x = df[(df.epoch_rule == epoch) & df.model.isin(main_models)]
            per_seed_mean = x.groupby(["model", "seed"])[col].mean()
            per_seed_worst = x.groupby(["model", "seed"])[col].min()
            mean = per_seed_mean.groupby("model").agg(["mean", "std"])
            worst = per_seed_worst.groupby("model").agg(["mean", "std"])
            lines += [f"## {epoch}-epoch x {agg}", "", "| Model | Mean over targets | Strict worst target |", "|---|---:|---:|"]
            for model in mean.sort_values("mean", ascending=False).index:
                lines.append(f"| {model} | {fmt(mean.loc[model, 'mean'], mean.loc[model, 'std'])} | {fmt(worst.loc[model, 'mean'], worst.loc[model, 'std'])} |" )
            lines.append("")

    err = df["soft_reproduction_abs_error"].fillna(0.0)
    discrepancy_rows = df[err > 0.001]
    reconciliation_path = tables / "metric_reconciliation.json"
    reconciled = set()
    if reconciliation_path.exists():
        payload = json.loads(reconciliation_path.read_text(encoding="utf-8"))
        reconciled = {
            (entry["model"], entry["target"], int(entry["seed"]), entry["epoch_rule"])
            for entry in payload.get("entries", [])
            if entry.get("accepted_exception")
        }
    root = tables.parents[2]
    master_path = root / "data/paper1_cicman/cache/views_v2/master.csv"
    index_path = root / "data/paper1_cicman/cache/windows/cross_dataset_task3_source_mixed/target_dataset_paderborn/test_windows.csv"
    with master_path.open(newline="", encoding="utf-8") as stream:
        cache_keys = {
            (row["dataset_id"], row["recording_id"], int(row["window_index"]))
            for row in csv.DictReader(stream)
        }
    with index_path.open(newline="", encoding="utf-8") as stream:
        paderborn_missing = sum(
            (row["dataset_id"], row["recording_id"], int(row["window_index"])) not in cache_keys
            for row in csv.DictReader(stream)
        )
    lines += [
        "## Integrity checks",
        "",
        f"- Rows: {len(df)}; duplicate keys: {int(df.duplicated(key).sum())}.",
        f"- Soft re-inference reproduction error: max={err.max():.6f}; rows > 0.001={int((err > 0.001).sum())}.",
        *[
            f"- Legacy soft discrepancy: `{row.model} / {row.target} / seed{row.seed} / {row.epoch_rule}` "
            f"(abs error={row.soft_reproduction_abs_error:.6f}); "
            + ("covered by the hash-bound canonical-environment reconciliation record."
               if (row.model, row.target, int(row.seed), row.epoch_rule) in reconciled
               else "NOT YET RECONCILED.")
            for row in discrepancy_rows.itertuples(index=False)
        ],
        f"- Canonical Paderborn test-index rows absent from the view cache after alignment: {paderborn_missing}.",
        "",
        "## Frozen official protocol",
        "",
        "Primary recording metric: **last-epoch checkpoint + true majority vote**. Probability-sum is retained as a sensitivity analysis. Window-level metrics are reported separately and are not used to define the recording-level primary result.",
        "",
    ]
    (tables / "recording_aggregation_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {tables / 'recording_aggregation_audit.md'}")


if __name__ == "__main__":
    main()
