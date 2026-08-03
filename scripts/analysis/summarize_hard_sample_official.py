#!/usr/bin/env python3
"""Summarize hard-sample control and focal variant under the official protocol."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables-dir", type=Path, default=Path(__file__).resolve().parents[2] / "outputs/tables")
    args = ap.parse_args()
    audit = pd.read_csv(args.tables_dir / "recording_aggregation_audit.csv")
    models = ["cicman_v6ic_t213", "cicman_v6ic_hard"]
    data = audit[
        audit["model"].isin(models)
        & audit["target"].eq("paderborn")
        & audit["seed"].isin([42, 2025, 2026])
        & audit["epoch_rule"].eq("last")
    ][["model", "seed", "majority_recording_macro_f1"]]
    if len(data) != 6:
        raise RuntimeError(f"hard-sample official audit incomplete: expected 6 rows, found {len(data)}")
    pivot = data.pivot(index="model", columns="seed", values="majority_recording_macro_f1")
    out = args.tables_dir / "v2_hard_sample_official.md"
    with out.open("w", encoding="utf-8") as handle:
        handle.write("# Hard-sample refinement (official last-epoch majority vote)\n\n")
        handle.write("| Configuration | seed 42 | seed 2025 | seed 2026 | Mean |\n|---|---:|---:|---:|---:|\n")
        for model, label in [("cicman_v6ic_t213", "CIC-MAN (IC), control"), ("cicman_v6ic_hard", "+ hard-sample reweighting")]:
            values = pivot.loc[model]
            handle.write(
                f"| {label} | {values[42]:.3f} | {values[2025]:.3f} | {values[2026]:.3f} | {values.mean():.3f} |\n"
            )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
