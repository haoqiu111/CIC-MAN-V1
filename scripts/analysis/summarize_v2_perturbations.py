#!/usr/bin/env python3
"""Aggregate experiment-5 perturbation JSONs into a paper-ready table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MODELS = ["cicman_v6ic", "cicman_v4", "single_env_order", "single_raw"]
TARGETS = ["hust", "ottawa", "paderborn"]
PERTS = ["clean", "gauss_snr10", "gauss_snr0", "impulse", "harmonic", "scale_0.5", "scale_2.0", "speed_jitter_3pct"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()

    pert_dir = args.project_root / "outputs/tables/v2_perturbations"
    out = args.project_root / "outputs/tables/v2_perturbation_summary.md"

    with out.open("w", encoding="utf-8") as f:
        f.write("# Experiment 5: Controlled Measurement-Intervention Robustness\n\n")
        f.write("Recording macro-F1 on the target domain (seed 42, last-epoch checkpoints,\n")
        f.write("perturbations injected into the time-domain signal, all views recomputed,\n")
        f.write("<=150 target recordings per task).\n\n")
        for target in TARGETS:
            f.write(f"## Target: {target}\n\n")
            f.write("| Model | " + " | ".join(PERTS) + " | Worst | Mean drop |\n")
            f.write("|---|" + "---:|" * (len(PERTS) + 2) + "\n")
            for model in MODELS:
                p = pert_dir / f"{model}_{target}_seed42.json"
                if not p.exists():
                    continue
                d = json.loads(p.read_text(encoding="utf-8"))
                vals = [d[k]["recording_macro_f1"] for k in PERTS if k in d]
                clean = d["clean"]["recording_macro_f1"]
                worst = min(vals)
                drops = [clean - d[k]["recording_macro_f1"] for k in PERTS[1:] if k in d]
                f.write(f"| {model} | " + " | ".join(f"{d[k]['recording_macro_f1']:.3f}" for k in PERTS if k in d)
                        + f" | {worst:.3f} | {sum(drops)/len(drops):+.3f} |\n")
            f.write("\n")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
