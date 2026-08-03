#!/usr/bin/env python3
"""Experiment 5 batch: perturbation robustness for the final method + baselines."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PY = sys.executable
HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
WINDOWS = ROOT / "data/paper1_cicman/cache/windows/cross_dataset_task3_source_mixed"
CKPT = ROOT / "outputs/checkpoints"

JOBS = []
for target in ["hust", "ottawa", "paderborn"]:
    for model in ["cicman_v6ic", "cicman_v4", "single_env_order", "single_raw"]:
        JOBS.append((target, model))

for target, model in JOBS:
    out_name = f"{model}_{target}_seed42"
    out_file = ROOT / f"outputs/tables/v2_perturbations/{out_name}.json"
    if out_file.exists():
        print(f"skip {out_name}", flush=True)
        continue
    cmd = [PY, "-u", str(HERE / "eval_v2_perturbations.py"),
           "--project-root", str(ROOT),
           "--task-dir", str(WINDOWS / f"target_dataset_{target}"),
           "--checkpoint", str(CKPT / f"v2_{model}_target_{target}_seed42"),
           "--which", "last", "--max-recordings", "150",
           "--output-name", out_name]
    print(f"##### perturbation {out_name}", flush=True)
    ret = subprocess.call(cmd)
    print(f"##### {out_name} exit={ret}", flush=True)
print("##### PERTURBATION BATCH DONE", flush=True)
