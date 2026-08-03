#!/usr/bin/env python3
"""Matched CIC-MAN v6ic ablations: 5 variants x 3 targets x 3 seeds."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PY = sys.executable
HERE = Path(__file__).resolve().parent
MODELS = ",".join([
    "v6ic_no_prior",
    "v6ic_no_consensus",
    "v6ic_no_disentangle",
    "v6ic_no_view_dropout",
    "v6ic_router_no_evidence",
])

for seed in (42, 2025, 2026):
    cmd = [
        PY, "-u", str(HERE / "run_v2_pilot.py"),
        "--project-root", str(HERE.parents[1]),
        "--epochs", "40",
        "--seed", str(seed),
        "--models", MODELS,
        "--isolate",
    ]
    print(f"##### matched v6ic ablation seed {seed}", flush=True)
    ret = subprocess.call(cmd)
    if ret != 0:
        raise SystemExit(ret)
print("##### MATCHED V6IC ABLATION BATCH DONE", flush=True)
