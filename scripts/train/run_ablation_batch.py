#!/usr/bin/env python3
"""P4: ablations of the final cicman_v4 over 3 targets, seeds 42/2025/2026."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PY = sys.executable
HERE = Path(__file__).resolve().parent
MODELS = "v4_no_prior,v4_no_consensus,v4_no_disentangle,v4_no_view_dropout,v4_router_no_evidence"

for seed in ["42", "2025", "2026"]:
    cmd = [PY, "-u", str(HERE / "run_v2_pilot.py"), "--project-root", str(HERE.parents[1]),
           "--epochs", "40", "--seed", seed, "--models", MODELS]
    print(f"##### ablation seed {seed}", flush=True)
    ret = subprocess.call(cmd)
    print(f"##### ablation seed {seed} exit={ret}", flush=True)
print("##### ABLATION BATCH DONE", flush=True)
