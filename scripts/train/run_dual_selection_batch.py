#!/usr/bin/env python3
"""P2 final: rerun the main matrix with dual selection recording (best/last epoch)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PY = sys.executable
HERE = Path(__file__).resolve().parent
MODELS = "cicman_v3,cicman_v4,moe,ensemble,single_env_order,single_raw"

for seed in ["42", "2025", "2026"]:
    cmd = [PY, "-u", str(HERE / "run_v2_pilot.py"), "--project-root", str(HERE.parents[1]),
           "--epochs", "40", "--seed", seed, "--models", MODELS, "--force"]
    print(f"##### dual-selection seed {seed}", flush=True)
    ret = subprocess.call(cmd)
    print(f"##### dual-selection seed {seed} exit={ret}", flush=True)
print("##### DUAL BATCH DONE", flush=True)
