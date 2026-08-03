#!/usr/bin/env python3
"""Blueprint statistics: extend the main matrix to 5 seeds (add 7, 123)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PY = sys.executable
HERE = Path(__file__).resolve().parent
MODELS = "cicman_v6ic,cicman_v4,single_env_order,single_raw,moe,ensemble"

for seed in ["7", "123"]:
    cmd = [PY, "-u", str(HERE / "run_v2_pilot.py"), "--project-root", str(HERE.parents[1]),
           "--epochs", "40", "--seed", seed, "--models", MODELS, "--isolate"]
    print(f"##### five-seed batch seed {seed}", flush=True)
    ret = subprocess.call(cmd)
    print(f"##### seed {seed} exit={ret}", flush=True)
print("##### FIVE-SEED BATCH DONE", flush=True)
