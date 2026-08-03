#!/usr/bin/env python3
"""Repair pass: redo dual-selection runs whose metrics.json was cleared."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PY = sys.executable
HERE = Path(__file__).resolve().parent
MODELS = "cicman_v3,cicman_v4,moe,ensemble,single_env_order,single_raw"

for seed in ["42", "2025", "2026"]:
    cmd = [PY, "-u", str(HERE / "run_v2_pilot.py"), "--project-root", str(HERE.parents[1]),
           "--epochs", "40", "--seed", seed, "--models", MODELS]
    print(f"##### repair seed {seed}", flush=True)
    ret = subprocess.call(cmd)
    print(f"##### repair seed {seed} exit={ret}", flush=True)
print("##### REPAIR DONE", flush=True)
