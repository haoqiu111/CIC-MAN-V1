#!/usr/bin/env python3
"""Run cicman_v4 (consensus-averaged reliability prior) over 3 targets x 3 seeds."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PY = sys.executable
HERE = Path(__file__).resolve().parent

for seed in ["42", "2025", "2026"]:
    cmd = [PY, "-u", str(HERE / "run_v2_pilot.py"), "--project-root", str(HERE.parents[1]),
           "--epochs", "40", "--seed", seed, "--models", "cicman_v4"]
    print(f"##### v4 seed {seed}", flush=True)
    ret = subprocess.call(cmd)
    print(f"##### v4 seed {seed} exit={ret}", flush=True)
print("##### V4 BATCH DONE", flush=True)
