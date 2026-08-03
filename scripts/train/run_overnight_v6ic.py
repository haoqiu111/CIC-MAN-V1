#!/usr/bin/env python3
"""v6ic multi-seed completion: seeds 2025/2026 over 3 targets (isolated)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PY = sys.executable
HERE = Path(__file__).resolve().parent

for seed in ["2025", "2026"]:
    cmd = [PY, "-u", str(HERE / "run_v2_pilot.py"), "--project-root", str(HERE.parents[1]),
           "--epochs", "40", "--seed", seed, "--models", "cicman_v6ic", "--isolate"]
    print(f"##### v6ic seed {seed}", flush=True)
    ret = subprocess.call(cmd)
    print(f"##### v6ic seed {seed} exit={ret}", flush=True)
print("##### V6IC SEEDS DONE", flush=True)
