#!/usr/bin/env python3
"""DG baselines: extra seeds 7/123 to reach 15 (target, seed) pairs for stats."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PY = sys.executable
HERE = Path(__file__).resolve().parent
MODELS = "dg_coral,dg_mmd,dg_irm,dg_groupdro"

for seed in ["7", "123"]:
    cmd = [PY, "-u", str(HERE / "run_v2_pilot.py"), "--project-root", str(HERE.parents[1]),
           "--epochs", "40", "--seed", seed, "--models", MODELS, "--isolate"]
    print(f"##### dg-extra seed {seed}", flush=True)
    ret = subprocess.call(cmd)
    print(f"##### seed {seed} exit={ret}", flush=True)
print("##### DG EXTRA SEEDS DONE", flush=True)
