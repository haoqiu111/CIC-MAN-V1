#!/usr/bin/env python3
"""Equal-backbone DG baseline family: CORAL/MMD/IRM/GroupDRO x 3 targets x 3 seeds."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PY = sys.executable
HERE = Path(__file__).resolve().parent
MODELS = "dg_coral,dg_mmd,dg_irm,dg_groupdro"

for seed in ["42", "2025", "2026"]:
    cmd = [PY, "-u", str(HERE / "run_v2_pilot.py"), "--project-root", str(HERE.parents[1]),
           "--epochs", "40", "--seed", seed, "--models", MODELS, "--isolate"]
    print(f"##### dg-baseline seed {seed}", flush=True)
    ret = subprocess.call(cmd)
    print(f"##### seed {seed} exit={ret}", flush=True)
print("##### DG BASELINE BATCH DONE", flush=True)
