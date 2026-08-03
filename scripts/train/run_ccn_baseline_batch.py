#!/usr/bin/env python3
"""Equal-backbone CCN reimplementation under the five-seed LODO protocol."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PY = sys.executable
HERE = Path(__file__).resolve().parent

for seed in (42, 2025, 2026, 7, 123):
    cmd = [
        PY, "-u", str(HERE / "run_v2_pilot.py"),
        "--project-root", str(HERE.parents[1]),
        "--epochs", "40",
        "--seed", str(seed),
        "--models", "dg_ccn",
        "--isolate",
    ]
    print(f"##### CCN baseline seed {seed}", flush=True)
    ret = subprocess.call(cmd)
    if ret != 0:
        raise SystemExit(ret)
print("##### CCN BASELINE BATCH DONE", flush=True)
