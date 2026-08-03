#!/usr/bin/env python3
"""Overnight batch: within-dataset protocols, then multi-seed cross-dataset repeats."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PY = sys.executable
HERE = Path(__file__).resolve().parent
ROOT = str(HERE.parents[1])

STEPS = [
    # P3: within-dataset DG protocols (seed 42)
    [PY, "-u", str(HERE / "run_v2_within.py"), "--project-root", ROOT, "--epochs", "40", "--seed", "42"],
    # P2: repeated seeds for the cross-dataset main matrix
    [PY, "-u", str(HERE / "run_v2_pilot.py"), "--project-root", ROOT, "--epochs", "40", "--seed", "2025",
     "--models", "cicman_v3,moe,ensemble,single_env_order,single_raw"],
    [PY, "-u", str(HERE / "run_v2_pilot.py"), "--project-root", ROOT, "--epochs", "40", "--seed", "2026",
     "--models", "cicman_v3,moe,ensemble,single_env_order,single_raw"],
]


def main() -> None:
    for i, cmd in enumerate(STEPS, 1):
        print(f"##### STEP {i}/{len(STEPS)}: {' '.join(cmd[2:])}", flush=True)
        ret = subprocess.call(cmd)
        print(f"##### STEP {i} exit={ret}", flush=True)
    print("##### BATCH DONE", flush=True)


if __name__ == "__main__":
    main()
