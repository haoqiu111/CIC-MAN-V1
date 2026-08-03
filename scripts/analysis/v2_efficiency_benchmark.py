#!/usr/bin/env python3
"""Efficiency table for CIC-MAN v2: params and inference latency, per view and full."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def add_src_to_path() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


add_src_to_path()

from cicman.v2.data import ALL_VIEWS  # noqa: E402
from cicman.v2.model import CICMANv2  # noqa: E402

SHAPES = {"raw": (4096,), "denoise": (4096,), "env_spec": (256,), "env_order": (256,), "stft": (64, 32), "cwt": (32, 32)}


def bench(model, views, device, batch: int, iters: int = 50):
    import torch

    x = {v: torch.randn((batch,) + SHAPES[v], device=device) for v in views}
    feats = torch.randn(batch, 16, device=device)
    with torch.no_grad():
        for _ in range(10):
            model(x, feats)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            model(x, feats)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000  # ms per batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for name, views in [("CIC-MAN (6 views)", ALL_VIEWS)] + [(f"single_{v}", [v]) for v in ALL_VIEWS]:
        model = CICMANv2(num_classes=3, views=views, num_domains=2, router_mode="causal" if len(views) > 1 else "uniform",
                         router_prior=[1.0 / len(views)] * len(views) if len(views) > 1 else None).to(device).eval()
        params = sum(p.numel() for p in model.parameters())
        lat1 = bench(model, views, device, 1)
        lat256 = bench(model, views, device, 256)
        rows.append({"model": name, "params_M": round(params / 1e6, 3),
                     "latency_b1_ms": round(lat1, 2), "latency_b256_ms": round(lat256, 2),
                     "throughput_win_per_s": int(256 / (lat256 / 1000))})
        print(rows[-1], flush=True)

    out = args.project_root / "outputs/tables/v2_efficiency.md"
    with out.open("w", encoding="utf-8") as f:
        f.write(f"# v2 Efficiency ({torch.cuda.get_device_name(0) if device.type=='cuda' else 'cpu'})\n\n")
        f.write("| Model | Params (M) | Latency b=1 (ms) | Latency b=256 (ms) | Windows/s |\n|---|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(f"| {r['model']} | {r['params_M']} | {r['latency_b1_ms']} | {r['latency_b256_ms']} | {r['throughput_win_per_s']} |\n")
    (args.project_root / "outputs/tables/v2_efficiency.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
