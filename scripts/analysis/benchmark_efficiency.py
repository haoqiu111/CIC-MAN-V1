#!/usr/bin/env python3
"""Benchmark model size and inference speed for paper-ready cost tables."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path


def add_src_to_path() -> None:
    script_path = Path(__file__).resolve()
    project_dir = script_path.parents[2]
    sys.path.insert(0, str(project_dir / "src"))


add_src_to_path()

from cicman.models.cic_man import build_cic_man  # noqa: E402
from cicman.models.cic_man_gated_filterbank import build_cic_man_gated_filterbank  # noqa: E402
from cicman.models.cic_man_heterogeneous import build_cic_man_heterogeneous  # noqa: E402
from cicman.models.raw_cnn import build_raw_cnn  # noqa: E402


DEFAULT_MODELS = [
    ("RawCNN", "raw_cnn", "raw_cnn_cross_dataset_task3_target_ottawa"),
    ("CIC-MAN-v1", "cic_man", "cic_man_v1_cross_dataset_task3_target_ottawa"),
    (
        "CIC-MAN-v5-class-router-style",
        "cic_man",
        "cic_man_v5_class_router_style_source_mixed_cross_dataset_task3_target_ottawa",
    ),
    (
        "CIC-MAN-heterogeneous",
        "cic_man",
        "cic_man_heterogeneous_source_mixed_cross_dataset_task3_target_ottawa",
    ),
    (
        "CIC-MAN-heterogeneous-v4-filterbank",
        "cic_man",
        "cic_man_heterogeneous_v4_filterbank_source_mixed_cross_dataset_task3_target_hust",
    ),
    (
        "CIC-MAN-gated-filterbank",
        "cic_man",
        "cic_man_gated_filterbank_source_mixed_cross_dataset_task3_target_hust",
    ),
    (
        "CIC-MAN-gated-filterbank-frozen-core",
        "cic_man",
        "cic_man_gated_filterbank_frozen_core_source_mixed_cross_dataset_task3_target_hust",
    ),
    (
        "CIC-MAN-gated-filterbank-calibrated",
        "cic_man",
        "cic_man_gated_filterbank_calibrated_source_mixed_cross_dataset_task3_target_hust",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--input-length", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--steps", type=int, default=80)
    return parser.parse_args()


def count_params(model) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


def build_model(model_name: str, checkpoint: dict[str, object]):
    num_classes = int(checkpoint["num_classes"])
    if model_name == "raw_cnn":
        return build_raw_cnn(num_classes=num_classes)
    config = checkpoint.get("config", {})
    num_agents = int(checkpoint.get("num_agents", config.get("num_agents", 4)))
    architecture = str(config.get("architecture", "minimal"))
    if architecture == "gated_filterbank":
        return build_cic_man_gated_filterbank(num_classes=num_classes, core_agents=max(1, num_agents - 1))
    if architecture == "heterogeneous":
        return build_cic_man_heterogeneous(num_classes=num_classes, num_agents=num_agents)
    return build_cic_man(num_classes=num_classes, num_agents=num_agents)


def benchmark(model, *, device: str, batch_size: int, input_length: int, warmup: int, steps: int) -> dict[str, float]:
    import torch

    model.eval()
    x = torch.randn(batch_size, 1, input_length, device=device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(steps):
            _ = model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    total_samples = batch_size * steps
    return {
        "batch_latency_ms": elapsed / steps * 1000.0,
        "sample_latency_ms": elapsed / total_samples * 1000.0,
        "throughput_samples_per_s": total_samples / elapsed if elapsed > 0 else 0.0,
    }


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    fieldnames = [
        "method",
        "model",
        "checkpoint",
        "device",
        "batch_size",
        "input_length",
        "num_parameters",
        "trainable_parameters",
        "batch_latency_ms",
        "sample_latency_ms",
        "throughput_samples_per_s",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, object]], path: Path) -> None:
    lines = [
        "# Efficiency and Complexity Summary",
        "",
        "| Method | Params | Batch Latency ms | Sample Latency ms | Throughput samples/s |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {int(row['num_parameters'])} | "
            f"{float(row['batch_latency_ms']):.3f} | {float(row['sample_latency_ms']):.6f} | "
            f"{float(row['throughput_samples_per_s']):.1f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()

    import torch

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    project_dir = args.project_dir.expanduser().resolve()
    output_dir = (args.output_dir or project_dir / "outputs" / "tables").expanduser().resolve()
    rows: list[dict[str, object]] = []
    for method, model_name, checkpoint_name in DEFAULT_MODELS:
        checkpoint_path = project_dir / "outputs" / "checkpoints" / checkpoint_name / "best.pt"
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = build_model(model_name, checkpoint).to(device)
        model.load_state_dict(checkpoint["model_state"])
        total, trainable = count_params(model)
        speed = benchmark(
            model,
            device=device,
            batch_size=args.batch_size,
            input_length=args.input_length,
            warmup=args.warmup,
            steps=args.steps,
        )
        rows.append(
            {
                "method": method,
                "model": model_name,
                "checkpoint": str(checkpoint_path),
                "device": device,
                "batch_size": args.batch_size,
                "input_length": args.input_length,
                "num_parameters": total,
                "trainable_parameters": trainable,
                **speed,
            }
        )

    csv_path = output_dir / "efficiency_summary.csv"
    md_path = output_dir / "efficiency_summary.md"
    json_path = output_dir / "efficiency_summary.json"
    write_csv(rows, csv_path)
    write_markdown(rows, md_path)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
