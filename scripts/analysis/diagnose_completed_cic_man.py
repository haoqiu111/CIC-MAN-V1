#!/usr/bin/env python3
"""Run CIC-MAN diagnostics for completed formal cross-dataset runs."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--splits", default="val,test", help="Comma-separated splits to diagnose.")
    return parser.parse_args()


def split_paths(project_root: Path, target: str, protocol: str) -> dict[str, Path]:
    root = project_root / "data" / "paper1_cicman" / "cache" / "windows" / protocol / f"target_dataset_{target}"
    return {
        "train": root / "train_windows.csv",
        "val": root / "val_windows.csv",
        "test": root / "test_windows.csv",
    }


def experiments(target: str) -> list[tuple[str, str]]:
    return [
        (f"cic_man_cross_dataset_task3_target_{target}", "cross_dataset_task3"),
        (f"cic_man_v1_cross_dataset_task3_target_{target}", "cross_dataset_task3"),
        (f"cic_man_v1_source_mixed_cross_dataset_task3_target_{target}", "cross_dataset_task3_source_mixed"),
    ]


def main() -> None:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    script = project_root  / "scripts" / "analysis" / "diagnose_cic_man.py"
    checkpoint_root = project_root  / "outputs" / "checkpoints"

    selected_splits = {split.strip() for split in args.splits.split(",") if split.strip()}
    for target in ["hust", "ottawa", "paderborn"]:
        for experiment, protocol in experiments(target):
            run_dir = checkpoint_root / experiment
            checkpoint = run_dir / "best.pt"
            if not checkpoint.exists():
                print(f"skip {experiment}: missing checkpoint {checkpoint}")
                continue
            for split_name, index_path in split_paths(project_root, target, protocol).items():
                if split_name not in selected_splits:
                    continue
                if not index_path.exists():
                    print(f"skip {experiment} split={split_name}: missing index {index_path}")
                    continue
                cmd = [
                    args.python,
                    str(script),
                    "--index",
                    str(index_path),
                    "--checkpoint",
                    str(checkpoint),
                    "--output-dir",
                    str(run_dir),
                    "--split-name",
                    split_name,
                    "--batch-size",
                    str(args.batch_size),
                    "--device",
                    args.device,
                    "--num-workers",
                    str(args.num_workers),
                ]
                print(" ".join(cmd))
                subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
