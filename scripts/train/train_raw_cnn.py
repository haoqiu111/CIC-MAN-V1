#!/usr/bin/env python3
"""Train the Raw 1D-CNN baseline from window-index CSV files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def add_src_to_path() -> None:
    script_path = Path(__file__).resolve()
    project_dir = script_path.parents[2]
    sys.path.insert(0, str(project_dir / "src"))


add_src_to_path()

from cicman.training.raw_cnn import train_raw_cnn  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-index", type=Path, required=True)
    parser.add_argument("--val-index", type=Path, required=True)
    parser.add_argument("--test-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-items", type=int, default=None)
    parser.add_argument("--max-eval-items", type=int, default=None)
    parser.add_argument("--shortcut-train-mode", choices=["none", "correlated", "reversed", "neutral"], default="none")
    parser.add_argument("--shortcut-val-mode", choices=["none", "correlated", "reversed", "neutral"], default="none")
    parser.add_argument("--shortcut-test-mode", choices=["none", "correlated", "reversed", "neutral"], default="none")
    parser.add_argument("--shortcut-amplitude", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_raw_cnn(
        train_index=args.train_index,
        val_index=args.val_index,
        test_index=args.test_index,
        output_dir=args.output_dir,
        num_classes=args.num_classes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        num_workers=args.num_workers,
        seed=args.seed,
        max_train_items=args.max_train_items,
        max_eval_items=args.max_eval_items,
        shortcut_train_mode=args.shortcut_train_mode,
        shortcut_val_mode=args.shortcut_val_mode,
        shortcut_test_mode=args.shortcut_test_mode,
        shortcut_amplitude=args.shortcut_amplitude,
    )
    print(json.dumps({"metrics": result["test_metrics"], "checkpoint": result["best_checkpoint"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
