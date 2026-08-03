#!/usr/bin/env python3
"""Run recording-level evaluation for completed CIC-MAN runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def add_project_paths() -> None:
    script_path = Path(__file__).resolve()
    project_dir = script_path.parents[2]
    sys.path.insert(0, str(project_dir / "src"))
    sys.path.insert(0, str(script_path.parent))


add_project_paths()

from eval_cic_man_recording import evaluate_recordings, write_per_recording  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--include-smoke", action="store_true")
    return parser.parse_args()


def infer_test_index(project_root: Path, experiment: str) -> Path | None:
    data_root = project_root / "data" / "paper1_cicman" / "cache" / "windows"
    if experiment.startswith("cic_man_v1_cross_dataset_task3_target_"):
        target = experiment.removeprefix("cic_man_v1_cross_dataset_task3_target_")
        return data_root / "cross_dataset_task3" / f"target_dataset_{target}" / "test_windows.csv"
    if experiment.startswith("cic_man_cross_dataset_task3_target_"):
        target = experiment.removeprefix("cic_man_cross_dataset_task3_target_")
        return data_root / "cross_dataset_task3" / f"target_dataset_{target}" / "test_windows.csv"
    return None


def should_skip(experiment: str, include_smoke: bool) -> bool:
    if include_smoke:
        return False
    name = experiment.lower()
    return "smoke" in name or "probe" in name


def main() -> None:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    checkpoint_root = project_root  / "outputs" / "checkpoints"

    completed = []
    skipped = []
    for checkpoint_path in sorted(checkpoint_root.glob("cic_man_*/best.pt")):
        experiment = checkpoint_path.parent.name
        if should_skip(experiment, args.include_smoke):
            skipped.append((experiment, "smoke_or_probe"))
            continue
        index_path = infer_test_index(project_root, experiment)
        if index_path is None or not index_path.exists():
            skipped.append((experiment, "missing_test_index"))
            continue
        result = evaluate_recordings(
            index_csv=index_path,
            checkpoint_path=checkpoint_path,
            batch_size=args.batch_size,
            device=args.device,
            num_workers=args.num_workers,
        )
        summary = {key: value for key, value in result.items() if key != "per_recording"}
        summary_path = checkpoint_path.parent / "recording_metrics.json"
        detail_path = checkpoint_path.parent / "recording_predictions.csv"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        write_per_recording(result["per_recording"], detail_path)
        mean_metrics = summary["mean_logits"]
        completed.append(
            (
                experiment,
                mean_metrics["accuracy"],
                mean_metrics["macro_f1"],
                mean_metrics["balanced_accuracy"],
                mean_metrics["num_recordings"],
            )
        )
        print(
            f"{experiment}: recording mean-logits "
            f"acc={mean_metrics['accuracy']:.6f} "
            f"macro_f1={mean_metrics['macro_f1']:.6f} "
            f"balanced_acc={mean_metrics['balanced_accuracy']:.6f} "
            f"n={mean_metrics['num_recordings']}"
        )

    print(f"Completed {len(completed)} recording-level evaluations.")
    if skipped:
        print(f"Skipped {len(skipped)} runs: {skipped}")


if __name__ == "__main__":
    main()
