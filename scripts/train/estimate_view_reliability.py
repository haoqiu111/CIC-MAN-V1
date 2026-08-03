#!/usr/bin/env python3
"""Estimate target-free per-view cross-domain reliability for one task.

For every intervention view, trains a small single-view model on one source
dataset and evaluates recording-level macro-F1 on the other source dataset
(both directions, averaged). This measures how well each view's health
information survives a domain shift WITHOUT touching the target domain, and
is used as the causal routing prior of CIC-MAN v3.

Output: <output>/view_reliability.json  {view: mean_cross_source_rec_f1}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def add_src_to_path() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


add_src_to_path()

from cicman.v2.data import ALL_VIEWS, CachedWindowDataset, ViewCache  # noqa: E402
from cicman.v2.model import CICMANv2  # noqa: E402
from cicman.v2.train import train_model  # noqa: E402


def estimate(task_dir: Path, cache_root, output_dir: Path, *, num_classes: int, epochs: int, seed: int, views=None, group_column: str = "dataset_id") -> dict:
    """Estimate target-free per-view reliability.

    ``cache_root`` may be a single cache dir or a list of cache dirs
    (measurement-mechanism intervention variants). With multiple variants the
    reliability of a view is the MINIMUM over variants of its mean LOSO
    recording macro-F1: a view whose cross-domain evidence disappears under a
    counterfactual intervention on the measurement mechanism was relying on a
    non-mechanism artifact (intervention-consistent reliability).

    ``group_column`` picks the leave-one-out unit: source datasets by default,
    or any window-index column (e.g. speed_profile_id / condition_id) when the
    task has a single source dataset.
    """
    out_path = output_dir / "view_reliability.json"
    if out_path.exists():
        return json.loads(out_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)

    views = views or ALL_VIEWS
    cache_roots = [Path(c) for c in (cache_root if isinstance(cache_root, (list, tuple)) else [cache_root])]

    probe_cache = ViewCache(cache_roots[0], views=views)
    full_train = CachedWindowDataset(task_dir / "train_windows.csv", probe_cache, domain_column=group_column)
    source_groups = full_train.domains
    if len(source_groups) < 2:
        raise RuntimeError(f"need >=2 source groups on {group_column}, got {source_groups}")

    variant_scores: dict[str, dict[str, list[float]]] = {}
    for root in cache_roots:
        variant = root.name
        scores: dict[str, list[float]] = {v: [] for v in views}
        for view in views:
            cache = ViewCache(root, views=[view])
            for held_out in source_groups:
                fit_sets = {d for d in source_groups if d != held_out}
                train_ds = CachedWindowDataset(task_dir / "train_windows.csv", cache, domain_column=group_column, domain_filter=fit_sets)
                val_ds = CachedWindowDataset(task_dir / "val_windows.csv", cache, domain_column=group_column, domain_filter=fit_sets)
                if len(val_ds) == 0:
                    # some splits keep only one group in val; probe selection then falls back to fit-train
                    val_ds = train_ds
                heldout_train = CachedWindowDataset(task_dir / "train_windows.csv", cache, domain_column=group_column, domain_filter={held_out})
                model = CICMANv2(num_classes=num_classes, views=[view], num_domains=max(len(train_ds.domains), 2), router_mode="uniform")
                result = train_model(
                    model=model,
                    train_dataset=train_ds,
                    val_dataset=val_ds,
                    test_dataset=heldout_train,
                    output_dir=output_dir / f"probe_{variant}_{view}_holdout_{held_out}",
                    views=[view],
                    num_classes=num_classes,
                    epochs=epochs,
                    seed=seed,
                    lambda_view_cls=0.0,
                    lambda_consensus=0.0,
                    lambda_adversarial=0.0,
                    lambda_domain_cls=0.0,
                    lambda_orthogonal=0.0,
                    lambda_balance=0.0,
                    recording_aggregation="probability_sum",
                    log_fn=lambda *a, **k: None,
                )
                f1 = result["test_metrics"]["recording_macro_f1"]
                scores[view].append(f1)
                print(f"  reliability [{variant}] {view} -> holdout {held_out}: rec_f1={f1:.4f}", flush=True)
        variant_scores[variant] = scores

    per_variant_mean = {
        variant: {v: float(sum(s) / len(s)) for v, s in scores.items()}
        for variant, scores in variant_scores.items()
    }
    reliability = {v: min(per_variant_mean[variant][v] for variant in per_variant_mean) for v in views}
    payload = {"reliability": reliability, "per_variant": per_variant_mean, "folds": variant_scores}
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"reliability (min over {len(cache_roots)} variants): {reliability}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group-column", default="dataset_id")
    args = parser.parse_args()
    estimate(args.task_dir, args.cache_dir, args.output_dir, num_classes=args.num_classes, epochs=args.epochs, seed=args.seed, group_column=args.group_column)


if __name__ == "__main__":
    main()
