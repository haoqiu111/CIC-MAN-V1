#!/usr/bin/env python3
"""Run the CIC-MAN v2 pilot matrix over the three cross-dataset targets.

Runs each (target, preset) combination sequentially in-process and writes a
combined markdown/CSV summary. Designed to be resumable: finished runs with a
metrics.json are skipped unless --force.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from pathlib import Path


def add_src_to_path() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


add_src_to_path()

from cicman.v2.data import ALL_VIEWS, CachedWindowDataset, ViewCache  # noqa: E402
from cicman.v2.model import CICMANv2  # noqa: E402
from cicman.v2.train import train_model  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from estimate_view_reliability import estimate as estimate_reliability  # noqa: E402
from train_cicman_v2 import PRESETS  # noqa: E402

TARGETS = ["hust", "ottawa", "paderborn"]
RELIABILITY_TEMPERATURE = 0.15

SINGLE_VIEW_RUNS = {f"single_{v}": v for v in ALL_VIEWS}


def run_one(*, task_dir: Path, cache_root: Path, output_dir: Path, preset_name: str, views_override, epochs: int, seed: int, force: bool, num_classes: int = 3, reliability_group: str = "dataset_id", tag: str = ""):
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists() and not force:
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    cfg = dict(PRESETS[preset_name])
    views = views_override or cfg["views"]

    router_prior = None
    if cfg.get("needs_reliability"):
        import numpy as np

        rel_seeds = cfg.get("reliability_seeds") or [seed]
        temperature = cfg.get("reliability_temperature", RELIABILITY_TEMPERATURE)
        extra_variants = cfg.get("reliability_extra_variants") or []
        rel_caches = [cache_root] + [cache_root.parent / v for v in extra_variants] if extra_variants else cache_root
        rel_tag = "_ic" if extra_variants else ""
        all_scores = []
        for rel_seed in rel_seeds:
            rel_dir = output_dir.parent / f"v2{tag}_reliability{rel_tag}_{task_dir.name}_seed{rel_seed}"
            rel = estimate_reliability(task_dir, rel_caches, rel_dir, num_classes=num_classes, epochs=8, seed=rel_seed, views=views, group_column=reliability_group)
            all_scores.append([rel["reliability"][v] for v in views])
        scores = np.mean(np.array(all_scores, dtype=np.float64), axis=0)
        prior = np.exp(scores / temperature)
        router_prior = (prior / prior.sum()).tolist()
        print(f"router prior ({views}, T={temperature}, rel_seeds={rel_seeds}): {[round(p, 3) for p in router_prior]}", flush=True)

    cache = ViewCache(cache_root, views=views)
    train_ds = CachedWindowDataset(task_dir / "train_windows.csv", cache)
    val_ds = CachedWindowDataset(task_dir / "val_windows.csv", cache)
    test_ds = CachedWindowDataset(task_dir / "test_windows.csv", cache)
    if cfg.get("augment_variants"):
        from cicman.v2.data import UnionWindowDataset

        parts = [train_ds]
        for variant in cfg["augment_variants"]:
            v_cache = ViewCache(cache_root.parent / variant, views=views)
            parts.append(CachedWindowDataset(task_dir / "train_windows.csv", v_cache))
        train_ds = UnionWindowDataset(parts)

    model = CICMANv2(
        num_classes=num_classes,
        views=views,
        num_domains=max(len(train_ds.domains), 2),
        router_mode=cfg["router_mode"],
        view_dropout=cfg.get("view_dropout", 0.0),
        router_prior=router_prior,
    )
    return train_model(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
        output_dir=output_dir,
        views=views,
        num_classes=num_classes,
        epochs=epochs,
        seed=seed,
        lambda_view_cls=cfg["lambda_view_cls"],
        lambda_consensus=cfg["lambda_consensus"],
        lambda_adversarial=cfg["lambda_adversarial"],
        lambda_domain_cls=cfg["lambda_domain_cls"],
        lambda_orthogonal=cfg["lambda_orthogonal"],
        lambda_balance=cfg["lambda_balance"],
        lambda_route_prior=cfg.get("lambda_route_prior", 0.0),
        focal_gamma=cfg.get("focal_gamma", 0.0),
        dg_method=cfg.get("dg_method"),
        lambda_dg=cfg.get("lambda_dg", 1.0),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    repo_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--project-root", type=Path, default=repo_root,
                        help="Root containing data/paper1_cicman (default: repository root).")
    parser.add_argument("--output-root", type=Path, default=repo_root / "outputs",
                        help="Checkpoint/table output root (default: <repository>/outputs).")
    parser.add_argument("--cache-name", default="views_v2")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--targets", default=",".join(TARGETS))
    parser.add_argument(
        "--models",
        default="cicman_v2,ensemble,moe,dann,single_raw,single_env_order,single_env_spec,single_stft,single_cwt,single_denoise",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--isolate",
        action="store_true",
        help="Run each (model,target) in a fresh subprocess so memory cannot accumulate across runs.",
    )
    parser.add_argument("--windows-name", default="cross_dataset_task3_source_mixed", help="Window-index protocol dir under cache/windows/.")
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--tag", default="", help="Run-name/summary suffix after 'v2', e.g. '_task4' (keeps task3 artifacts untouched).")
    parser.add_argument("--reliability-group", default="dataset_id", help="Leave-one-out group column for the reliability prior (single-source tasks need a within-dataset column).")
    args = parser.parse_args()

    cache_root = args.project_root / "data/paper1_cicman/cache" / args.cache_name
    windows_root = args.project_root / "data/paper1_cicman/cache/windows" / args.windows_name
    out_root = args.output_root / "checkpoints"
    tables = args.output_root / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    rows = []
    failures = []
    for target in args.targets.split(","):
        task_dir = windows_root / f"target_dataset_{target}"
        for model_name in args.models.split(","):
            if model_name in SINGLE_VIEW_RUNS:
                preset, views = "single_view", [SINGLE_VIEW_RUNS[model_name]]
            else:
                preset, views = model_name, None
            run_name = f"v2{args.tag}_{model_name}_target_{target}_seed{args.seed}"
            output_dir = out_root / run_name
            print(f"=== {run_name} ===", flush=True)
            metrics_path = output_dir / "metrics.json"
            force_run = args.force
            if args.isolate and not (metrics_path.exists() and not args.force):
                import subprocess

                sub_cmd = [
                    sys.executable, "-u", str(Path(__file__).resolve()),
                    "--project-root", str(args.project_root),
                    "--cache-name", args.cache_name,
                    "--output-root", str(args.output_root),
                    "--epochs", str(args.epochs),
                    "--seed", str(args.seed),
                    "--targets", target,
                    "--models", model_name,
                    "--windows-name", args.windows_name,
                    "--num-classes", str(args.num_classes),
                    "--tag", args.tag,
                    "--reliability-group", args.reliability_group,
                ]
                if args.force:
                    sub_cmd.append("--force")
                ret = subprocess.call(sub_cmd)
                if ret != 0 or not metrics_path.exists():
                    print(f"isolated run failed: {run_name} (exit {ret})", flush=True)
                    failures.append(run_name)
                    continue
                force_run = False  # child already retrained; parent only loads its metrics
            try:
                result = run_one(
                    task_dir=task_dir,
                    cache_root=cache_root,
                    output_dir=output_dir,
                    preset_name=preset,
                    views_override=views,
                    epochs=args.epochs,
                    seed=args.seed,
                    force=force_run,
                    num_classes=args.num_classes,
                    reliability_group=args.reliability_group,
                    tag=args.tag,
                )
            except Exception:
                traceback.print_exc()
                failures.append(run_name)
                continue
            finally:
                import gc

                gc.collect()
                try:
                    import torch

                    torch.cuda.empty_cache()
                except Exception:
                    pass
            t = result["test_metrics"]
            t_last = result.get("test_metrics_last")
            rows.append(
                {
                    "model": model_name,
                    "target": target,
                    "seed": args.seed,
                    "best_epoch": result["best_epoch"],
                    "window_acc": round(t["window_accuracy"], 4),
                    "window_macro_f1": round(t["window_macro_f1"], 4),
                    "recording_acc": round(t["recording_accuracy"], 4),
                    "recording_macro_f1": round(t["recording_macro_f1"], 4),
                    "window_macro_f1_last": round(t_last["window_macro_f1"], 4) if t_last else "",
                    "recording_macro_f1_last": round(t_last["recording_macro_f1"], 4) if t_last else "",
                    "num_recordings": t["num_recordings"],
                    "router_weights": json.dumps([round(w, 3) for w in (t["mean_router_weights"] or [])]),
                }
            )
            print(
                f"--> {run_name}: win_f1={t['window_macro_f1']:.4f} rec_f1={t['recording_macro_f1']:.4f}",
                flush=True,
            )

    if not rows:
        print("no results")
        if failures:
            raise SystemExit(f"failed runs: {', '.join(failures)}")
        return

    csv_path = tables / f"v2{args.tag}_pilot_summary_seed{args.seed}.csv"
    if csv_path.exists():
        new_keys = {(r["model"], r["target"]) for r in rows}
        with csv_path.open(newline="", encoding="utf-8") as f:
            for old in csv.DictReader(f):
                if (old["model"], old["target"]) not in new_keys:
                    rows.append(old)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_path = tables / f"v2{args.tag}_pilot_summary_seed{args.seed}.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# CIC-MAN v2 Pilot (seed {args.seed}, {args.epochs} epochs)\n\n")
        f.write(f"Windows: {args.windows_name}, {args.num_classes}-class, strict target-free.\n\n")
        f.write("| Model | Target | Win Acc | Win Macro-F1 | Rec Acc | Rec Macro-F1 | Best Ep | Router |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---|\n")
        for r in sorted(rows, key=lambda r: (r["target"], -float(r["window_macro_f1"]))):
            f.write(
                f"| {r['model']} | {r['target']} | {r['window_acc']} | {r['window_macro_f1']} | "
                f"{r['recording_acc']} | {r['recording_macro_f1']} | {r['best_epoch']} | {r['router_weights']} |\n"
            )
    print(f"summary -> {md_path}")
    if failures:
        raise SystemExit(f"failed runs: {', '.join(failures)}")


if __name__ == "__main__":
    main()
