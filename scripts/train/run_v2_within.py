#!/usr/bin/env python3
"""Run CIC-MAN v2 + baselines on the within-dataset DG protocols.

Protocols (window indexes must already exist under cache/windows/):
  ottawa_leave_speed          4-class, domain = speed_profile_id
  hust_leave_bearing_type     4-class, domain = bearing_type_id
  paderborn_leave_condition   3-class effective (no rolling), domain = condition_id

Writes one summary table per protocol.
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

from cicman.v2.data import CachedWindowDataset, UnionWindowDataset, ViewCache  # noqa: E402
from cicman.v2.model import CICMANv2  # noqa: E402
from cicman.v2.train import train_model  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from estimate_view_reliability import estimate as estimate_reliability  # noqa: E402
from train_cicman_v2 import PRESETS  # noqa: E402

PROTOCOLS = {
    "ottawa_leave_speed": {"num_classes": 4, "domain_column": "speed_profile_id"},
    "hust_leave_bearing_type": {"num_classes": 4, "domain_column": "bearing_type_id"},
    "paderborn_leave_condition": {"num_classes": 3, "domain_column": "condition_id"},
}

SINGLE_VIEW_RUNS = {
    "single_raw": "raw",
    "single_env_order": "env_order",
    "single_env_spec": "env_spec",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--cache-name", default="views_v2")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--protocols", default=",".join(PROTOCOLS))
    parser.add_argument("--models", default="cicman_v2,ensemble,moe,single_raw,single_env_order,single_env_spec")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    windows_root = args.project_root / "data/paper1_cicman/cache/windows"
    cache_root = args.project_root / "data/paper1_cicman/cache" / args.cache_name
    repo_root = Path(__file__).resolve().parents[2]
    out_root = repo_root / "outputs/checkpoints"
    tables = repo_root / "outputs/tables"

    for protocol in args.protocols.split(","):
        pconf = PROTOCOLS[protocol]
        rows = []
        for task_dir in sorted((windows_root / protocol).iterdir()):
            if not (task_dir / "train_windows.csv").exists():
                continue
            for model_name in args.models.split(","):
                if model_name in SINGLE_VIEW_RUNS:
                    preset, views = "single_view", [SINGLE_VIEW_RUNS[model_name]]
                else:
                    preset, views = model_name, None
                cfg = dict(PRESETS[preset])
                use_views = views or cfg["views"]
                run_name = f"v2w_{model_name}_{protocol}_{task_dir.name}_seed{args.seed}"
                output_dir = out_root / run_name
                metrics_path = output_dir / "metrics.json"
                print(f"=== {run_name} ===", flush=True)
                try:
                    if metrics_path.exists() and not args.force:
                        result = json.loads(metrics_path.read_text(encoding="utf-8"))
                    else:
                        router_prior = None
                        if cfg.get("needs_reliability"):
                            import numpy as np

                            rel_seeds = cfg.get("reliability_seeds") or [args.seed]
                            temperature = cfg.get("reliability_temperature", 0.15)
                            extra_variants = cfg.get("reliability_extra_variants") or []
                            rel_caches = [cache_root] + [cache_root.parent / v for v in extra_variants] if extra_variants else cache_root
                            rel_tag = "_ic" if extra_variants else ""
                            all_scores = []
                            for rel_seed in rel_seeds:
                                rel_dir = out_root / f"v2w_reliability{rel_tag}_{protocol}_{task_dir.name}_seed{rel_seed}"
                                rel = estimate_reliability(
                                    task_dir, rel_caches, rel_dir,
                                    num_classes=pconf["num_classes"], epochs=8, seed=rel_seed,
                                    views=use_views, group_column=pconf["domain_column"],
                                )
                                all_scores.append([rel["reliability"][v] for v in use_views])
                            scores = np.mean(np.array(all_scores, dtype=np.float64), axis=0)
                            prior = np.exp(scores / temperature)
                            router_prior = (prior / prior.sum()).tolist()
                            print(f"router prior ({use_views}, T={temperature}): {[round(p, 3) for p in router_prior]}", flush=True)

                        cache = ViewCache(cache_root, views=use_views)
                        train_ds = CachedWindowDataset(task_dir / "train_windows.csv", cache, domain_column=pconf["domain_column"])
                        val_ds = CachedWindowDataset(task_dir / "val_windows.csv", cache, domain_column=pconf["domain_column"])
                        test_ds = CachedWindowDataset(task_dir / "test_windows.csv", cache, domain_column=pconf["domain_column"])
                        if cfg.get("augment_variants"):
                            parts = [train_ds]
                            for variant in cfg["augment_variants"]:
                                v_cache = ViewCache(cache_root.parent / variant, views=use_views)
                                parts.append(CachedWindowDataset(task_dir / "train_windows.csv", v_cache, domain_column=pconf["domain_column"]))
                            train_ds = UnionWindowDataset(parts)
                        model = CICMANv2(
                            num_classes=pconf["num_classes"],
                            views=use_views,
                            num_domains=max(len(train_ds.domains), 2),
                            router_mode=cfg["router_mode"],
                            view_dropout=cfg.get("view_dropout", 0.0),
                            router_prior=router_prior,
                        )
                        result = train_model(
                            model=model,
                            train_dataset=train_ds,
                            val_dataset=val_ds,
                            test_dataset=test_ds,
                            output_dir=output_dir,
                            views=use_views,
                            num_classes=pconf["num_classes"],
                            epochs=args.epochs,
                            seed=args.seed,
                            lambda_view_cls=cfg["lambda_view_cls"],
                            lambda_consensus=cfg["lambda_consensus"],
                            lambda_adversarial=cfg["lambda_adversarial"],
                            lambda_domain_cls=cfg["lambda_domain_cls"],
                            lambda_orthogonal=cfg["lambda_orthogonal"],
                            lambda_balance=cfg["lambda_balance"],
                            lambda_route_prior=cfg.get("lambda_route_prior", 0.0),
                            dg_method=cfg.get("dg_method"),
                            lambda_dg=cfg.get("lambda_dg", 1.0),
                        )
                except Exception:
                    traceback.print_exc()
                    continue
                t = result["test_metrics"]
                t_last = result.get("test_metrics_last")
                rows.append(
                    {
                        "model": model_name,
                        "task": task_dir.name,
                        "seed": args.seed,
                        "window_macro_f1": round(t["window_macro_f1"], 4),
                        "window_acc": round(t["window_accuracy"], 4),
                        "recording_macro_f1": round(t["recording_macro_f1"], 4),
                        "recording_acc": round(t["recording_accuracy"], 4),
                        "window_macro_f1_last": round(t_last["window_macro_f1"], 4) if t_last else "",
                        "recording_macro_f1_last": round(t_last["recording_macro_f1"], 4) if t_last else "",
                        "num_recordings": t["num_recordings"],
                    }
                )
                print(f"--> {run_name}: win_f1={t['window_macro_f1']:.4f} rec_f1={t['recording_macro_f1']:.4f}", flush=True)

        if not rows:
            continue
        csv_path = tables / f"v2_within_{protocol}_seed{args.seed}.csv"
        fieldnames = list(rows[0].keys())
        if csv_path.exists():
            # merge, never overwrite: keep old rows whose (model, task) was not re-run
            new_keys = {(r["model"], r["task"]) for r in rows}
            with csv_path.open(newline="", encoding="utf-8") as f:
                for old in csv.DictReader(f):
                    if (old["model"], old["task"]) not in new_keys:
                        rows.append({k: old.get(k, "") for k in fieldnames})
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        md_path = tables / f"v2_within_{protocol}_seed{args.seed}.md"
        with md_path.open("w", encoding="utf-8") as f:
            f.write(f"# v2 Within-Dataset: {protocol} (seed {args.seed}, {args.epochs} epochs)\n\n")
            f.write("| Model | Task | Win Macro-F1 | Win Acc | Rec Macro-F1 | Rec Acc |\n|---|---|---:|---:|---:|---:|\n")
            for r in sorted(rows, key=lambda r: (r["task"], -float(r["window_macro_f1"]))):
                f.write(f"| {r['model']} | {r['task']} | {r['window_macro_f1']} | {r['window_acc']} | {r['recording_macro_f1']} | {r['recording_acc']} |\n")
        print(f"summary -> {md_path}", flush=True)


if __name__ == "__main__":
    main()
