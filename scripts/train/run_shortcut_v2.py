#!/usr/bin/env python3
"""Experiment 6: shortcut reversal on the v2 pipeline (target=paderborn task).

Trains on sources injected with a class-correlated low-frequency tone, then
evaluates the trained model on target caches where the tone is correlated,
reversed, or neutral. A shortcut-reliant model shows a large correlated vs
reversed gap; a causal model should be nearly invariant.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def add_src_to_path() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


add_src_to_path()

from cicman.v2.data import ALL_VIEWS, CachedWindowDataset, UnionWindowDataset, ViewCache  # noqa: E402
from cicman.v2.model import CICMANv2  # noqa: E402
from cicman.v2.train import evaluate, train_model  # noqa: E402
from recording_protocol import OFFICIAL_AGGREGATION, PROTOCOL_VERSION  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from estimate_view_reliability import estimate as estimate_reliability  # noqa: E402
from train_cicman_v2 import PRESETS  # noqa: E402

MODELS = ["single_raw", "single_env_order", "moe", "cicman_v4", "cicman_v5ic"]
HERO_MODELS = [
    "single_raw", "dann", "dg_irm", "dg_coral", "dg_mmd", "dg_groupdro",
    "moe", "ensemble", "cicman_v4", "single_env_order", "cicman_v6ic",
]
DG_EXPECTED = {
    "dg_irm": "irm",
    "dg_coral": "coral",
    "dg_mmd": "mmd",
    "dg_groupdro": "groupdro",
}
SINGLE = {"single_raw": "raw", "single_env_order": "env_order"}
# cicman_v5ic = v4 config with the intervention-consistent reliability prior
# (min LOSO F1 over {identity, highpass-800Hz} measurement interventions)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--reevaluate-only", action="store_true", help="Re-score existing last.pt files without training.")
    parser.add_argument("--output-prefix", default="v2sc", help="Run-directory prefix; use a non-default value for smoke tests.")
    args = parser.parse_args()

    task_dir = args.project_root / "data/paper1_cicman/cache/windows/cross_dataset_task3_source_mixed/target_dataset_paderborn"
    sc_root = args.project_root / "data/paper1_cicman/cache/views_v2_shortcut"
    repo_root = Path(__file__).resolve().parents[2]
    out_root = repo_root / "outputs/checkpoints"
    tables = repo_root / "outputs/tables"
    target_recordings = set(json.loads((sc_root / "target_recordings.json").read_text(encoding="utf-8")))

    results = []
    for model_name in args.models.split(","):
        if model_name in SINGLE:
            preset, views = "single_view", [SINGLE[model_name]]
        elif model_name in ("cicman_v5ic", "cicman_v6ic"):
            preset, views = "cicman_v4", None
        else:
            preset, views = model_name, None
        cfg = dict(PRESETS[preset])
        expected_dg = DG_EXPECTED.get(model_name)
        if expected_dg is not None and cfg.get("dg_method") != expected_dg:
            raise RuntimeError(
                f"DG configuration mismatch for {model_name}: "
                f"expected {expected_dg}, got {cfg.get('dg_method')}"
            )
        use_views = views or cfg["views"]
        run_dir = out_root / f"{args.output_prefix}_{model_name}_seed{args.seed}"
        metrics_path = run_dir / "shortcut_metrics.json"
        if metrics_path.exists() and not args.reevaluate_only:
            entry = json.loads(metrics_path.read_text(encoding="utf-8"))
            results.append(entry)
            print(f"cached {model_name}: {entry['recording_macro_f1']}")
            continue

        if args.reevaluate_only:
            import torch
            from torch.utils.data import DataLoader

            checkpoint = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=True)
            state = checkpoint["model"]
            use_views = list(checkpoint["views"])
            has_prior = "router_log_prior" in state
            prior = torch.exp(state["router_log_prior"]).tolist() if has_prior else None
            router_mode = "feats_only" if model_name == "moe" else ("causal" if has_prior else "uniform")
            num_domains = int(state["domain_adv_head.2.weight"].shape[0])
            model = CICMANv2(
                num_classes=3,
                views=use_views,
                num_domains=num_domains,
                router_mode=router_mode,
                router_prior=prior,
            )
            model.load_state_dict(state, strict=True)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device).eval()
            entry = {
                "model": model_name,
                "seed": args.seed,
                "recording_macro_f1": {},
                "window_macro_f1": {},
                "recording_aggregation": OFFICIAL_AGGREGATION,
                "recording_protocol_version": PROTOCOL_VERSION,
                "checkpoint_epoch": int(checkpoint["epoch"]),
                "checkpoint_sha256": sha256(run_dir / "last.pt"),
            }
            config_path = run_dir / "run_config.json"
            if config_path.exists():
                entry["training_config"] = json.loads(config_path.read_text(encoding="utf-8"))
            for mode in ["correlated", "reversed", "neutral"]:
                cache_m = ViewCache(sc_root / f"sc_target_{mode}", views=use_views)
                ds_m = CachedWindowDataset(
                    task_dir / "test_windows.csv", cache_m, recording_filter=target_recordings
                )
                loader = DataLoader(ds_m, batch_size=256, shuffle=False)
                metrics = evaluate(model, loader, device, 3, use_views)
                entry["recording_macro_f1"][mode] = metrics["recording_macro_f1"]
                entry["window_macro_f1"][mode] = metrics["window_macro_f1"]
            entry["recording_tie_break"] = "cumulative_probability_then_lowest_class_id"
            entry["num_target_recordings"] = len(target_recordings)
            entry["reversal_gap_rec"] = (
                entry["recording_macro_f1"]["correlated"] - entry["recording_macro_f1"]["reversed"]
            )
            metrics_path.write_text(json.dumps(entry, indent=2), encoding="utf-8")
            results.append(entry)
            print(f"re-evaluated {model_name}: {entry['recording_macro_f1']}", flush=True)
            continue

        run_dir.mkdir(parents=True, exist_ok=True)
        run_config = {
            "model": model_name,
            "preset": preset,
            "seed": args.seed,
            "epochs": args.epochs,
            "views": list(use_views),
            "router_mode": cfg["router_mode"],
            "dg_method": cfg.get("dg_method"),
            "lambda_dg": cfg.get("lambda_dg", 1.0),
            "recording_aggregation": OFFICIAL_AGGREGATION,
            "recording_protocol_version": PROTOCOL_VERSION,
            "target_subset_size": len(target_recordings),
        }
        (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

        router_prior = None
        if cfg.get("needs_reliability"):
            if model_name in ("cicman_v5ic", "cicman_v6ic"):
                rel_dir = out_root / f"v2sc_reliability_ic_seed{args.seed}"
                rel_caches = [sc_root / "sc_sources_correlated", sc_root / "sc_sources_correlated_hp800"]
            else:
                rel_dir = out_root / f"v2sc_reliability_seed{args.seed}"
                rel_caches = sc_root / "sc_sources_correlated"
            rel = estimate_reliability(task_dir, rel_caches, rel_dir,
                                       num_classes=3, epochs=8, seed=args.seed, views=use_views)
            scores = np.array([rel["reliability"][v] for v in use_views])
            prior = np.exp(scores / cfg.get("reliability_temperature", 0.08))
            router_prior = (prior / prior.sum()).tolist()
            print(f"shortcut-world router prior: {[round(p, 3) for p in router_prior]}")

        src_cache = ViewCache(sc_root / "sc_sources_correlated", views=use_views)
        train_ds = CachedWindowDataset(task_dir / "train_windows.csv", src_cache)
        val_ds = CachedWindowDataset(task_dir / "val_windows.csv", src_cache)
        if model_name == "cicman_v6ic":
            # intervention-augmented training: the model must be consistent
            # across {identity, highpass-800Hz} measurement interventions
            hp_cache = ViewCache(sc_root / "sc_sources_correlated_hp800", views=use_views)
            train_ds = UnionWindowDataset([train_ds, CachedWindowDataset(task_dir / "train_windows.csv", hp_cache)])
        corr_cache = ViewCache(sc_root / "sc_target_correlated", views=use_views)
        test_corr = CachedWindowDataset(
            task_dir / "test_windows.csv", corr_cache, recording_filter=target_recordings
        )

        model = CICMANv2(
            num_classes=3,
            views=use_views,
            num_domains=max(len(train_ds.domains), 2),
            router_mode=cfg["router_mode"],
            view_dropout=cfg.get("view_dropout", 0.0),
            router_prior=router_prior,
        )
        result = train_model(
            model=model, train_dataset=train_ds, val_dataset=val_ds, test_dataset=test_corr,
            output_dir=run_dir, views=use_views, num_classes=3, epochs=args.epochs, seed=args.seed,
            lambda_view_cls=cfg["lambda_view_cls"], lambda_consensus=cfg["lambda_consensus"],
            lambda_adversarial=cfg["lambda_adversarial"], lambda_domain_cls=cfg["lambda_domain_cls"],
            lambda_orthogonal=cfg["lambda_orthogonal"], lambda_balance=cfg["lambda_balance"],
            lambda_route_prior=cfg.get("lambda_route_prior", 0.0),
            dg_method=cfg.get("dg_method"), lambda_dg=cfg.get("lambda_dg", 1.0),
        )

        import torch
        from torch.utils.data import DataLoader

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # train_model evaluates last.pt but returns with best.pt loaded. Reload
        # last.pt explicitly so correlated/reversed/neutral all use one frozen
        # checkpoint under the manuscript's strict last-epoch protocol.
        last_checkpoint = torch.load(run_dir / "last.pt", map_location=device, weights_only=True)
        model.load_state_dict(last_checkpoint["model"], strict=True)
        model.to(device).eval()
        entry = {"model": model_name, "seed": args.seed, "recording_macro_f1": {}, "window_macro_f1": {},
                 "recording_aggregation": OFFICIAL_AGGREGATION,
                 "recording_protocol_version": PROTOCOL_VERSION,
                 "recording_tie_break": "cumulative_probability_then_lowest_class_id",
                 "num_target_recordings": len(target_recordings),
                 "checkpoint_epoch": int(last_checkpoint["epoch"]),
                 "checkpoint_sha256": sha256(run_dir / "last.pt"),
                 "training_config": {
                     "preset": preset,
                     "epochs": args.epochs,
                     "views": list(use_views),
                     "router_mode": cfg["router_mode"],
                     "dg_method": cfg.get("dg_method"),
                     "lambda_dg": cfg.get("lambda_dg", 1.0),
                 }}
        for mode in ["correlated", "reversed", "neutral"]:
            cache_m = ViewCache(sc_root / f"sc_target_{mode}", views=use_views)
            ds_m = CachedWindowDataset(
                task_dir / "test_windows.csv", cache_m, recording_filter=target_recordings
            )
            loader = DataLoader(ds_m, batch_size=256, shuffle=False)
            m = evaluate(model, loader, device, 3, use_views)
            entry["recording_macro_f1"][mode] = m["recording_macro_f1"]
            entry["window_macro_f1"][mode] = m["window_macro_f1"]
        entry["reversal_gap_rec"] = entry["recording_macro_f1"]["correlated"] - entry["recording_macro_f1"]["reversed"]
        metrics_path.write_text(json.dumps(entry, indent=2), encoding="utf-8")
        results.append(entry)
        print(f"{model_name}: {json.dumps(entry['recording_macro_f1'])} gap={entry['reversal_gap_rec']:.4f}")

    if args.output_prefix != "v2sc":
        print(f"smoke prefix {args.output_prefix}: summary table intentionally not overwritten")
        return

    # Merge every available hero-model artifact so a partial invocation cannot
    # silently erase historical rows from the official shortcut table.
    merged = []
    for model_name in HERO_MODELS:
        path = out_root / f"v2sc_{model_name}_seed{args.seed}" / "shortcut_metrics.json"
        if path.exists():
            merged.append(json.loads(path.read_text(encoding="utf-8")))
    results = merged
    md = tables / f"v2_shortcut_reversal_seed{args.seed}.md"
    with md.open("w", encoding="utf-8") as f:
        f.write(f"# v2 Shortcut Reversal (target=paderborn task, seed {args.seed})\n\n")
        f.write(
            "Class-correlated 35/55/75 Hz tone injected into source recordings; "
            f"target tone correlated/reversed/neutral on the fixed seed-{args.seed} "
            f"subset (n={len(target_recordings)} recordings).\n\n"
        )
        f.write("| Model | Corr Rec-F1 | Reversed Rec-F1 | Neutral Rec-F1 | Reversal Gap |\n|---|---:|---:|---:|---:|\n")
        for e in results:
            r = e["recording_macro_f1"]
            f.write(f"| {e['model']} | {r['correlated']:.4f} | {r['reversed']:.4f} | {r['neutral']:.4f} | {e['reversal_gap_rec']:.4f} |\n")
    print(f"-> {md}")


if __name__ == "__main__":
    main()
