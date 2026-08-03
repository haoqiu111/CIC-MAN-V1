#!/usr/bin/env python3
"""Train CIC-MAN v2 or a baseline from the precomputed view cache.

Model presets
  cicman_v2   : all six views, causal router, consensus + adversarial + disentangle
  single_view : one view (set --views), plain classifier path
  ensemble    : all views, uniform router, no causal losses
  moe         : all views, feats-only router, no causal losses
  dann        : raw view only + domain adversarial

Example:
  python train_cicman_v2.py --task-dir data/paper1_cicman/cache/windows/cross_dataset_task3_source_mixed/target_dataset_hust \
      --cache-dir data/paper1_cicman/cache/views_v2 --preset cicman_v2 \
      --output-dir outputs/checkpoints/v2_cicman_hust --num-classes 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def add_src_to_path() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


add_src_to_path()

from cicman.v2.data import ALL_VIEWS, CachedWindowDataset, ViewCache  # noqa: E402
from cicman.v2.model import CICMANv2  # noqa: E402
from cicman.v2.train import train_model  # noqa: E402

PRESETS = {
    "cicman_v2": {
        "views": ALL_VIEWS,
        "router_mode": "causal",
        "lambda_view_cls": 0.3,
        "lambda_consensus": 0.1,
        "lambda_adversarial": 0.1,
        "lambda_domain_cls": 0.1,
        "lambda_orthogonal": 0.05,
        "lambda_balance": 0.01,
    },
    "single_view": {
        "views": ["env_order"],
        "router_mode": "uniform",
        "lambda_view_cls": 0.0,
        "lambda_consensus": 0.0,
        "lambda_adversarial": 0.0,
        "lambda_domain_cls": 0.0,
        "lambda_orthogonal": 0.0,
        "lambda_balance": 0.0,
    },
    "ensemble": {
        "views": ALL_VIEWS,
        "router_mode": "uniform",
        "lambda_view_cls": 0.3,
        "lambda_consensus": 0.0,
        "lambda_adversarial": 0.0,
        "lambda_domain_cls": 0.0,
        "lambda_orthogonal": 0.0,
        "lambda_balance": 0.0,
    },
    "moe": {
        "views": ALL_VIEWS,
        "router_mode": "feats_only",
        "lambda_view_cls": 0.3,
        "lambda_consensus": 0.0,
        "lambda_adversarial": 0.0,
        "lambda_domain_cls": 0.0,
        "lambda_orthogonal": 0.0,
        "lambda_balance": 0.01,
    },
    "cicman_v3": {
        "views": ALL_VIEWS,
        "router_mode": "causal",
        "lambda_view_cls": 0.3,
        "lambda_consensus": 0.1,
        "lambda_adversarial": 0.1,
        "lambda_domain_cls": 0.1,
        "lambda_orthogonal": 0.05,
        "lambda_balance": 0.0,
        "lambda_route_prior": 0.5,
        "view_dropout": 0.15,
        "needs_reliability": True,
    },
    "cicman_v4": {
        "views": ALL_VIEWS,
        "router_mode": "causal",
        "lambda_view_cls": 0.3,
        "lambda_consensus": 0.1,
        "lambda_adversarial": 0.1,
        "lambda_domain_cls": 0.1,
        "lambda_orthogonal": 0.05,
        "lambda_balance": 0.0,
        "lambda_route_prior": 1.0,
        "view_dropout": 0.15,
        "needs_reliability": True,
        "reliability_seeds": [42, 2025, 2026],
        "reliability_temperature": 0.08,
    },
    "cicman_v6ic": {
        # v4 + intervention-consistent reliability prior + intervention-augmented training
        "views": ALL_VIEWS,
        "router_mode": "causal",
        "lambda_view_cls": 0.3,
        "lambda_consensus": 0.1,
        "lambda_adversarial": 0.1,
        "lambda_domain_cls": 0.1,
        "lambda_orthogonal": 0.05,
        "lambda_balance": 0.0,
        "lambda_route_prior": 1.0,
        "view_dropout": 0.15,
        "needs_reliability": True,
        "reliability_seeds": [42],
        "reliability_temperature": 0.08,
        "reliability_extra_variants": ["views_v2_hp800"],
        "augment_variants": ["views_v2_hp800"],
    },
    "cicman_v6ic_hard": {
        # R1 revision: v6ic + hard-sample adaptive reweighting (focal gamma)
        "views": ALL_VIEWS,
        "router_mode": "causal",
        "lambda_view_cls": 0.3,
        "lambda_consensus": 0.1,
        "lambda_adversarial": 0.1,
        "lambda_domain_cls": 0.1,
        "lambda_orthogonal": 0.05,
        "lambda_balance": 0.0,
        "lambda_route_prior": 1.0,
        "view_dropout": 0.15,
        "needs_reliability": True,
        "reliability_seeds": [42],
        "reliability_temperature": 0.08,
        "reliability_extra_variants": ["views_v2_hp800"],
        "augment_variants": ["views_v2_hp800"],
        "focal_gamma": 1.5,
    },
    "cicman_v6ic_t213": {
        # torch-2.13 control twin of cicman_v6ic (identical config; separate
        # run name so torch-1.12 originals are never overwritten)
        "views": ALL_VIEWS,
        "router_mode": "causal",
        "lambda_view_cls": 0.3,
        "lambda_consensus": 0.1,
        "lambda_adversarial": 0.1,
        "lambda_domain_cls": 0.1,
        "lambda_orthogonal": 0.05,
        "lambda_balance": 0.0,
        "lambda_route_prior": 1.0,
        "view_dropout": 0.15,
        "needs_reliability": True,
        "reliability_seeds": [42],
        "reliability_temperature": 0.08,
        "reliability_extra_variants": ["views_v2_hp800"],
        "augment_variants": ["views_v2_hp800"],
    },
    # Matched ablations of the final CIC-MAN v6ic configuration.  Every
    # variant retains the HP800 intervention-augmented training union; only the
    # named component is removed.  This prevents the legacy v4 ablation table
    # from being used as evidence for the final v6ic model.
    "v6ic_no_prior": {
        "views": ALL_VIEWS,
        "router_mode": "causal",
        "lambda_view_cls": 0.3,
        "lambda_consensus": 0.1,
        "lambda_adversarial": 0.1,
        "lambda_domain_cls": 0.1,
        "lambda_orthogonal": 0.05,
        "lambda_balance": 0.0,
        "lambda_route_prior": 0.0,
        "view_dropout": 0.15,
        "augment_variants": ["views_v2_hp800"],
    },
    "v6ic_no_consensus": {
        "views": ALL_VIEWS,
        "router_mode": "causal",
        "lambda_view_cls": 0.3,
        "lambda_consensus": 0.0,
        "lambda_adversarial": 0.1,
        "lambda_domain_cls": 0.1,
        "lambda_orthogonal": 0.05,
        "lambda_balance": 0.0,
        "lambda_route_prior": 1.0,
        "view_dropout": 0.15,
        "needs_reliability": True,
        "reliability_seeds": [42],
        "reliability_temperature": 0.08,
        "reliability_extra_variants": ["views_v2_hp800"],
        "augment_variants": ["views_v2_hp800"],
    },
    "v6ic_no_disentangle": {
        "views": ALL_VIEWS,
        "router_mode": "causal",
        "lambda_view_cls": 0.3,
        "lambda_consensus": 0.1,
        "lambda_adversarial": 0.0,
        "lambda_domain_cls": 0.0,
        "lambda_orthogonal": 0.0,
        "lambda_balance": 0.0,
        "lambda_route_prior": 1.0,
        "view_dropout": 0.15,
        "needs_reliability": True,
        "reliability_seeds": [42],
        "reliability_temperature": 0.08,
        "reliability_extra_variants": ["views_v2_hp800"],
        "augment_variants": ["views_v2_hp800"],
    },
    "v6ic_no_view_dropout": {
        "views": ALL_VIEWS,
        "router_mode": "causal",
        "lambda_view_cls": 0.3,
        "lambda_consensus": 0.1,
        "lambda_adversarial": 0.1,
        "lambda_domain_cls": 0.1,
        "lambda_orthogonal": 0.05,
        "lambda_balance": 0.0,
        "lambda_route_prior": 1.0,
        "view_dropout": 0.0,
        "needs_reliability": True,
        "reliability_seeds": [42],
        "reliability_temperature": 0.08,
        "reliability_extra_variants": ["views_v2_hp800"],
        "augment_variants": ["views_v2_hp800"],
    },
    "v6ic_router_no_evidence": {
        "views": ALL_VIEWS,
        "router_mode": "feats_only",
        "lambda_view_cls": 0.3,
        "lambda_consensus": 0.1,
        "lambda_adversarial": 0.1,
        "lambda_domain_cls": 0.1,
        "lambda_orthogonal": 0.05,
        "lambda_balance": 0.0,
        "lambda_route_prior": 1.0,
        "view_dropout": 0.15,
        "needs_reliability": True,
        "reliability_seeds": [42],
        "reliability_temperature": 0.08,
        "reliability_extra_variants": ["views_v2_hp800"],
        "augment_variants": ["views_v2_hp800"],
    },
    # Ablations of the final cicman_v4 configuration
    "v4_no_prior": {
        "views": ALL_VIEWS,
        "router_mode": "causal",
        "lambda_view_cls": 0.3,
        "lambda_consensus": 0.1,
        "lambda_adversarial": 0.1,
        "lambda_domain_cls": 0.1,
        "lambda_orthogonal": 0.05,
        "lambda_balance": 0.0,
        "lambda_route_prior": 0.0,
        "view_dropout": 0.15,
    },
    "v4_no_consensus": {
        "views": ALL_VIEWS,
        "router_mode": "causal",
        "lambda_view_cls": 0.3,
        "lambda_consensus": 0.0,
        "lambda_adversarial": 0.1,
        "lambda_domain_cls": 0.1,
        "lambda_orthogonal": 0.05,
        "lambda_balance": 0.0,
        "lambda_route_prior": 1.0,
        "view_dropout": 0.15,
        "needs_reliability": True,
        "reliability_seeds": [42, 2025, 2026],
        "reliability_temperature": 0.08,
    },
    "v4_no_disentangle": {
        "views": ALL_VIEWS,
        "router_mode": "causal",
        "lambda_view_cls": 0.3,
        "lambda_consensus": 0.1,
        "lambda_adversarial": 0.0,
        "lambda_domain_cls": 0.0,
        "lambda_orthogonal": 0.0,
        "lambda_balance": 0.0,
        "lambda_route_prior": 1.0,
        "view_dropout": 0.15,
        "needs_reliability": True,
        "reliability_seeds": [42, 2025, 2026],
        "reliability_temperature": 0.08,
    },
    "v4_no_view_dropout": {
        "views": ALL_VIEWS,
        "router_mode": "causal",
        "lambda_view_cls": 0.3,
        "lambda_consensus": 0.1,
        "lambda_adversarial": 0.1,
        "lambda_domain_cls": 0.1,
        "lambda_orthogonal": 0.05,
        "lambda_balance": 0.0,
        "lambda_route_prior": 1.0,
        "view_dropout": 0.0,
        "needs_reliability": True,
        "reliability_seeds": [42, 2025, 2026],
        "reliability_temperature": 0.08,
    },
    "v4_router_no_evidence": {
        "views": ALL_VIEWS,
        "router_mode": "feats_only",
        "lambda_view_cls": 0.3,
        "lambda_consensus": 0.1,
        "lambda_adversarial": 0.1,
        "lambda_domain_cls": 0.1,
        "lambda_orthogonal": 0.05,
        "lambda_balance": 0.0,
        "lambda_route_prior": 1.0,
        "view_dropout": 0.15,
        "needs_reliability": True,
        "reliability_seeds": [42, 2025, 2026],
        "reliability_temperature": 0.08,
    },
    # Equal-backbone DG baselines: same six-view backbone + uniform fusion,
    # only the DG objective differs (fair comparison family)
    "dg_coral": {
        "views": ALL_VIEWS, "router_mode": "uniform", "lambda_view_cls": 0.3,
        "lambda_consensus": 0.0, "lambda_adversarial": 0.0, "lambda_domain_cls": 0.0,
        "lambda_orthogonal": 0.0, "lambda_balance": 0.0,
        "dg_method": "coral", "lambda_dg": 1.0,
    },
    "dg_mmd": {
        "views": ALL_VIEWS, "router_mode": "uniform", "lambda_view_cls": 0.3,
        "lambda_consensus": 0.0, "lambda_adversarial": 0.0, "lambda_domain_cls": 0.0,
        "lambda_orthogonal": 0.0, "lambda_balance": 0.0,
        "dg_method": "mmd", "lambda_dg": 1.0,
    },
    "dg_irm": {
        "views": ALL_VIEWS, "router_mode": "uniform", "lambda_view_cls": 0.3,
        "lambda_consensus": 0.0, "lambda_adversarial": 0.0, "lambda_domain_cls": 0.0,
        "lambda_orthogonal": 0.0, "lambda_balance": 0.0,
        "dg_method": "irm", "lambda_dg": 1.0,
    },
    "dg_groupdro": {
        "views": ALL_VIEWS, "router_mode": "uniform", "lambda_view_cls": 0.3,
        "lambda_consensus": 0.0, "lambda_adversarial": 0.0, "lambda_domain_cls": 0.0,
        "lambda_orthogonal": 0.0, "lambda_balance": 0.0,
        "dg_method": "groupdro", "lambda_dg": 1.0,
    },
    "dg_ccn": {
        # Equal-backbone reimplementation of Li et al.'s Causal Consistency
        # Network: class-conditional causal-consistency plus variance of
        # per-source classification losses.  The published lambda1=10 and
        # lambda2=1 are applied inside train.py.
        "views": ALL_VIEWS, "router_mode": "uniform", "lambda_view_cls": 0.3,
        "lambda_consensus": 0.0, "lambda_adversarial": 0.0, "lambda_domain_cls": 0.0,
        "lambda_orthogonal": 0.0, "lambda_balance": 0.0,
        "dg_method": "ccn", "lambda_dg": 1.0,
    },
    "dann": {
        "views": ["raw"],
        "router_mode": "uniform",
        "lambda_view_cls": 0.0,
        "lambda_consensus": 0.0,
        "lambda_adversarial": 0.5,
        "lambda_domain_cls": 0.0,
        "lambda_orthogonal": 0.0,
        "lambda_balance": 0.0,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", type=Path, required=True, help="Dir with train/val/test_windows.csv")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="cicman_v2")
    parser.add_argument("--views", default=None, help="Comma-separated view override.")
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-balanced-sampling", action="store_true")
    parser.add_argument(
        "--domain-column",
        default="dataset_id",
        help="Window-index column used as the adversarial domain label "
        "(dataset_id for cross-dataset; condition_id / speed_profile_id / bearing_type_id within-dataset).",
    )
    args = parser.parse_args()

    cfg = dict(PRESETS[args.preset])
    if args.views:
        cfg["views"] = [v.strip() for v in args.views.split(",") if v.strip()]
    views = cfg["views"]

    cache = ViewCache(args.cache_dir, views=views)
    train_ds = CachedWindowDataset(args.task_dir / "train_windows.csv", cache, domain_column=args.domain_column)
    val_ds = CachedWindowDataset(args.task_dir / "val_windows.csv", cache, domain_column=args.domain_column)
    test_ds = CachedWindowDataset(args.task_dir / "test_windows.csv", cache, domain_column=args.domain_column)
    print(f"preset={args.preset} views={views} train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    num_domains = max(len(train_ds.domains), 2)
    model = CICMANv2(
        num_classes=args.num_classes,
        views=views,
        num_domains=num_domains,
        router_mode=cfg["router_mode"],
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params/1e6:.2f}M, domains={train_ds.domains}")

    result = train_model(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
        output_dir=args.output_dir,
        views=views,
        num_classes=args.num_classes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        num_workers=args.num_workers,
        seed=args.seed,
        balanced_sampling=not args.no_balanced_sampling,
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
    t = result["test_metrics"]
    print(
        f"TEST window_f1={t['window_macro_f1']:.4f} acc={t['window_accuracy']:.4f} "
        f"rec_f1={t['recording_macro_f1']:.4f} rec_acc={t['recording_accuracy']:.4f} "
        f"router={t['mean_router_weights']}"
    )


if __name__ == "__main__":
    main()
