#!/usr/bin/env python3
"""Train the minimal CIC-MAN model from window-index CSV files."""

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

from cicman.training.cic_man import train_cic_man  # noqa: E402


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
    parser.add_argument("--num-agents", type=int, default=4)
    parser.add_argument(
        "--architecture",
        choices=["minimal", "heterogeneous", "gated_filterbank", "gated_viewbank", "vfinal"],
        default="minimal",
    )
    parser.add_argument(
        "--core-checkpoint",
        type=Path,
        default=None,
        help="Trained v5 checkpoint used to initialize the gated-filterbank core.",
    )
    parser.add_argument(
        "--freeze-core",
        action="store_true",
        help="Freeze the initialized v5 core and train only optional branch/gate parameters.",
    )
    parser.add_argument(
        "--view-agent-checkpoint",
        type=Path,
        default=None,
        help="Pretrained vFinal view-agent checkpoint used to initialize intervention branches.",
    )
    parser.add_argument(
        "--freeze-view-agents",
        action="store_true",
        help="Freeze pretrained vFinal view encoders/heads and train only gates/auxiliary heads.",
    )
    parser.add_argument(
        "--view-bank-views",
        default="envelope,order,denoise",
        help="Comma-separated intervention views for architecture=gated_viewbank.",
    )
    parser.add_argument("--max-total-gate", type=float, default=0.35)
    parser.add_argument("--lambda-pred", type=float, default=0.5)
    parser.add_argument("--lambda-router", type=float, default=0.1)
    parser.add_argument("--lambda-router-balance", type=float, default=0.1)
    parser.add_argument("--lambda-class-router-balance", type=float, default=0.0)
    parser.add_argument("--lambda-agent", type=float, default=0.5)
    parser.add_argument("--lambda-diversity", type=float, default=0.01)
    parser.add_argument("--lambda-class-agent-diversity", type=float, default=0.0)
    parser.add_argument("--lambda-class-prototype", type=float, default=0.0)
    parser.add_argument("--lambda-prototype-coverage", type=float, default=0.0)
    parser.add_argument("--lambda-class-semantic-coverage", type=float, default=0.0)
    parser.add_argument("--class-semantic-margin", type=float, default=0.2)
    parser.add_argument("--class-semantic-fault-weight", type=float, default=2.0)
    parser.add_argument("--class-semantic-class-weights", default="")
    parser.add_argument("--lambda-view-reliability", type=float, default=0.0)
    parser.add_argument("--lambda-physics-fidelity", type=float, default=0.0)
    parser.add_argument("--lambda-gate-calibration", type=float, default=0.0)
    parser.add_argument("--gate-improvement-margin", type=float, default=0.05)
    parser.add_argument("--gate-physics-z-threshold", type=float, default=0.0)
    parser.add_argument("--gate-min-target-rate", type=float, default=0.0)
    parser.add_argument("--gate-physics-weight", type=float, default=0.25)
    parser.add_argument("--gate-health-confidence-threshold", type=float, default=-1.0)
    parser.add_argument("--gate-health-loss-margin", type=float, default=-1.0)
    parser.add_argument("--gate-health-weight", type=float, default=0.0)
    parser.add_argument("--lambda-mechanism-gate", type=float, default=0.0)
    parser.add_argument("--mechanism-gate-improvement-margin", type=float, default=0.03)
    parser.add_argument("--mechanism-gate-fidelity-z-threshold", type=float, default=0.0)
    parser.add_argument("--mechanism-gate-class-guard-margin", type=float, default=0.05)
    parser.add_argument("--mechanism-gate-domain-wise-guard", action="store_true")
    parser.add_argument("--lambda-class-conditional-gate", type=float, default=0.0)
    parser.add_argument("--class-gate-improvement-margin", type=float, default=0.02)
    parser.add_argument("--class-gate-fidelity-z-threshold", type=float, default=-0.25)
    parser.add_argument("--class-gate-inner-margin-guard", type=float, default=0.05)
    parser.add_argument("--class-gate-outer-guard-margin", type=float, default=0.05)
    parser.add_argument("--domain-mixup", action="store_true")
    parser.add_argument("--domain-style", action="store_true")
    parser.add_argument("--domain-mixup-alpha", type=float, default=0.4)
    parser.add_argument("--lambda-domain-cls", type=float, default=0.0)
    parser.add_argument("--lambda-domain-adversarial", type=float, default=0.0)
    parser.add_argument("--lambda-class-domain-adversarial", type=float, default=0.0)
    parser.add_argument("--lambda-style-domain", type=float, default=0.0)
    parser.add_argument("--lambda-health-style-orthogonal", type=float, default=0.0)
    parser.add_argument("--lambda-health-cls", type=float, default=0.0)
    parser.add_argument("--lambda-counterfactual-separation", type=float, default=0.0)
    parser.add_argument("--counterfactual-inner-margin", type=float, default=1.0)
    parser.add_argument("--counterfactual-outer-margin", type=float, default=1.0)
    parser.add_argument("--domain-adversarial-alpha", type=float, default=1.0)
    parser.add_argument("--use-health-style-split", action="store_true")
    parser.add_argument("--health-logit-weight", type=float, default=0.0)
    parser.add_argument("--lambda-domain-pred", type=float, default=0.5)
    parser.add_argument("--lambda-domain-router", type=float, default=0.1)
    parser.add_argument("--noise-std", type=float, default=0.03)
    parser.add_argument("--scale-std", type=float, default=0.10)
    parser.add_argument("--mask-ratio", type=float, default=0.05)
    parser.add_argument("--shortcut-train-mode", choices=["none", "correlated", "reversed", "neutral"], default="none")
    parser.add_argument("--shortcut-val-mode", choices=["none", "correlated", "reversed", "neutral"], default="none")
    parser.add_argument("--shortcut-test-mode", choices=["none", "correlated", "reversed", "neutral"], default="none")
    parser.add_argument("--shortcut-amplitude", type=float, default=1.0)
    parser.add_argument("--balanced-source-sampling", action="store_true")
    parser.add_argument("--balanced-domain-class-sampling", action="store_true")
    parser.add_argument(
        "--selection-metric",
        choices=["macro_f1", "mean_dataset_macro_f1", "worst_dataset_macro_f1", "last_epoch"],
        default="macro_f1",
    )
    parser.add_argument("--max-train-items", type=int, default=None)
    parser.add_argument("--max-eval-items", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    class_semantic_class_weights = (
        [float(item.strip()) for item in args.class_semantic_class_weights.split(",") if item.strip()]
        if args.class_semantic_class_weights.strip()
        else None
    )
    result = train_cic_man(
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
        num_agents=args.num_agents,
        lambda_pred=args.lambda_pred,
        lambda_router=args.lambda_router,
        lambda_router_balance=args.lambda_router_balance,
        lambda_class_router_balance=args.lambda_class_router_balance,
        lambda_agent=args.lambda_agent,
        lambda_diversity=args.lambda_diversity,
        lambda_class_agent_diversity=args.lambda_class_agent_diversity,
        lambda_class_prototype=args.lambda_class_prototype,
        lambda_prototype_coverage=args.lambda_prototype_coverage,
        lambda_class_semantic_coverage=args.lambda_class_semantic_coverage,
        class_semantic_margin=args.class_semantic_margin,
        class_semantic_fault_weight=args.class_semantic_fault_weight,
        class_semantic_class_weights=class_semantic_class_weights,
        lambda_view_reliability=args.lambda_view_reliability,
        lambda_physics_fidelity=args.lambda_physics_fidelity,
        lambda_gate_calibration=args.lambda_gate_calibration,
        gate_improvement_margin=args.gate_improvement_margin,
        gate_physics_z_threshold=args.gate_physics_z_threshold,
        gate_min_target_rate=args.gate_min_target_rate,
        gate_physics_weight=args.gate_physics_weight,
        gate_health_confidence_threshold=args.gate_health_confidence_threshold,
        gate_health_loss_margin=args.gate_health_loss_margin,
        gate_health_weight=args.gate_health_weight,
        lambda_mechanism_gate=args.lambda_mechanism_gate,
        mechanism_gate_improvement_margin=args.mechanism_gate_improvement_margin,
        mechanism_gate_fidelity_z_threshold=args.mechanism_gate_fidelity_z_threshold,
        mechanism_gate_class_guard_margin=args.mechanism_gate_class_guard_margin,
        mechanism_gate_domain_wise_guard=args.mechanism_gate_domain_wise_guard,
        lambda_class_conditional_gate=args.lambda_class_conditional_gate,
        class_gate_improvement_margin=args.class_gate_improvement_margin,
        class_gate_fidelity_z_threshold=args.class_gate_fidelity_z_threshold,
        class_gate_inner_margin_guard=args.class_gate_inner_margin_guard,
        class_gate_outer_guard_margin=args.class_gate_outer_guard_margin,
        lambda_domain_adversarial=args.lambda_domain_adversarial,
        lambda_class_domain_adversarial=args.lambda_class_domain_adversarial,
        lambda_style_domain=args.lambda_style_domain,
        lambda_health_style_orthogonal=args.lambda_health_style_orthogonal,
        lambda_health_cls=args.lambda_health_cls,
        lambda_domain_cls=args.lambda_domain_cls,
        lambda_domain_pred=args.lambda_domain_pred,
        lambda_domain_router=args.lambda_domain_router,
        lambda_counterfactual_separation=args.lambda_counterfactual_separation,
        counterfactual_inner_margin=args.counterfactual_inner_margin,
        counterfactual_outer_margin=args.counterfactual_outer_margin,
        domain_mixup_alpha=args.domain_mixup_alpha,
        domain_mixup=args.domain_mixup,
        domain_style=args.domain_style,
        noise_std=args.noise_std,
        scale_std=args.scale_std,
        mask_ratio=args.mask_ratio,
        balanced_source_sampling=args.balanced_source_sampling,
        balanced_domain_class_sampling=args.balanced_domain_class_sampling,
        selection_metric=args.selection_metric,
        max_train_items=args.max_train_items,
        max_eval_items=args.max_eval_items,
        architecture=args.architecture,
        core_checkpoint=args.core_checkpoint,
        freeze_core=args.freeze_core,
        view_agent_checkpoint=args.view_agent_checkpoint,
        freeze_view_agents=args.freeze_view_agents,
        view_bank_views=[item.strip() for item in args.view_bank_views.split(",") if item.strip()],
        max_total_gate=args.max_total_gate,
        domain_adversarial_alpha=args.domain_adversarial_alpha,
        use_health_style_split=args.use_health_style_split,
        health_logit_weight=args.health_logit_weight,
        shortcut_train_mode=args.shortcut_train_mode,
        shortcut_val_mode=args.shortcut_val_mode,
        shortcut_test_mode=args.shortcut_test_mode,
        shortcut_amplitude=args.shortcut_amplitude,
    )
    print(json.dumps({"metrics": result["test_metrics"], "checkpoint": result["best_checkpoint"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
