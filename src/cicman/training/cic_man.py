"""Training utilities for the minimal CIC-MAN model."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from cicman.data.dataset import WindowIndexDataset
from cicman.evaluation.metrics import accuracy, balanced_accuracy, confusion_matrix, macro_f1
from cicman.evaluation.shortcuts import apply_label_shortcut
from cicman.losses import (
    agent_diversity_loss,
    agent_supervision_loss,
    class_conditional_agent_diversity_loss,
    class_conditional_domain_adversarial_loss,
    class_conditional_router_balance_loss,
    class_conditional_soft_gate_loss,
    class_domain_prototype_consistency_loss,
    class_domain_prototype_coverage_loss,
    class_semantic_coverage_loss,
    domain_adversarial_loss,
    health_style_orthogonality_loss,
    mechanism_fidelity_guided_gate_loss,
    prediction_consistency_loss,
    physics_fidelity_router_loss,
    router_balance_loss,
    router_consistency_loss,
    source_calibrated_filterbank_gate_loss,
    source_calibrated_viewbank_gate_loss,
    source_view_reliability_loss,
    style_domain_classification_loss,
)
from cicman.models.cic_man import build_cic_man
from cicman.models.cic_man_gated_filterbank import build_cic_man_gated_filterbank
from cicman.models.cic_man_gated_viewbank import build_cic_man_gated_viewbank
from cicman.models.cic_man_heterogeneous import build_cic_man_heterogeneous
from cicman.models.cic_man_vfinal import build_cic_man_vfinal
from cicman.training.raw_cnn import limit_dataset, make_loader, set_seed


def source_intervention(x, *, noise_std: float, scale_std: float, mask_ratio: float):
    """Apply lightweight source-side signal interventions."""

    import torch

    view = x
    if scale_std > 0:
        scales = 1.0 + torch.randn(x.size(0), 1, 1, device=x.device, dtype=x.dtype) * scale_std
        view = view * scales
    if noise_std > 0:
        noise = torch.randn_like(view) * noise_std
        view = view + noise
    if mask_ratio > 0:
        length = view.size(-1)
        mask_length = max(1, int(length * mask_ratio))
        starts = torch.randint(0, max(1, length - mask_length + 1), (view.size(0),), device=view.device)
        mask = torch.ones_like(view)
        positions = torch.arange(mask_length, device=view.device).unsqueeze(0) + starts.unsqueeze(1)
        mask.scatter_(2, positions.unsqueeze(1), 0)
        view = view * mask
    return view


def domain_intervention_mixup(
    x,
    y,
    dataset_ids: list[str],
    *,
    alpha: float,
):
    """Mix same-label windows from different source domains within a batch."""

    import torch

    batch_size = x.size(0)
    partners = torch.arange(batch_size, device=x.device)
    active = torch.zeros(batch_size, dtype=torch.bool, device=x.device)
    labels = y.detach().cpu().tolist()
    for index in range(batch_size):
        candidates = [
            candidate
            for candidate in range(batch_size)
            if labels[candidate] == labels[index] and dataset_ids[candidate] != dataset_ids[index]
        ]
        if candidates:
            chosen = candidates[int(torch.randint(0, len(candidates), (1,), device=x.device).item())]
            partners[index] = chosen
            active[index] = True

    if not bool(active.any()):
        return x, active

    if alpha > 0:
        beta = torch.distributions.Beta(alpha, alpha)
        lam = beta.sample((batch_size,)).to(device=x.device, dtype=x.dtype)
    else:
        lam = torch.full((batch_size,), 0.5, device=x.device, dtype=x.dtype)
    lam = torch.maximum(lam, 1.0 - lam).view(batch_size, 1, 1)
    mixed = lam * x + (1.0 - lam) * x[partners]
    mixed = torch.where(active.view(batch_size, 1, 1), mixed, x)
    return mixed, active


def domain_style_perturbation(
    x,
    y,
    dataset_ids: list[str],
    *,
    alpha: float,
    eps: float = 1e-6,
):
    """Apply same-label cross-domain MixStyle using window mean/std statistics."""

    import torch

    batch_size = x.size(0)
    partners = torch.arange(batch_size, device=x.device)
    active = torch.zeros(batch_size, dtype=torch.bool, device=x.device)
    labels = y.detach().cpu().tolist()
    for index in range(batch_size):
        candidates = [
            candidate
            for candidate in range(batch_size)
            if labels[candidate] == labels[index] and dataset_ids[candidate] != dataset_ids[index]
        ]
        if candidates:
            chosen = candidates[int(torch.randint(0, len(candidates), (1,), device=x.device).item())]
            partners[index] = chosen
            active[index] = True

    if not bool(active.any()):
        return x, active

    if alpha > 0:
        beta = torch.distributions.Beta(alpha, alpha)
        lam = beta.sample((batch_size,)).to(device=x.device, dtype=x.dtype).view(batch_size, 1, 1)
    else:
        lam = torch.full((batch_size, 1, 1), 0.5, device=x.device, dtype=x.dtype)

    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True, unbiased=False).clamp_min(eps)
    partner_mean = mean[partners]
    partner_std = std[partners]
    mixed_mean = lam * mean + (1.0 - lam) * partner_mean
    mixed_std = lam * std + (1.0 - lam) * partner_std
    styled = (x - mean) / std * mixed_std + mixed_mean
    styled = torch.where(active.view(batch_size, 1, 1), styled, x)
    return styled, active


def counterfactual_separation_loss(
    logits,
    labels,
    active,
    *,
    inner_margin: float,
    outer_margin: float,
):
    """Preserve inner evidence and protect outer-vs-inner separation on source counterfactuals."""

    import torch
    import torch.nn.functional as F

    if not bool(active.any()):
        return logits.new_tensor(0.0), {
            "counterfactual_active_rate": 0.0,
            "counterfactual_inner_active_rate": 0.0,
            "counterfactual_outer_active_rate": 0.0,
        }
    active_logits = logits[active]
    active_labels = labels[active]
    losses = []
    inner_mask = active_labels == 1
    if bool(inner_mask.any()):
        inner_logits = active_logits[inner_mask]
        inner_score = inner_logits[:, 1] - torch.maximum(inner_logits[:, 0], inner_logits[:, 2])
        losses.append(F.relu(float(inner_margin) - inner_score).mean())
    outer_mask = active_labels == 2
    if bool(outer_mask.any()):
        outer_logits = active_logits[outer_mask]
        outer_score = outer_logits[:, 2] - outer_logits[:, 1]
        losses.append(F.relu(float(outer_margin) - outer_score).mean())
    if not losses:
        loss = logits.new_tensor(0.0)
    else:
        loss = sum(losses) / len(losses)
    return loss, {
        "counterfactual_active_rate": float(active.detach().float().mean().cpu().item()),
        "counterfactual_inner_active_rate": float((active & (labels == 1)).detach().float().mean().cpu().item()),
        "counterfactual_outer_active_rate": float((active & (labels == 2)).detach().float().mean().cpu().item()),
    }


def compute_metrics(y_true: list[int], y_pred: list[int], num_classes: int, loss: float) -> dict[str, object]:
    return {
        "loss": loss,
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred, num_classes),
        "balanced_accuracy": balanced_accuracy(y_true, y_pred, num_classes),
        "num_samples": len(y_true),
        "confusion_matrix": confusion_matrix(y_true, y_pred, num_classes).tolist(),
    }


def metadata_column(metadata: dict[str, object], key: str) -> list[str]:
    values = metadata[key]
    if isinstance(values, list):
        return [str(value) for value in values]
    return [str(value) for value in list(values)]


def grouped_metrics(
    group_true: dict[str, list[int]],
    group_pred: dict[str, list[int]],
    num_classes: int,
) -> dict[str, dict[str, object]]:
    return {
        group: compute_metrics(group_true[group], group_pred[group], num_classes, 0.0)
        for group in sorted(group_true)
    }


def rows_for_dataset(dataset) -> list[dict[str, str]]:
    if hasattr(dataset, "indices") and hasattr(dataset, "dataset"):
        return [dataset.dataset.rows[index] for index in dataset.indices]
    return dataset.rows


def limit_dataset_stratified(dataset, max_items: int | None, keys: tuple[str, ...]):
    if max_items is None or max_items <= 0 or max_items >= len(dataset):
        return dataset
    import torch

    groups: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for index, row in enumerate(dataset.rows):
        groups[tuple(str(row.get(key, "")) for key in keys)].append(index)
    if not groups:
        return limit_dataset(dataset, max_items)
    selected: list[int] = []
    leftovers: list[int] = []
    per_group = max(1, max_items // len(groups))
    for group_indices in groups.values():
        selected.extend(group_indices[:per_group])
        leftovers.extend(group_indices[per_group:])
    if len(selected) < max_items:
        selected.extend(leftovers[: max_items - len(selected)])
    return torch.utils.data.Subset(dataset, selected[:max_items])


def source_domain_map(index_csv: Path) -> dict[str, int]:
    with Path(index_csv).open(newline="", encoding="utf-8") as f:
        domains = sorted({row["dataset_id"] for row in csv.DictReader(f)})
    return {domain_id: idx for idx, domain_id in enumerate(domains)}


def make_cic_loader(
    index_csv: Path,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    max_items: int | None,
    *,
    balanced_source_sampling: bool = False,
    balanced_domain_class_sampling: bool = False,
):
    import torch

    base_dataset = WindowIndexDataset(index_csv)
    if balanced_domain_class_sampling:
        dataset = limit_dataset_stratified(base_dataset, max_items, ("dataset_id", "label_id"))
    elif balanced_source_sampling:
        dataset = limit_dataset_stratified(base_dataset, max_items, ("dataset_id",))
    else:
        dataset = limit_dataset(base_dataset, max_items)
    sampler = None
    if balanced_domain_class_sampling:
        rows = rows_for_dataset(dataset)
        counts = Counter((row["dataset_id"], row["label_id"]) for row in rows)
        weights = torch.tensor([1.0 / counts[(row["dataset_id"], row["label_id"])] for row in rows], dtype=torch.double)
        sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        shuffle = False
    elif balanced_source_sampling:
        rows = rows_for_dataset(dataset)
        counts = Counter(row["dataset_id"] for row in rows)
        weights = torch.tensor([1.0 / counts[row["dataset_id"]] for row in rows], dtype=torch.double)
        sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        shuffle = False
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def run_eval_epoch(model, loader, criterion, device: str, num_classes: int) -> dict[str, object]:
    import torch

    model.eval()
    losses = []
    all_true: list[int] = []
    all_pred: list[int] = []
    dataset_true: dict[str, list[int]] = defaultdict(list)
    dataset_pred: dict[str, list[int]] = defaultdict(list)
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            logits = model(x)
            loss = criterion(logits, y)
            losses.append(float(loss.detach().cpu().item()))
            pred = logits.argmax(dim=1).detach().cpu().numpy().tolist()
            true = y.detach().cpu().numpy().tolist()
            datasets = metadata_column(batch["metadata"], "dataset_id")
            all_pred.extend(pred)
            all_true.extend(true)
            for dataset_id, true_label, pred_label in zip(datasets, true, pred):
                dataset_true[dataset_id].append(true_label)
                dataset_pred[dataset_id].append(pred_label)
    metrics = compute_metrics(
        all_true,
        all_pred,
        num_classes,
        float(np.mean(losses)) if losses else 0.0,
    )
    by_dataset = grouped_metrics(dataset_true, dataset_pred, num_classes)
    metrics["by_dataset"] = by_dataset
    if by_dataset:
        dataset_f1 = [float(item["macro_f1"]) for item in by_dataset.values()]
        metrics["mean_dataset_macro_f1"] = float(np.mean(dataset_f1))
        metrics["worst_dataset_macro_f1"] = float(np.min(dataset_f1))
    else:
        metrics["mean_dataset_macro_f1"] = metrics["macro_f1"]
        metrics["worst_dataset_macro_f1"] = metrics["macro_f1"]
    return metrics


def run_shortcut_eval_epoch(
    model,
    loader,
    criterion,
    device: str,
    num_classes: int,
    *,
    shortcut_mode: str,
    shortcut_amplitude: float,
) -> dict[str, object]:
    import torch

    model.eval()
    losses = []
    all_true: list[int] = []
    all_pred: list[int] = []
    dataset_true: dict[str, list[int]] = defaultdict(list)
    dataset_pred: dict[str, list[int]] = defaultdict(list)
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            x = apply_label_shortcut(
                x,
                y,
                mode=shortcut_mode,
                num_classes=num_classes,
                amplitude=shortcut_amplitude,
            )
            logits = model(x)
            loss = criterion(logits, y)
            losses.append(float(loss.detach().cpu().item()))
            pred = logits.argmax(dim=1).detach().cpu().numpy().tolist()
            true = y.detach().cpu().numpy().tolist()
            datasets = metadata_column(batch["metadata"], "dataset_id")
            all_pred.extend(pred)
            all_true.extend(true)
            for dataset_id, true_label, pred_label in zip(datasets, true, pred):
                dataset_true[dataset_id].append(true_label)
                dataset_pred[dataset_id].append(pred_label)
    metrics = compute_metrics(
        all_true,
        all_pred,
        num_classes,
        float(np.mean(losses)) if losses else 0.0,
    )
    by_dataset = grouped_metrics(dataset_true, dataset_pred, num_classes)
    metrics["by_dataset"] = by_dataset
    if by_dataset:
        dataset_f1 = [float(item["macro_f1"]) for item in by_dataset.values()]
        metrics["mean_dataset_macro_f1"] = float(np.mean(dataset_f1))
        metrics["worst_dataset_macro_f1"] = float(np.min(dataset_f1))
    else:
        metrics["mean_dataset_macro_f1"] = metrics["macro_f1"]
        metrics["worst_dataset_macro_f1"] = metrics["macro_f1"]
    return metrics


def load_gated_core_checkpoint(model, checkpoint_path: Path, device: str) -> None:
    """Initialize a gated model's stable core from a trained v5 checkpoint."""

    import torch

    checkpoint = torch.load(checkpoint_path, map_location=device)
    source_state = checkpoint["model_state"]
    target_state = model.state_dict()
    core_prefixes = ("encoder.", "router.", "agents.")
    copied = {}
    missing = []
    mismatched = []
    for key, value in target_state.items():
        if not key.startswith(core_prefixes):
            continue
        if key not in source_state:
            missing.append(key)
            continue
        if tuple(source_state[key].shape) != tuple(value.shape):
            mismatched.append((key, tuple(source_state[key].shape), tuple(value.shape)))
            continue
        copied[key] = source_state[key]
    if missing or mismatched:
        raise ValueError(
            "Cannot initialize gated core from checkpoint; "
            f"missing={missing[:5]}, mismatched={mismatched[:5]}"
        )
    target_state.update(copied)
    model.load_state_dict(target_state)


def freeze_gated_core(model) -> None:
    """Freeze the v5 core so only optional branches and gates can learn."""

    model.core_frozen = True
    for module_name in ("encoder", "router", "agents"):
        module = getattr(model, module_name)
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad = False


def load_vfinal_view_agent_checkpoint(model, checkpoint_path: Path, device: str) -> None:
    """Initialize vFinal intervention view encoders/heads from pretraining."""

    import torch

    checkpoint = torch.load(checkpoint_path, map_location=device)
    source_state = checkpoint["model_state"]
    target_state = model.state_dict()
    prefixes = ("view_encoders.", "view_agents.")
    copied = {}
    missing = []
    mismatched = []
    for key, value in target_state.items():
        if not key.startswith(prefixes):
            continue
        if key not in source_state:
            missing.append(key)
            continue
        if tuple(source_state[key].shape) != tuple(value.shape):
            mismatched.append((key, tuple(source_state[key].shape), tuple(value.shape)))
            continue
        copied[key] = source_state[key]
    if missing or mismatched:
        raise ValueError(
            "Cannot initialize vFinal view agents from checkpoint; "
            f"missing={missing[:5]}, mismatched={mismatched[:5]}"
        )
    target_state.update(copied)
    model.load_state_dict(target_state)


def freeze_vfinal_view_agents(model) -> None:
    """Freeze pretrained intervention encoders/heads while gates learn."""

    model.view_agents_frozen = True
    for module_name in ("view_encoders", "view_agents"):
        module = getattr(model, module_name)
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad = False


def keep_frozen_core_eval(model) -> None:
    if not getattr(model, "core_frozen", False):
        pass
    else:
        for module_name in ("encoder", "router", "agents", "feature_dropout"):
            getattr(model, module_name).eval()
    if getattr(model, "view_agents_frozen", False):
        for module_name in ("view_encoders", "view_agents"):
            getattr(model, module_name).eval()


def validation_score(metrics: dict[str, object], selection_metric: str) -> float:
    if selection_metric == "last_epoch":
        return float(metrics["macro_f1"])
    if selection_metric == "worst_dataset_macro_f1":
        return float(metrics.get("worst_dataset_macro_f1", metrics["macro_f1"]))
    if selection_metric == "mean_dataset_macro_f1":
        return float(metrics.get("mean_dataset_macro_f1", metrics["macro_f1"]))
    if selection_metric == "macro_f1":
        return float(metrics["macro_f1"])
    raise ValueError(f"Unknown selection metric: {selection_metric}")


def run_train_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device: str,
    num_classes: int,
    *,
    lambda_pred: float,
    lambda_router: float,
    lambda_router_balance: float,
    lambda_class_router_balance: float,
    lambda_agent: float,
    lambda_diversity: float,
    lambda_class_agent_diversity: float,
    lambda_class_prototype: float,
    lambda_prototype_coverage: float,
    lambda_class_semantic_coverage: float,
    class_semantic_margin: float,
    class_semantic_fault_weight: float,
    class_semantic_class_weights: list[float] | None,
    lambda_view_reliability: float,
    lambda_physics_fidelity: float,
    lambda_gate_calibration: float,
    gate_improvement_margin: float,
    gate_physics_z_threshold: float,
    gate_min_target_rate: float,
    gate_physics_weight: float,
    gate_health_confidence_threshold: float,
    gate_health_loss_margin: float,
    gate_health_weight: float,
    lambda_mechanism_gate: float,
    mechanism_gate_improvement_margin: float,
    mechanism_gate_fidelity_z_threshold: float,
    mechanism_gate_class_guard_margin: float,
    mechanism_gate_domain_wise_guard: bool,
    lambda_class_conditional_gate: float,
    class_gate_improvement_margin: float,
    class_gate_fidelity_z_threshold: float,
    class_gate_inner_margin_guard: float,
    class_gate_outer_guard_margin: float,
    lambda_domain_adversarial: float,
    lambda_class_domain_adversarial: float,
    lambda_style_domain: float,
    lambda_health_style_orthogonal: float,
    lambda_health_cls: float,
    source_domain_to_id: dict[str, int],
    lambda_domain_cls: float,
    lambda_domain_pred: float,
    lambda_domain_router: float,
    lambda_counterfactual_separation: float,
    counterfactual_inner_margin: float,
    counterfactual_outer_margin: float,
    domain_mixup_alpha: float,
    domain_mixup: bool,
    domain_style: bool,
    noise_std: float,
    scale_std: float,
    mask_ratio: float,
    shortcut_mode: str,
    shortcut_amplitude: float,
) -> dict[str, object]:
    model.train()
    keep_frozen_core_eval(model)
    losses = []
    cls_losses = []
    pred_losses = []
    router_losses = []
    balance_losses = []
    class_balance_losses = []
    agent_losses = []
    diversity_losses = []
    class_agent_diversity_losses = []
    class_prototype_losses = []
    prototype_coverage_losses = []
    class_semantic_coverage_losses = []
    view_reliability_losses = []
    physics_fidelity_losses = []
    gate_calibration_losses = []
    gate_target_rates = []
    gate_means = []
    gate_positive_margins = []
    mechanism_gate_losses = []
    mechanism_gate_target_rates = []
    mechanism_gate_means = []
    mechanism_gate_positive_margins = []
    mechanism_gate_fidelity_means = []
    mechanism_gate_guard_rates = []
    class_gate_losses = []
    class_gate_target_rates = []
    class_gate_inner_target_rates = []
    class_gate_outer_block_rates = []
    class_gate_means = []
    class_gate_positive_margins = []
    domain_cls_losses = []
    domain_adv_losses = []
    domain_adv_accs = []
    class_domain_adv_losses = []
    class_domain_adv_accs = []
    style_domain_losses = []
    style_domain_accs = []
    health_style_orthogonal_losses = []
    health_cls_losses = []
    domain_pred_losses = []
    domain_router_losses = []
    domain_mixup_rates = []
    counterfactual_separation_losses = []
    counterfactual_active_rates = []
    counterfactual_inner_active_rates = []
    counterfactual_outer_active_rates = []
    all_true: list[int] = []
    all_pred: list[int] = []

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        dataset_ids = metadata_column(batch["metadata"], "dataset_id")
        x = apply_label_shortcut(
            x,
            y,
            mode=shortcut_mode,
            num_classes=num_classes,
            amplitude=shortcut_amplitude,
        )
        x_intervened = source_intervention(x, noise_std=noise_std, scale_std=scale_std, mask_ratio=mask_ratio)

        clean = model(x, return_details=True)
        intervened = model(x_intervened, return_details=True)

        cls_loss = criterion(clean["logits"], y) + 0.5 * criterion(intervened["logits"], y)
        pred_loss = prediction_consistency_loss(clean["logits"], intervened["logits"])
        route_loss = router_consistency_loss(clean["router_weights"], intervened["router_weights"])
        zero = clean["logits"].new_tensor(0.0)
        balance_loss = router_balance_loss(clean["router_weights"]) if lambda_router_balance > 0 else zero
        class_balance_loss = (
            class_conditional_router_balance_loss(clean["router_weights"], y, num_classes)
            if lambda_class_router_balance > 0
            else zero
        )
        agent_loss = agent_supervision_loss(clean["agent_logits"], y) if lambda_agent > 0 else zero
        div_loss = agent_diversity_loss(clean["agent_logits"]) if lambda_diversity > 0 else zero
        class_agent_div_loss = (
            class_conditional_agent_diversity_loss(clean["agent_logits"], y, num_classes)
            if lambda_class_agent_diversity > 0
            else zero
        )
        class_proto_loss = (
            class_domain_prototype_consistency_loss(clean["features"], y, dataset_ids, num_classes)
            if lambda_class_prototype > 0
            else zero
        )
        proto_coverage_loss = (
            class_domain_prototype_coverage_loss(clean["features"], y, dataset_ids, num_classes)
            if lambda_prototype_coverage > 0
            else zero
        )
        semantic_coverage_loss = (
            class_semantic_coverage_loss(
                clean["features"],
                y,
                dataset_ids,
                num_classes,
                margin=class_semantic_margin,
                fault_weight=class_semantic_fault_weight,
                class_weights=class_semantic_class_weights,
            )
            if lambda_class_semantic_coverage > 0
            else zero
        )
        view_reliability_loss = (
            source_view_reliability_loss(clean["agent_logits"], clean["router_weights"], y)
            if lambda_view_reliability > 0
            else zero
        )
        physics_fidelity_loss = (
            physics_fidelity_router_loss(x, clean["router_weights"], y)
            if lambda_physics_fidelity > 0
            else zero
        )
        if lambda_gate_calibration > 0 and "view_logits" in clean:
            gate_calibration_loss, gate_calibration_stats = source_calibrated_viewbank_gate_loss(
                clean,
                y,
                improvement_margin=gate_improvement_margin,
                physics_z_threshold=gate_physics_z_threshold,
                min_target_rate=gate_min_target_rate,
                physics_weight=gate_physics_weight,
                health_confidence_threshold=gate_health_confidence_threshold,
                health_loss_margin=gate_health_loss_margin,
                health_weight=gate_health_weight,
            )
        elif lambda_gate_calibration > 0:
            gate_calibration_loss, gate_calibration_stats = source_calibrated_filterbank_gate_loss(
                clean,
                y,
                improvement_margin=gate_improvement_margin,
                physics_z_threshold=gate_physics_z_threshold,
            )
        else:
            gate_calibration_loss = zero
            gate_calibration_stats = {
                "gate_target_rate": 0.0,
                "gate_mean": 0.0,
                "gate_positive_margin": 0.0,
            }
        if lambda_mechanism_gate > 0:
            mechanism_gate_loss, mechanism_gate_stats = mechanism_fidelity_guided_gate_loss(
                clean,
                y,
                dataset_ids,
                improvement_margin=mechanism_gate_improvement_margin,
                fidelity_z_threshold=mechanism_gate_fidelity_z_threshold,
                class_guard_margin=mechanism_gate_class_guard_margin,
                domain_wise_guard=mechanism_gate_domain_wise_guard,
            )
        else:
            mechanism_gate_loss = zero
            mechanism_gate_stats = {
                "mechanism_gate_target_rate": 0.0,
                "mechanism_gate_mean": 0.0,
                "mechanism_gate_positive_margin": 0.0,
                "mechanism_gate_fidelity_mean": 0.0,
                "mechanism_gate_guard_rate": 0.0,
            }
        if lambda_class_conditional_gate > 0:
            class_gate_loss, class_gate_stats = class_conditional_soft_gate_loss(
                clean,
                y,
                dataset_ids,
                improvement_margin=class_gate_improvement_margin,
                fidelity_z_threshold=class_gate_fidelity_z_threshold,
                inner_margin_guard=class_gate_inner_margin_guard,
                outer_guard_margin=class_gate_outer_guard_margin,
            )
        else:
            class_gate_loss = zero
            class_gate_stats = {
                "class_gate_target_rate": 0.0,
                "class_gate_inner_target_rate": 0.0,
                "class_gate_outer_block_rate": 0.0,
                "class_gate_mean": 0.0,
                "class_gate_positive_margin": 0.0,
            }
        domain_adv_loss, domain_adv_acc = (
            domain_adversarial_loss(clean, dataset_ids, source_domain_to_id)
            if lambda_domain_adversarial > 0
            else (zero, 0.0)
        )
        class_domain_adv_loss, class_domain_adv_acc = (
            class_conditional_domain_adversarial_loss(clean, y, dataset_ids, source_domain_to_id)
            if lambda_class_domain_adversarial > 0
            else (zero, 0.0)
        )
        style_domain_loss, style_domain_acc = (
            style_domain_classification_loss(clean, dataset_ids, source_domain_to_id)
            if lambda_style_domain > 0
            else (zero, 0.0)
        )
        health_style_orthogonal_loss = (
            health_style_orthogonality_loss(clean)
            if lambda_health_style_orthogonal > 0
            else zero
        )
        health_cls_loss = (
            criterion(clean["health_logits"], y)
            if lambda_health_cls > 0 and "health_logits" in clean
            else zero
        )
        domain_pred_loss = clean["logits"].new_tensor(0.0)
        domain_route_loss = clean["logits"].new_tensor(0.0)
        domain_cls_loss = clean["logits"].new_tensor(0.0)
        domain_rate = 0.0
        counterfactual_sep_loss = clean["logits"].new_tensor(0.0)
        counterfactual_sep_stats = {
            "counterfactual_active_rate": 0.0,
            "counterfactual_inner_active_rate": 0.0,
            "counterfactual_outer_active_rate": 0.0,
        }
        if domain_mixup or domain_style:
            if domain_style:
                x_domain, active = domain_style_perturbation(
                    x,
                    y,
                    dataset_ids,
                    alpha=domain_mixup_alpha,
                )
            else:
                x_domain, active = domain_intervention_mixup(
                    x,
                    y,
                    dataset_ids,
                    alpha=domain_mixup_alpha,
                )
            if bool(active.any()):
                domain = model(x_domain, return_details=True)
                if lambda_domain_cls > 0:
                    domain_cls_loss = criterion(domain["logits"][active], y[active])
                domain_pred_loss = prediction_consistency_loss(clean["logits"][active], domain["logits"][active])
                domain_route_loss = router_consistency_loss(
                    clean["router_weights"][active],
                    domain["router_weights"][active],
                )
                domain_rate = float(active.detach().float().mean().cpu().item())
        if lambda_counterfactual_separation > 0:
            x_cf, cf_active = domain_style_perturbation(
                x,
                y,
                dataset_ids,
                alpha=domain_mixup_alpha,
            )
            if bool(cf_active.any()):
                cf = model(x_cf, return_details=True)
                counterfactual_sep_loss, counterfactual_sep_stats = counterfactual_separation_loss(
                    cf["logits"],
                    y,
                    cf_active,
                    inner_margin=counterfactual_inner_margin,
                    outer_margin=counterfactual_outer_margin,
                )
        loss = (
            cls_loss
            + lambda_pred * pred_loss
            + lambda_router * route_loss
            + lambda_router_balance * balance_loss
            + lambda_class_router_balance * class_balance_loss
            + lambda_agent * agent_loss
            + lambda_diversity * div_loss
            + lambda_class_agent_diversity * class_agent_div_loss
            + lambda_class_prototype * class_proto_loss
            + lambda_prototype_coverage * proto_coverage_loss
            + lambda_class_semantic_coverage * semantic_coverage_loss
            + lambda_view_reliability * view_reliability_loss
            + lambda_physics_fidelity * physics_fidelity_loss
            + lambda_gate_calibration * gate_calibration_loss
            + lambda_mechanism_gate * mechanism_gate_loss
            + lambda_class_conditional_gate * class_gate_loss
            + lambda_domain_adversarial * domain_adv_loss
            + lambda_class_domain_adversarial * class_domain_adv_loss
            + lambda_style_domain * style_domain_loss
            + lambda_health_style_orthogonal * health_style_orthogonal_loss
            + lambda_health_cls * health_cls_loss
            + lambda_domain_pred * domain_pred_loss
            + lambda_domain_router * domain_route_loss
            + lambda_counterfactual_separation * counterfactual_sep_loss
        )
        if lambda_domain_cls > 0:
            loss = loss + lambda_domain_cls * domain_cls_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        losses.append(float(loss.detach().cpu().item()))
        cls_losses.append(float(cls_loss.detach().cpu().item()))
        pred_losses.append(float(pred_loss.detach().cpu().item()))
        router_losses.append(float(route_loss.detach().cpu().item()))
        balance_losses.append(float(balance_loss.detach().cpu().item()))
        class_balance_losses.append(float(class_balance_loss.detach().cpu().item()))
        agent_losses.append(float(agent_loss.detach().cpu().item()))
        diversity_losses.append(float(div_loss.detach().cpu().item()))
        class_agent_diversity_losses.append(float(class_agent_div_loss.detach().cpu().item()))
        class_prototype_losses.append(float(class_proto_loss.detach().cpu().item()))
        prototype_coverage_losses.append(float(proto_coverage_loss.detach().cpu().item()))
        class_semantic_coverage_losses.append(float(semantic_coverage_loss.detach().cpu().item()))
        view_reliability_losses.append(float(view_reliability_loss.detach().cpu().item()))
        physics_fidelity_losses.append(float(physics_fidelity_loss.detach().cpu().item()))
        gate_calibration_losses.append(float(gate_calibration_loss.detach().cpu().item()))
        gate_target_rates.append(gate_calibration_stats["gate_target_rate"])
        gate_means.append(gate_calibration_stats["gate_mean"])
        gate_positive_margins.append(gate_calibration_stats["gate_positive_margin"])
        mechanism_gate_losses.append(float(mechanism_gate_loss.detach().cpu().item()))
        mechanism_gate_target_rates.append(mechanism_gate_stats["mechanism_gate_target_rate"])
        mechanism_gate_means.append(mechanism_gate_stats["mechanism_gate_mean"])
        mechanism_gate_positive_margins.append(mechanism_gate_stats["mechanism_gate_positive_margin"])
        mechanism_gate_fidelity_means.append(mechanism_gate_stats["mechanism_gate_fidelity_mean"])
        mechanism_gate_guard_rates.append(mechanism_gate_stats["mechanism_gate_guard_rate"])
        class_gate_losses.append(float(class_gate_loss.detach().cpu().item()))
        class_gate_target_rates.append(class_gate_stats["class_gate_target_rate"])
        class_gate_inner_target_rates.append(class_gate_stats["class_gate_inner_target_rate"])
        class_gate_outer_block_rates.append(class_gate_stats["class_gate_outer_block_rate"])
        class_gate_means.append(class_gate_stats["class_gate_mean"])
        class_gate_positive_margins.append(class_gate_stats["class_gate_positive_margin"])
        domain_adv_losses.append(float(domain_adv_loss.detach().cpu().item()))
        domain_adv_accs.append(domain_adv_acc)
        class_domain_adv_losses.append(float(class_domain_adv_loss.detach().cpu().item()))
        class_domain_adv_accs.append(class_domain_adv_acc)
        style_domain_losses.append(float(style_domain_loss.detach().cpu().item()))
        style_domain_accs.append(style_domain_acc)
        health_style_orthogonal_losses.append(float(health_style_orthogonal_loss.detach().cpu().item()))
        health_cls_losses.append(float(health_cls_loss.detach().cpu().item()))
        domain_cls_losses.append(float(domain_cls_loss.detach().cpu().item()))
        domain_pred_losses.append(float(domain_pred_loss.detach().cpu().item()))
        domain_router_losses.append(float(domain_route_loss.detach().cpu().item()))
        domain_mixup_rates.append(domain_rate)
        counterfactual_separation_losses.append(float(counterfactual_sep_loss.detach().cpu().item()))
        counterfactual_active_rates.append(counterfactual_sep_stats["counterfactual_active_rate"])
        counterfactual_inner_active_rates.append(counterfactual_sep_stats["counterfactual_inner_active_rate"])
        counterfactual_outer_active_rates.append(counterfactual_sep_stats["counterfactual_outer_active_rate"])
        all_pred.extend(clean["logits"].argmax(dim=1).detach().cpu().numpy().tolist())
        all_true.extend(y.detach().cpu().numpy().tolist())

    metrics = compute_metrics(
        all_true,
        all_pred,
        num_classes,
        float(np.mean(losses)) if losses else 0.0,
    )
    metrics.update(
        {
            "classification_loss": float(np.mean(cls_losses)) if cls_losses else 0.0,
            "prediction_consistency_loss": float(np.mean(pred_losses)) if pred_losses else 0.0,
            "router_consistency_loss": float(np.mean(router_losses)) if router_losses else 0.0,
            "router_balance_loss": float(np.mean(balance_losses)) if balance_losses else 0.0,
            "class_router_balance_loss": float(np.mean(class_balance_losses)) if class_balance_losses else 0.0,
            "agent_supervision_loss": float(np.mean(agent_losses)) if agent_losses else 0.0,
            "agent_diversity_loss": float(np.mean(diversity_losses)) if diversity_losses else 0.0,
            "class_agent_diversity_loss": (
                float(np.mean(class_agent_diversity_losses)) if class_agent_diversity_losses else 0.0
            ),
            "class_prototype_consistency_loss": (
                float(np.mean(class_prototype_losses)) if class_prototype_losses else 0.0
            ),
            "prototype_coverage_loss": float(np.mean(prototype_coverage_losses)) if prototype_coverage_losses else 0.0,
            "class_semantic_coverage_loss": (
                float(np.mean(class_semantic_coverage_losses)) if class_semantic_coverage_losses else 0.0
            ),
            "view_reliability_loss": float(np.mean(view_reliability_losses)) if view_reliability_losses else 0.0,
            "physics_fidelity_loss": float(np.mean(physics_fidelity_losses)) if physics_fidelity_losses else 0.0,
            "gate_calibration_loss": float(np.mean(gate_calibration_losses)) if gate_calibration_losses else 0.0,
            "gate_target_rate": float(np.mean(gate_target_rates)) if gate_target_rates else 0.0,
            "filterbank_gate_mean": float(np.mean(gate_means)) if gate_means else 0.0,
            "gate_positive_margin": float(np.mean(gate_positive_margins)) if gate_positive_margins else 0.0,
            "mechanism_gate_loss": float(np.mean(mechanism_gate_losses)) if mechanism_gate_losses else 0.0,
            "mechanism_gate_target_rate": (
                float(np.mean(mechanism_gate_target_rates)) if mechanism_gate_target_rates else 0.0
            ),
            "mechanism_gate_mean": float(np.mean(mechanism_gate_means)) if mechanism_gate_means else 0.0,
            "mechanism_gate_positive_margin": (
                float(np.mean(mechanism_gate_positive_margins)) if mechanism_gate_positive_margins else 0.0
            ),
            "mechanism_gate_fidelity_mean": (
                float(np.mean(mechanism_gate_fidelity_means)) if mechanism_gate_fidelity_means else 0.0
            ),
            "mechanism_gate_guard_rate": (
                float(np.mean(mechanism_gate_guard_rates)) if mechanism_gate_guard_rates else 0.0
            ),
            "class_gate_loss": float(np.mean(class_gate_losses)) if class_gate_losses else 0.0,
            "class_gate_target_rate": float(np.mean(class_gate_target_rates)) if class_gate_target_rates else 0.0,
            "class_gate_inner_target_rate": (
                float(np.mean(class_gate_inner_target_rates)) if class_gate_inner_target_rates else 0.0
            ),
            "class_gate_outer_block_rate": (
                float(np.mean(class_gate_outer_block_rates)) if class_gate_outer_block_rates else 0.0
            ),
            "class_gate_mean": float(np.mean(class_gate_means)) if class_gate_means else 0.0,
            "class_gate_positive_margin": (
                float(np.mean(class_gate_positive_margins)) if class_gate_positive_margins else 0.0
            ),
            "domain_adversarial_loss": float(np.mean(domain_adv_losses)) if domain_adv_losses else 0.0,
            "domain_adversarial_accuracy": float(np.mean(domain_adv_accs)) if domain_adv_accs else 0.0,
            "class_domain_adversarial_loss": (
                float(np.mean(class_domain_adv_losses)) if class_domain_adv_losses else 0.0
            ),
            "class_domain_adversarial_accuracy": (
                float(np.mean(class_domain_adv_accs)) if class_domain_adv_accs else 0.0
            ),
            "style_domain_loss": float(np.mean(style_domain_losses)) if style_domain_losses else 0.0,
            "style_domain_accuracy": float(np.mean(style_domain_accs)) if style_domain_accs else 0.0,
            "health_style_orthogonality_loss": (
                float(np.mean(health_style_orthogonal_losses)) if health_style_orthogonal_losses else 0.0
            ),
            "health_classification_loss": float(np.mean(health_cls_losses)) if health_cls_losses else 0.0,
            "domain_classification_loss": float(np.mean(domain_cls_losses)) if domain_cls_losses else 0.0,
            "domain_prediction_consistency_loss": float(np.mean(domain_pred_losses)) if domain_pred_losses else 0.0,
            "domain_router_consistency_loss": float(np.mean(domain_router_losses)) if domain_router_losses else 0.0,
            "domain_mixup_rate": float(np.mean(domain_mixup_rates)) if domain_mixup_rates else 0.0,
            "counterfactual_separation_loss": (
                float(np.mean(counterfactual_separation_losses)) if counterfactual_separation_losses else 0.0
            ),
            "counterfactual_active_rate": (
                float(np.mean(counterfactual_active_rates)) if counterfactual_active_rates else 0.0
            ),
            "counterfactual_inner_active_rate": (
                float(np.mean(counterfactual_inner_active_rates)) if counterfactual_inner_active_rates else 0.0
            ),
            "counterfactual_outer_active_rate": (
                float(np.mean(counterfactual_outer_active_rates)) if counterfactual_outer_active_rates else 0.0
            ),
        }
    )
    return metrics


def train_cic_man(
    *,
    train_index: Path,
    val_index: Path,
    test_index: Path,
    output_dir: Path,
    num_classes: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    num_workers: int,
    seed: int,
    num_agents: int,
    lambda_pred: float,
    lambda_router: float,
    lambda_router_balance: float,
    lambda_class_router_balance: float,
    lambda_agent: float,
    lambda_diversity: float,
    lambda_class_agent_diversity: float,
    lambda_class_prototype: float,
    lambda_prototype_coverage: float,
    lambda_class_semantic_coverage: float,
    class_semantic_margin: float,
    class_semantic_fault_weight: float,
    class_semantic_class_weights: list[float] | None,
    lambda_view_reliability: float,
    lambda_physics_fidelity: float,
    lambda_gate_calibration: float,
    gate_improvement_margin: float,
    gate_physics_z_threshold: float,
    gate_min_target_rate: float,
    gate_physics_weight: float,
    gate_health_confidence_threshold: float,
    gate_health_loss_margin: float,
    gate_health_weight: float,
    lambda_mechanism_gate: float,
    mechanism_gate_improvement_margin: float,
    mechanism_gate_fidelity_z_threshold: float,
    mechanism_gate_class_guard_margin: float,
    mechanism_gate_domain_wise_guard: bool,
    lambda_class_conditional_gate: float,
    class_gate_improvement_margin: float,
    class_gate_fidelity_z_threshold: float,
    class_gate_inner_margin_guard: float,
    class_gate_outer_guard_margin: float,
    lambda_domain_adversarial: float,
    lambda_class_domain_adversarial: float,
    lambda_style_domain: float,
    lambda_health_style_orthogonal: float,
    lambda_health_cls: float,
    lambda_domain_cls: float,
    lambda_domain_pred: float,
    lambda_domain_router: float,
    lambda_counterfactual_separation: float,
    counterfactual_inner_margin: float,
    counterfactual_outer_margin: float,
    domain_mixup_alpha: float,
    domain_mixup: bool,
    domain_style: bool,
    noise_std: float,
    scale_std: float,
    mask_ratio: float,
    balanced_source_sampling: bool = False,
    balanced_domain_class_sampling: bool = False,
    selection_metric: str = "macro_f1",
    max_train_items: int | None = None,
    max_eval_items: int | None = None,
    architecture: str = "minimal",
    core_checkpoint: Path | None = None,
    freeze_core: bool = False,
    view_agent_checkpoint: Path | None = None,
    freeze_view_agents: bool = False,
    view_bank_views: list[str] | None = None,
    max_total_gate: float = 0.35,
    domain_adversarial_alpha: float = 1.0,
    use_health_style_split: bool = False,
    health_logit_weight: float = 0.0,
    shortcut_train_mode: str = "none",
    shortcut_val_mode: str = "none",
    shortcut_test_mode: str = "none",
    shortcut_amplitude: float = 1.0,
) -> dict[str, object]:
    import torch

    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader = make_cic_loader(
        train_index,
        batch_size,
        True,
        num_workers,
        max_train_items,
        balanced_source_sampling=balanced_source_sampling,
        balanced_domain_class_sampling=balanced_domain_class_sampling,
    )
    val_loader = make_loader(val_index, batch_size, False, num_workers, max_eval_items)
    test_loader = make_loader(test_index, batch_size, False, num_workers, max_eval_items)
    domain_to_id = source_domain_map(train_index)

    if architecture == "gated_filterbank":
        model = build_cic_man_gated_filterbank(num_classes=num_classes, core_agents=max(1, num_agents - 1)).to(device)
        num_agents = model.num_agents
        model_name = "CIC-MAN-gated-filterbank"
    elif architecture == "vfinal":
        model = build_cic_man_vfinal(
            num_classes=num_classes,
            core_agents=4,
            view_names=view_bank_views
            or ["envelope", "stft", "wavelet", "order", "denoise", "filterbank"],
            max_total_gate=max_total_gate,
            num_domains=max(1, len(domain_to_id)),
            domain_adversarial_alpha=domain_adversarial_alpha,
            use_health_style_split=use_health_style_split,
            health_logit_weight=health_logit_weight,
        ).to(device)
        num_agents = model.num_agents
        model_name = "CIC-MAN-vFinal-mechanism-viewbank"
    elif architecture == "gated_viewbank":
        model = build_cic_man_gated_viewbank(
            num_classes=num_classes,
            core_agents=4,
            view_names=view_bank_views or ["envelope", "order", "denoise"],
            max_total_gate=max_total_gate,
            num_domains=max(1, len(domain_to_id)),
            domain_adversarial_alpha=domain_adversarial_alpha,
            use_health_style_split=use_health_style_split,
            health_logit_weight=health_logit_weight,
        ).to(device)
        num_agents = model.num_agents
        model_name = "CIC-MAN-gated-viewbank"
    elif architecture == "heterogeneous":
        model = build_cic_man_heterogeneous(num_classes=num_classes, num_agents=num_agents).to(device)
        model_name = "CIC-MAN-heterogeneous"
    elif architecture == "minimal":
        model = build_cic_man(num_classes=num_classes, num_agents=num_agents).to(device)
        model_name = "CIC-MAN-minimal"
    else:
        raise ValueError(f"Unknown CIC-MAN architecture: {architecture}")
    if core_checkpoint is not None:
        if architecture not in {"gated_filterbank", "gated_viewbank", "vfinal"}:
            raise ValueError("--core-checkpoint is only supported for gated architectures")
        load_gated_core_checkpoint(model, core_checkpoint, device)
    if view_agent_checkpoint is not None:
        if architecture != "vfinal":
            raise ValueError("--view-agent-checkpoint is only supported for architecture=vfinal")
        load_vfinal_view_agent_checkpoint(model, view_agent_checkpoint, device)
    if freeze_core:
        if architecture not in {"gated_filterbank", "gated_viewbank", "vfinal"}:
            raise ValueError("--freeze-core is only supported for gated architectures")
        freeze_gated_core(model)
    if freeze_view_agents:
        if architecture != "vfinal":
            raise ValueError("--freeze-view-agents is only supported for architecture=vfinal")
        freeze_vfinal_view_agents(model)
    criterion = torch.nn.CrossEntropyLoss()
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("No trainable parameters remain after applying freeze settings.")
    optimizer = torch.optim.AdamW(trainable_parameters, lr=lr, weight_decay=1e-4)

    config = {
        "model": model_name,
        "architecture": architecture,
        "num_classes": num_classes,
        "num_agents": num_agents,
        "lambda_pred": lambda_pred,
        "lambda_router": lambda_router,
        "lambda_router_balance": lambda_router_balance,
        "lambda_class_router_balance": lambda_class_router_balance,
        "lambda_agent": lambda_agent,
        "lambda_diversity": lambda_diversity,
        "lambda_class_agent_diversity": lambda_class_agent_diversity,
        "lambda_class_prototype": lambda_class_prototype,
        "lambda_prototype_coverage": lambda_prototype_coverage,
        "lambda_class_semantic_coverage": lambda_class_semantic_coverage,
        "class_semantic_margin": class_semantic_margin,
        "class_semantic_fault_weight": class_semantic_fault_weight,
        "class_semantic_class_weights": class_semantic_class_weights or [],
        "lambda_view_reliability": lambda_view_reliability,
        "lambda_physics_fidelity": lambda_physics_fidelity,
        "lambda_gate_calibration": lambda_gate_calibration,
        "gate_improvement_margin": gate_improvement_margin,
        "gate_physics_z_threshold": gate_physics_z_threshold,
        "gate_min_target_rate": gate_min_target_rate,
        "gate_physics_weight": gate_physics_weight,
        "gate_health_confidence_threshold": gate_health_confidence_threshold,
        "gate_health_loss_margin": gate_health_loss_margin,
        "gate_health_weight": gate_health_weight,
        "lambda_mechanism_gate": lambda_mechanism_gate,
        "mechanism_gate_improvement_margin": mechanism_gate_improvement_margin,
        "mechanism_gate_fidelity_z_threshold": mechanism_gate_fidelity_z_threshold,
        "mechanism_gate_class_guard_margin": mechanism_gate_class_guard_margin,
        "mechanism_gate_domain_wise_guard": mechanism_gate_domain_wise_guard,
        "lambda_class_conditional_gate": lambda_class_conditional_gate,
        "class_gate_improvement_margin": class_gate_improvement_margin,
        "class_gate_fidelity_z_threshold": class_gate_fidelity_z_threshold,
        "class_gate_inner_margin_guard": class_gate_inner_margin_guard,
        "class_gate_outer_guard_margin": class_gate_outer_guard_margin,
        "lambda_domain_adversarial": lambda_domain_adversarial,
        "lambda_class_domain_adversarial": lambda_class_domain_adversarial,
        "lambda_style_domain": lambda_style_domain,
        "lambda_health_style_orthogonal": lambda_health_style_orthogonal,
        "lambda_health_cls": lambda_health_cls,
        "domain_adversarial_alpha": domain_adversarial_alpha,
        "use_health_style_split": use_health_style_split,
        "health_logit_weight": health_logit_weight,
        "source_domain_to_id": domain_to_id,
        "lambda_domain_cls": lambda_domain_cls,
        "lambda_domain_pred": lambda_domain_pred,
        "lambda_domain_router": lambda_domain_router,
        "lambda_counterfactual_separation": lambda_counterfactual_separation,
        "counterfactual_inner_margin": counterfactual_inner_margin,
        "counterfactual_outer_margin": counterfactual_outer_margin,
        "domain_mixup_alpha": domain_mixup_alpha,
        "domain_mixup": domain_mixup,
        "domain_style": domain_style,
        "noise_std": noise_std,
        "scale_std": scale_std,
        "mask_ratio": mask_ratio,
        "balanced_source_sampling": balanced_source_sampling,
        "balanced_domain_class_sampling": balanced_domain_class_sampling,
        "selection_metric": selection_metric,
        "core_checkpoint": str(core_checkpoint) if core_checkpoint is not None else "",
        "freeze_core": freeze_core,
        "view_agent_checkpoint": str(view_agent_checkpoint) if view_agent_checkpoint is not None else "",
        "freeze_view_agents": freeze_view_agents,
        "view_bank_views": view_bank_views or [],
        "max_total_gate": max_total_gate,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "shortcut_train_mode": shortcut_train_mode,
        "shortcut_val_mode": shortcut_val_mode,
        "shortcut_test_mode": shortcut_test_mode,
        "shortcut_amplitude": shortcut_amplitude,
    }

    best_val = -1.0
    best_path = output_dir / "best.pt"
    history = []
    for epoch in range(1, epochs + 1):
        train_metrics = run_train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            num_classes,
            lambda_pred=lambda_pred,
            lambda_router=lambda_router,
            lambda_router_balance=lambda_router_balance,
            lambda_class_router_balance=lambda_class_router_balance,
            lambda_agent=lambda_agent,
            lambda_diversity=lambda_diversity,
            lambda_class_agent_diversity=lambda_class_agent_diversity,
            lambda_class_prototype=lambda_class_prototype,
            lambda_prototype_coverage=lambda_prototype_coverage,
            lambda_class_semantic_coverage=lambda_class_semantic_coverage,
            class_semantic_margin=class_semantic_margin,
            class_semantic_fault_weight=class_semantic_fault_weight,
            class_semantic_class_weights=class_semantic_class_weights,
            lambda_view_reliability=lambda_view_reliability,
            lambda_physics_fidelity=lambda_physics_fidelity,
            lambda_gate_calibration=lambda_gate_calibration,
            gate_improvement_margin=gate_improvement_margin,
            gate_physics_z_threshold=gate_physics_z_threshold,
            gate_min_target_rate=gate_min_target_rate,
            gate_physics_weight=gate_physics_weight,
            gate_health_confidence_threshold=gate_health_confidence_threshold,
            gate_health_loss_margin=gate_health_loss_margin,
            gate_health_weight=gate_health_weight,
            lambda_mechanism_gate=lambda_mechanism_gate,
            mechanism_gate_improvement_margin=mechanism_gate_improvement_margin,
            mechanism_gate_fidelity_z_threshold=mechanism_gate_fidelity_z_threshold,
            mechanism_gate_class_guard_margin=mechanism_gate_class_guard_margin,
            mechanism_gate_domain_wise_guard=mechanism_gate_domain_wise_guard,
            lambda_class_conditional_gate=lambda_class_conditional_gate,
            class_gate_improvement_margin=class_gate_improvement_margin,
            class_gate_fidelity_z_threshold=class_gate_fidelity_z_threshold,
            class_gate_inner_margin_guard=class_gate_inner_margin_guard,
            class_gate_outer_guard_margin=class_gate_outer_guard_margin,
            lambda_domain_adversarial=lambda_domain_adversarial,
            lambda_class_domain_adversarial=lambda_class_domain_adversarial,
            lambda_style_domain=lambda_style_domain,
            lambda_health_style_orthogonal=lambda_health_style_orthogonal,
            lambda_health_cls=lambda_health_cls,
            source_domain_to_id=domain_to_id,
            lambda_domain_cls=lambda_domain_cls,
            lambda_domain_pred=lambda_domain_pred,
            lambda_domain_router=lambda_domain_router,
            lambda_counterfactual_separation=lambda_counterfactual_separation,
            counterfactual_inner_margin=counterfactual_inner_margin,
            counterfactual_outer_margin=counterfactual_outer_margin,
            domain_mixup_alpha=domain_mixup_alpha,
            domain_mixup=domain_mixup,
            domain_style=domain_style,
            noise_std=noise_std,
            scale_std=scale_std,
            mask_ratio=mask_ratio,
            shortcut_mode=shortcut_train_mode,
            shortcut_amplitude=shortcut_amplitude,
        )
        if shortcut_val_mode in {"none", "clean", ""}:
            val_metrics = run_eval_epoch(model, val_loader, criterion, device, num_classes)
        else:
            val_metrics = run_shortcut_eval_epoch(
                model,
                val_loader,
                criterion,
                device,
                num_classes,
                shortcut_mode=shortcut_val_mode,
                shortcut_amplitude=shortcut_amplitude,
            )
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))
        score = validation_score(val_metrics, selection_metric)
        record["selection_score"] = score
        if selection_metric == "last_epoch" or score > best_val:
            best_val = score
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "selection_score": score,
                    "num_classes": num_classes,
                    "num_agents": num_agents,
                    "config": config,
                },
                best_path,
            )

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    if shortcut_test_mode in {"none", "clean", ""}:
        test_metrics = run_eval_epoch(model, test_loader, criterion, device, num_classes)
    else:
        test_metrics = run_shortcut_eval_epoch(
            model,
            test_loader,
            criterion,
            device,
            num_classes,
            shortcut_mode=shortcut_test_mode,
            shortcut_amplitude=shortcut_amplitude,
        )
    result = {
        "best_checkpoint": str(best_path),
        "best_epoch": checkpoint["epoch"],
        "best_val_metrics": checkpoint["val_metrics"],
        "test_metrics": test_metrics,
        "history": history,
        "config": config,
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
