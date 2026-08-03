"""Loss terms for the minimal CIC-MAN training objective."""

from __future__ import annotations


def prediction_consistency_loss(logits_a, logits_b, temperature: float = 1.0):
    """Symmetric KL consistency between two intervened source views."""

    import torch.nn.functional as F

    log_prob_a = F.log_softmax(logits_a / temperature, dim=1)
    log_prob_b = F.log_softmax(logits_b / temperature, dim=1)
    prob_a = log_prob_a.exp().detach()
    prob_b = log_prob_b.exp().detach()
    loss_ab = F.kl_div(log_prob_a, prob_b, reduction="batchmean")
    loss_ba = F.kl_div(log_prob_b, prob_a, reduction="batchmean")
    return 0.5 * (loss_ab + loss_ba) * (temperature**2)


def router_consistency_loss(weights_a, weights_b):
    """Keep routing decisions stable under source-side interventions."""

    import torch.nn.functional as F

    return F.mse_loss(weights_a, weights_b)


def router_balance_loss(router_weights):
    """Encourage a batch to use all agents instead of collapsing to one."""

    import torch
    import torch.nn.functional as F

    mean_weights = router_weights.mean(dim=0)
    target = torch.full_like(mean_weights, 1.0 / router_weights.size(1))
    return F.mse_loss(mean_weights, target)


def class_conditional_router_balance_loss(router_weights, labels, num_classes: int):
    """Encourage every class present in a batch to use all agents."""

    import torch
    import torch.nn.functional as F

    losses = []
    for label_id in range(num_classes):
        mask = labels == label_id
        if bool(mask.any()):
            class_mean = router_weights[mask].mean(dim=0)
            target = torch.full_like(class_mean, 1.0 / router_weights.size(1))
            losses.append(F.mse_loss(class_mean, target))
    if not losses:
        return router_weights.new_tensor(0.0)
    return sum(losses) / len(losses)


def agent_supervision_loss(agent_logits, labels):
    """Apply source label supervision to every agent head."""

    import torch.nn.functional as F

    losses = [F.cross_entropy(agent_logits[:, idx, :], labels) for idx in range(agent_logits.size(1))]
    return sum(losses) / len(losses)


def source_view_reliability_loss(agent_logits, router_weights, labels, temperature: float = 0.5):
    """Teach the router to prefer source views whose agent predicts reliably.

    For each source sample, per-agent cross entropy gives a supervised estimate
    of which intervention view preserved class-discriminative fault evidence.
    The target reliability distribution is detached so the router learns from
    agent reliability without letting agents game the target.
    """

    import torch
    import torch.nn.functional as F

    per_agent_losses = []
    for idx in range(agent_logits.size(1)):
        per_agent_losses.append(F.cross_entropy(agent_logits[:, idx, :], labels, reduction="none"))
    loss_matrix = torch.stack(per_agent_losses, dim=1)
    reliability_target = torch.softmax(-loss_matrix.detach() / temperature, dim=1)
    return F.kl_div((router_weights + 1e-12).log(), reliability_target, reduction="batchmean")


def physics_fidelity_router_loss(x, router_weights, labels, *, temperature: float = 0.5):
    """Guide routing with lightweight fault-mechanism fidelity scores.

    The scores correspond to the heterogeneous view order: raw, smoothed,
    high-pass, envelope-like, and optionally filterbank-envelope. Fault samples
    are encouraged to route toward views with stronger impulsive/filterbank
    evidence, while normal samples prefer the complementary stable views.
    """

    import torch
    import torch.nn.functional as F

    def smooth(signal, kernel_size: int = 33):
        return F.avg_pool1d(signal, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def metrics(signal):
        centered = signal - signal.mean(dim=-1, keepdim=True)
        std = centered.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
        normalized = centered / std
        kurtosis = normalized.pow(4).mean(dim=-1)
        rms = centered.pow(2).mean(dim=-1).sqrt().clamp_min(1e-6)
        crest = centered.abs().amax(dim=-1) / rms
        roughness = (signal[:, :, 1:] - signal[:, :, :-1]).abs().mean(dim=-1) / signal.abs().mean(dim=-1).clamp_min(1e-6)
        local_smooth = smooth(signal)
        high_energy = (signal - local_smooth).pow(2).mean(dim=-1) / signal.pow(2).mean(dim=-1).clamp_min(1e-6)
        return kurtosis.squeeze(1), crest.squeeze(1), roughness.squeeze(1), high_energy.squeeze(1)

    raw = x
    smoothed = smooth(raw, 33)
    highpass = raw - smoothed
    envelope_like = torch.sqrt(highpass.pow(2) + 1e-6)
    smooth_short = smooth(raw, 9)
    smooth_long = smooth(raw, 129)
    high_band = raw - smooth_short
    mid_band = smooth_short - smoothed
    low_band = smoothed - smooth_long
    filterbank_envelope = torch.sqrt(high_band.pow(2) + 0.5 * mid_band.pow(2) + 0.25 * low_band.pow(2) + 1e-6)
    views = [raw, smoothed, highpass, envelope_like, filterbank_envelope]
    view_scores = []
    for view in views[: router_weights.size(1)]:
        kurtosis, crest, roughness, high_energy = metrics(view)
        score = torch.log1p(kurtosis) + torch.log1p(crest) + roughness + high_energy
        view_scores.append(score)
    scores = torch.stack(view_scores, dim=1)
    scores = (scores - scores.mean(dim=1, keepdim=True)) / scores.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
    normal_mask = labels == 0
    scores = torch.where(normal_mask.unsqueeze(1), -scores, scores)
    target = torch.softmax(scores.detach() / temperature, dim=1)
    return F.kl_div((router_weights + 1e-12).log(), target, reduction="batchmean")


def source_calibrated_filterbank_gate_loss(
    details,
    labels,
    *,
    improvement_margin: float = 0.05,
    physics_z_threshold: float = 0.0,
    positive_weight: float = 2.0,
):
    """Open the filterbank gate only when source evidence supports it.

    A source sample is eligible for filterbank intervention when the filterbank
    head improves per-sample cross entropy over the frozen core by a margin and
    its lightweight physics score is above the batch-normalized threshold.
    """

    import torch
    import torch.nn.functional as F

    required = ("core_logits", "filterbank_logits", "filterbank_gate_logits", "physics_score")
    if any(key not in details for key in required):
        zero = details["logits"].new_tensor(0.0)
        return zero, {
            "gate_target_rate": 0.0,
            "gate_mean": 0.0,
            "gate_positive_margin": 0.0,
        }

    core_loss = F.cross_entropy(details["core_logits"], labels, reduction="none")
    filterbank_loss = F.cross_entropy(details["filterbank_logits"], labels, reduction="none")
    improvement = core_loss - filterbank_loss
    physics = details["physics_score"].squeeze(1)
    physics_z = (physics - physics.mean()).div(physics.std(unbiased=False).clamp_min(1e-6))
    target = ((improvement.detach() >= improvement_margin) & (physics_z.detach() >= physics_z_threshold)).float()
    logits = details["filterbank_gate_logits"].squeeze(1)
    weight = torch.where(target > 0, torch.full_like(target, positive_weight), torch.ones_like(target))
    loss = F.binary_cross_entropy_with_logits(logits, target, weight=weight)
    positive_margin = improvement[target > 0].mean() if bool((target > 0).any()) else improvement.new_tensor(0.0)
    return loss, {
        "gate_target_rate": float(target.detach().mean().cpu().item()),
        "gate_mean": float(torch.sigmoid(logits.detach()).mean().cpu().item()),
        "gate_positive_margin": float(positive_margin.detach().cpu().item()),
    }


def source_calibrated_viewbank_gate_loss(
    details,
    labels,
    *,
    improvement_margin: float = 0.05,
    physics_z_threshold: float = 0.0,
    min_target_rate: float = 0.0,
    physics_weight: float = 0.25,
    health_confidence_threshold: float = -1.0,
    health_loss_margin: float = -1.0,
    health_weight: float = 0.0,
    positive_weight: float = 2.0,
):
    """Open optional view-bank gates only with source loss, fidelity, and health evidence."""

    import torch
    import torch.nn.functional as F

    required = ("core_logits", "view_logits", "view_gate_logits", "view_physics_score")
    if any(key not in details for key in required):
        return source_calibrated_filterbank_gate_loss(
            details,
            labels,
            improvement_margin=improvement_margin,
            physics_z_threshold=physics_z_threshold,
            positive_weight=positive_weight,
        )

    view_logits = details["view_logits"]
    if view_logits.size(1) == 0:
        zero = details["logits"].new_tensor(0.0)
        return zero, {"gate_target_rate": 0.0, "gate_mean": 0.0, "gate_positive_margin": 0.0}

    core_loss = F.cross_entropy(details["core_logits"], labels, reduction="none").unsqueeze(1)
    view_losses = []
    for view_idx in range(view_logits.size(1)):
        view_losses.append(F.cross_entropy(view_logits[:, view_idx, :], labels, reduction="none"))
    view_loss = torch.stack(view_losses, dim=1)
    improvement = core_loss - view_loss
    physics = details["view_physics_score"]
    physics_z = (physics - physics.mean(dim=0, keepdim=True)).div(
        physics.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    )
    target = ((improvement.detach() >= improvement_margin) & (physics_z.detach() >= physics_z_threshold)).float()
    health_score = torch.zeros_like(core_loss)
    if "health_logits" in details and (health_confidence_threshold >= 0 or health_loss_margin >= 0 or health_weight > 0):
        health_prob = F.softmax(details["health_logits"], dim=1).gather(1, labels.unsqueeze(1)).clamp_min(1e-12)
        health_loss = F.cross_entropy(details["health_logits"], labels, reduction="none").unsqueeze(1)
        health_score = health_prob.detach().expand_as(improvement)
        health_support = torch.ones_like(target, dtype=torch.bool)
        if health_confidence_threshold >= 0:
            health_support = health_support & (health_score >= float(health_confidence_threshold))
        if health_loss_margin >= 0:
            health_support = health_support & (health_loss.detach().expand_as(improvement) <= core_loss.detach() + float(health_loss_margin))
        target = target * health_support.float()
    if min_target_rate > 0:
        min_count = max(1, int(round(float(min_target_rate) * target.size(0))))
        if health_weight > 0 and "health_logits" in details:
            health_z = (health_score - health_score.mean(dim=0, keepdim=True)).div(
                health_score.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
            )
        else:
            health_z = torch.zeros_like(improvement)
        score = improvement.detach() + float(physics_weight) * physics_z.detach() + float(health_weight) * health_z.detach()
        for view_idx in range(target.size(1)):
            current = int(target[:, view_idx].sum().item())
            if current >= min_count:
                continue
            needed = min_count - current
            inactive_score = score[:, view_idx].clone()
            inactive_score[target[:, view_idx] > 0] = -float("inf")
            if "health_logits" in details and (health_confidence_threshold >= 0 or health_loss_margin >= 0):
                inactive_score[~health_support[:, view_idx]] = -float("inf")
            needed = min(needed, int(torch.isfinite(inactive_score).sum().item()))
            if needed <= 0:
                continue
            selected = torch.topk(inactive_score, k=needed).indices
            target[selected, view_idx] = 1.0
    logits = details["view_gate_logits"]
    weight = torch.where(target > 0, torch.full_like(target, positive_weight), torch.ones_like(target))
    loss = F.binary_cross_entropy_with_logits(logits, target, weight=weight)
    positive_margin = improvement[target > 0].mean() if bool((target > 0).any()) else improvement.new_tensor(0.0)
    return loss, {
        "gate_target_rate": float(target.detach().mean().cpu().item()),
        "gate_mean": float(torch.sigmoid(logits.detach()).mean().cpu().item()),
        "gate_positive_margin": float(positive_margin.detach().cpu().item()),
    }


def mechanism_fidelity_guided_gate_loss(
    details,
    labels,
    dataset_ids: list[str] | None = None,
    *,
    improvement_margin: float = 0.03,
    fidelity_z_threshold: float = 0.0,
    class_guard_margin: float = 0.05,
    domain_wise_guard: bool = False,
    positive_weight: float = 2.0,
):
    """Train vFinal optional gates with source-only mechanism and class-safety evidence.

    A view is eligible when it improves source CE over the core, has above-batch
    mechanism fidelity, and does not erode the true-class margin. Inner/outer
    samples receive an explicit pairwise guard so a view cannot be selected if
    it blurs the most fragile boundary observed in the current experiments.
    """

    import torch
    import torch.nn.functional as F

    required = ("core_logits", "view_logits", "view_gate_logits", "view_mechanism_fidelity")
    if any(key not in details for key in required):
        zero = details["logits"].new_tensor(0.0)
        return zero, {
            "mechanism_gate_target_rate": 0.0,
            "mechanism_gate_mean": 0.0,
            "mechanism_gate_positive_margin": 0.0,
            "mechanism_gate_fidelity_mean": 0.0,
            "mechanism_gate_guard_rate": 0.0,
        }

    view_logits = details["view_logits"]
    if view_logits.size(1) == 0:
        zero = details["logits"].new_tensor(0.0)
        return zero, {
            "mechanism_gate_target_rate": 0.0,
            "mechanism_gate_mean": 0.0,
            "mechanism_gate_positive_margin": 0.0,
            "mechanism_gate_fidelity_mean": 0.0,
            "mechanism_gate_guard_rate": 0.0,
        }

    core_loss = F.cross_entropy(details["core_logits"], labels, reduction="none").unsqueeze(1)
    view_loss = torch.stack(
        [F.cross_entropy(view_logits[:, idx, :], labels, reduction="none") for idx in range(view_logits.size(1))],
        dim=1,
    )
    improvement = core_loss - view_loss
    fidelity = details["view_mechanism_fidelity"]
    fidelity_z = (fidelity - fidelity.mean(dim=0, keepdim=True)).div(
        fidelity.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    )

    true_index = labels.view(-1, 1, 1).expand(-1, view_logits.size(1), 1)
    view_true = view_logits.gather(2, true_index).squeeze(2)
    core_true = details["core_logits"].gather(1, labels.view(-1, 1)).expand_as(view_true)
    class_mask = torch.ones_like(view_logits, dtype=torch.bool)
    class_mask.scatter_(2, true_index, False)
    view_other = view_logits.masked_fill(~class_mask, -float("inf")).max(dim=2).values
    core_other = details["core_logits"].masked_fill(
        torch.nn.functional.one_hot(labels, num_classes=details["core_logits"].size(1)).bool(),
        -float("inf"),
    ).max(dim=1, keepdim=True).values.expand_as(view_true)
    view_margin = view_true - view_other
    core_margin = core_true - core_other
    margin_guard = view_margin >= (core_margin.detach() - float(class_guard_margin))

    boundary_guard = torch.ones_like(margin_guard)
    if details["core_logits"].size(1) > 2:
        inner_mask = labels == 1
        outer_mask = labels == 2
        if bool(inner_mask.any()):
            view_inner_gap = view_logits[inner_mask, :, 1] - view_logits[inner_mask, :, 2]
            core_inner_gap = (details["core_logits"][inner_mask, 1] - details["core_logits"][inner_mask, 2]).unsqueeze(1)
            boundary_guard[inner_mask] = boundary_guard[inner_mask] & (
                view_inner_gap >= core_inner_gap.detach() - float(class_guard_margin)
            )
        if bool(outer_mask.any()):
            view_outer_gap = view_logits[outer_mask, :, 2] - view_logits[outer_mask, :, 1]
            core_outer_gap = (details["core_logits"][outer_mask, 2] - details["core_logits"][outer_mask, 1]).unsqueeze(1)
            boundary_guard[outer_mask] = boundary_guard[outer_mask] & (
                view_outer_gap >= core_outer_gap.detach() - float(class_guard_margin)
            )

    guard = margin_guard & boundary_guard
    if domain_wise_guard and dataset_ids is not None:
        domain_class_guard = torch.zeros_like(guard)
        for domain_id in sorted(set(dataset_ids)):
            domain_mask = torch.tensor(
                [value == domain_id for value in dataset_ids],
                dtype=torch.bool,
                device=labels.device,
            )
            for label_id in sorted(set(labels.detach().cpu().tolist())):
                mask = domain_mask & (labels == int(label_id))
                if not bool(mask.any()):
                    continue
                mean_guard = guard[mask].float().mean(dim=0) >= 0.5
                mean_improvement = improvement[mask].mean(dim=0) >= -float(class_guard_margin)
                class_safe = mean_guard & mean_improvement.detach()
                domain_class_guard[mask] = class_safe.unsqueeze(0).expand(int(mask.sum().item()), -1)
        guard = guard & domain_class_guard
    target = (
        (improvement.detach() >= float(improvement_margin))
        & (fidelity_z.detach() >= float(fidelity_z_threshold))
        & guard.detach()
    ).float()
    logits = details["view_gate_logits"]
    weight = torch.where(target > 0, torch.full_like(target, positive_weight), torch.ones_like(target))
    bce = F.binary_cross_entropy_with_logits(logits, target, weight=weight)
    gate = torch.sigmoid(logits)
    fidelity_reward = -(gate * fidelity_z.detach()).mean()
    loss = bce + 0.05 * fidelity_reward
    positive_margin = improvement[target > 0].mean() if bool((target > 0).any()) else improvement.new_tensor(0.0)
    return loss, {
        "mechanism_gate_target_rate": float(target.detach().mean().cpu().item()),
        "mechanism_gate_mean": float(gate.detach().mean().cpu().item()),
        "mechanism_gate_positive_margin": float(positive_margin.detach().cpu().item()),
        "mechanism_gate_fidelity_mean": float(fidelity.detach().mean().cpu().item()),
        "mechanism_gate_guard_rate": float(guard.detach().float().mean().cpu().item()),
    }


def class_conditional_soft_gate_loss(
    details,
    labels,
    dataset_ids: list[str] | None = None,
    *,
    inner_class: int = 1,
    outer_class: int = 2,
    improvement_margin: float = 0.02,
    fidelity_z_threshold: float = -0.25,
    inner_margin_guard: float = 0.05,
    outer_guard_margin: float = 0.05,
    positive_weight: float = 3.0,
    negative_weight: float = 1.0,
):
    """Class-conditional soft gate target for pretrained view experts.

    The loss is deliberately asymmetric: optional views are encouraged to open
    mainly for source inner samples when a pretrained view improves inner
    evidence, while outer samples are explicitly discouraged from opening if a
    view weakens outer-vs-inner separation. This keeps the v5/A10 core as the
    default normal/outer path.
    """

    import torch
    import torch.nn.functional as F

    required = ("core_logits", "view_logits", "view_gate_logits", "view_mechanism_fidelity")
    if any(key not in details for key in required):
        zero = details["logits"].new_tensor(0.0)
        return zero, {
            "class_gate_target_rate": 0.0,
            "class_gate_inner_target_rate": 0.0,
            "class_gate_outer_block_rate": 0.0,
            "class_gate_mean": 0.0,
            "class_gate_positive_margin": 0.0,
        }
    view_logits = details["view_logits"]
    if view_logits.size(1) == 0:
        zero = details["logits"].new_tensor(0.0)
        return zero, {
            "class_gate_target_rate": 0.0,
            "class_gate_inner_target_rate": 0.0,
            "class_gate_outer_block_rate": 0.0,
            "class_gate_mean": 0.0,
            "class_gate_positive_margin": 0.0,
        }

    core_loss = F.cross_entropy(details["core_logits"], labels, reduction="none").unsqueeze(1)
    view_loss = torch.stack(
        [F.cross_entropy(view_logits[:, idx, :], labels, reduction="none") for idx in range(view_logits.size(1))],
        dim=1,
    )
    improvement = core_loss - view_loss
    fidelity = details["view_mechanism_fidelity"]
    fidelity_z = (fidelity - fidelity.mean(dim=0, keepdim=True)).div(
        fidelity.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    )

    inner_mask = labels == int(inner_class)
    outer_mask = labels == int(outer_class)
    inner_gap_view = view_logits[:, :, inner_class] - torch.maximum(
        view_logits[:, :, 0], view_logits[:, :, outer_class]
    )
    inner_gap_core = (
        details["core_logits"][:, inner_class]
        - torch.maximum(details["core_logits"][:, 0], details["core_logits"][:, outer_class])
    ).unsqueeze(1)
    outer_gap_view = view_logits[:, :, outer_class] - view_logits[:, :, inner_class]
    outer_gap_core = (details["core_logits"][:, outer_class] - details["core_logits"][:, inner_class]).unsqueeze(1)

    inner_target = (
        inner_mask.unsqueeze(1)
        & (improvement.detach() >= float(improvement_margin))
        & (fidelity_z.detach() >= float(fidelity_z_threshold))
        & (inner_gap_view.detach() >= inner_gap_core.detach() - float(inner_margin_guard))
    )

    outer_block = outer_mask.unsqueeze(1) & (
        outer_gap_view.detach() < outer_gap_core.detach() - float(outer_guard_margin)
    )
    normal_block = (labels == 0).unsqueeze(1) & (improvement.detach() < float(improvement_margin))
    target = inner_target.float()
    target = torch.where(outer_block | normal_block, torch.zeros_like(target), target)

    domain_risk = torch.zeros_like(target)
    if dataset_ids is not None:
        # Source-domain class-safety is soft: unsafe outer behavior increases
        # the penalty on gates for that source domain, but does not erase every
        # inner target. Hard erasure made the target rate collapse to zero.
        for domain_id in sorted(set(dataset_ids)):
            domain_mask = torch.tensor(
                [value == domain_id for value in dataset_ids],
                dtype=torch.bool,
                device=labels.device,
            )
            domain_outer = domain_mask & outer_mask
            if bool(domain_outer.any()):
                unsafe_rate = (
                    outer_gap_view[domain_outer] < outer_gap_core[domain_outer].detach() - float(outer_guard_margin)
                ).float().mean(dim=0)
                domain_risk[domain_mask] = unsafe_rate.unsqueeze(0).expand(int(domain_mask.sum().item()), -1)

    logits = details["view_gate_logits"]
    gate = torch.sigmoid(logits)
    weight = torch.where(target > 0, torch.full_like(target, positive_weight), torch.full_like(target, negative_weight))
    weight = weight + domain_risk.detach()
    loss = F.binary_cross_entropy_with_logits(logits, target, weight=weight)
    positive_margin = improvement[target > 0].mean() if bool((target > 0).any()) else improvement.new_tensor(0.0)
    return loss, {
        "class_gate_target_rate": float(target.detach().mean().cpu().item()),
        "class_gate_inner_target_rate": (
            float(target[inner_mask].detach().mean().cpu().item()) if bool(inner_mask.any()) else 0.0
        ),
        "class_gate_outer_block_rate": (
            float(outer_block[outer_mask].detach().float().mean().cpu().item()) if bool(outer_mask.any()) else 0.0
        ),
        "class_gate_mean": float(gate.detach().mean().cpu().item()),
        "class_gate_positive_margin": float(positive_margin.detach().cpu().item()),
    }


def domain_adversarial_loss(details, dataset_ids: list[str], domain_to_id: dict[str, int]):
    """Predict source domain through a gradient-reversal discriminator."""

    import torch
    import torch.nn.functional as F

    if "domain_logits" not in details or not domain_to_id:
        return details["logits"].new_tensor(0.0), 0.0
    labels = torch.tensor(
        [domain_to_id.get(domain_id, 0) for domain_id in dataset_ids],
        dtype=torch.long,
        device=details["domain_logits"].device,
    )
    logits = details["domain_logits"]
    loss = F.cross_entropy(logits, labels)
    pred = logits.detach().argmax(dim=1)
    acc = float((pred == labels).float().mean().cpu().item())
    return loss, acc


def class_conditional_domain_adversarial_loss(
    details,
    labels,
    dataset_ids: list[str],
    domain_to_id: dict[str, int],
):
    """Class-balanced health-domain adversarial loss.

    Domain confusion is averaged within each health class present in the batch
    so the majority fault/normal class cannot dominate the disentanglement
    signal.
    """

    import torch
    import torch.nn.functional as F

    if "health_domain_logits" not in details or not domain_to_id:
        return details["logits"].new_tensor(0.0), 0.0
    domain_labels = torch.tensor(
        [domain_to_id.get(domain_id, 0) for domain_id in dataset_ids],
        dtype=torch.long,
        device=details["health_domain_logits"].device,
    )
    logits = details["health_domain_logits"]
    losses = []
    for label_id in sorted(set(labels.detach().cpu().tolist())):
        mask = labels == int(label_id)
        if bool(mask.any()):
            losses.append(F.cross_entropy(logits[mask], domain_labels[mask]))
    loss = sum(losses) / len(losses) if losses else logits.new_tensor(0.0)
    pred = logits.detach().argmax(dim=1)
    acc = float((pred == domain_labels).float().mean().cpu().item())
    return loss, acc


def style_domain_classification_loss(details, dataset_ids: list[str], domain_to_id: dict[str, int]):
    """Preserve domain/style information in the style branch."""

    import torch
    import torch.nn.functional as F

    if "style_domain_logits" not in details or not domain_to_id:
        return details["logits"].new_tensor(0.0), 0.0
    labels = torch.tensor(
        [domain_to_id.get(domain_id, 0) for domain_id in dataset_ids],
        dtype=torch.long,
        device=details["style_domain_logits"].device,
    )
    logits = details["style_domain_logits"]
    loss = F.cross_entropy(logits, labels)
    pred = logits.detach().argmax(dim=1)
    acc = float((pred == labels).float().mean().cpu().item())
    return loss, acc


def health_style_orthogonality_loss(details):
    """Discourage health and style branches from encoding the same directions."""

    import torch.nn.functional as F

    if "health_features" not in details or "style_features" not in details:
        return details["logits"].new_tensor(0.0)
    health = details["health_features"] - details["health_features"].mean(dim=0, keepdim=True)
    style = details["style_features"] - details["style_features"].mean(dim=0, keepdim=True)
    health = F.normalize(health, dim=1)
    style = F.normalize(style, dim=1)
    return (health * style).sum(dim=1).pow(2).mean()


def agent_diversity_loss(agent_logits):
    """Encourage agents to avoid identical class distributions.

    The returned value is minimized, so lower cosine similarity between agent
    probability vectors gives a smaller penalty.
    """

    import torch
    import torch.nn.functional as F

    if agent_logits.size(1) < 2:
        return agent_logits.new_tensor(0.0)
    probs = F.softmax(agent_logits, dim=-1)
    probs = F.normalize(probs, dim=-1)
    similarities = torch.matmul(probs, probs.transpose(1, 2))
    num_agents = agent_logits.size(1)
    mask = ~torch.eye(num_agents, dtype=torch.bool, device=agent_logits.device)
    return similarities[:, mask].mean()


def class_conditional_agent_diversity_loss(agent_logits, labels, num_classes: int):
    """Encourage agents to learn complementary non-target boundaries per class."""

    import torch
    import torch.nn.functional as F

    if agent_logits.size(1) < 2 or num_classes < 3:
        return agent_logits.new_tensor(0.0)
    losses = []
    num_agents = agent_logits.size(1)
    off_diag = ~torch.eye(num_agents, dtype=torch.bool, device=agent_logits.device)
    for label_id in range(num_classes):
        mask = labels == label_id
        if bool(mask.any()):
            non_target_classes = [class_id for class_id in range(num_classes) if class_id != label_id]
            class_logits = agent_logits[mask][:, :, non_target_classes]
            class_agent_probs = F.softmax(class_logits, dim=-1).mean(dim=0)
            similarities = torch.matmul(class_agent_probs, class_agent_probs.transpose(0, 1))
            losses.append(similarities[off_diag].mean())
    if not losses:
        return agent_logits.new_tensor(0.0)
    return sum(losses) / len(losses)


def class_domain_prototype_consistency_loss(features, labels, dataset_ids: list[str], num_classes: int):
    """Align same-class feature prototypes across source domains within a batch."""

    import torch
    import torch.nn.functional as F

    normalized = F.normalize(features, dim=1)
    losses = []
    for label_id in range(num_classes):
        label_mask = labels == label_id
        if not bool(label_mask.any()):
            continue
        present_domains = sorted({dataset_ids[index] for index in torch.where(label_mask)[0].detach().cpu().tolist()})
        if len(present_domains) < 2:
            continue
        prototypes = []
        for domain_id in present_domains:
            domain_mask_values = [value == domain_id for value in dataset_ids]
            domain_mask = torch.tensor(domain_mask_values, dtype=torch.bool, device=features.device)
            mask = label_mask & domain_mask
            if bool(mask.any()):
                prototypes.append(normalized[mask].mean(dim=0))
        if len(prototypes) < 2:
            continue
        stacked = F.normalize(torch.stack(prototypes, dim=0), dim=1)
        class_center = F.normalize(stacked.mean(dim=0, keepdim=True), dim=1)
        losses.append(1.0 - (stacked * class_center).sum(dim=1).mean())
    if not losses:
        return features.new_tensor(0.0)
    return sum(losses) / len(losses)


def class_domain_prototype_coverage_loss(
    features,
    labels,
    dataset_ids: list[str],
    num_classes: int,
    *,
    temperature: float = 0.2,
):
    """Contrast each sample against cross-domain class prototypes.

    Positives are same-label prototypes from other source domains. Negatives are
    all different-label prototypes. This encourages source classes to form a
    geometry that is both cross-domain connected and class separated.
    """

    import torch
    import torch.nn.functional as F

    normalized = F.normalize(features, dim=1)
    prototype_rows = []
    unique_domains = sorted(set(dataset_ids))
    for domain_id in unique_domains:
        domain_mask = torch.tensor(
            [value == domain_id for value in dataset_ids],
            dtype=torch.bool,
            device=features.device,
        )
        for label_id in range(num_classes):
            mask = domain_mask & (labels == label_id)
            if bool(mask.any()):
                prototype = F.normalize(normalized[mask].mean(dim=0, keepdim=True), dim=1).squeeze(0)
                prototype_rows.append((domain_id, label_id, prototype))

    if len(prototype_rows) < 2:
        return features.new_tensor(0.0)

    prototype_domains = [row[0] for row in prototype_rows]
    prototype_labels = torch.tensor([row[1] for row in prototype_rows], dtype=torch.long, device=features.device)
    prototype_features = torch.stack([row[2] for row in prototype_rows], dim=0)
    logits = torch.matmul(normalized, prototype_features.transpose(0, 1)) / temperature

    losses = []
    for index, (label_id, domain_id) in enumerate(zip(labels.detach().cpu().tolist(), dataset_ids)):
        positive_mask = (prototype_labels == int(label_id)) & torch.tensor(
            [proto_domain != domain_id for proto_domain in prototype_domains],
            dtype=torch.bool,
            device=features.device,
        )
        negative_mask = prototype_labels != int(label_id)
        candidate_mask = positive_mask | negative_mask
        if not bool(positive_mask.any()) or not bool(negative_mask.any()):
            continue
        candidate_logits = logits[index][candidate_mask]
        candidate_positive = positive_mask[candidate_mask]
        log_probs = F.log_softmax(candidate_logits, dim=0)
        losses.append(-log_probs[candidate_positive].mean())

    if not losses:
        return features.new_tensor(0.0)
    return sum(losses) / len(losses)


def class_semantic_coverage_loss(
    features,
    labels,
    dataset_ids: list[str],
    num_classes: int,
    *,
    margin: float = 0.2,
    fault_weight: float = 2.0,
    class_weights: list[float] | None = None,
):
    """Expand source-only class support with domain-class prototypes.

    For every source sample, the nearest same-class domain prototype should be
    closer than the nearest different-class prototype by a margin. Fault classes
    can be up-weighted because the target failures currently concentrate on
    inner/outer semantic collapse rather than normal recognition.
    """

    import torch
    import torch.nn.functional as F

    normalized = F.normalize(features, dim=1)
    prototype_rows = []
    for domain_id in sorted(set(dataset_ids)):
        domain_mask = torch.tensor(
            [value == domain_id for value in dataset_ids],
            dtype=torch.bool,
            device=features.device,
        )
        for label_id in range(num_classes):
            mask = domain_mask & (labels == label_id)
            if bool(mask.any()):
                prototype = F.normalize(normalized[mask].mean(dim=0, keepdim=True), dim=1).squeeze(0)
                prototype_rows.append((domain_id, label_id, prototype))

    if len(prototype_rows) < 2:
        return features.new_tensor(0.0)

    prototype_labels = torch.tensor([row[1] for row in prototype_rows], dtype=torch.long, device=features.device)
    prototype_features = torch.stack([row[2] for row in prototype_rows], dim=0)
    similarities = torch.matmul(normalized, prototype_features.transpose(0, 1))

    losses = []
    for index, label_id in enumerate(labels.detach().cpu().tolist()):
        positive_mask = prototype_labels == int(label_id)
        negative_mask = prototype_labels != int(label_id)
        if not bool(positive_mask.any()) or not bool(negative_mask.any()):
            continue
        positive_sim = similarities[index][positive_mask].max()
        negative_sim = similarities[index][negative_mask].max()
        sample_loss = F.relu(float(margin) + negative_sim - positive_sim)
        if class_weights is not None and int(label_id) < len(class_weights):
            sample_loss = sample_loss * float(class_weights[int(label_id)])
        elif int(label_id) != 0:
            sample_loss = sample_loss * float(fault_weight)
        losses.append(sample_loss)

    if not losses:
        return features.new_tensor(0.0)
    return sum(losses) / len(losses)
