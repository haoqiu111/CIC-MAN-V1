"""CIC-MAN v2: heterogeneous intervention-view agents with causal routing.

Each precomputed signal-processing view (soft intervention on the measurement
mechanism) gets its own encoder. Per view, features split into a health
representation z_h (fault-causal, consistent across views) and a domain
representation z_d (rig/measurement-specific, free to vary). A shared
classifier acts on z_h of every view; the router fuses per-view logits using
signal quality/fidelity features, cross-view consensus distances, and
per-view prediction uncertainty.

The same class also implements the ablation/baseline family:
  - single view  : views=[one], routing trivial
  - avg ensemble : uniform router
  - plain MoE    : router on features only, no consensus/adversarial terms
  - DANN         : views=[raw], adversarial on
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

VIEW_KINDS = {
    "raw": "wave",
    "denoise": "wave",
    "env_spec": "spec",
    "env_order": "spec",
    "stft": "img",
    "cwt": "img",
}


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.alpha * grad, None


def grad_reverse(x: torch.Tensor, alpha: float) -> torch.Tensor:
    return GradientReversal.apply(x, alpha)


def wave_encoder(feature_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv1d(1, 32, 15, stride=2, padding=7, bias=False),
        nn.BatchNorm1d(32),
        nn.ReLU(inplace=True),
        nn.MaxPool1d(2),
        nn.Conv1d(32, 64, 9, stride=2, padding=4, bias=False),
        nn.BatchNorm1d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool1d(2),
        nn.Conv1d(64, 128, 7, stride=2, padding=3, bias=False),
        nn.BatchNorm1d(128),
        nn.ReLU(inplace=True),
        nn.Conv1d(128, feature_dim, 5, stride=1, padding=2, bias=False),
        nn.BatchNorm1d(feature_dim),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool1d(1),
        nn.Flatten(),
    )


def spec_encoder(feature_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv1d(1, 32, 9, stride=2, padding=4, bias=False),
        nn.BatchNorm1d(32),
        nn.ReLU(inplace=True),
        nn.Conv1d(32, 64, 7, stride=2, padding=3, bias=False),
        nn.BatchNorm1d(64),
        nn.ReLU(inplace=True),
        nn.Conv1d(64, feature_dim, 5, stride=2, padding=2, bias=False),
        nn.BatchNorm1d(feature_dim),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool1d(1),
        nn.Flatten(),
    )


def img_encoder(feature_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(1, 32, 3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Conv2d(64, feature_dim, 3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(feature_dim),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
    )


class CICMANv2(nn.Module):
    def __init__(
        self,
        num_classes: int,
        views: list[str],
        *,
        feature_dim: int = 128,
        health_dim: int = 64,
        domain_dim: int = 32,
        feat_dim: int = 16,
        num_domains: int = 3,
        router_mode: str = "causal",  # causal | feats_only | uniform
        dropout: float = 0.2,
        view_dropout: float = 0.0,
        router_prior: list[float] | None = None,
    ):
        super().__init__()
        self.views = list(views)
        self.num_classes = num_classes
        self.router_mode = router_mode
        self.view_dropout = view_dropout
        if router_prior is not None:
            prior = torch.tensor(router_prior, dtype=torch.float32)
            prior = prior / prior.sum()
            self.register_buffer("router_log_prior", torch.log(prior.clamp_min(1e-6)))
        else:
            self.router_log_prior = None

        encoders = {}
        for v in self.views:
            kind = VIEW_KINDS[v]
            if kind == "wave":
                encoders[v] = wave_encoder(feature_dim)
            elif kind == "spec":
                encoders[v] = spec_encoder(feature_dim)
            else:
                encoders[v] = img_encoder(feature_dim)
        self.encoders = nn.ModuleDict(encoders)

        self.health_proj = nn.ModuleDict(
            {v: nn.Sequential(nn.Linear(feature_dim, health_dim), nn.ReLU(inplace=True), nn.Linear(health_dim, health_dim)) for v in self.views}
        )
        self.domain_proj = nn.ModuleDict(
            {v: nn.Sequential(nn.Linear(feature_dim, domain_dim), nn.ReLU(inplace=True), nn.Linear(domain_dim, domain_dim)) for v in self.views}
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(health_dim, num_classes)
        self.domain_adv_head = nn.Sequential(nn.Linear(health_dim, 64), nn.ReLU(inplace=True), nn.Linear(64, num_domains))
        self.domain_cls_head = nn.Sequential(nn.Linear(domain_dim, 64), nn.ReLU(inplace=True), nn.Linear(64, num_domains))

        k = len(self.views)
        router_in = feat_dim + 2 * k
        self.router = nn.Sequential(nn.Linear(router_in, 64), nn.ReLU(inplace=True), nn.Linear(64, k))

    def forward(self, views: dict[str, torch.Tensor], feats: torch.Tensor, *, adv_alpha: float = 0.0):
        z_h, z_d, logits_per_view = [], [], []
        for v in self.views:
            x = views[v]
            if VIEW_KINDS[v] in ("wave", "spec") and x.dim() == 2:
                x = x.unsqueeze(1)
            elif VIEW_KINDS[v] == "img" and x.dim() == 3:
                x = x.unsqueeze(1)
            f = self.encoders[v](x)
            h = self.health_proj[v](f)
            d = self.domain_proj[v](f)
            z_h.append(h)
            z_d.append(d)
            logits_per_view.append(self.classifier(self.dropout(h)))

        z_h = torch.stack(z_h, dim=1)  # (B, K, H)
        z_d = torch.stack(z_d, dim=1)  # (B, K, D)
        logits_pv = torch.stack(logits_per_view, dim=1)  # (B, K, C)

        # router evidence: consensus distance + prediction entropy per view
        center = z_h.mean(dim=1, keepdim=True)
        cons_dist = torch.norm(z_h - center, dim=-1)  # (B, K)
        probs = torch.softmax(logits_pv, dim=-1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-9))).sum(-1)  # (B, K)

        if self.router_mode == "uniform":
            weights = torch.full_like(cons_dist, 1.0 / len(self.views))
            router_logits = torch.zeros_like(cons_dist)
        else:
            if self.router_mode == "feats_only":
                evidence = torch.cat([feats, torch.zeros_like(cons_dist), torch.zeros_like(entropy)], dim=1)
            else:
                evidence = torch.cat([feats, cons_dist.detach(), entropy.detach()], dim=1)
            router_logits = self.router(evidence)
            if self.router_log_prior is not None:
                router_logits = router_logits + self.router_log_prior
            weights = torch.softmax(router_logits, dim=1)

        if self.training and self.view_dropout > 0 and len(self.views) > 1:
            keep = (torch.rand_like(weights) > self.view_dropout).float()
            keep[keep.sum(1) == 0] = 1.0
            weights = weights * keep
            weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-9)

        fused_logits = (logits_pv * weights.unsqueeze(-1)).sum(dim=1)
        fused_h = (z_h * weights.detach().unsqueeze(-1)).sum(dim=1)

        adv_logits = None
        if adv_alpha > 0:
            adv_logits = self.domain_adv_head(grad_reverse(fused_h, adv_alpha))
        dom_logits = self.domain_cls_head(z_d.reshape(-1, z_d.size(-1)))

        return {
            "logits": fused_logits,
            "logits_per_view": logits_pv,
            "router_weights": weights,
            "router_logits": router_logits,
            "z_h": z_h,
            "z_d": z_d,
            "adv_logits": adv_logits,
            "domain_logits": dom_logits,
            "consensus_distance": cons_dist,
        }
