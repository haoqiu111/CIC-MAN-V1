"""Blueprint-oriented CIC-MAN vFinal with mechanism-guided multi-view agents."""

from __future__ import annotations


def build_cic_man_vfinal(
    num_classes: int,
    *,
    in_channels: int = 1,
    core_agents: int = 4,
    feature_dim: int = 128,
    dropout: float = 0.2,
    view_names: list[str] | tuple[str, ...] = (
        "envelope",
        "stft",
        "wavelet",
        "order",
        "denoise",
        "filterbank",
    ),
    initial_gate_bias: float = -4.0,
    max_total_gate: float = 0.30,
    num_domains: int = 2,
    domain_adversarial_alpha: float = 1.0,
    use_health_style_split: bool = True,
    health_logit_weight: float = 0.0,
):
    """Build the first full-blueprint CIC-MAN candidate.

    The model keeps the v5 raw core as the default decision path. Optional
    intervention views are allowed to contribute only through small gates whose
    inputs include source-learned branch evidence and differentiable
    fault-mechanism fidelity scores.
    """

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class _GradientReverse(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, alpha: float):
            ctx.alpha = float(alpha)
            return x.view_as(x)

        @staticmethod
        def backward(ctx, grad_output):
            return -ctx.alpha * grad_output, None

    def grad_reverse(x, alpha: float):
        return _GradientReverse.apply(x, alpha)

    valid_views = {"envelope", "stft", "wavelet", "order", "denoise", "filterbank"}
    view_names = tuple(view_names)
    unknown = sorted(set(view_names) - valid_views)
    if unknown:
        raise ValueError(f"Unknown vFinal intervention views: {unknown}")

    def encoder_block():
        return nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, feature_dim, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

    class CICMANvFinal(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_classes = num_classes
            self.core_agents = core_agents
            self.view_names = list(view_names)
            self.num_views = len(self.view_names)
            self.num_agents = core_agents + self.num_views
            self.max_total_gate = float(max_total_gate)
            self.domain_adversarial_alpha = float(domain_adversarial_alpha)
            self.use_health_style_split = bool(use_health_style_split)
            self.health_logit_weight = float(health_logit_weight)

            self.encoder = encoder_block()
            self.feature_dropout = nn.Dropout(p=dropout)
            self.router = nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(feature_dim // 2, core_agents),
            )
            self.agents = nn.ModuleList(
                [nn.Sequential(nn.Dropout(p=dropout), nn.Linear(feature_dim, num_classes)) for _ in range(core_agents)]
            )

            self.view_encoders = nn.ModuleDict({name: encoder_block() for name in self.view_names})
            self.view_agents = nn.ModuleDict(
                {name: nn.Sequential(nn.Dropout(p=dropout), nn.Linear(feature_dim, num_classes)) for name in self.view_names}
            )
            self.view_gates = nn.ModuleDict(
                {
                    name: nn.Sequential(
                        nn.Linear(feature_dim * 2 + 8, feature_dim // 2),
                        nn.ReLU(inplace=True),
                        nn.Linear(feature_dim // 2, 1),
                    )
                    for name in self.view_names
                }
            )
            for gate in self.view_gates.values():
                nn.init.constant_(gate[-1].bias, initial_gate_bias)

            self.domain_classifier = nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(feature_dim // 2, max(1, int(num_domains))),
            )
            if self.use_health_style_split:
                self.health_projector = nn.Sequential(
                    nn.Linear(feature_dim, feature_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(feature_dim, feature_dim),
                )
                self.style_projector = nn.Sequential(
                    nn.Linear(feature_dim, feature_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(feature_dim, feature_dim),
                )
                self.health_classifier = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(feature_dim, num_classes))
                self.health_domain_classifier = nn.Sequential(
                    nn.Linear(feature_dim, feature_dim // 2),
                    nn.ReLU(inplace=True),
                    nn.Linear(feature_dim // 2, max(1, int(num_domains))),
                )
                self.style_domain_classifier = nn.Sequential(
                    nn.Linear(feature_dim, feature_dim // 2),
                    nn.ReLU(inplace=True),
                    nn.Linear(feature_dim // 2, max(1, int(num_domains))),
                )

        @staticmethod
        def _smooth(x, kernel_size: int):
            return F.avg_pool1d(x, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

        @staticmethod
        def _standardize_column(x):
            return (x - x.mean(dim=0, keepdim=True)).div(x.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6))

        def make_view(self, x, name: str):
            smooth_short = self._smooth(x, 9)
            smooth_mid = self._smooth(x, 33)
            smooth_long = self._smooth(x, 129)
            high = x - smooth_short
            if name == "envelope":
                return torch.sqrt(high.pow(2) + 1e-6)
            if name == "stft":
                mid_band = smooth_short - smooth_mid
                low_band = smooth_mid - smooth_long
                return torch.sqrt(high.pow(2) + mid_band.pow(2) + 0.25 * low_band.pow(2) + 1e-6)
            if name == "wavelet":
                even = x[:, :, 0::2]
                odd = x[:, :, 1::2]
                length = min(even.size(-1), odd.size(-1))
                detail = even[:, :, :length] - odd[:, :, :length]
                return F.interpolate(detail, size=x.size(-1), mode="linear", align_corners=False)
            if name == "order":
                return x - smooth_long
            if name == "denoise":
                return x - smooth_mid
            if name == "filterbank":
                mid_band = smooth_short - smooth_mid
                low_band = smooth_mid - smooth_long
                return torch.sqrt(high.pow(2) + 0.5 * mid_band.pow(2) + 0.25 * low_band.pow(2) + 1e-6)
            raise ValueError(f"Unknown view: {name}")

        def mechanism_features(self, view):
            centered = view - view.mean(dim=-1, keepdim=True)
            abs_mean = centered.abs().mean(dim=-1).clamp_min(1e-6)
            rms = centered.pow(2).mean(dim=-1).sqrt().clamp_min(1e-6)
            std = centered.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
            normalized = centered / std
            kurtosis = torch.log1p(normalized.pow(4).mean(dim=-1))
            crest = torch.log1p(centered.abs().amax(dim=-1) / rms)
            impulse = torch.log1p(centered.abs().amax(dim=-1) / abs_mean)
            roughness = (view[:, :, 1:] - view[:, :, :-1]).abs().mean(dim=-1) / view.abs().mean(dim=-1).clamp_min(1e-6)
            smooth_mid = self._smooth(view, 33)
            high_energy = (view - smooth_mid).pow(2).mean(dim=-1) / view.pow(2).mean(dim=-1).clamp_min(1e-6)

            smooth_short = self._smooth(view, 9)
            smooth_long = self._smooth(view, 129)
            mid_band = smooth_short - smooth_mid
            low_band = smooth_mid - smooth_long
            high_band = view - smooth_short
            band_energy = torch.cat(
                [
                    low_band.pow(2).mean(dim=-1),
                    mid_band.pow(2).mean(dim=-1),
                    high_band.pow(2).mean(dim=-1),
                ],
                dim=1,
            )
            band_prob = band_energy / band_energy.sum(dim=1, keepdim=True).clamp_min(1e-6)
            band_entropy = -(band_prob * (band_prob + 1e-12).log()).sum(dim=1, keepdim=True)
            spectral_concentration = 1.0 - band_entropy / torch.log(
                torch.tensor(3.0, device=view.device, dtype=view.dtype)
            )
            peak_ratio = torch.log1p(band_energy.max(dim=1, keepdim=True).values / band_energy.mean(dim=1, keepdim=True).clamp_min(1e-6))

            features = torch.cat(
                [
                    kurtosis,
                    crest,
                    impulse,
                    roughness,
                    high_energy,
                    spectral_concentration,
                    peak_ratio,
                ],
                dim=1,
            )
            fidelity = (
                features[:, 0]
                + features[:, 1]
                + 0.5 * features[:, 2]
                + features[:, 3]
                + features[:, 4]
                + features[:, 5]
                + features[:, 6]
            )
            return features, fidelity

        def forward(self, x, *, return_details: bool = False):
            features = self.encoder(x)
            core_router_logits = self.router(self.feature_dropout(features))
            core_router_weights = torch.softmax(core_router_logits, dim=1)
            core_agent_logits = torch.stack([agent(features) for agent in self.agents], dim=1)
            core_logits = torch.sum(core_agent_logits * core_router_weights.unsqueeze(-1), dim=1)

            view_features = []
            view_logits = []
            view_gate_logits = []
            view_mechanism_features = []
            view_fidelity = []
            for name in self.view_names:
                view = self.make_view(x, name)
                mechanism, fidelity = self.mechanism_features(view)
                mechanism_z = self._standardize_column(mechanism)
                fidelity_z = self._standardize_column(fidelity.unsqueeze(1))
                branch_features = self.view_encoders[name](view)
                branch_logits = self.view_agents[name](branch_features)
                gate_input = torch.cat([features, branch_features, mechanism_z, fidelity_z], dim=1)
                view_features.append(branch_features)
                view_logits.append(branch_logits)
                view_gate_logits.append(self.view_gates[name](gate_input))
                view_mechanism_features.append(mechanism)
                view_fidelity.append(fidelity.unsqueeze(1))

            if self.num_views:
                view_features_tensor = torch.stack(view_features, dim=1)
                view_logits_tensor = torch.stack(view_logits, dim=1)
                gate_logits = torch.cat(view_gate_logits, dim=1)
                raw_gates = torch.sigmoid(gate_logits)
                gate_sum = raw_gates.sum(dim=1, keepdim=True).clamp_min(1e-6)
                scale = torch.clamp(self.max_total_gate / gate_sum, max=1.0)
                view_gates = raw_gates * scale
                core_gate = (1.0 - view_gates.sum(dim=1, keepdim=True)).clamp_min(0.0)
                logits = core_gate * core_logits + torch.sum(view_gates.unsqueeze(-1) * view_logits_tensor, dim=1)
                combined_features = core_gate * features + torch.sum(view_gates.unsqueeze(-1) * view_features_tensor, dim=1)
                router_weights = torch.cat([core_router_weights * core_gate, view_gates], dim=1)
                agent_logits = torch.cat([core_agent_logits, view_logits_tensor], dim=1)
                router_logits = torch.cat([core_router_logits, gate_logits], dim=1)
                mechanism_tensor = torch.stack(view_mechanism_features, dim=1)
                fidelity_tensor = torch.cat(view_fidelity, dim=1)
            else:
                logits = core_logits
                combined_features = features
                gate_logits = features.new_zeros((features.size(0), 0))
                view_gates = gate_logits
                view_logits_tensor = features.new_zeros((features.size(0), 0, self.num_classes))
                view_features_tensor = features.new_zeros((features.size(0), 0, features.size(1)))
                router_weights = core_router_weights
                agent_logits = core_agent_logits
                router_logits = core_router_logits
                mechanism_tensor = features.new_zeros((features.size(0), 0, 7))
                fidelity_tensor = gate_logits

            if self.use_health_style_split:
                health_features = self.health_projector(combined_features)
                style_features = self.style_projector(combined_features)
                health_logits = self.health_classifier(health_features)
                if self.health_logit_weight > 0:
                    logits = (1.0 - self.health_logit_weight) * logits + self.health_logit_weight * health_logits

            if not return_details:
                return logits

            details = {
                "logits": logits,
                "features": combined_features,
                "core_features": features,
                "core_logits": core_logits,
                "router_logits": router_logits,
                "router_weights": router_weights,
                "agent_logits": agent_logits,
                "view_names": self.view_names,
                "view_features": view_features_tensor,
                "view_logits": view_logits_tensor,
                "view_gate": view_gates,
                "view_gate_logits": gate_logits,
                "view_physics_score": fidelity_tensor,
                "view_mechanism_features": mechanism_tensor,
                "view_mechanism_fidelity": fidelity_tensor,
                "domain_logits": self.domain_classifier(grad_reverse(combined_features, self.domain_adversarial_alpha)),
            }
            if self.use_health_style_split:
                details.update(
                    {
                        "health_features": health_features,
                        "style_features": style_features,
                        "health_logits": health_logits,
                        "health_domain_logits": self.health_domain_classifier(
                            grad_reverse(health_features, self.domain_adversarial_alpha)
                        ),
                        "style_domain_logits": self.style_domain_classifier(style_features),
                    }
                )
            return details

    return CICMANvFinal()
