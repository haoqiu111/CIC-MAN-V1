"""CIC-MAN with a stable raw core and optional filterbank intervention gate."""

from __future__ import annotations


def build_cic_man_gated_filterbank(
    num_classes: int,
    *,
    in_channels: int = 1,
    core_agents: int = 4,
    feature_dim: int = 128,
    dropout: float = 0.2,
    initial_gate_bias: float = -4.0,
):
    """Build CIC-MAN-v5 style core plus a gated filterbank branch.

    The first four agents are the stable raw-signal core. The fifth branch is
    a multi-scale filterbank-envelope intervention that is mixed in only through
    a learned sample-level gate.
    """

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class GatedFilterbankCICMAN(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_classes = num_classes
            self.core_agents = core_agents
            self.num_agents = core_agents + 1
            self.encoder = nn.Sequential(
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
            self.filterbank_encoder = nn.Sequential(
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
            self.feature_dropout = nn.Dropout(p=dropout)
            self.router = nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(feature_dim // 2, core_agents),
            )
            self.filterbank_gate = nn.Sequential(
                nn.Linear(feature_dim * 2 + 1, feature_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(feature_dim // 2, 1),
            )
            nn.init.constant_(self.filterbank_gate[-1].bias, initial_gate_bias)
            self.agents = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Dropout(p=dropout),
                        nn.Linear(feature_dim, num_classes),
                    )
                    for _ in range(core_agents)
                ]
            )
            self.filterbank_agent = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(feature_dim, num_classes),
            )

        @staticmethod
        def _smooth(x, kernel_size: int):
            return F.avg_pool1d(x, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

        def filterbank_view(self, x):
            smooth_short = self._smooth(x, 9)
            smooth_mid = self._smooth(x, 33)
            smooth_long = self._smooth(x, 129)
            high_band = x - smooth_short
            mid_band = smooth_short - smooth_mid
            low_band = smooth_mid - smooth_long
            return torch.sqrt(high_band.pow(2) + 0.5 * mid_band.pow(2) + 0.25 * low_band.pow(2) + 1e-6)

        def physics_score(self, x):
            centered = x - x.mean(dim=-1, keepdim=True)
            rms = centered.pow(2).mean(dim=-1).sqrt().clamp_min(1e-6)
            std = centered.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
            normalized = centered / std
            kurtosis = torch.log1p(normalized.pow(4).mean(dim=-1))
            crest = torch.log1p(centered.abs().amax(dim=-1) / rms)
            roughness = (x[:, :, 1:] - x[:, :, :-1]).abs().mean(dim=-1) / x.abs().mean(dim=-1).clamp_min(1e-6)
            smooth_mid = self._smooth(x, 33)
            high_energy = (x - smooth_mid).pow(2).mean(dim=-1) / x.pow(2).mean(dim=-1).clamp_min(1e-6)
            score = kurtosis + crest + roughness + high_energy
            return score.squeeze(1).unsqueeze(1)

        def forward(self, x, *, return_details: bool = False):
            features = self.encoder(x)
            routed_features = self.feature_dropout(features)
            core_router_logits = self.router(routed_features)
            core_router_weights = torch.softmax(core_router_logits, dim=1)
            core_agent_logits = torch.stack([agent(features) for agent in self.agents], dim=1)
            core_logits = torch.sum(core_agent_logits * core_router_weights.unsqueeze(-1), dim=1)

            filterbank = self.filterbank_view(x)
            filter_features = self.filterbank_encoder(filterbank)
            filter_logits = self.filterbank_agent(filter_features)
            physics = self.physics_score(x)
            normalized_physics = (physics - physics.mean()).div(physics.std(unbiased=False).clamp_min(1e-6))
            gate_input = torch.cat([features, filter_features, normalized_physics], dim=1)
            gate = torch.sigmoid(self.filterbank_gate(gate_input))
            logits = (1.0 - gate) * core_logits + gate * filter_logits

            router_weights = torch.cat([core_router_weights * (1.0 - gate), gate], dim=1)
            agent_logits = torch.cat([core_agent_logits, filter_logits.unsqueeze(1)], dim=1)
            combined_features = (1.0 - gate) * features + gate * filter_features
            gate_logits = self.filterbank_gate(gate_input)
            if return_details:
                return {
                    "logits": logits,
                    "features": combined_features,
                    "core_features": features,
                    "filterbank_features": filter_features,
                    "core_logits": core_logits,
                    "filterbank_logits": filter_logits,
                    "router_logits": torch.cat([core_router_logits, self.filterbank_gate(gate_input)], dim=1),
                    "router_weights": router_weights,
                    "agent_logits": agent_logits,
                    "filterbank_gate": gate,
                    "filterbank_gate_logits": gate_logits,
                    "physics_score": physics,
                }
            return logits

    return GatedFilterbankCICMAN()
