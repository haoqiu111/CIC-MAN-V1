"""Heterogeneous-intervention CIC-MAN model.

This version gives each agent a distinct signal view, making the agents actual
measurement-intervention experts rather than only separate classifier heads.
"""

from __future__ import annotations


def build_cic_man_heterogeneous(
    num_classes: int,
    *,
    in_channels: int = 1,
    num_agents: int = 4,
    feature_dim: int = 128,
    dropout: float = 0.2,
):
    """Build a CIC-MAN model with raw/smoothed/high-pass/envelope/filterbank views."""

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def make_encoder():
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

    class HeterogeneousCICMAN(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_classes = num_classes
            self.num_agents = num_agents
            self.encoders = nn.ModuleList([make_encoder() for _ in range(num_agents)])
            self.feature_dropout = nn.Dropout(p=dropout)
            self.router = nn.Sequential(
                nn.Linear(feature_dim * num_agents, feature_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feature_dim, num_agents),
            )
            self.agents = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Dropout(p=dropout),
                        nn.Linear(feature_dim, num_classes),
                    )
                    for _ in range(num_agents)
                ]
            )

        @staticmethod
        def _smooth(x, kernel_size: int = 33):
            return F.avg_pool1d(x, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

        def intervention_views(self, x):
            smoothed = self._smooth(x, 33)
            highpass = x - smoothed
            envelope_like = torch.sqrt(highpass.pow(2) + 1e-6)
            smooth_short = self._smooth(x, 9)
            smooth_long = self._smooth(x, 129)
            high_band = x - smooth_short
            mid_band = smooth_short - smoothed
            low_band = smoothed - smooth_long
            filterbank_envelope = torch.sqrt(
                high_band.pow(2) + 0.5 * mid_band.pow(2) + 0.25 * low_band.pow(2) + 1e-6
            )
            views = [x, smoothed, highpass, envelope_like, filterbank_envelope]
            if self.num_agents <= len(views):
                return views[: self.num_agents]
            extra = [x] * (self.num_agents - len(views))
            return views + extra

        def forward(self, x, *, return_details: bool = False):
            views = self.intervention_views(x)
            branch_features = torch.stack(
                [encoder(view) for encoder, view in zip(self.encoders, views)],
                dim=1,
            )
            router_input = self.feature_dropout(branch_features.flatten(start_dim=1))
            router_logits = self.router(router_input)
            router_weights = torch.softmax(router_logits, dim=1)
            agent_logits = torch.stack(
                [agent(branch_features[:, idx, :]) for idx, agent in enumerate(self.agents)],
                dim=1,
            )
            logits = torch.sum(agent_logits * router_weights.unsqueeze(-1), dim=1)
            features = torch.sum(branch_features * router_weights.unsqueeze(-1), dim=1)
            if return_details:
                return {
                    "logits": logits,
                    "features": features,
                    "branch_features": branch_features,
                    "router_logits": router_logits,
                    "router_weights": router_weights,
                    "agent_logits": agent_logits,
                }
            return logits

    return HeterogeneousCICMAN()
