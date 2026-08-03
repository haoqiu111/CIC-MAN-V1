"""Minimal CIC-MAN model for target-free domain-generalized fault diagnosis."""

from __future__ import annotations


def build_cic_man(
    num_classes: int,
    *,
    in_channels: int = 1,
    num_agents: int = 4,
    feature_dim: int = 128,
    dropout: float = 0.2,
):
    """Build a minimal causal intervention-consistent multi-agent router.

    Torch is imported inside the function so data-preparation scripts can run
    without PyTorch installed.
    """

    import torch
    import torch.nn as nn

    class CICMAN(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_classes = num_classes
            self.num_agents = num_agents
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
            self.feature_dropout = nn.Dropout(p=dropout)
            self.router = nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(feature_dim // 2, num_agents),
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

        def forward(self, x, *, return_details: bool = False):
            features = self.encoder(x)
            routed_features = self.feature_dropout(features)
            router_logits = self.router(routed_features)
            router_weights = torch.softmax(router_logits, dim=1)
            agent_logits = torch.stack([agent(features) for agent in self.agents], dim=1)
            logits = torch.sum(agent_logits * router_weights.unsqueeze(-1), dim=1)
            if return_details:
                return {
                    "logits": logits,
                    "features": features,
                    "router_logits": router_logits,
                    "router_weights": router_weights,
                    "agent_logits": agent_logits,
                }
            return logits

    return CICMAN()

