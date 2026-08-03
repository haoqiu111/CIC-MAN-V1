"""Simple Raw 1D-CNN baseline."""

from __future__ import annotations


def build_raw_cnn(num_classes: int, in_channels: int = 1):
    """Build the Raw 1D-CNN.

    Torch is imported inside the function so non-training data-preparation
    scripts can run without PyTorch installed.
    """

    import torch.nn as nn

    class RawCNN1D(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
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
                nn.Conv1d(128, 128, kernel_size=5, stride=1, padding=2, bias=False),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool1d(1),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(p=0.2),
                nn.Linear(128, num_classes),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    return RawCNN1D()

