"""Small generic factory for exercising the standalone HAWQ-v2 CLI."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 16, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def build(num_samples: int = 32, batch_size: int = 8, seed: int = 7):
    generator = torch.Generator().manual_seed(seed)
    inputs = torch.randn(num_samples, 2, 16, generator=generator)
    targets = torch.randint(0, 4, (num_samples,), generator=generator)
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=batch_size, shuffle=False)
    return TinyNet(), loader, nn.CrossEntropyLoss()
