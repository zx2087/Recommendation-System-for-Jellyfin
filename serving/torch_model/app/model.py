import torch
import torch.nn as nn


FEATURE_DIM = 8


class ToyScorer(nn.Module):
    def __init__(self, input_dim: int = FEATURE_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [num_candidates, feature_dim]
        return self.net(x).squeeze(-1)