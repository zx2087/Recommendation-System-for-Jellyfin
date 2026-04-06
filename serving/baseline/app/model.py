import torch
import torch.nn as nn

EMBEDDING_DIM = 384
FEATURE_DIM = EMBEDDING_DIM * 2 + 3  # user_emb + movie_emb + [cosine, dot, l2]


class RecommenderMLP(nn.Module):
    def __init__(self, input_dim: int = FEATURE_DIM) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),   # 0
            nn.BatchNorm1d(512),         # 1
            nn.ReLU(),                   # 2
            nn.Dropout(0.3),             # 3

            nn.Linear(512, 256),         # 4
            nn.BatchNorm1d(256),         # 5
            nn.ReLU(),                   # 6
            nn.Dropout(0.3),             # 7

            nn.Linear(256, 128),         # 8
            nn.BatchNorm1d(128),         # 9
            nn.ReLU(),                   # 10
            nn.Dropout(0.2),             # 11

            nn.Linear(128, 1),           # 12
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)