import torch
import torch.nn as nn

EMBEDDING_DIM = 384
FEATURE_DIM = EMBEDDING_DIM * 2 + 3  # user_emb + movie_emb + [cosine, dot, l2]


class RecommenderMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = FEATURE_DIM,
        hidden_dims: tuple[int, int, int] = (512, 256, 128),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.BatchNorm1d(hidden_dims[2]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[2], 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)