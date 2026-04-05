import torch.nn as nn

EMBEDDING_DIM = 384
FEATURE_DIM = EMBEDDING_DIM * 2 + 3  # 771


def RecommenderMLP(input_dim: int = FEATURE_DIM):
    return nn.Sequential(
        nn.Linear(input_dim, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.3),

        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.3),

        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.2),

        nn.Linear(128, 1),
    )