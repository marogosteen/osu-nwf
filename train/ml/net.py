from torch import nn
import torch


class NNWFNet(nn.Module):
    def __init__(self, feature_size: int, output_size: int):
        super(NNWFNet, self).__init__()
        self.linearSequential = nn.Sequential(
            nn.Linear(feature_size, 64),
            nn.ReLU(),
            nn.Linear(64, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, output_size)
        )

    def forward(self, x) -> torch.Tensor:
        x: torch.Tensor = self.linearSequential(x)
        return x
