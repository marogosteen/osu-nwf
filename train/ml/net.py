from torch import nn
import torch


class NWFNet(nn.Module):
    # WARNING: num_classはpytorch提供モデルなどの他のモデルと共通命名になっている。
    def __init__(self, feature_size: int, num_class: int):
        super(NWFNet, self).__init__()
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
            nn.Linear(16, num_class)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x: torch.Tensor = self.linearSequential(x)
        return x
