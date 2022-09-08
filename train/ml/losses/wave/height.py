import torch
from torch import nn


class WaveHeightLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.__lossfunc = torch.nn.MSELoss()

    def forward(
        self, p: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        loss = self.__lossfunc(p, t)
        return loss
