import torch
from torch import nn


class WaveHeightLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.__lossfunc = torch.nn.MSELoss()

    def forward(
        self, p: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        t = t.to(torch.float32)
        loss = self.__lossfunc(p, t)
        return loss


class WaveHeightClassLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.__lossfunc = torch.nn.CrossEntropyLoss()

    def forward(
        self, p: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        t = t.to(torch.long)
        loss = self.__lossfunc(p[:, 0:9], t[:, 0])
        return loss
