import torch
from torch import nn


class WindVelocityClassLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.__lossfunc = torch.nn.CrossEntropyLoss()

    def forward(
        self, p: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        loss = self.__lossfunc(p[:, 0:20], t[:, 0])
        loss += self.__lossfunc(p[:, 20:40], t[:, 1])
        loss += self.__lossfunc(p[:, 40:60], t[:, 2])
        return loss
