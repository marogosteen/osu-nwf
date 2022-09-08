import torch
from torch import nn


class WaveHeightClassLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.__lossfunc = torch.nn.CrossEntropyLoss()

    def forward(
        self, p: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        loss = self.__lossfunc(p[:, 0:9], t[:, 0])
        loss += self.__lossfunc(p[:, 9:18], t[:, 1])
        loss += self.__lossfunc(p[:, 18:27], t[:, 2])
        return loss
