import torch
from torch import nn


class WindDirectionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.__lossfunc = torch.nn.CrossEntropyLoss()

    def forward(
        self, p: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        t = t.to(torch.long)
        loss = self.__lossfunc(p[:, 0:17], t[:, 0])
        loss += self.__lossfunc(p[:, 17:34], t[:, 1])
        loss += self.__lossfunc(p[:, 34:51], t[:, 2])
        return loss
