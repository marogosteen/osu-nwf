"""
REWET is real time weather table dataset.
"""

import os

import torch
from torchvision import transforms


class UVVelo:
    dataset_store_dir = "../assets/dataset_store/"

    def __init__(
        self,
        case_dir: str,
        eval_year: int,
        mode: str
    ) -> None:

        if not (mode == "train" or mode == "eval"):
            ValueError("mode value is must be 'train' or 'eval'")
        case_dir += mode
        self.dataset_path = os.path.join(self.dataset_store_dir, case_dir)

        if not os.path.exists(self.dataset_path):
            # generate
            pass

        self.header: list = open(
            self.dataset_path).readline().strip().split(",")
        self.dataset = list(map(
            lambda line: list(map(float, line.strip().split())),
            open(self.dataset_path).readlines()[1:]))

        self.__len = len(self.dataset)
        # TODO: transforms実装
        mean = torch.Tensor([])
        std = torch.Tensor([])
        self.__transforms = transforms.Lambda(
            lambda feature: (feature - mean) / std)

    def __len__(self):
        return self.__len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor]:
        # TODO: colをどう指定するか
        pred = self.__transforms(self.dataset[idx][:])
        truth = self.dataset[idx][:]
        return torch.Tensor(pred), torch.Tensor(truth)
