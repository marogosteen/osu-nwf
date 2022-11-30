"""
REWET is real time weather table dataset.
"""
import numpy as np
import torch
from torchvision import transforms

from ml.dataset.base_dataset import BaseNWFDataset
from ml.dataset.generator import DatasetGenerator


class ThreePointUVVelo(BaseNWFDataset):
    def __init__(self, generator: DatasetGenerator) -> None:
        super().__init__(generator)

        feature_array = np.array(self.features, dtype=float)
        mean = torch.from_numpy(feature_array.mean(axis=0))
        std = torch.from_numpy(feature_array.std(axis=0))
        self.__transforms = transforms.Lambda(
            lambda feature: (feature - mean) / std)

    def __len__(self) -> int:
        return super().__len__()

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feature = torch.Tensor(list(map(float, self.features[idx])))
        feature = self.__transforms(feature)
        truth_item = list(map(float, self.truths[idx]))
        return feature, torch.Tensor(truth_item)
