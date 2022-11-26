import typing

import torch
from PIL import Image
from torchvision import transforms

from ml.dataset.base_dataset import BaseNWFDataset
from ml.dataset.generator.generator import DatasetGenerator


class NWFPressureMap(BaseNWFDataset):
    def __init__(self, generator: DatasetGenerator) -> None:
        super().__init__(generator)
        self.__transforms = transforms.ToTensor()

    def __len__(self) -> int:
        return super().__len__()

    def __getitem__(
        self, idx: int
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        image_path = self.features[idx][0]
        image = Image.open(image_path).convert("RGB")
        image = self.__transforms(image)
        truth_item = list(map(float, self.truths[idx]))
        return image, torch.Tensor(truth_item)
