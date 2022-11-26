import typing

import torch
from PIL import Image
from torchvision import transforms

from ml.dataset.base_dataset import BaseNWFDataset
from ml.dataset.generator.generator import DatasetGenerator


class NWFPressureMap(BaseNWFDataset):
    def __init__(self, generator: DatasetGenerator) -> None:
        super(NWFPressureMap).__init__(generator)

        self.__transforms = transforms.ToTensor()

    def __getitem__(
        self, idx: int
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        image_path = self.__image_path_list[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.__transforms(image)
        truth_item = list(map(float, self.__truths[idx]))
        return image, torch.Tensor(truth_item)
