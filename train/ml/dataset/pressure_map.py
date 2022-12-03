import torch
from PIL import Image
from torchvision import transforms

from ml.dataset.dataset_base import NWFDatasetBase
from ml.dataset.generator import DatasetGenerator


class NWFPressureMap(NWFDatasetBase):
    def __init__(self, generator: DatasetGenerator) -> None:
        super().__init__(generator)
        self.__transforms = transforms.ToTensor()

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_path = self.features[idx][1]
        image = Image.open(image_path).convert("RGB")
        image = self.__transforms(image)
        truth_item = list(map(float, self.truths[idx][1:]))
        return image, torch.Tensor(truth_item)
