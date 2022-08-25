import os
import typing

import torch
from torch.utils.data import IterableDataset
from torchvision import transforms
from PIL import Image

from ml.datasets import wind_dataset
from infrastructure import dataset_store


class WindNWFDataset(IterableDataset):
    def __init__(
        self,
        datasetname: str,
        generator: wind_dataset.DatasetGenerator
    ) -> None:
        generator.generate()
        self.datasetpath = os.path.join(
            dataset_store.DATASET_STORE_DIR,
            f"{datasetname}.csv")

        if not os.path.exists(self.datasetpath):
            mse = f"dataset fileが見つかりません。path: {self.datasetpath} cwd: {os.getcwd()}"
            raise FileExistsError(mse)

        self.__device = "cuda" if torch.cuda.is_available() else "cpu"
        self.__len = self.__get_length()
        self.__transforms = transforms.ToTensor()
        self.__truth_size = len(
            open(self.datasetpath).readline().strip().split(",")[2:])

    @property
    def truth_size(self):
        return self.__truth_size

    def __get_length(self):
        length = 0
        with open(self.datasetpath) as f:
            while True:
                line = f.readline()
                if not line:
                    break
                length += 1
        return length

    def __len__(self) -> int:
        return self.__len

    def __iter__(self):
        self.datasetfile = open(self.datasetpath)
        return self

    def __next__(self) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        line = self.datasetfile.readline()
        if not line:
            raise StopIteration

        line = line.strip().split(",")
        imagepath = line[1]
        truth = list(map(float, line[2:]))

        image = Image.open(imagepath).convert("RGB")
        image = self.__transforms(image).to(self.__device)

        return image, torch.Tensor(truth).to(self.__device)

    def get_datasettime(self) -> list[str]:
        datetimes = []
        with open(self.datasetpath) as f:
            while True:
                line = f.readline()
                if not line:
                    break
                datetimes.append(line.strip().split()[0])
        return datetimes
