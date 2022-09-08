import os
import typing

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class NWFDataset(Dataset):
    label_startcol = 2

    def __init__(
        self,
        datasetfile_path: str
    ) -> None:

        self.datasetpath = datasetfile_path
        if not os.path.exists(self.datasetpath):
            mse = "dataset fileが見つかりません。path: {} cwd: {}".format(
                self.datasetpath,
                os.getcwd()
            )
            raise FileExistsError(mse)

        self.dataset_list: list = list(map(
            lambda l: l.strip().split(","),
            open(self.datasetpath).readlines()))
        self.__len = len(self.dataset_list)
        self.__transforms = transforms.ToTensor()
        self.__truth_size = len(
            self.dataset_list[0][self.label_startcol:])

    @property
    def truth_size(self):
        return self.__truth_size

    def __len__(self) -> int:
        return self.__len

    def __getitem__(
        self, idx: int
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        line = self.dataset_list[idx]
        image = Image.open(line[1]).convert("RGB")
        image = self.__transforms(image)
        truth = list(map(float, line[self.label_startcol:]))
        return image, torch.Tensor(truth)

    def get_datasettimes(self) -> list[str]:
        datetimes = []
        with open(self.datasetpath) as f:
            while True:
                line = f.readline()
                if not line:
                    break
                datetimes.append(line.strip().split(",")[0])
        return datetimes
