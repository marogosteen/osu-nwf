import os

from torch.utils.data import Dataset

from ml.dataset.generator import DatasetGenerator


class BaseNWFDataset(Dataset):
    def __init__(self, generator: DatasetGenerator) -> None:
        super().__init__()

        generator.generate()
        self.__feature_names, self.features = self.__read_dataset(
            generator.feature_path)
        self.__truth_names, self.truths = self.__read_dataset(
            generator.truth_path)

        self.__len = len(self.features)
        self.__truth_size = len(self.features[0])

    @property
    def feature_names(self) -> list:
        return self.__feature_names

    @property
    def truth_names(self) -> list:
        return self.__truth_names

    @property
    def truth_size(self):
        return self.__truth_size

    def __len__(self) -> int:
        return self.__len

    def get_datasettimes(self) -> list[str]:
        return list(map(lambda line: line[0], self.features))

    def __read_dataset(self, path: str) -> tuple[list, list]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                "dataset fileが見つかりません。path: {} cwd: {}".format(
                    path, os.getcwd()))
        return (
            open(path).readline().strip().split(","),
            list(map(
                lambda l: l.strip().split(",")[1:],
                open(path).readlines()[1:])
            )
        )
