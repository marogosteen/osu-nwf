import torch

from torch.utils.data import IterableDataset
from torchvision import transforms

from services.application import generator
from services.application.generator import generate_mode
import config
from services import domain


class NWFWaveDataset(IterableDataset):
    def __init__(
        self, train_config: config.NwfConfig, generator_mode: str
    ) -> None:

        super().__init__()
        dataset_generator = generator.DatasetGenerator(
            train_config,
            generator.query.generate_query(train_config, generator_mode),
            generator_mode)

        generate_result = dataset_generator.generate()
        self.__feature_path = generate_result.feature_path
        self.__truth_path = generate_result.truth_path
        self.__feature_service = domain.DatasetFileReadService(
            self.__feature_path)
        self.__truth_service = domain.DatasetFileReadService(
            self.__truth_path)

        feature = None
        truth = None
        record_count = 0
        iter(self)
        while True:
            try:
                feature, truth = self.next()
                record_count += 1
            except StopIteration:
                break

        self.__len = record_count
        self.__feature_size = len(feature)
        self.__truth_size = len(truth)

    def close(self) -> None:
        self.__feature_service.remove_datasetfile()
        self.__truth_service.remove_datasetfile()

    @property
    def feature_size(self) -> int:
        return self.__feature_size

    @property
    def truth_size(self) -> int:
        return self.__truth_size

    def __len__(self) -> None:
        return self.__len

    def __iter__(self):
        self.__feature_service = domain.DatasetFileReadService(
            self.__feature_path)
        self.__truth_service = domain.DatasetFileReadService(
            self.__truth_path)
        return self

    def __next__(self):
        return self.next()

    def next_service(self) -> tuple[torch.Tensor, torch.Tensor]:
        return next(self.__feature_service), next(self.__truth_service)

    def next(self) -> tuple[torch.Tensor, torch.Tensor]:
        features, truths = self.next_service()
        while None in features or None in truths:
            features, truths = self.next_service()

        return (
            torch.tensor(features, dtype=torch.float),
            torch.tensor(truths, dtype=torch.float))


class WaveTrainDataset(NWFWaveDataset):
    def __init__(self, train_config: config.NwfConfig) -> None:
        try:
            super().__init__(train_config, generate_mode.train)
            self.__calc_means()
            self.__calc_stds()
            self.__normalizer = transforms.Lambda(
                lambda x: ((x - self.mean_tensor) / self.std_tensor))

        except Exception as e:
            self.close()
            print(e)
            raise

    @property
    def mean_tensor(self) -> list:
        return self.__mean_tensor.to(torch.float32)

    @property
    def std_tensor(self) -> list:
        return self.__std_tensor.to(torch.float32)

    @property
    def normalizer(self):
        return self.__normalizer

    def __calc_means(self) -> None:
        iter(self)
        sum_tensor = torch.zeros(self.feature_size, dtype=torch.double)
        while True:
            try:
                feature_tensor, _ = self.next()
                sum_tensor += feature_tensor
            except StopIteration:
                break
        self.__mean_tensor = sum_tensor / len(self)

    def __calc_stds(self) -> None:
        iter(self)
        var_tensor = torch.zeros(self.feature_size, dtype=torch.double)
        while True:
            try:
                feature_tensor, _ = self.next()
                var_tensor += (feature_tensor - self.mean_tensor)**2
            except StopIteration:
                break
        var_tensor /= len(self)
        self.__std_tensor = var_tensor**0.5

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        feature, truths = self.next()
        return self.__normalizer(feature), truths


class WaveEvalDataset(NWFWaveDataset):
    def __init__(
        self,
        train_config: config.NwfConfig,
        normalizer: transforms.Lambda
    ) -> None:

        try:
            super().__init__(train_config, generate_mode.eval)
            self.__normalizer = normalizer

        except StopIteration:
            self.close()

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        feature, truths = self.next()
        return self.__normalizer(feature), truths
