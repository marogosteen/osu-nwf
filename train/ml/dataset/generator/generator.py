import os

import infrastructure
from ml.dataset.generator.fetcher.base_fetcher import Fetcher


class DatasetGenerator:
    def __init__(
        self,
        dataset_dir: str,
        feature_fetcher: Fetcher,
        truth_fetcher: Fetcher
    ) -> None:
        self.__feature_path = os.path.join(
            infrastructure.DATASET_STORE_DIR, dataset_dir + "feature.csv")
        self.__truth_path = os.path.join(
            infrastructure.DATASET_STORE_DIR, dataset_dir + "truth.csv")
        self.__feature_fetcher = feature_fetcher
        self.__truth_fetcher = truth_fetcher

        dataset_dir = os.path.dirname(self.__feature_path)
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)

    @property
    def feature_path(self) -> str:
        return self.__feature_path

    @property
    def truth_path(self) -> str:
        return self.__truth_path

    def __delete_dataset(self):
        if os.path.exists(self.__feature_path):
            self.__feature_file.close()
            os.remove(self.__feature_path)

        if os.path.exists(self.__truth_path):
            self.__truth_file.close()
            os.remove(self.__truth_path)

    def generate(self) -> None:
        if self.is_generated():
            return

        # if an error occurs during generate,
        # delete the incomplete dataset.
        is_done = False
        try:
            self.__generate()
            is_done = True
        except Exception as e:
            self.__delete_dataset()
            raise e
        finally:
            if not is_done:
                self.__delete_dataset()

    def __generate(self):
        self.__feature_file = open(self.__feature_path, "w")
        self.__truth_file = open(self.__truth_path, "w")

        while True:
            record_time, feature = self.__feature_fetcher.fetch()
            _, truth = self.__truth_fetcher.fetch()

            if not len(feature) or not len(truth):
                break

            if None in feature or None in truth:
                continue

            feature_line = ",".join(map(str, feature))
            truth_line = ",".join(map(str, truth))

            self.__feature_file.write(f"{record_time},{feature_line}\n")
            self.__truth_file.write(f"{record_time},{truth_line}\n")

        self.__feature_file.close()
        self.__truth_file.close()
        print("generate complete! ({}, {})".format(
            self.__feature_path, self.__truth_path))

    def is_generated(self) -> bool:
        return os.path.exists(
            self.__feature_path) and os.path.exists(self.__truth_path)
