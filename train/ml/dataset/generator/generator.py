import os

import infrastructure
from ml.dataset.generator.fetcher.fetcher_base import FetcherBase


class DatasetGenerator:
    def __init__(
        self,
        dataset_dir: str,
        feature_fetcher: FetcherBase,
        truth_fetcher: FetcherBase,
        mode: str
    ) -> None:
        if not (mode == "train" or mode == "eval"):
            raise ValueError("mode value must be train or eval.")

        self.__feature_path = os.path.join(
            infrastructure.DATASET_STORE_DIR,
            dataset_dir + f"feature_{mode}.csv")
        self.__truth_path = os.path.join(
            infrastructure.DATASET_STORE_DIR,
            dataset_dir + f"truth_{mode}.csv")
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
        print("generating dataset... ({}, {})".format(
            self.__feature_path, self.__truth_path))

        self.__feature_file = open(self.__feature_path, "w")
        self.__truth_file = open(self.__truth_path, "w")

        self.__feature_file.write("{}\n".format(
            ",".join(self.__feature_fetcher.header)))
        self.__truth_file.write("{}\n".format(
            ",".join(self.__truth_fetcher.header)))

        while True:
            record_times, features = self.__feature_fetcher.fetch_many()
            _, truths = self.__truth_fetcher.fetch_many()

            if not features or not truths:
                break

            for record_time, feature, truth in zip(
                record_times, features, truths
            ):
                if None in feature or None in truth:
                    continue

                feature_line = ",".join(map(str, feature))
                truth_line = ",".join(map(str, truth))

                self.__feature_file.write(f"{record_time},{feature_line}\n")
                self.__truth_file.write(f"{record_time},{truth_line}\n")

        self.__feature_file.close()
        self.__truth_file.close()
        print("generate complete!")

    def is_generated(self) -> bool:
        return os.path.exists(
            self.__feature_path) and os.path.exists(self.__truth_path)
