import os
import random
import string

from infrastructure import dataset_store


class DatasetFileWriteService:
    dataset_dir = dataset_store.DATASET_STORE_DIR
    str_length = 8

    def __init__(self, mode: str) -> None:
        if not os.path.exists(self.dataset_dir):
            os.mkdir(self.dataset_dir)

        while True:
            random_str = "".join(
                random.choices(
                    string.ascii_lowercase+string.digits,
                    k=self.str_length))

            self.feature_path: str = \
                self.dataset_dir + f"{random_str}_{mode}feature.csv"
            self.truth_path: str = \
                self.dataset_dir + f"{random_str}_{mode}truth.csv"
            if os.path.exists(self.feature_path):
                continue

            break

        self.data_file = open(self.feature_path, "w")
        self.truth_file = open(self.truth_path, "w")

    def close(self) -> None:
        self.data_file.close()
        self.truth_file.close()

    def __del__(self) -> None:
        self.close()

    def write_features(self, features: list) -> None:
        self.data_file.write(
            ",".join(list(map(str, features)))+"\n")

    def write_truths(self, truths: list) -> None:
        self.truth_file.write(
            ",".join(list(map(str, truths)))+"\n")
