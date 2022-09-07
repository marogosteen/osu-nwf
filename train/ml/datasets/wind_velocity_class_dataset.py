import datetime
import math
import os
import typing

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from infrastructure import pressure_images, weather_db, dataset_store


class DatasetGenerator:
    filenamepattern = "%Y%m%d%H%M"
    recordpattern = "%Y-%m-%d %H:%M"

    def __init__(
        self,
        datasetname: str,
    ) -> None:
        self.datasetfile_path = os.path.join(
            dataset_store.DATASET_STORE_DIR, f"{datasetname}.csv")
        self.dataset_dir = os.path.dirname(self.datasetfile_path)
        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)
        self.__dbconnect = weather_db.DbContext()

    def __conv_directionclass(self, record: list) -> list:
        for i, v in enumerate(record):
            v = int(v) + 1
            v = 2 if v < 2 else v
            v = 21 if v > 20 else v
            v -= 2
            record[i] = v
        return record

    def delete_dataset(self):
        if self.is_generated():
            self.datasetfile.close()
            os.remove(self.datasetfile_path)

    def generate(
        self,
        begin_year: int,
        end_year: int,
        target_year: int | None = None,
        forecast_timedelta: int = 1
    ) -> None:
        if not self.is_generated():
            try:
                self.__generate(
                    begin_year=begin_year,
                    end_year=end_year,
                    target_year=target_year,
                    forecast_timedelta=forecast_timedelta,
                )
            except Exception as e:
                self.delete_dataset()
                raise e

    def __generate(
        self,
        begin_year: int,
        end_year: int,
        target_year: int | None = None,
        forecast_timedelta: int = 1
    ) -> None:
        self.__init_record_buf(begin_year)
        forecast_timedelta: datetime.timedelta = datetime.timedelta(
            hours=forecast_timedelta)

        print(f"generating dataset({self.datasetfile_path})...")
        self.datasetfile = open(self.datasetfile_path, mode="w")
        while True:
            record = self.next_buf()

            # 終了条件
            if record is None:
                break
            record_datetime = datetime.datetime.strptime(
                record[0], self.recordpattern)
            # evaldatasetの場合、時間ベースで停止すれば全てFetchする必要がなくなる。
            if record_datetime.year == end_year:
                break

            # continue条件
            feature_datetime = record_datetime - forecast_timedelta
            if record_datetime.year == target_year or feature_datetime.year == target_year:
                continue
            if None in record:
                continue
            imagepath = self.__generate_imagepath(feature_datetime)
            if not os.path.exists(imagepath):
                continue

            record = list(record[1:])
            record = self.__conv_directionclass(record)
            direction_class = ",".join(map(str, record))
            self.datasetfile.write(
                f"{record_datetime.strftime(self.recordpattern)},{imagepath},{direction_class}\n")

        self.datasetfile.close()
        print(f"generate complete! ({self.datasetfile_path})")

    def is_generated(self) -> bool:
        return os.path.exists(self.datasetfile_path)

    def __init_record_buf(self, begin_year: int):
        query = self.__query()
        self.__dbconnect.cursor.execute(query)
        self.record_buf = []
        while True:
            record = self.next_buf()
            record_datetime = datetime.datetime.strptime(
                record[0], self.recordpattern)
            if record_datetime.year == begin_year:
                self.record_buf.insert(0, record)
                break

    def next_buf(self) -> list:
        if len(self.record_buf) == 0:
            self.record_buf: list = self.__dbconnect.cursor.fetchmany(1000)
            if len(self.record_buf) == 0:
                return None
        return self.record_buf.pop(0)

    def __generate_imagepath(self, fetchtime: datetime.datetime) -> str:
        return os.path.join(
            pressure_images.IMAGEDIR,
            str(fetchtime.year),
            str(fetchtime.month).zfill(2),
            fetchtime.strftime(self.filenamepattern)+".jpg"
        )

    def __query(self) -> str:
        return f"""
SELECT
    ukb.datetime,
    ukb.velocity,
    kix.velocity,
    tomogashima.velocity
FROM
    wind AS ukb
    INNER JOIN wind AS kix ON ukb.datetime == kix.datetime
    INNER JOIN wind AS tomogashima ON ukb.datetime == tomogashima.datetime
WHERE
    ukb.place == "ukb" AND
    kix.place == "kix" AND
    tomogashima.place == "tomogashima"
ORDER BY
    ukb.datetime
;
"""


class WindNWFDataset(Dataset):
    label_startcol = 2

    def __init__(
        self,
        datasetfile_path: str
    ) -> None:

        self.datasetpath = datasetfile_path
        if not os.path.exists(self.datasetpath):
            mse = f"dataset fileが見つかりません。path: {self.datasetpath} cwd: {os.getcwd()}"
            raise FileExistsError(mse)

        self.dataset_list: list = list(map(
            lambda l: l.strip().split(","), open(self.datasetpath).readlines()))
        self.__len = len(self.dataset_list)
        self.__transforms = transforms.ToTensor()
        self.__truth_size = len(
            self.dataset_list[0][self.label_startcol:])

    @property
    def truth_size(self):
        return self.__truth_size

    def __len__(self) -> int:
        return self.__len

    def __getitem__(self, idx: int) -> typing.Tuple[torch.Tensor, torch.Tensor]:
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
