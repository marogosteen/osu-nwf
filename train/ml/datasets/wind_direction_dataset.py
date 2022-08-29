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
    dataset_dir = os.path.join(dataset_store.DATASET_STORE_DIR, "direction")
    filenamepattern = "%Y%m%d%H%M"
    recordpattern = "%Y-%m-%d %H:%M"
    select_records = [
        "ukb.sin_direction",
        "ukb.cos_direction",
        "kix.sin_direction",
        "kix.cos_direction",
        "tomogashima.sin_direction",
        "tomogashima.cos_direction"
    ]

    def __init__(
        self,
        datasetname: str,
        begin_year: int,
        end_year: int,
        target_year: int | None = None,
        forecast_timedelta: int = 1
    ) -> None:

        self.datasetfile_path = os.path.join(
            self.dataset_dir, f"{datasetname}.csv")
        self.__target_year = target_year
        self.__end_year = end_year
        self.__timedelta_1hour = datetime.timedelta(hours=1)
        self.__forecast_timedelta = datetime.timedelta(
            hours=forecast_timedelta)
        self.__currenttime = datetime.datetime(
            year=begin_year, month=1, day=1, hour=0, minute=0)
        self.__dbconnect = weather_db.DbContext()
        self.__init_record_buf()

    def delete_dataset(self):
        if self.is_generated():
            self.datasetfile.close()
            os.remove(self.datasetfile_path)

    def generate(self) -> None:
        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)

        print(f"generating dataset({self.datasetfile_path})...")
        self.datasetfile = open(self.datasetfile_path, mode="w")
        while True:
            record = self.next_buf()
            if record is None:
                break
            record_datetime = datetime.datetime.strptime(
                record[0], self.recordpattern)

            if record_datetime.year == self.__target_year:
                continue

            forecast_time = self.__forecast_datetime()
            # evaldatasetの場合、時間ベースで停止すれば全てFetchする必要がなくなる。
            if forecast_time.year == self.__end_year:
                break
            # eval dataを学習に使用しない。
            if forecast_time.year == self.__target_year:
                self.__currenttime = datetime.datetime(
                    year=self.__currenttime.year+1, month=1, day=1, hour=0, minute=0)
                forecast_time = self.__forecast_datetime()
                continue

            # 観測データが抜けて予測時差が一致しないととデータセットとして破綻する。
            while record_datetime != forecast_time:
                print(f"気象データの欠損があります。 \ndatetime: {forecast_time}")
                self.__currenttime += self.__timedelta_1hour

            imagepath = self.__generate_imagepath(self.__currenttime)
            if not os.path.exists(imagepath):
                raise FileExistsError(f"学習用画像ファイルがありません。 path:{imagepath}")

            self.__currenttime += self.__timedelta_1hour
            if None in record:
                continue

            direction_class = ",".join(map(str, [
                self.__conv_directionclass(record[1], record[2]),
                self.__conv_directionclass(record[3], record[4]),
                self.__conv_directionclass(record[5], record[6])
            ]))
            self.datasetfile.write(
                f"{forecast_time.strftime(self.recordpattern)},{imagepath},{direction_class}\n")

        self.datasetfile.close()
        print(f"generate complete! ({self.datasetfile_path})")

    def is_generated(self) -> bool:
        return os.path.exists(self.datasetfile_path)

    def __init_record_buf(self):
        query = self.__query()
        self.__dbconnect.cursor.execute(query)
        self.records_buf = []
        first_forecastdatetime = self.__forecast_datetime().strftime(self.recordpattern)
        self.records_buf: list = self.__dbconnect.cursor.fetchmany(1000)
        record = self.next_buf()
        while record[0] != first_forecastdatetime:
            record = self.next_buf()
        self.records_buf.insert(0, record)

    def next_buf(self) -> list:
        if len(self.records_buf) == 0:
            self.records_buf: list = self.__dbconnect.cursor.fetchmany(1000)
            if len(self.records_buf) == 0:
                return None
        return self.records_buf.pop(0)

    def __forecast_datetime(self) -> datetime.datetime:
        return self.__currenttime + self.__forecast_timedelta

    def __generate_imagepath(self, fetchtime: datetime.datetime) -> str:
        return os.path.join(
            pressure_images.IMAGEDIR,
            str(fetchtime.year),
            str(fetchtime.month).zfill(2),
            fetchtime.strftime(self.filenamepattern)+".jpg"
        )

    # TODO magic number
    def __conv_directionclass(self, sinv: float, cosv: float) -> int:
        if sinv == 0 and cosv == 0:
            return 0

        asv = math.asin(sinv)
        acv = math.acos(cosv)
        sd = asv / (2 * math.pi) * 360
        cd = acv / (2 * math.pi) * 360
        r = cd if sd >= 0 else 360 - cd
        r = round(r, 1)
        if r % (360 / 16) != 0:
            raise ValueError("")
        c = int(r // (360 / 16) + 1 - 8)
        if c <= 0:
            c += 16

        return c

    def __query(self) -> str:
        return f"""
SELECT
    ukb.datetime,
    ukb.sin_direction,
    ukb.cos_direction,
    kix.sin_direction,
    kix.cos_direction,
    tomogashima.sin_direction,
    tomogashima.cos_direction
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
        generator: DatasetGenerator
    ) -> None:
        if not generator.is_generated():
            try:
                generator.generate()
            except Exception as e:
                generator.delete_dataset()
                raise e

        self.datasetpath = generator.datasetfile_path
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
