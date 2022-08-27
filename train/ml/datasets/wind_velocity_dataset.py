import datetime
import os
import typing

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from infrastructure import pressure_images, weather_db, dataset_store


class DatasetGenerator:
    dataset_dir = dataset_store.DATASET_STORE_DIR
    filepattern = "%Y%m%d%H%M"
    recordpattern = "%Y-%m-%d %H:%M"
    select_records = [
        "ukb.velocity",
        "ukb.sin_direction",
        "ukb.cos_direction",
        "kix.velocity",
        "kix.sin_direction",
        "kix.cos_direction",
        "tomogashima.velocity",
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

        self.dataset_name = f"{datasetname}.csv"
        self.__end_year = end_year
        self.__target_year = target_year
        self.__timestep = datetime.timedelta(hours=1)
        self.__forecast_timedelta = datetime.timedelta(
            hours=forecast_timedelta)
        self.__currenttime = datetime.datetime(
            year=begin_year, month=1, day=1, hour=0, minute=0)
        self.__dbconnect = weather_db.DbContext()

    def generate(self) -> None:
        if not os.path.exists(self.dataset_dir):
            os.mkdir(self.dataset_dir)

        datasetfile_path = os.path.join(self.dataset_dir, self.dataset_name)
        if os.path.exists(datasetfile_path):
            return

        print(f"generating dataset({self.dataset_name})...")
        datasetfile = open(datasetfile_path, mode="w")
        while True:
            forecast_time = self.__currenttime + self.__forecast_timedelta

            if forecast_time.year == self.__end_year:
                datasetfile.close()
                print(f"generate complete! ({self.dataset_name})")
                return

            # eval dataを学習に使用しない。
            if forecast_time.year == self.__target_year:
                self.__currenttime = datetime.datetime(
                    year=self.__currenttime.year+1, month=1, day=1, hour=0, minute=0)
                continue

            imagepath = self.__generate_imagepath(self.__currenttime)
            if not os.path.exists(imagepath):
                raise FileExistsError(f"学習用画像ファイルがありません。 path:{imagepath}")

            wind_query = self.__wind_query(forecast_time)
            self.__dbconnect.cursor.execute(wind_query)
            wind_record = self.__dbconnect.cursor.fetchone()
            self.__currenttime += self.__timestep
            if None in wind_record:
                continue

            datasetfile.write(
                f"{forecast_time.strftime(self.recordpattern)},{imagepath},{wind_record[0]},{wind_record[3]},{wind_record[6]}\n")

    def __generate_imagepath(self, fetchtime: datetime.datetime) -> str:
        return os.path.join(
            pressure_images.IMAGEDIR,
            str(fetchtime.year),
            str(fetchtime.month).zfill(2),
            fetchtime.strftime(self.filepattern)+".jpg"
        )

    def __wind_query(self, forecasttime: datetime.datetime) -> str:
        return f"""
select
    {",".join(self.select_records)}
from 
    wind as ukb
    inner join wind as kix on ukb.datetime == kix.datetime
    inner join wind as tomogashima on ukb.datetime == tomogashima.datetime
where
    ukb.place == "ukb" and
    kix.place == "kix" and
    tomogashima.place == "tomogashima" and
    ukb.datetime == '{forecasttime.strftime(self.recordpattern)}';
"""


class WindNWFDataset(Dataset):
    label_startcol = 2

    def __init__(
        self,
        datasetname: str,
        generator: DatasetGenerator
    ) -> None:
        generator.generate()
        self.datasetpath = os.path.join(
            dataset_store.DATASET_STORE_DIR, f"{datasetname}.csv")

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
