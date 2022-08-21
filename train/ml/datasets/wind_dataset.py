import datetime
import os

import torch
from torch.utils.data import IterableDataset
from torchvision import transforms
from PIL import Image

from infrastructure import pressure_images, weather_db


class WindRecord:
    ukb_velocity: float
    ukb_sin_direction: float
    ukb_cos_direction: float
    kix_velocity: float
    kix_sin_direction: float
    kix_cos_direction: float
    tomogashima_velocity: float
    tomogashima_sin_direction: float
    tomogashima_cos_directio: float


def from_record(record: list) -> WindRecord:
    wind_record = WindRecord()
    wind_record.ukb_velocity = record[0]
    wind_record.ukb_sin_direction = record[1]
    wind_record.ukb_cos_direction = record[2]
    wind_record.kix_velocity = record[3]
    wind_record.kix_sin_direction = record[4]
    wind_record.kix_cos_direction = record[5]
    wind_record.tomogashima_velocity = record[6]
    wind_record.tomogashima_sin_direction = record[7]
    wind_record.tomogashima_cos_directio = record[8]
    return wind_record


class DatasetGenerator:
    dataset_dir = "../dataset_store"
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

    def generate(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not os.path.exists(self.dataset_dir):
            os.mkdir(self.dataset_dir)
        datasetfile = open(
            os.path.join(self.dataset_dir, self.dataset_name),
            mode="w")

        while True:
            if (self.__currenttime + self.__forecast_timedelta).year == self.__end_year:
                datasetfile.close()
                return

            # eval dataを学習に使用しない。
            if self.__currenttime.year == self.__target_year:
                self.__currenttime = datetime.datetime(
                    year=self.__currenttime.year+1, month=1, day=1, hour=0, minute=0)

            imagepath = self.__generate_imagepath(self.__currenttime)
            if not os.path.exists(imagepath):
                exit(f"学習用画像ファイルがありません。 path:{imagepath}")

            wind_query = self.__wind_query(
                self.__currenttime + self.__forecast_timedelta)
            self.__dbconnect.cursor.execute(wind_query)
            wind_record = self.__dbconnect.cursor.fetchone()
            self.__currenttime += self.__timestep
            if None in wind_record:
                continue

            datasetfile.write(
                f"{imagepath},{wind_record[0]},{wind_record[3]},{wind_record[6]}\n")

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


class WindNWFDataset(IterableDataset):
    def __init__(
        self,
        datasetname: str,
        generator: DatasetGenerator | None = None
    ) -> None:
        if not generator is None:
            generator.generate()

        self.datasetpath = f"../dataset_store/{datasetname}.csv"
        if not os.path.exists(self.datasetpath):
            mse = "\n".join(
                [
                    "dataset fileが見つかりません。",
                    "path: "+self.datasetpath,
                    "cwd: "+os.getcwd()])
            raise FileNotFoundError(mse)

        self.__device = "cuda" if torch.cuda.is_available() else "cpu"
        self.__len = self.__get_length()
        self.__transforms = transforms.ToTensor()
        self.__truth_size = len(
            open(self.datasetpath).readline().strip().split(",")[1:])

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

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        line = self.datasetfile.readline()
        if not line:
            raise StopIteration

        line = line.strip().split(",")
        imagepath = line[0]
        truth = list(map(float, line[1:]))

        image = Image.open(imagepath).convert("RGB")
        image = self.__transforms(image).to(self.__device)

        return image, torch.Tensor(truth).to(self.__device)
