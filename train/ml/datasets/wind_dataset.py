import datetime
import os

import torch
from torch.utils.data import IterableDataset
from torchvision import transforms

from services.domain.pressure_image import PressureImageReadService
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

    def from_record(self, record: list):
        self.ukb_velocity = record[0]
        self.ukb_sin_direction = record[1]
        self.ukb_cos_direction = record[2]
        self.kix_velocity = record[3]
        self.kix_sin_direction = record[4]
        self.kix_cos_direction = record[5]
        self.tomogashima_velocity = record[6]
        self.tomogashima_sin_direction = record[7]
        self.tomogashima_cos_directio = record[8]

class WindNWFDataset(IterableDataset):
    pattern = "%Y-%m-%d %H:%M"
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
        begin_year: int,
        end_year: int,
        target_year: int | None = None,
        forecast_timedelta: int = 1
    ) -> None:

        self.__end_year = end_year
        self.__target_year = target_year
        # self.__timestep = datetime.timedelta(hours=1)
        self.__timestep = datetime.timedelta(hours=4)
        self.__forecast_timedelta = datetime.timedelta(
            hours=forecast_timedelta
        )
        self.__currenttime = datetime.datetime(
            year=begin_year, month=1, day=1, hour=0, minute=0
        )
        self.__pressure_readservice = PressureImageReadService()
        self.__dbconnect = weather_db.DbContext()
        self.__transforms = transforms.ToTensor()
        self.truth_size = len(self.select_records)

    def close(self) -> None:
        self.__dbconnect.close()

    def __len__(self) -> int:
        return

    def __iter__(self):
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        while True:
            if self.__currenttime.year == self.__target_year:
                self.__currenttime = datetime.datetime(
                    year=self.__currenttime.year+1, month=1, day=1, hour=0, minute=0
                )
            if self.__currenttime.year == self.__end_year:
                raise StopIteration
            image = self.__pressure_readservice.fetch(self.__currenttime)
            image = self.__transforms(image).to(self.__device)

            wind_query = self.__wind_query(
                self.__currenttime + self.__forecast_timedelta
            )
            self.__dbconnect.cursor.execute(wind_query)
            truth = self.__dbconnect.cursor.fetchone()
            self.__currenttime += self.__timestep

            if None in truth:
                continue
            break

        return image, torch.Tensor(truth).to(self.__device)

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
    ukb.datetime == '{forecasttime.strftime(self.pattern)}';
"""


class DatasetGenerator:
    pattern = "%Y-%m-%d %H:%M"
    filepattern = "%Y%m%d%H%M"

    def __init__(
        self,
        begin_year: int,
        end_year: int,
        target_year: int | None = None,
        forecast_timedelta: int = 1
    ) -> None:

        self.__device = "cuda" if torch.cuda.is_available() else "cpu"
        self.__end_year = end_year
        self.__target_year = target_year
        # self.__timestep = datetime.timedelta(hours=1)
        self.__timestep = datetime.timedelta(hours=4)
        self.__forecast_timedelta = datetime.timedelta(
            hours=forecast_timedelta
        )
        self.__currenttime = datetime.datetime(
            year=begin_year, month=1, day=1, hour=0, minute=0
        )
        self.__pressure_readservice = PressureImageReadService()
        self.__dbconnect = weather_db.DbContext()
        self.__transforms = transforms.ToTensor()
        self.truth_size = len(self.select_records)

    def hoge(self) -> tuple[torch.Tensor, torch.Tensor]:
        while True:
            if self.__currenttime.year == self.__end_year:
                break

            # eval dataを学習に使用しない。
            if self.__currenttime.year == self.__target_year:
                self.__currenttime = datetime.datetime(
                    year=self.__currenttime.year+1, month=1, day=1, hour=0, minute=0
                )

            imagepath = self.__generate_imagepath(self.__currenttime)
            if not os.path.exists(imagepath):
                exit(f"学習用画像ファイルがありません。 path:{imagepath}")

            wind_query = self.__wind_query(
                self.__currenttime + self.__forecast_timedelta
            )
            self.__dbconnect.cursor.execute(wind_query)
            wind_record = WindRecord.from_record(self.__dbconnect.cursor.fetchone())
            self.__currenttime += self.__timestep

            if None in wind_record:
                continue

            self.__conv_directionclass()

        return image, torch.Tensor(wind_record).to(self.__device)

    def __generate_imagepath(self, fetchtime: datetime.datetime) -> str:
        return os.path.join(
            pressure_images.IMAGEDIR,
            str(fetchtime.year),
            str(fetchtime.month).zfill(2),
            fetchtime.strftime(self.filepattern)+".jpg"
        )

    def __conv_directionclass(sin_direction: float, cos_direction: float):
        pass

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
    ukb.datetime == '{forecasttime.strftime(self.pattern)}';
"""
