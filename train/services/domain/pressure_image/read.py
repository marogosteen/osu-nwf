import datetime
import os

from PIL import Image

from infrastructure import pressure_images


class PressureImageReadService():
    pattern = "%Y%m%d%H%M"

    def __init__(self) -> None:
        pass

    def fetch(self, fetchtime: datetime.datetime) -> Image.Image:
        image_path = self.__generate_imagepath(fetchtime)
        return Image.open(image_path).convert("RGB")

    def __generate_imagepath(self, fetchtime: datetime.datetime) -> str:
        return os.path.join(
            pressure_images.IMAGEDIR,
            str(fetchtime.year),
            str(fetchtime.month).zfill(2),
            fetchtime.strftime(self.pattern)+".jpg"
        )
