import os
import sqlite3

import infrastructure
from train.ml.dataset.generator.fetcher.base_fetcher import Fetcher


class PressureImagePathFetcher(Fetcher):
    def __init__(self, target_yaer: int, mode: str):
        if not (mode == "train" or mode == "eval"):
            raise ValueError("mode should be 'train' or 'eval'.")

        super(PressureImagePathFetcher)
        self.__db = sqlite3.connect(infrastructure.DBPATH)
        self.__feature_corsor = self.__db.cursor().execute(
            self.__pressure_time_query(target_yaer, mode))

    def fetch(self) -> tuple[str, list]:
        record_time: str = self.__feature_corsor.fetchone()
        pressure_image_path = self.__get_image_path(record_time)
        if not os.path.exists(pressure_image_path):
            raise FileNotFoundError(f"not exists {pressure_image_path}")
        return record_time, [pressure_image_path]

    def __get_image_path(self, record_time: str) -> str:
        pressure_image_path = record_time.replace("-", "/").replace(
            " ", "/").replace(":", "")
        pressure_image_path = os.path.join(
            infrastructure.IMAGEDIR, pressure_image_path)
        return pressure_image_path

    def __pressure_time_query(self, target_year: int, mode: str) -> str:
        operator = "!=" if mode == "train" else "=="
        return f"""
SELECT datetime
FROM air_pressure
WHERE place == 'kobe'
    AND datetime between '2016' and '2020'
    AND strftime("%Y", datetime) {operator} {target_year}
;
"""
