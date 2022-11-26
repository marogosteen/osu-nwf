import os
import sqlite3

import infrastructure
from ml.dataset.generator.fetcher.base_fetcher import Fetcher


class PressureImagePathFetcher(Fetcher):
    def __init__(self, target_yaer: int, mode: str):
        super()
        if not (mode == "train" or mode == "eval"):
            raise ValueError("mode should be 'train' or 'eval'.")

        self.__db = sqlite3.connect(infrastructure.DBPATH)
        self.__corsor = self.__db.cursor().execute(
            self.__pressure_time_query(target_yaer, mode))
        self.__buf = []

    def fetch(self) -> tuple[str, list]:
        if not self.__buf:
            self.__buf = self.__corsor.fetchmany(10000)
            if not self.__buf:
                return ["", []]

        record_time: str = self.__buf.pop(0)[0]
        pressure_image_path = self.__get_image_path(record_time)
        if not os.path.exists(pressure_image_path):
            raise FileNotFoundError(f"not exists {pressure_image_path}")
        return record_time, [pressure_image_path]

    def __get_image_path(self, record_time: str) -> str:
        pressure_image_path = record_time.replace("-", "/").replace(
            " ", "/").replace(":", "")
        pressure_image_path += ".jpg"
        pressure_image_path = os.path.join(
            infrastructure.IMAGEDIR, pressure_image_path)
        return pressure_image_path

    def __pressure_time_query(self, target_year: int, mode: str) -> str:
        operator = "!=" if mode == "train" else "=="
        return f"""
SELECT air_pressure.datetime
FROM air_pressure
    inner join amedas_station
        on amedas_station.id == air_pressure.amedas_station
WHERE
    amedas_station.station_name_alphabet == 'kobe'
    AND amedas_station.amedas_subid == 0
    AND air_pressure.datetime between '2016' and '2020'
    AND strftime('%Y', air_pressure.datetime) {operator} '{target_year}'
;
"""
