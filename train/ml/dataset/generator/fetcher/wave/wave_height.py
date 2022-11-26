import os
import sqlite3

import infrastructure
from train.ml.dataset.generator.fetcher.base_fetcher import Fetcher


class WaveHeightFetcher(Fetcher):
    def __init__(
        self, target_year: int, forecast_timedelta: int, mode: str
    ):
        super(WaveHeightFetcher)
        self.__db = sqlite3.connect(infrastructure.DBPATH)
        self.__feature_corsor = self.__db.cursor().execute(
            open(self.__wave_height_query(
                target_year, forecast_timedelta, mode)).read())

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

    def __wave_height_query(
        self, target_year: int, forecast_timedelta: int, mode: str
    ) -> str:
        operator = "!=" if mode == "train" else "=="
        return f"""
SELECT
    datetime, height
FROM
    (
        SELECT
            datetime(datetime, '-{forecast_timedelta} hours') as datetime,
            significant_height as height
        FROM wave
        WHERE strftime(%M, datetime) == '00'
    )
WHERE
    AND datetime between '2016' and '2020'
    AND strftime("%Y", datetime) {operator} {target_year}
;
"""
