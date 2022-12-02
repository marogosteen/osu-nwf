import os

import infrastructure
from ml.dataset.generator.fetcher import FetcherBase


class PressureImagePathFetcher(FetcherBase):
    header = ["datetime", "image_path"]

    def __init__(
        self, target_year: int, forecast_timedelta: int, mode: str
    ) -> None:
        super().__init__(target_year, forecast_timedelta, mode)

    def conv_record(self, record: list) -> list:
        record_time: str = record[0]
        image_path = record_time.replace("-", "/").replace(
            " ", "/").replace(":", "")
        image_path += ".jpg"
        image_path = os.path.join(
            infrastructure.IMAGEDIR, image_path)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"not exists {image_path}")
        return [image_path]

    def query(
        self, target_year: int, forecast_timedelta: int, mode: str
    ) -> str:
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
