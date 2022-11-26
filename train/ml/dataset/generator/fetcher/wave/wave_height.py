import sqlite3

import infrastructure
from ml.dataset.generator.fetcher.base_fetcher import Fetcher


class WaveHeightFetcher(Fetcher):
    def __init__(
        self, target_year: int, forecast_timedelta: int, mode: str
    ):
        self.__db = sqlite3.connect(infrastructure.DBPATH)
        self.__corsor = self.__db.cursor().execute(
            self.__wave_height_query(target_year, forecast_timedelta, mode))
        self.__buf = []

    def fetch(self) -> tuple[str, list]:
        if not self.__buf:
            self.__buf = self.__corsor.fetchmany(10000)
            if not self.__buf:
                return ["", []]
        record = self.__buf.pop(0)
        return record[0], record[1:]

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
        WHERE strftime('%M', datetime) == '00'
    )
WHERE
    datetime between '2016' and '2020'
    AND strftime("%Y", datetime) {operator} '{target_year}'
;
"""
