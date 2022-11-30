import sqlite3

import infrastructure


class Fetcher:
    header = []

    def __init__(
        self, target_year: int, forecast_timedelta: int, mode: str
    ) -> None:
        if not (mode == "train" or mode == "eval"):
            raise ValueError("mode value must be train or eval.")

        self.__db = sqlite3.connect(infrastructure.DBPATH)
        self.__corsor = self.__db.cursor().execute(
            self.query(target_year, forecast_timedelta, mode))

    def conv_record(self, record: list) -> list:
        pass

    def fetch_many(self) -> tuple[list, list]:
        record_times = []
        image_paths = []

        records = self.__corsor.fetchmany(5000)
        if not records:
            return record_times, image_paths

        record_times = list(map(lambda record: record[0], records))
        weather_values = list(map(self.conv_record, records))
        return record_times, weather_values

    def query(
        self, target_year: int, forecast_timedelta: int, mode: str
    ) -> str:
        pass
