from ml.dataset.generator.fetcher.base_fetcher import Fetcher


class WaveHeightFetcher(Fetcher):
    header = ["datetime", "height"]

    def __init__(
        self, target_year: int, forecast_timedelta: int, mode: str
    ):
        super().__init__(target_year, forecast_timedelta, mode)

    def conv_record(self, record: list) -> list:
        return record[1:]

    def query(
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


class WaveHeightClassFetcher(Fetcher):
    header = ["datetime", "wave_height_class"]

    def __init__(
        self, target_year: int, forecast_timedelta: int, mode: str
    ) -> None:
        super().__init__(target_year, forecast_timedelta, mode)

    def conv_record(self, record: list) -> list:
        height = record[1]
        hc = height // 0.5
        hc = 0 if hc < 0 else hc
        hc = 8 if hc > 8 else hc
        return [int(hc)]

    def query(
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
