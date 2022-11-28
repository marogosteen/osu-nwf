from ml.dataset.generator.fetcher.base_fetcher import Fetcher


class UV(Fetcher):
    header = [
        "datetime", "ukb_direction", "kix_direction", "tomogashima_direction"]

    def __init__(
        self, target_year: int, forecast_timedelta: int, mode: str
    ) -> None:
        super().__init__(target_year, forecast_timedelta, mode)

    def conv_record(self, record: list) -> list:
        record
        return

    def query(
        self, target_year: int, forecast_timedelta: int, mode: str
    ) -> str:
        operator = "!=" if mode == "train" else "=="
        return f"""
SELECT *
FROM
(
    SELECT
        datetime(ukb.datetime, '-{forecast_timedelta} hours') AS datetime,
        ukb.direction,
        kix.direction,
        tomogashima.direction
    FROM
        (
            SELECT datetime, amedas_station, direction
            FROM wind_direction
        ) AS ukb
        INNER JOIN
        (
            SELECT datetime, amedas_station, direction
            FROM wind_direction
        ) AS kix ON kix.datetime == ukb.datetime
        INNER JOIN
        (
            SELECT datetime, amedas_station, direction
            FROM wind_direction
        ) AS tomogashima ON tomogashima.datetime == ukb.datetime
    WHERE
        ukb.amedas_station == 879
        AND kix.amedas_station == 855
        AND tomogashima.amedas_station == 899
)
WHERE
    datetime between '2016' and '2020'
    AND strftime("%Y", datetime) {operator} '{target_year}'
;
"""