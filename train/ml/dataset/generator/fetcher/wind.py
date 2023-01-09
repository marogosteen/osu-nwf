from ml.dataset.generator.fetcher.fetcher_base import FetcherBase


class WindDirectionFetcher(FetcherBase):
    header = [
        "feature_datetime", "ukb_direction", "kix_direction",
        "tomogashima_direction"
    ]

    def __init__(
        self, target_year: int, forecast_timedelta: int, mode: str
    ) -> None:
        super().__init__(target_year, forecast_timedelta, mode)

    def conv_record(self, record: list) -> list:
        return record[1:]

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


class WindVelocityFetcher(FetcherBase):
    header = [
        "feature_datetime", "ukb_direction", "kix_direction",
        "tomogashima_direction"
    ]

    def __init__(
        self, target_year: int, forecast_timedelta: int, mode: str
    ) -> None:
        super().__init__(target_year, forecast_timedelta, mode)

    def conv_record(self, record: list) -> list:
        return record[1:]

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
        ukb.velocity,
        kix.velocity,
        tomogashima.velocity
    FROM
        wind AS ukb
        inner join wind AS kix ON ukb.datetime == kix.datetime
        inner join wind AS tomogashima ON ukb.datetime == tomogashima.datetime
    WHERE
        ukb.place == "ukb"
        AND kix.place == "kix"
        AND tomogashima.place == "tomogashima"
)
WHERE
    datetime between '2016' and '2020'
    AND strftime("%Y", datetime) {operator} '{target_year}'
;
"""


class WindVelocityClassFetcher(FetcherBase):
    header = [
        "feature_datetime", "ukb_direction", "kix_direction",
        "tomogashima_direction"
    ]

    def __init__(
        self, target_year: int, forecast_timedelta: int, mode: str
    ) -> None:
        super().__init__(target_year, forecast_timedelta, mode)

    def record_conv(self, record: list) -> list:
        for i in range(1, len(record)):
            velo = record[i]
            velo_class = velo // 1
            velo_class = 1 if velo_class < 2 else velo_class
            velo_class = 20 if velo_class > 20 else velo_class
            velo_class -= 1
            record[i] = int(velo_class)
        return record

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
        ukb.velocity,
        kix.velocity,
        tomogashima.velocity
    FROM
        wind AS ukb
        inner join wind AS kix ON ukb.datetime == kix.datetime
        inner join wind AS tomogashima ON ukb.datetime == tomogashima.datetime
    WHERE
        ukb.place == "ukb"
        AND kix.place == "kix"
        AND tomogashima.place == "tomogashima"
)
WHERE
    datetime between '2016' and '2020'
    AND strftime("%Y", datetime) {operator} '{target_year}'
;
"""
