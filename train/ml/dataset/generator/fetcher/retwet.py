import datetime
import math

from ml.dataset.generator.fetcher.fetcher_base import FetcherBase


class ThereePoint(FetcherBase):
    header = [
        "datetime",
        "month_sin",
        "month_cos",
        "hour_sin",
        "hour_cos",
        "ukb_velocity",
        "ukb_sin_direction",
        "ukb_cos_direction",
        "kix_velocity",
        "kix_sin_direction",
        "kix_cos_direction",
        "tomogashima_velocity",
        "tomogashima_sin_direction",
        "tomogashima_cos_direction",
        "is_surge",
        "is_wind_wave",
        "temperature",
        "pressure",
        "wave_height",
        "wave_period"
    ]

    def __init__(
        self, target_year: int, forecast_timedelta: int, mode: str
    ) -> None:
        super().__init__(target_year, forecast_timedelta, mode)

    def conv_record(self, record: list) -> list:
        next_record = []
        next_record.extend(self.conv_datetime(record[0]))
        next_record.extend(self.conv_wind(record[1], record[2]))
        next_record.extend(self.conv_wind(record[3], record[5]))
        next_record.extend(self.conv_wind(record[5], record[6]))
        next_record.extend(self.conv_wave_class(record[-2], record[-1]))
        next_record.extend(record[7:])
        return next_record

    def conv_datetime(
        self, record_time: str
    ) -> tuple[float, float, float, float]:
        # datetimeは欠損値がない前提
        dt = datetime.datetime.strptime(record_time, "%Y-%m-%d %H:%M:%S")
        manth_rad = dt.month / 12 * 2 * math.pi
        hour_rad = dt.hour / 24 * 2 * math.pi

        return (
            math.sin(manth_rad),
            math.cos(manth_rad),
            math.sin(hour_rad),
            math.cos(hour_rad)
        )

    def conv_wind(
        self, velocity: float | None, direction: int | None
    ) -> tuple[float | None, float | None, float | None]:
        if direction is None:
            return velocity, None, None

        if direction == 0:
            return velocity, 0, 0

        rad = 2 * math.pi * (direction - 1) / 16
        return velocity, math.sin(rad), math.cos(rad)

    def conv_wave_class(
        self, height: float | None, period: float | None
    ) -> tuple[int | None, int | None]:
        if height is None or period is None:
            return None, None

        is_surge = 1 if 4 * height + 2 < period else 0
        is_wind_wave = 0 if is_surge else 1
        return is_surge, is_wind_wave

    def query(
        self, target_year: int, forecast_timedelta: int, mode: str
    ) -> str:
        operator = "!=" if mode == "train" else "=="
        return f"""
SELECT *
FROM
(
    SELECT datetime(ukb.datetime, '-{forecast_timedelta} hours') as datetime,
        ukb.velocity,
        ukb_direction.direction,
        kix.velocity,
        kix_direction.direction,
        tomogashima.velocity,
        tomogashima_direction.direction,
        temperature.temperature,
        air_pressure.air_pressure,
        wave.significant_height,
        wave.significant_period
    FROM wind AS ukb
    INNER JOIN wind AS kix ON kix.datetime == ukb.datetime
    INNER JOIN wind AS tomogashima ON tomogashima.datetime == ukb.datetime
    INNER JOIN wind_direction AS ukb_direction
        ON ukb_direction.datetime == ukb.datetime
    INNER JOIN wind_direction AS kix_direction
        ON kix_direction.datetime == ukb.datetime
    INNER JOIN wind_direction AS tomogashima_direction
        ON tomogashima_direction.datetime == ukb.datetime
    INNER JOIN temperature ON temperature.datetime == ukb.datetime
    INNER JOIN air_pressure ON air_pressure.datetime == ukb.datetime
    INNER JOIN wave ON wave.datetime == ukb.datetime
    WHERE ukb.place == "ukb"
        AND kix.place == "kix"
        AND tomogashima.place == "tomogashima"
        AND ukb_direction.amedas_station == 879
        AND kix_direction.amedas_station == 855
        AND tomogashima_direction.amedas_station == 899
        AND air_pressure.amedas_station == 880
)
WHERE
    datetime between '2016' and '2020'
    AND strftime("%Y", datetime) {operator} '{target_year}'
;
"""


class ThereePointUV(ThereePoint):
    header = [
        "datetime",
        "month_sin",
        "month_cos",
        "hour_sin",
        "hour_cos",
        "ukb_u",
        "ukb_v",
        "kix_u",
        "kix_v",
        "tomogashima_u",
        "tomogashima_v",
        "is_surge",
        "is_wind_wave",
        "temperature",
        "pressure",
        "wave_height",
        "wave_period"
    ]

    def __init__(
        self, target_year: int, forecast_timedelta: int, mode: str
    ) -> None:
        super().__init__(target_year, forecast_timedelta, mode)

    def conv_record(self, record: list) -> list:
        next_record = []
        next_record.extend(self.conv_datetime(record[0]))
        next_record.extend(self.conv_uv(record[1], record[2]))
        next_record.extend(self.conv_uv(record[3], record[4]))
        next_record.extend(self.conv_uv(record[5], record[6]))
        next_record.extend(super().conv_wave_class(record[-2], record[-1]))
        next_record.extend(record[7:])
        return next_record

    def conv_uv(
        self, velocity: float | None, direction: int | None
    ) -> tuple[float, float]:
        if velocity is None or direction is None:
            return None, None

        if direction == 0:
            return 0, 0

        rad = 2 * math.pi * (direction - 1) / 16
        u = velocity * math.cos(rad)
        v = velocity * math.sin(rad)
        return u, v
