import datetime
import math

from ml.dataset.generator.fetcher.fetcher_base import FetcherBase


class RetwetBaseFetcher(FetcherBase):
    """
    気象テーブルデータセットの基準となるRecordFetcher
    """

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
        "osaka_velocity",
        "osaka_sin_direction",
        "osaka_cos_direction",
        "akashi_velocity",
        "akashi_sin_direction",
        "akashi_cos_direction",
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
        next_record.extend(self.conv_wind(record[7], record[8]))
        next_record.extend(self.conv_wind(record[9], record[10]))
        next_record.extend(self.conv_wave_class(record[-2], record[-1]))
        next_record.extend(record[11:])
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
        osaka.velocity,
        osaka_direction.direction,
        akashi.velocity,
        akashi_direction.direction,
        temperature.temperature,
        air_pressure.air_pressure,
        wave.significant_height,
        wave.significant_period
    FROM wind AS ukb
    INNER JOIN wind AS kix ON kix.datetime == ukb.datetime
    INNER JOIN wind AS tomogashima ON tomogashima.datetime == ukb.datetime
    INNER JOIN wind AS osaka ON osaka.datetime == ukb.datetime
    INNER JOIN wind AS akashi ON akashi.datetime == ukb.datetime
    INNER JOIN wind_direction AS ukb_direction
        ON ukb_direction.datetime == ukb.datetime
    INNER JOIN wind_direction AS kix_direction
        ON kix_direction.datetime == ukb.datetime
    INNER JOIN wind_direction AS tomogashima_direction
        ON tomogashima_direction.datetime == ukb.datetime
    INNER JOIN wind_direction AS osaka_direction
        ON osaka_direction.datetime == ukb.datetime
    INNER JOIN wind_direction AS akashi_direction
        ON akashi_direction.datetime == ukb.datetime
    INNER JOIN temperature ON temperature.datetime == ukb.datetime
    INNER JOIN air_pressure ON air_pressure.datetime == ukb.datetime
    INNER JOIN wave ON wave.datetime == ukb.datetime
    WHERE ukb.place == "ukb"
        AND kix.place == "kix"
        AND tomogashima.place == "tomogashima"
        AND osaka.place == "osaka"
        AND akashi.place == "akashi"
        AND ukb_direction.amedas_station == 879
        AND kix_direction.amedas_station == 855
        AND tomogashima_direction.amedas_station == 899
        AND osaka_direction.amedas_station == 850
        AND akashi_direction.amedas_station == 878
        AND air_pressure.amedas_station == 880
)
WHERE
    datetime between '2016' and '2020'
    AND strftime("%Y", datetime) {operator} '{target_year}'
;
"""


class ThereePointFetcher(RetwetBaseFetcher):
    """
    RetwetBaseFetcherを基準に風観測データが神戸、関空島、友ヶ島の3地点のみのRecordFetcher
    """

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
        next_record.extend(record[11:])
        return next_record


class ThereePointUVFetcher(ThereePointFetcher):
    """
    ThreePointFetcherを基準に特徴量の風速と風向が緯度成分Uと経度成分Vの風速であるRecordFetcher
    """

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
        next_record.extend(self.conv_uv(record[3], record[5]))
        next_record.extend(self.conv_uv(record[5], record[6]))
        next_record.extend(self.conv_wave_class(record[-2], record[-1]))
        next_record.extend(record[11:])
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


class OnePointFetcher(RetwetBaseFetcher):
    """
    RetwetBaseFetcherを基準に風観測データが神戸の1地点のみのRecordFetcher
    """

    header = [
        "datetime",
        "month_sin",
        "month_cos",
        "hour_sin",
        "hour_cos",
        "ukb_velocity",
        "ukb_sin_direction",
        "ukb_cos_direction",
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
        next_record.extend(self.conv_wave_class(record[-2], record[-1]))
        next_record.extend(record[11:])
        return next_record


class NotContainDatetimeFetcher(RetwetBaseFetcher):
    """
    RetwetBaseFetcherを基準に時間情報を含めないRecordFetcher
    """

    header = [
        "datetime",
        "ukb_velocity",
        "ukb_sin_direction",
        "ukb_cos_direction",
        "kix_velocity",
        "kix_sin_direction",
        "kix_cos_direction",
        "tomogashima_velocity",
        "tomogashima_sin_direction",
        "tomogashima_cos_direction",
        "osaka_velocity",
        "osaka_sin_direction",
        "osaka_cos_direction",
        "akashi_velocity",
        "akashi_sin_direction",
        "akashi_cos_direction",
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
        next_record.extend(self.conv_wind(record[1], record[2]))
        next_record.extend(self.conv_wind(record[3], record[5]))
        next_record.extend(self.conv_wind(record[5], record[6]))
        next_record.extend(self.conv_wind(record[7], record[8]))
        next_record.extend(self.conv_wind(record[9], record[10]))
        next_record.extend(self.conv_wave_class(record[-2], record[-1]))
        next_record.extend(record[11:])
        return next_record


class NotContainWaveClassFetcher(RetwetBaseFetcher):
    """
    RetwetBaseFetcherを基準に波クラスを含めないRecordFetcher
    """

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
        "osaka_velocity",
        "osaka_sin_direction",
        "osaka_cos_direction",
        "akashi_velocity",
        "akashi_sin_direction",
        "akashi_cos_direction",
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
        next_record.extend(self.conv_wind(record[7], record[8]))
        next_record.extend(self.conv_wind(record[9], record[10]))
        next_record.extend(record[11:])
        return next_record


class NotContainTemperatureFetcher(RetwetBaseFetcher):
    """
    RetwetBaseFetcherを基準に気温を含めないRecordFetcher
    """

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
        "osaka_velocity",
        "osaka_sin_direction",
        "osaka_cos_direction",
        "akashi_velocity",
        "akashi_sin_direction",
        "akashi_cos_direction",
        "is_surge",
        "is_wind_wave",
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
        next_record.extend(self.conv_wind(record[7], record[8]))
        next_record.extend(self.conv_wind(record[9], record[10]))
        next_record.extend(self.conv_wave_class(record[-2], record[-1]))
        next_record.extend(record[12:])
        return next_record


class NotContainPressureFetcher(RetwetBaseFetcher):
    """
    RetwetBaseFetcherを基準に気圧を含めないRecordFetcher
    """

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
        "osaka_velocity",
        "osaka_sin_direction",
        "osaka_cos_direction",
        "akashi_velocity",
        "akashi_sin_direction",
        "akashi_cos_direction",
        "is_surge",
        "is_wind_wave",
        "temperature",
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
        next_record.extend(self.conv_wind(record[7], record[8]))
        next_record.extend(self.conv_wind(record[9], record[10]))
        next_record.extend(self.conv_wave_class(record[-2], record[-1]))
        next_record.append(record[11])
        next_record.extend(record[13:])
        return next_record


class NotContainNowphasFetcher(RetwetBaseFetcher):
    """
    RetwetBaseFetcherを基準にナウファス観測データを含めないRecordFetcher
    """

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
        "osaka_velocity",
        "osaka_sin_direction",
        "osaka_cos_direction",
        "akashi_velocity",
        "akashi_sin_direction",
        "akashi_cos_direction",
        "temperature",
        "pressure"
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
        next_record.extend(self.conv_wind(record[7], record[8]))
        next_record.extend(self.conv_wind(record[9], record[10]))
        next_record.extend(record[11:13])
        return next_record
