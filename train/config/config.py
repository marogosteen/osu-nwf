import os

import torch
from torchvision import models

from ml import dataset
from ml.dataset.dataset_base import NWFDatasetBase
from ml.dataset.generator import fetcher
from ml.dataset.generator.fetcher.fetcher_base import FetcherBase
from ml import losses
from ml.net import NWFNet


class DatasetEnum:
    PRESSURE_MAP = "pressure_map"
    RETWET = "retwet"


class FetcherEnum:
    RETWET_BASE = "retwet_base"
    THREE_POINT = "three_point"
    THREE_POINT_UV = "three_point_uv"
    ONE_POINT = "one_point"
    NOT_CONTAIN_WIND = "not_contain_wind"
    NOT_CONTAIN_DATETIME = "not_contain_datetime"
    NOT_CONTAIN_WAVE_CLASS = "not_contain_wave_class"
    NOT_CONTAIN_TEMPERATURE = "not_contain_temperature"
    NOT_CONTAIN_PRESSURE = "not_contain_pressure"
    NOT_CONTAIN_NOWPHAS = "not_contain_nowphas"
    PRESSURE_MAP = "pressure_map"
    WAVE_HEIGHT = "wave_height"
    WAVE_PERIOD = "wave_period"
    WAVE_HEIGHT_CLASS = "wave_height_class"
    WIND_DIRECTION = "wind_direction"
    WIND_VELOCITY = "wind_velocity"
    WIND_VELOCITY_CLASS = "wind_velocity_class"


class NWFConfig:
    __dataset_type_key = "dataset_type"
    __feature_fetcher_key = "feature_fetcher"
    __truth_fetcher_key = "truth_fetcher"
    __feature_timerange_key = "feature_timerange"
    __forecast_time_delta_key = "forecast_time_delta"
    __target_year_key = "target_year"

    def __init__(self, config_json: dict) -> None:
        self.__config_json = config_json

    @property
    def config_json(self) -> dict:
        return self.__config_json

    @property
    def dataset_name(self) -> str:
        return os.path.join(
            self.__config_json[self.__dataset_type_key],
            self.__config_json[self.__feature_fetcher_key],
            f"timerange_{self.feature_timerange}",
            self.__config_json[self.__truth_fetcher_key],
            f"{self.forecast_time_delta}hour_later",
            str(self.target_year)
        )

    @property
    def dataset_type(self) -> DatasetEnum:
        match self.__config_json[self.__dataset_type_key]:
            case DatasetEnum.PRESSURE_MAP:
                return DatasetEnum.PRESSURE_MAP
            case DatasetEnum.RETWET:
                return DatasetEnum.RETWET
            case name:
                raise ValueError(f"not match dataset type ({name}).")

    @property
    def nwf_dataset(self) -> NWFDatasetBase:
        match self.dataset_type:
            case DatasetEnum.PRESSURE_MAP:
                return dataset.NWFPressureMap
            case DatasetEnum.RETWET:
                return dataset.NWFRetwet
            case name:
                raise ValueError(f"not match dataset ({name}).")

    @property
    def feature_fetcher(self) -> FetcherBase:
        match self.__config_json[self.__feature_fetcher_key]:
            case FetcherEnum.RETWET_BASE:
                return fetcher.retwet.RetwetBaseFetcher
            case FetcherEnum.THREE_POINT:
                return fetcher.retwet.ThereePointFetcher
            case FetcherEnum.THREE_POINT_UV:
                return fetcher.retwet.ThereePointUVFetcher
            case FetcherEnum.ONE_POINT:
                return fetcher.retwet.OnePointFetcher
            case FetcherEnum.NOT_CONTAIN_WIND:
                return fetcher.retwet.NotContainWindFetcher
            case FetcherEnum.NOT_CONTAIN_DATETIME:
                return fetcher.retwet.NotContainDatetimeFetcher
            case FetcherEnum.NOT_CONTAIN_WAVE_CLASS:
                return fetcher.retwet.NotContainWaveClassFetcher
            case FetcherEnum.NOT_CONTAIN_TEMPERATURE:
                return fetcher.retwet.NotContainTemperatureFetcher
            case FetcherEnum.NOT_CONTAIN_PRESSURE:
                return fetcher.retwet.NotContainPressureFetcher
            case FetcherEnum.NOT_CONTAIN_NOWPHAS:
                return fetcher.retwet.NotContainNowphasFetcher
            case FetcherEnum.PRESSURE_MAP:
                return fetcher.pressure_map.PressureImagePathFetcher
            case name:
                raise ValueError(f"not match feature fetcher ({name}).")

    @property
    def truth_fetcher(self) -> FetcherBase:
        match self.__config_json[self.__truth_fetcher_key]:
            case FetcherEnum.WAVE_HEIGHT:
                return fetcher.wave.WaveHeightFetcher
            case FetcherEnum.WAVE_PERIOD:
                return fetcher.wave.WavePeriodFetcher
            case FetcherEnum.WAVE_HEIGHT_CLASS:
                return fetcher.wave.WaveHeightClassFetcher
            case FetcherEnum.WIND_DIRECTION:
                return fetcher.wind.WindDirectionFetcher
            case FetcherEnum.WIND_VELOCITY:
                return fetcher.wind.WindVelocityFetcher
            case FetcherEnum.WIND_VELOCITY:
                return fetcher.wind.WindVelocityClassFetcher
            case name:
                raise ValueError(f"not match truth fetcher ({name}).")

    @property
    def net(self) -> torch.nn.Module:
        match self.dataset_type:
            case DatasetEnum.PRESSURE_MAP:
                return models.DenseNet
            case DatasetEnum.RETWET:
                return NWFNet
            case name:
                raise ValueError(f"not match net ({name}).")

    @property
    def loss_func(self) -> torch.nn.Module:
        # NOTE: loss funcはtruthに依存している。
        match self.__config_json[self.__truth_fetcher_key]:
            case FetcherEnum.WAVE_HEIGHT:
                return losses.wave.WaveHeightLoss
            case FetcherEnum.WAVE_PERIOD:
                return losses.wave.WavePeriodLoss
            case FetcherEnum.WAVE_HEIGHT_CLASS:
                return losses.wave.WaveHeightClassLoss
            case FetcherEnum.WIND_DIRECTION:
                return losses.wind.WindDirectionLoss
            case FetcherEnum.WIND_VELOCITY:
                return losses.wind.WindVelocityLoss
            case FetcherEnum.WIND_VELOCITY:
                return losses.wind.WindVelocityClassLoss
            case name:
                raise ValueError(f"not match loss func ({name}).")

    @property
    def num_class(self) -> int:
        # NOTE: num classはtruthに依存している。
        match self.__config_json[self.__truth_fetcher_key]:
            case FetcherEnum.WAVE_HEIGHT | FetcherEnum.WAVE_PERIOD:
                return 1
            case FetcherEnum.WAVE_HEIGHT_CLASS:
                return 27
            case FetcherEnum.WIND_DIRECTION:
                return 51
            case FetcherEnum.WIND_VELOCITY:
                return 3
            case FetcherEnum.WIND_VELOCITY:
                return 60
            case name:
                raise ValueError(f"not match num class ({name}).")

    @property
    def feature_timerange(self) -> int:
        return self.__config_json[self.__feature_timerange_key]

    @property
    def forecast_time_delta(self) -> int:
        return self.__config_json[self.__forecast_time_delta_key]

    @property
    def target_year(self) -> int:
        return self.__config_json[self.__target_year_key]
