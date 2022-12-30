import json
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
    THREE_POINT_BASE = "three_point"
    THREE_POINT_UV = "three_point_uv"
    PRESSURE_MAP = "pressure_map"
    WAVE_HEIGHT = "wave_height"
    WAVE_PERIOD = "wave_period"
    WAVE_HEIGHT_CLASS = "wave_height_class"
    WIND_DIRECTION = "wind_direction"
    WIND_VELOCITY = "wind_velocity"
    WIND_VELOCITY_CLASS = "wind_velocity_class"


class NWFConfig:
    __cnofig_path = "config.json"

    def __init__(self) -> None:
        self.__config_json = json.load(open(self.__cnofig_path))
        self.__dataset_type: str = self.__config_json["dataset_type"]
        self.__feature_fetcher: str = self.__config_json["feature_fetcher"]
        self.__truth_fetcher: str = self.__config_json["truth_fetcher"]

    @property
    def config_json(self) -> dict:
        return self.__config_json

    @property
    def dataset_name(self) -> str:
        return os.path.join(
            self.__dataset_type,
            self.__feature_fetcher,
            self.__truth_fetcher
        )

    @property
    def dataset_type(self) -> str:
        return self.__dataset_type

    @property
    def nwf_dataset(self) -> NWFDatasetBase:
        match self.__dataset_type:
            case DatasetEnum.PRESSURE_MAP:
                return dataset.NWFPressureMap
            case DatasetEnum.RETWET:
                return dataset.NWFRetwet
            case name:
                raise ValueError(f"not match dataset ({name}).")

    @property
    def feature_fetcher(self) -> FetcherBase:
        match self.__feature_fetcher:
            case FetcherEnum.RETWET_BASE:
                return fetcher.retwet.RetwetBaseFetcher
            case FetcherEnum.THREE_POINT_BASE:
                return fetcher.retwet.ThereePointFetcher
            case FetcherEnum.THREE_POINT_UV:
                return fetcher.retwet.ThereePointUVFetcher
            case FetcherEnum.PRESSURE_MAP:
                return fetcher.pressure_map.PressureImagePathFetcher
            case name:
                raise ValueError(f"not match feature fetcher ({name}).")

    @property
    def truth_fetcher(self) -> FetcherBase:
        match self.__truth_fetcher:
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
        match self.__dataset_type:
            case DatasetEnum.PRESSURE_MAP:
                return models.DenseNet
            case DatasetEnum.RETWET:
                return NWFNet
            case name:
                raise ValueError(f"not match net ({name}).")

    @property
    def loss_func(self) -> torch.nn.Module:
        # NOTE: loss funcはtruthに依存している。
        match self.__truth_fetcher:
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
        match self.__truth_fetcher:
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
