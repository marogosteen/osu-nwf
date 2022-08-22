import config
import datetime
import math

from services.domain.record_db.service_models.record_model import RecordFetchServiceModel

class RecordTransformer:
    def __init__(self, dataset_config: config.Explanatory | config.Target) -> None:
        self.__transform_funcs = []

        if type(dataset_config) is config.Explanatory:
            self.__generate_feature_trainsform_funcs(dataset_config)
        elif type(dataset_config) is config.Target:
            self.__generate_truth_transform_funcs(dataset_config)
        else:
            raise ValueError()

    def __generate_feature_trainsform_funcs(
        self, explanatory_config: config.Explanatory
    ) -> None:

        if explanatory_config.datetime:
            self.__transform_funcs.append(record_time)

        if explanatory_config.wave_class:
            self.__transform_funcs.append(wave_class)

        if explanatory_config.temperature:
            self.__transform_funcs.append(temperature)

        if explanatory_config.kobe_air_pressure:
            self.__transform_funcs.append(kobe_air_pressure)

        if explanatory_config.ukb:
            self.__transform_funcs.append(ukb)

        if explanatory_config.kix:
            self.__transform_funcs.append(kix)

        if explanatory_config.tomogashima:
            self.__transform_funcs.append(tomogashima)

        if explanatory_config.akashi:
            self.__transform_funcs.append(akashi)

        if explanatory_config.osaka:
            self.__transform_funcs.append(osaka)

        if explanatory_config.wave_significant_height:
            self.__transform_funcs.append(wave_height)

        if explanatory_config.wave_significant_period:
            self.__transform_funcs.append(wave_period)

    def __generate_truth_transform_funcs(
        self, target_config: config.Target
    ) -> None:
        if target_config.height:
            self.__transform_funcs.append(
                wave_height)
        if target_config.period:
            self.__transform_funcs.append(
                wave_period)

    def transform(self, record: RecordFetchServiceModel) -> list:
        results = []
        for func in self.__transform_funcs:
            results = func(results, record)

        return results


MAX_MONTH = 12
MAX_HOUR = 23


def record_time(
    features: list, record: RecordFetchServiceModel
) -> list:

    record_time = datetime.datetime.strptime(record.time, "%Y-%m-%d %H:%M")

    scaled_month = 2 * math.pi * record_time.month / MAX_MONTH
    scaled_hour = 2 * math.pi * record_time.hour / MAX_HOUR

    features.append(math.sin(scaled_month))
    features.append(math.cos(scaled_month))
    features.append(math.sin(scaled_hour))
    features.append(math.cos(scaled_hour))

    return features


def wave_class(features: list, record: RecordFetchServiceModel) -> list:
    if not record.height or not record.period:
        # wind wave
        features.append(None)
        # swell_wave
        features.append(None)
        return features

    is_wind_wave = record.period > record.height * 4 + 2
    # wind wave
    features.append(int(is_wind_wave))
    # swell_wave
    features.append(int(not is_wind_wave))
    return features


def temperature(features: list, record: RecordFetchServiceModel) -> list:
    features.append(record.temperature)
    return features


def kobe_air_pressure(features: list, record: RecordFetchServiceModel) -> list:
    features.append(record.kobe_pressure)
    return features


def ukb(features: list, record: RecordFetchServiceModel) -> list:
    features.append(record.ukb_velocity)
    features.append(record.ukb_sin_direction)
    features.append(record.ukb_cos_direction)
    return features


def kix(features: list, record: RecordFetchServiceModel) -> list:
    features.append(record.kix_velocity)
    features.append(record.kix_sin_direction)
    features.append(record.kix_cos_direction)
    return features


def tomogashima(features: list, record: RecordFetchServiceModel) -> list:
    features.append(record.tomogashima_velocity)
    features.append(record.tomogashima_sin_direction)
    features.append(record.tomogashima_cos_direction)
    return features


def akashi(features: list, record: RecordFetchServiceModel) -> list:
    features.append(record.akashi_velocity)
    features.append(record.akashi_sin_direction)
    features.append(record.akashi_cos_direction)
    return features


def osaka(features: list, record: RecordFetchServiceModel) -> list:
    features.append(record.osaka_velocity)
    features.append(record.osaka_sin_direction)
    features.append(record.osaka_cos_direction)
    return features


def wave_height(features: list, record: RecordFetchServiceModel) -> list:
    if record.height:
        record.height *= 100
    features.append(record.height)
    return features


def wave_period(features: list, record: RecordFetchServiceModel) -> list:
    features.append(record.period)
    return features
