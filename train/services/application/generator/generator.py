import datetime

import config
from services import domain
from services.application.generator import transformer


class GenerateResult:
    feature_path: str
    truth_path: str
    generator_mode: str


class RecordBuffer:
    def __init__(self) -> int:
        self.buffer: list[domain.RecordFetchServiceModel] = []

    def push(self, record):
        self.buffer.append(record)
        self.buffer.pop(0)

    def __iter__(self):
        self.iter_queue = iter(self.buffer)
        return self

    def __next__(self) -> domain.RecordFetchServiceModel:
        return next(self.iter_queue)

    def __getitem__(self, num: int) -> domain.RecordFetchServiceModel:
        return self.buffer[num]


class DatasetGenerator:
    record_time_format = "%Y-%m-%d %H:%M"

    def __init__(self, nwf_config: config.NwfConfig, sql_query: str, mode: str) -> None:
        self.mode = mode
        self.forecast_hour = nwf_config.forecast_hour

        self.__write_service = domain.DatasetFileWriteService(mode)
        self.__explanatory_transformer = transformer.RecordTransformer(
            nwf_config.explanatory)
        self.__target_transformer = transformer.RecordTransformer(
            nwf_config.target)

        self.__feature_fetch_service = domain.RecordFetchService(
            sql_query)
        self.__truth_fetch_service = domain.RecordFetchService(
            sql_query)

        # truthは数時刻先のRecordを返す
        for _ in range(nwf_config.train_span-1+nwf_config.forecast_hour):
            next(self.__truth_fetch_service)

        self.__feature_buffer = RecordBuffer()
        for _ in range(nwf_config.train_span):
            self.__feature_buffer.buffer.append(
                next(self.__feature_fetch_service))

        self.__truth_buffer = RecordBuffer()
        for _ in range(nwf_config.forecast_span):
            self.__truth_buffer.buffer.append(
                next(self.__truth_fetch_service))

    def generate(self) -> GenerateResult:
        while True:
            try:
                if self.__is_inferiority():
                    self.__feature_buffer.push(
                        next(self.__feature_fetch_service))
                    self.__truth_buffer.push(
                        next(self.__truth_fetch_service))
                    continue

                features = [self.__feature_buffer[-1].time]
                for record in self.__feature_buffer:
                    features.extend(
                        self.__explanatory_transformer.transform(record))

                self.__write_service.write_features(features)

                truths = [self.__truth_buffer[0].time]
                for record in self.__truth_buffer:
                    truths.extend(
                        self.__target_transformer.transform(record))
                self.__write_service.write_truths(truths)

                self.__feature_buffer.push(
                    next(self.__feature_fetch_service))
                self.__truth_buffer.push(
                    next(self.__truth_fetch_service))

            except StopIteration:
                self.__write_service.close()
                break

        result = GenerateResult()
        result.feature_path = self.__write_service.feature_path
        result.truth_path = self.__write_service.truth_path
        result.generator_mode = self.mode

        return result

    # record must be continuous as a time series.
    def __is_inferiority(self) -> bool:
        record: domain.RecordFetchServiceModel

        time_delta = datetime.timedelta(hours=1)
        confirm_time: datetime.datetime = None
        for record in self.__feature_buffer:
            if not confirm_time:
                confirm_time = datetime.datetime.strptime(
                    record.time, self.record_time_format)
            else:
                confirm_time += time_delta

            if record.time != datetime.datetime.strftime(
                confirm_time, self.record_time_format
            ):
                return True

        confirm_time += datetime.timedelta(hours=self.forecast_hour)
        for record in self.__truth_buffer:
            if record.time != datetime.datetime.strftime(
                confirm_time, self.record_time_format
            ):
                return True
            confirm_time += time_delta

        return False
