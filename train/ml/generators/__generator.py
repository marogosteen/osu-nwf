import datetime
import os
import sqlite3

import infrastructure


class DatasetGenerator:
    filenamepattern = "%Y/%m/%d/%H%M"
    recordpattern = "%Y-%m-%d %H:%M"

    def __init__(self, datasetname: str) -> None:
        self.datasetfile_path = os.path.join(
            infrastructure.DATASET_STORE_DIR,
            f"{datasetname}.csv"
        )
        self.dataset_dir = os.path.dirname(self.datasetfile_path)

        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)
        self.__db = sqlite3.connect(infrastructure.DBPATH)
        self.__dbcursor = self.__db.cursor()

    def __delete_dataset(self):
        if self.__is_generated():
            self.datasetfile.close()
            os.remove(self.datasetfile_path)

    def generate(
        self,
        begin_year: int,
        end_year: int,
        target_year: int | None = None,
        forecast_timedelta: int = 1
    ) -> None:

        if not self.__is_generated():
            # if an error occurs during generate,
            # delete the incomplete dataset.
            is_done = False
            try:
                self.__generate(
                    begin_year=begin_year,
                    end_year=end_year,
                    target_year=target_year,
                    forecast_timedelta=forecast_timedelta,
                )
                is_done = True
            except Exception as e:
                self.__delete_dataset()
                raise e
            finally:
                if not is_done:
                    self.__delete_dataset()

    def __generate(
        self,
        begin_year: int,
        end_year: int,
        target_year: int | None = None,
        forecast_timedelta: int = 1
    ) -> None:
        self.__init_record_buf(begin_year)
        forecast_timedelta: datetime.timedelta = datetime.timedelta(
            hours=forecast_timedelta)

        print(f"generating dataset ({self.datasetfile_path})...")
        self.datasetfile = open(self.datasetfile_path, mode="w")
        while True:
            record = self.__next_record()

            # 終了条件
            if record is None:
                break
            record_datetime = datetime.datetime.strptime(
                record[0], self.recordpattern)
            # evaldatasetの場合、時間ベースで停止すれば全てFetchする必要がなくなる。
            if record_datetime.year == end_year:
                break

            # continue条件
            feature_datetime = record_datetime - forecast_timedelta
            if record_datetime.year == target_year or \
                    feature_datetime.year == target_year:
                continue
            if None in record:
                continue
            imagepath = self.__get_imagepath(feature_datetime)
            if not os.path.exists(imagepath):
                continue

            record = list(record)
            record = self.record_conv(record)
            direction_class = ",".join(map(str, record))
            self.datasetfile.write(
                "{},{},{}\n".format(
                    record_datetime.strftime(self.recordpattern),
                    imagepath,
                    direction_class
                )
            )

        self.datasetfile.close()
        print(f"generate complete! ({self.datasetfile_path})")

    def __get_imagepath(self, fetchtime: datetime.datetime) -> str:
        return os.path.join(
            infrastructure.IMAGEDIR,
            fetchtime.strftime(self.filenamepattern)+".jpg"
        )

    def __is_generated(self) -> bool:
        return os.path.exists(self.datasetfile_path)

    def __init_record_buf(self, begin_year: int):
        query = self.query()
        self.__dbcursor = self.__dbcursor.execute(query)
        self.record_buf = []
        while True:
            record = self.__next_record()
            record_datetime = datetime.datetime.strptime(
                record[0], self.recordpattern)
            if record_datetime.year == begin_year:
                self.record_buf.insert(0, record)
                break

    def __next_record(self) -> list:
        if len(self.record_buf) == 0:
            self.record_buf: list = self.__dbcursor.fetchmany(1000)
            if len(self.record_buf) == 0:
                return None
        return self.record_buf.pop(0)

    def record_conv(self, record):
        return record

    def query(self) -> str:
        return ""
