import os


class DatasetFileReadService:
    def __init__(self, file_path) -> None:
        self.__dataset_file = open(file_path)

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        self.__dataset_file.close()

    def remove_datasetfile(self) -> None:
        try:
            self.__dataset_file.close()
        finally:
            if os.path.exists(self.__dataset_file.name):
                os.remove(self.__dataset_file.name)

    def __next__(self) -> list:
        line = self.__dataset_file.readline().strip()
        if not line:
            raise StopIteration

        line = line.split(",")
        line.pop(0)
        return list(map(self.__nullable_float, line))

    def __nullable_float(self, value: str) -> float | None:
        if value == str(None):
            return None
        return float(value)
