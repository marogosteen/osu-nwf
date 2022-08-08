from domain.tables import *
from infrastructure import database
from services import domain


class RecordFetchService:
    fetch_count = 10000

    def __init__(self, query: str) -> None:
        self.query = query
        self.dbContext = database.DbContext()
        self.execute_sql()
        self.__buffer = iter([])

    def close(self) -> None:
        self.dbContext.close()

    def __del__(self):
        self.close()

    def execute_sql(self) -> None:
        self.dbContext.cursor.execute(self.query)

    def fetch_many(self) -> None:
        buffer = self.dbContext.cursor.fetchmany(self.fetch_count)
        if not buffer:
            raise StopIteration
        self.__buffer = iter(buffer)

    def __next__(self):
        try:
            record = next(self.__buffer)
        except StopIteration:
            self.fetch_many()
            record = next(self.__buffer)

        return domain.RecordFetchServiceModel(record)
