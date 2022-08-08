import sqlite3


DBPATH = "../weather.sqlite"


class DbContext:
    def __init__(self) -> None:
        self.__connect = sqlite3.connect(DBPATH)
        self.__cursor = self.__connect.cursor()

    @property
    def cursor(self):
        return self.__cursor

    def close(self) -> None:
        """
        databaseをcloseする．
        """

        self.__connect.close()

    def __del__(self):
        self.close()
