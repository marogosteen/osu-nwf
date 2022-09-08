from ml.generators.__generator import DatasetGenerator


class WaveHeightDatasetGenerator(DatasetGenerator):
    def __init__(self, datasetname: str) -> None:
        super().__init__(datasetname)

    def record_conv(self, record) -> list:
        return record[1]

    def query(self) -> str:
        return """
SELECT
    datetime,
    significant_height
FROM
    wave
WHERE
    place == "kobe"
ORDER BY
    datetime
"""
