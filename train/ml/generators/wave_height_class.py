from ml.generators.__generator import DatasetGenerator


class WaveHeightClassDatasetGenerator(DatasetGenerator):
    def __init__(self, datasetname: str) -> None:
        super().__init__(datasetname)

    def record_conv(self, record) -> list:
        h = record[1]
        h = h // 0.5
        return

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
;
"""
