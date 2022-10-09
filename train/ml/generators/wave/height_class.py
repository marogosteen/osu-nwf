from ml.generators.__generator import DatasetGenerator


class WaveHeightClassDatasetGenerator(DatasetGenerator):
    def __init__(self, datasetname: str) -> None:
        super().__init__(datasetname)

    def record_conv(self, record) -> list:
        h = record[1]
        hc = h // 0.5
        hc = 0 if hc < 0 else hc
        hc = 8 if hc > 8 else hc
        return [int(hc)]

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
