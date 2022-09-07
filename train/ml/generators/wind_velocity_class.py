from ml.generators.__generator import DatasetGenerator


class WindVelocityClassDatasetGenerator(DatasetGenerator):
    def __init__(self, datasetname: str) -> None:
        super().__init__(datasetname)

    def record_conv(self, record) -> list:
        for i, v in enumerate(record[1:]):
            v = int(v) + 1
            v = 2 if v < 2 else v
            v = 21 if v > 20 else v
            v -= 2
            record[i] = v
        return record

    def query(self) -> str:
        return """
SELECT
    ukb.datetime,
    ukb.velocity,
    kix.velocity,
    tomogashima.velocity
FROM
    wind AS ukb
    inner join wind AS kix ON ukb.datetime == kix.datetime
    inner join wind AS tomogashima ON ukb.datetime == tomogashima.datetime
WHERE
    ukb.place == "ukb"
    AND kix.place == "kix"
    AND tomogashima.place == "tomogashima"
ORDER BY 
    ukb.datetime
;
"""
