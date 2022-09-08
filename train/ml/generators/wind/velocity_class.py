from ml.generators.__generator import DatasetGenerator


class WindVelocityClassDatasetGenerator(DatasetGenerator):
    def __init__(self, datasetname: str) -> None:
        super().__init__(datasetname)

    def record_conv(self, record) -> list:
        for i, v in enumerate(record[1:]):
            vc = v // 1
            vc = 1 if vc < 2 else v
            v = 20 if v > 20 else v
            v -= 1
            record[i] = int(vc)
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
