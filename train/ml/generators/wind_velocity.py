from ml.generators.__generator import DatasetGenerator


class WindVelocityDatasetGenerator(DatasetGenerator):
    def __init__(self, datasetname: str) -> None:
        super().__init__(datasetname)

    def record_conv(self, record) -> list:
        return record[1:]

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
