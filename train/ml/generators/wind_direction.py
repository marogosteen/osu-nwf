import math

from ml.generators.__generator import DatasetGenerator


class WindDirectionDatasetGenerator(DatasetGenerator):
    def __init__(self, datasetname: str) -> None:
        super().__init__(datasetname)

    def record_conv(self, record) -> list:
        return [
            self.conv_direction(record[1], record[2]),
            self.conv_direction(record[3], record[4]),
            self.conv_direction(record[5], record[6])
        ]

    def conv_direction(self, sinv: float, cosv: float) -> int:
        # 静穏の場合
        if sinv == 0 and cosv == 0:
            return 0

        asv = math.asin(sinv)
        acv = math.acos(cosv)
        sd = asv / (2 * math.pi) * 360
        cd = acv / (2 * math.pi) * 360
        r = cd if sd >= 0 else 360 - cd
        r = round(r, 1)
        if r % (360 / 16) != 0:
            raise ValueError("")
        c = int(r // (360 / 16) + 1 - 8)
        if c <= 0:
            c += 16

        return c

    def query(self) -> str:
        return """
SELECT
    ukb.datetime,
    ukb.sin_direction,
    ukb.cos_direction,
    kix.sin_direction,
    kix.cos_direction,
    tomogashima.sin_direction,
    tomogashima.cos_direction
FROM
    wind AS ukb
    INNER JOIN wind AS kix ON ukb.datetime == kix.datetime
    INNER JOIN wind AS tomogashima ON ukb.datetime == tomogashima.datetime
WHERE
    ukb.place == "ukb" AND
    kix.place == "kix" AND
    tomogashima.place == "tomogashima"
ORDER BY
    ukb.datetime
;
"""
