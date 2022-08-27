class RecordFetchServiceModel:
    time: str
    ukb_velocity: float
    ukb_sin_direction: float
    ukb_cos_direction: float
    kix_velocity: float
    kix_sin_direction: float
    kix_cos_direction: float
    tomogashima_velocity: float
    tomogashima_sin_direction: float
    tomogashima_cos_direction: float
    akashi_velocity: float
    akashi_sin_direction: float
    akashi_cos_direction: float
    osaka_velocity: float
    osaka_sin_direction: float
    osaka_cos_direction: float
    temperature: float
    kobe_pressure: float
    height: float
    period: float

    attribute_names = [
        "time",
        "ukb_velocity",
        "ukb_sin_direction",
        "ukb_cos_direction",
        "kix_velocity",
        "kix_sin_direction",
        "kix_cos_direction",
        "tomogashima_velocity",
        "tomogashima_sin_direction",
        "tomogashima_cos_direction",
        "akashi_velocity",
        "akashi_sin_direction",
        "akashi_cos_direction",
        "osaka_velocity",
        "osaka_sin_direction",
        "osaka_cos_direction",
        "temperature",
        "kobe_pressure",
        "height",
        "period"
    ]

    def __init__(self, record: list) -> None:
        self.__dict__ = dict(zip(self.attribute_names, record))
