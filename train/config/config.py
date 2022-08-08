import json


class Explanatory:
    def __init__(self, config_dict: dict) -> None:
        self.explanatory_dict_key = "explanatory"
        config_dict: dict = config_dict[self.explanatory_dict_key]

        self.datetime: bool = None
        self.wave_class: bool = None
        self.ukb: bool = None
        self.kix: bool = None
        self.tomogashima: bool = None
        self.akashi: bool = None
        self.osaka: bool = None
        self.temperature: bool = None
        self.kobe_air_pressure: bool = None
        self.wave_significant_height: bool = None
        self.wave_significant_period: bool = None

        self_dict = vars(self)
        for key in config_dict.keys():
            if not key in self_dict.keys():
                raise Exception(key+" is not match")

        self_dict.update(config_dict)


class Target:
    def __init__(self, config_dict: dict) -> None:
        self.target_dict_key = "target"
        config_dict: dict = config_dict[self.target_dict_key]

        self.height: bool = None
        self.period: bool = None

        self_dict = vars(self)
        for key in config_dict.keys():
            if not key in self_dict.keys():
                raise Exception(key+" is not match")

        self_dict.update(config_dict)

        # if list(vars(self).values()).count(True) != 1:
        #     raise Exception("設定できる予測値は1つのみです。")


class NwfConfig:
    def __init__(self, file_path: str):
        self.case_title: str = None
        self.epochs: int = None
        self.learning_rate: float = None
        self.batch_size: int = None
        self.eval_year: int = None
        self.forecast_hour: int = None
        self.train_span: int = None
        self.forecast_span: int = None
        self.earlystop_endure: int = None
        self.explanatory: Explanatory = None
        self.target: Target = None

        with open(file_path) as f:
            nwf_config_dict: dict = json.load(f)
            self_dict = vars(self)
            for key in nwf_config_dict.keys():
                if not key in self_dict.keys():
                    raise Exception(key+" is not match")

            self_dict.update(nwf_config_dict)

            self.explanatory = Explanatory(nwf_config_dict)
            self.target = Target(nwf_config_dict)

        if not self.case_title:
            raise ValueError("please set case_title.")

    @property
    def report_name(self):
        return f"{self.case_title}_train{self.train_span}_forecast{self.forecast_hour}"
