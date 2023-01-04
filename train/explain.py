import os

import numpy as np
import shap
import torch

from config import config
from ml import dataset
from ml.dataset.dataset_base import NWFDatasetBase
from ml.net import NWFNet
from services.trainreport_writeservice import TrainReportWriteService


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
FEATURE_FETCHER_NAMES = [
    "retwet_base", "not_contain_nowphas", "not_contain_wind"]
YEARS = [2016, 2017, 2018, 2019]
TRUTH_FETCHER = "wave_period"

for feature_fetcher_name in FEATURE_FETCHER_NAMES:
    for year in YEARS:
        print(feature_fetcher_name, TRUTH_FETCHER, year)
        config_json = {
            "dataset_type": "retwet",
            "feature_fetcher": feature_fetcher_name,
            "truth_fetcher": TRUTH_FETCHER,
            "forecast_time_delta": 1,
            "feature_timerange": 1,
            "target_year": year
        }
        nwf_config = config.NWFConfig(config_json)
        report_service = TrainReportWriteService(
            reportname=nwf_config.dataset_name)
        state_dict_path = report_service.state_dict_path()

        mode = "eval"
        dataset_generator = dataset.generator.DatasetGenerator(
            dataset_dir=nwf_config.dataset_name,
            feature_fetcher=nwf_config.feature_fetcher(
                nwf_config.target_year, 0, mode
            ),
            truth_fetcher=nwf_config.truth_fetcher(
                nwf_config.target_year, nwf_config.forecast_time_delta, mode
            ),
            feature_timerange=nwf_config.feature_timerange,
            mode=mode
        )
        eval_dataset: NWFDatasetBase = nwf_config.nwf_dataset(
            dataset_generator)

        net = NWFNet(
            feature_size=eval_dataset.feature_size,
            num_class=nwf_config.num_class
        ).to(DEVICE)
        net.eval()
        net.load_state_dict(torch.load(state_dict_path))
        print("net loaded.")

        feature_list = []
        for data, label in eval_dataset:
            feature_list.append(data)

        input_tensor = torch.stack(feature_list, dim=0).to(DEVICE)
        print("input tensor shape: ", input_tensor.shape)

        explainer = shap.DeepExplainer(net, input_tensor)

        shap_values = explainer.shap_values(input_tensor)
        shap_file_path = os.path.join(
            os.path.dirname(state_dict_path),
            "shap.csv"
        )

        np.savetxt(shap_file_path, shap_values, delimiter=",")
