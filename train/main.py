import os

import torch
from torchvision import models

from config import config
from ml import dataset
from ml.dataset.dataset_base import NWFDatasetBase
from ml.net import NWFNet
from ml.train_controller import TrainController
from services.trainreport_writeservice import TrainReportWriteService


def train_nwf(
    forecast_time_delta: int, feature_timerange: int, year: int,
    nwf_config: config.NWFConfig
) -> None:
    dataset_name = os.path.join(
        nwf_config.dataset_name,
        f"{forecast_time_delta}hour_later",
        str(year)
    )

    report_service = TrainReportWriteService(
        reportname=dataset_name, target_year=year)
    report_service.save_config(nwf_config)

    mode = "train"
    dataset_generator = dataset.generator.DatasetGenerator(
        dataset_dir=dataset_name,
        feature_fetcher=nwf_config.feature_fetcher(year, 0, mode),
        truth_fetcher=nwf_config.truth_fetcher(
            year, forecast_time_delta, mode),
        feature_timerange=feature_timerange,
        mode=mode
    )
    train_dataset: NWFDatasetBase = nwf_config.nwf_dataset(dataset_generator)

    dataset_generator = dataset.generator.DatasetGenerator(
        dataset_dir=dataset_name,
        feature_fetcher=nwf_config.feature_fetcher(year, 0, mode),
        truth_fetcher=nwf_config.truth_fetcher(
            year, forecast_time_delta, mode),
        feature_timerange=feature_timerange,
        mode="eval"
    )
    eval_dataset: NWFDatasetBase = nwf_config.nwf_dataset(dataset_generator)

    match nwf_config.dataset_type:
        case config.DatasetEnum.PRESSURE_MAP:
            net = models.DenseNet(num_classes=nwf_config.num_class)
        case config.DatasetEnum.RETWET:
            net = NWFNet(
                feature_size=train_dataset.feature_size,
                num_class=nwf_config.num_class
            )
        case name:
            raise ValueError(f"not match net ({name}).")

    controller = TrainController(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        net=net,
        loss_func=nwf_config.loss_func()
    )

    state_dict_path = report_service.state_dict_path()
    if not os.path.exists(state_dict_path):
        net, loss_history, state_dict = controller.train_model()

        report_service.state_dict(state_dict)
        report_service.loss_history(loss_history)
        best_trainloss = min(loss_history)
        print("best epoch: ", loss_history.index(best_trainloss) + 1)
        print("best train loss: ", round(best_trainloss, 5))
        print("best train RMSE: ", round(best_trainloss**0.5, 5))

    else:
        net.load_state_dict(torch.load(state_dict_path))

    mode = "eval"
    truths, predicts, eval_loss = controller.eval()
    datetimes = train_dataset.get_datasettimes()
    report_service.save_truths(truths, datetimes)
    report_service.save_preds(predicts, datetimes)

    print("eval RMSE:", round(eval_loss**0.5, 5))


if __name__ == "__main__":
    nwf_config = config.NWFConfig()

    for forecast_time_delta in [1, 3, 6, 9, 12]:
        for feature_timerange in [1]:
            for year in [2016, 2017, 2018, 2019]:
                train_nwf(
                    forecast_time_delta, feature_timerange, year, nwf_config)
