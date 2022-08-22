import os

import torch
from torch.utils.data import DataLoader
from torchvision import models

from ml.datasets import wind_dataset
from ml.controllers.wind_controller import WindTrainController
from services.application import report


reportname = "windvelocity_1hourlater"

# for year in [2016, 2017, 2018, 2019]:
for year in [2016]:
    print(year)

    report_service = report.WindReportWriteService(
        os.path.join(reportname, str(year)))

    datasetname = reportname+str(year)
    train_dataset = wind_dataset.WindNWFDataset(
        generator=wind_dataset.DatasetGenerator(
            begin_year=2016,
            end_year=2020,
            target_year=year,
            forecast_timedelta=1,
            datasetname=datasetname+"train"),
        datasetname=datasetname+"train")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    learning_rate = 0.0005
    net = models.DenseNet(
        num_classes=train_dataset.truth_size).to(device)
    optimizer = torch.optim.Adam(
        net.parameters(), lr=learning_rate)
    loss_func = torch.nn.MSELoss()

    controller = WindTrainController(
        train_dataset=train_dataset,
        net=net,
        optimizer=optimizer,
        loss_func=loss_func)
    net, loss_history, state_dict = controller.train_model()

    eval_dataset = wind_dataset.WindNWFDataset(
        generator=wind_dataset.DatasetGenerator(
            begin_year=year,
            end_year=year+1,
            forecast_timedelta=1,
            datasetname=datasetname+"eval"),
        datasetname=datasetname+"eval")
    dataloader = DataLoader(eval_dataset)

    truths = []
    predicts = []
    net.eval()
    for feature, truth in dataloader:
        pred: torch.Tensor = net(feature)
        truths.append(truth)
        predicts.append(pred)
    truths = torch.cat(truths)
    predicts = torch.cat(predicts)

    report_service.state_dict(state_dict)
    report_service.loss_history(loss_history)
    datetimes = eval_dataset.get_datasettime()
    report_service.save_truths(truths.tolist(), datetimes)
    report_service.save_preds(predicts.tolist(), datetimes)

    best_trainloss = min(loss_history)
    print("done")
    print("best epoch: ", loss_history.index(best_trainloss) + 1)
    print("best train loss: ", round(min(best_trainloss), 5))
    print("best train RMSE: ", round(best_trainloss**0.5, 5))
    print("eval RMSE:", round(float(loss_func(truths, predicts))**0.5), 5)
