import os

import torch
from torch.utils.data import DataLoader
from torchvision import models

from ml.datasets import wind_velocity_dataset
from ml.controllers.wind_controller import WindTrainController
from services.application import report


forecast_timedelta = 1
learning_rate = 0.0005
reportname = f"windvelocity_{forecast_timedelta}hourlater"

if __name__ == "__main__":
    for year in [2016, 2017, 2018, 2019]:
        print(reportname, year)

        report_service = report.WindReportWriteService(
            reportname=reportname, target_year=year)
        datasetname = reportname+str(year)

        # IterableDatasetをDatasetにしたい
        train_dataset = wind_velocity_dataset.WindNWFDataset(
            generator=wind_velocity_dataset.DatasetGenerator(
                begin_year=2016,
                end_year=2020,
                target_year=year,
                forecast_timedelta=forecast_timedelta,
                datasetname=datasetname+"train"),
            datasetname=datasetname+"train")
        net = models.DenseNet(
            num_classes=train_dataset.truth_size)
        optimizer = torch.optim.Adam(
            net.parameters(), lr=learning_rate)
        loss_func = torch.nn.MSELoss()
        controller = WindTrainController(
            train_dataset=train_dataset,
            net=net,
            optimizer=optimizer,
            loss_func=loss_func)

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

        eval_dataset = wind_velocity_dataset.WindNWFDataset(
            generator=wind_velocity_dataset.DatasetGenerator(
                begin_year=year,
                end_year=year+1,
                forecast_timedelta=forecast_timedelta,
                datasetname=datasetname+"eval"),
            datasetname=datasetname+"eval")
        eval_dataloader = DataLoader(
            eval_dataset, batch_size=controller.batch_size)

        # このメソッド作りたい
        truths = []
        predicts = []
        net.eval()
        truth: torch.Tensor
        pred: torch.Tensor
        eval_loss = 0
        for feature, truth in eval_dataloader:
            pred = net(feature)
            eval_loss += float(loss_func(truth, pred))
            truths.extend(truth.tolist())
            predicts.extend(pred.tolist())
        eval_loss /= len(eval_dataloader)

        datetimes = eval_dataset.get_datasettimes()
        report_service.save_truths(truths, datetimes)
        report_service.save_preds(predicts, datetimes)

        print("eval RMSE:", round(eval_loss**0.5, 5))
        print("done")
