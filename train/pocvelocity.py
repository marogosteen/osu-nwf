import os

import torch
from torch.utils.data import DataLoader
from torchvision import models

from ml.generators.wind_velocity import WindVelocityDatasetGenerator as DSGenerator
from ml.dataset import NWFDataset
from ml.controllers.wind
from services.application import report


learning_rate = 0.0005
if __name__ == "__main__":
    for forecast_timedelta in [1]:
        for year in [2016]:
            datasetname = f"hogevelocity/{forecast_timedelta}hourlater/{year}"
            print(datasetname)

            report_service = report.WindReportWriteService(
                reportname=datasetname, target_year=year)

            generator = DSGenerator(
                datasetname=datasetname+"train")
            generator.generate(
                begin_year=2016,
                end_year=2020,
                target_year=year,
                forecast_timedelta=forecast_timedelta,
            )
            train_dataset = NWFDataset(
                generator.datasetfile_path)

            net = models.DenseNet(num_classes=3)
            optimizer = torch.optim.Adam(
                net.parameters(), lr=learning_rate)
            loss_func = torch.nn.MSELoss()
            controller = WindDirectionTrainController(
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

            generator = DSGenerator(
                datasetname=datasetname+"eval")
            generator.generate(
                begin_year=year,
                end_year=year+1,
                forecast_timedelta=forecast_timedelta,
            )
            eval_dataset = NWFDataset(
                generator.datasetfile_path)
            eval_dataloader = DataLoader(
                eval_dataset, batch_size=controller.batch_size)

            # このメソッド作りたい
            device = "cuda" if torch.cuda.is_available() else "cpu"
            truths = []
            predicts = []
            net.eval()
            truth: torch.Tensor
            pred: torch.Tensor
            eval_loss, ukb_correct, kix_correct, tomogashima_correct = 0, 0, 0, 0
            for feature, truth in eval_dataloader:
                feature = feature.to(device)
                truth = truth.to(device).to(torch.long)
                pred = net(feature)
                loss = float(loss_func(pred, truth))
                eval_loss += loss
                truths.extend(truth.tolist())
                predicts.extend(pred.tolist())

            eval_loss /= len(eval_dataloader)
            datetimes = eval_dataset.get_datasettimes()
            report_service.save_truths(truths, datetimes)
            report_service.save_preds(predicts, datetimes)

            print("eval RMSE:", round(eval_loss**0.5, 5))
