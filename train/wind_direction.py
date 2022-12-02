import os

import torch
from torch.utils.data import DataLoader
from torchvision import models

from ml.dataset import NWFPressureMap
from ml.dataset.generator import DatasetGeneratorBase
from ml.dataset.generator.fetcher.pressure_map import PressureImagePathFetcher
from ml.dataset.generator.fetcher.wind import WindDirectionFetcher
from ml.losses.wind.direction import WindDirectionLoss
from ml.train_controller import TrainController
from services.trainreport_writeservice import TrainReportWriteService


if __name__ == "__main__":
    for forecast_timedelta in [1, 3, 6, 9, 12]:
        for year in [2016, 2017, 2018, 2019]:
            datasetname = "wind/direction/{}hourlater/{}".format(
                forecast_timedelta, year)
            print(datasetname)

            report_service = TrainReportWriteService(
                reportname=datasetname, target_year=year)

            dataset_generator = DatasetGeneratorBase(
                dataset_dir=datasetname,
                feature_fetcher=PressureImagePathFetcher(year, 0, "train"),
                truth_fetcher=WindDirectionFetcher(
                    year, forecast_timedelta, "train"),
                mode="train"
            )

            train_dataset = NWFPressureMap(
                generator=dataset_generator)

            device = "cuda" if torch.cuda.is_available() else "cpu"
            net = models.DenseNet(num_classes=51).to(device)
            lossfunc = WindDirectionLoss()
            controller = TrainController(
                train_dataset=train_dataset, device=device,
                net=net, lossfunc=lossfunc)

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

            dataset_generator = DatasetGeneratorBase(
                dataset_dir=datasetname,
                feature_fetcher=PressureImagePathFetcher(year, 0, "eval"),
                truth_fetcher=WindDirectionFetcher(
                    year, forecast_timedelta, "eval"),
                mode="eval"
            )

            eval_dataset = NWFPressureMap(
                generator=dataset_generator)
            eval_dataloader = DataLoader(
                eval_dataset, batch_size=controller.batch_size)

            # このメソッド作りたい
            truths = []
            predicts = []
            net.eval()
            truth: torch.Tensor
            pred: torch.Tensor
            eval_loss, u_correct, k_correct, t_correct = 0, 0, 0, 0
            with torch.no_grad():
                for feature, truth in eval_dataloader:
                    feature = feature.to(device)
                    truth = truth.to(device).to(torch.long)
                    pred = net(feature)
                    loss = float(lossfunc(pred, truth))
                    eval_loss += loss
                    truths.extend(truth.tolist())
                    predicts.extend(pred.tolist())

                    u_correct += float((
                        pred[:, 0:17].argmax(1) == truth[:, 0]
                    ).type(torch.float).sum())
                    k_correct += float((
                        pred[:, 17:34].argmax(1) == truth[:, 1]
                    ).type(torch.float).sum())
                    t_correct += float((
                        pred[:, 34:51].argmax(1) == truth[:, 2]
                    ).type(torch.float).sum())

                eval_loss /= len(eval_dataloader)
                u_correct /= len(eval_dataset)
                k_correct /= len(eval_dataset)
                t_correct /= len(eval_dataset)
                print(f"ukb Accuracy: {(100*u_correct):>0.1f}%")
                print(f"kix Accuracy: {(100*k_correct):>0.1f}%")
                print(f"tomogashima Accuracy: {(100*t_correct):>0.1f}%")

                datetimes = eval_dataset.get_datasettimes()
                report_service.save_truths(truths, datetimes)
                report_service.save_preds(predicts, datetimes)

                print("eval RMSE:", round(eval_loss**0.5, 5))
