from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from ml.datasets import wind_dataset
from torchvision import models
from services.application import report


class WindTrainController:
    epochs = 10000
    batch_size = 128
    learning_rate = 0.0005
    earlystop_endure = 10

    def __init__(
        self,
        train_dataset: wind_dataset.WindNWFDataset,
        eval_dataset: wind_dataset.WindNWFDataset,
    ):

        self.__train_dataset = train_dataset
        self.__eval_dataset = eval_dataset

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type:
            raise exc_type(exc_value) from exc_type
        # self.__train_dataset.close()
        # self.__eval_dataset.close()
        return True

    def train(self):
        train_dataloader = DataLoader(
            self.__train_dataset,
            batch_size=self.batch_size
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        net = models.DenseNet(
            num_classes=self.__train_dataset.truth_size
        ).to(device)
        optimizer = torch.optim.Adam(
            net.parameters(),
            lr=self.learning_rate)
        loss_func = torch.nn.MSELoss()

        best_epoch = None
        best_state_dict = None
        best_loss = None
        loss_history = []
        for epoch in tqdm(range(self.epochs)):
            # train
            net.train()
            loss: torch.nn.MSELoss = 0
            for feature, truth in train_dataloader:
                pred = net(feature)
                loss = loss_func(pred, truth)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            loss_history.append(float(loss))

            if not best_loss:
                best_epoch = epoch
                best_loss = float(loss)
                best_state_dict = net.state_dict()

            elif best_loss >= loss:
                best_epoch = epoch
                best_loss = float(loss)
                best_state_dict = net.state_dict()

            if self.earlystop_endure < epoch - loss_history.index(best_loss):
                print("Early Stop \n")
                break

        eval_dataloader = DataLoader(
            self.__train_dataset,
            batch_size=self.batch_size
        )
        truths = []
        predicts = []
        net.eval()
        for feature, truth in eval_dataloader:
            pred: torch.Tensor = net(feature)
            truths.extend(truth.tolist())
            predicts.extend(pred.tolist())
        report_service = report.WindReportWriteService("sample_predwind", 2017)
        report_service.state_dict(best_state_dict)
        report_service.loss_history(loss_history)
        report_service.save_truths(truths)
        report_service.save_preds(predicts)

        print("done")
        print("best epoch: ", best_epoch + 1)
        print("best eval loss: ", round(best_loss, 5))
        print("best eval RMSE: ", round(best_loss**0.5, 5))
