import os

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from ml.dataset.dataset_base import NWFDatasetBase
from torchvision import models


class TrainController:
    __epochs = 1000
    __batch_size = 64
    __schedule_gamma = 0.85

    def __init__(
        self,
        train_dataset: NWFDatasetBase,
        eval_dataset: NWFDatasetBase,
        net: models.DenseNet,
        loss_func: torch.nn.Module,
        learning_rate: float = 0.01,
        max_endure: int = 10,
    ):
        self.__device = "cuda" if torch.cuda.is_available() else "cpu"
        self.__train_dataloader = DataLoader(
            train_dataset, batch_size=self.__batch_size, shuffle=True,
            num_workers=os.cpu_count(), pin_memory=True)
        self.__eval_dataloader = DataLoader(
            eval_dataset, batch_size=self.__batch_size,
            num_workers=os.cpu_count(), pin_memory=True)
        self.__net = net.to(self.__device)
        self.__loss_func = loss_func
        self.__max_endure = max_endure
        self.__learning_rate = learning_rate

        self.__optimizer = torch.optim.Adam(
            net.parameters(), lr=self.__learning_rate)
        self.__scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.__optimizer, gamma=self.__schedule_gamma)

    def train_model(self) -> tuple[models.DenseNet, list, dict]:
        print("traning model...")
        best_state_dict = None
        best_loss = None
        loss_history = []
        endure = 0
        for _ in tqdm(range(self.__epochs)):
            self.__net.train()
            feature: torch.Tensor
            truth: torch.Tensor
            pred: torch.Tensor
            for feature, truth in self.__train_dataloader:
                feature = feature.to(self.__device)
                truth = truth.to(self.__device)
                pred = self.__net(feature)
                loss = self.__loss_func(pred, truth)
                self.__optimizer.zero_grad()
                loss.backward()
                self.__optimizer.step()
            self.__scheduler.step()

            _, _, monitor_loss = self.eval()
            loss_history.append(monitor_loss)

            if not best_loss:
                best_loss = float(monitor_loss)
                best_state_dict = self.__net.state_dict()
                endure = 0
            elif best_loss >= monitor_loss:
                best_loss = float(monitor_loss)
                best_state_dict = self.__net.state_dict()
                endure = 0
            else:
                endure += 1

            if endure > self.__max_endure:
                print("Early Stop \n")
                break

        print("complete train!")
        return self.__net, loss_history, best_state_dict

    def eval(self) -> tuple[list, list, float]:
        truths = []
        predicts = []
        self.__net.eval()
        feature: torch.Tensor
        truth: torch.Tensor
        pred: torch.Tensor
        eval_loss = 0
        with torch.no_grad():
            for feature, truth in self.__eval_dataloader:
                feature = feature.to(self.__device)
                truth = truth.to(self.__device)
                pred = self.__net(feature)
                loss = float(self.__loss_func(pred, truth))
                eval_loss += loss
                truths.extend(truth.tolist())
                predicts.extend(pred.tolist())
            eval_loss /= len(self.__eval_dataloader)

        return truths, predicts, eval_loss
