from typing import Tuple

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from ml.datasets import wind_velocity
from torchvision import models


class WindVelocityClassTrainController:
    epochs = 1000
    batch_size = 256
    earlystop_endure = 10

    def __init__(
        self,
        train_dataset: wind_velocity.WindNWFDataset,
        net: models.DenseNet,
        optimizer: torch.optim.Adam,
        loss_func: torch.nn.CrossEntropyLoss
    ):
        self.__device = "cuda" if torch.cuda.is_available() else "cpu"
        self.__train_dataset = train_dataset
        self.__net = net.to(self.__device)
        self.__optimizer = optimizer
        self.__loss_func = loss_func

    def train_model(self) -> Tuple[models.DenseNet, list, dict]:
        print("traning model...")
        train_dataloader = DataLoader(
            self.__train_dataset, batch_size=self.batch_size, shuffle=True)
        best_state_dict = None
        best_loss = None
        loss_history = []
        for epoch in tqdm(range(self.epochs)):
            self.__net.train()
            sumloss = 0
            feature: torch.Tensor
            truth: torch.Tensor
            for feature, truth in train_dataloader:
                feature = feature.to(self.__device)
                truth = truth.to(self.__device).to(torch.long)
                pred = self.__net(feature)
                loss = self.__loss_func(pred[:, 0:20], truth[:, 0])
                loss = self.__loss_func(pred[:, 20:40], truth[:, 0])
                loss += self.__loss_func(pred[:, 40:60], truth[:, 2])
                sumloss += float(loss) / 3.
                self.__optimizer.zero_grad()
                loss.backward()
                self.__optimizer.step()

            meanloss = sumloss / len(train_dataloader)
            loss_history.append(meanloss)

            if not best_loss:
                best_loss = float(meanloss)
                best_state_dict = self.__net.state_dict()
            elif best_loss >= meanloss:
                best_loss = float(meanloss)
                best_state_dict = self.__net.state_dict()

            if self.earlystop_endure < epoch - loss_history.index(best_loss):
                print("Early Stop \n")
                break

        print("complete train!")
        return self.__net, loss_history, best_state_dict
