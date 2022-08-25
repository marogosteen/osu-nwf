from typing import Tuple

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from ml.datasets import wind_dataset
from torchvision import models


class WindTrainController:
    epochs = 1000
    batch_size = 256
    learning_rate = 0.0005
    earlystop_endure = 10

    def __init__(
        self,
        train_dataset: wind_dataset.WindNWFDataset,
        net: models.DenseNet,
        optimizer: torch.optim.Adam,
        loss_func: torch.nn.MSELoss
    ):
        self.__train_dataset = train_dataset
        self.__net = net
        self.__optimizer = optimizer
        self.__loss_func = loss_func

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type:
            raise exc_type(exc_value) from exc_type
        # self.__train_dataset.close()
        # self.__eval_dataset.close()
        return True

    def train_model(self) -> Tuple[models.DenseNet, list, dict]:
        print("traning model...")
        train_dataloader = DataLoader(
            self.__train_dataset, batch_size=self.batch_size, shuffle=True)
        best_state_dict = None
        best_loss = None
        loss_history = []
        for epoch in tqdm(range(self.epochs)):
            # train
            self.__net.train()
            sumloss = 0
            for feature, truth in train_dataloader:
                pred = self.__net(feature)
                loss = self.__loss_func(pred, truth)
                sumloss += float(loss)
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
