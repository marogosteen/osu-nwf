import os

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from ml.dataset.base_dataset import BaseNWFDataset
from torchvision import models


class TrainController:
    epochs = 1000
    batch_size = 256
    schedule_gamma = 0.9

    def __init__(
        self,
        train_dataset: BaseNWFDataset,
        device: str,
        net: models.DenseNet,
        lossfunc: torch.nn.Module,
        learning_rate: float = 0.01,
        max_endure: int = 10,
    ):
        self.__device = device
        self.__train_dataset = train_dataset
        self.__net = net
        self.__lossfunc = lossfunc
        self.max_endure = max_endure
        self.learning_rate = learning_rate

        self.__optimizer = torch.optim.Adam(
            net.parameters(), lr=self.learning_rate)
        self.__scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.__optimizer, gamma=self.schedule_gamma)

    def train_model(self) -> tuple[models.DenseNet, list, dict]:
        print("traning model...")
        train_dataloader = DataLoader(
            self.__train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=os.cpu_count(), pin_memory=True)
        best_state_dict = None
        best_loss = None
        loss_history = []
        endure = 0
        for _ in tqdm(range(self.epochs)):
            self.__net.train()
            sumloss = 0
            feature: torch.Tensor
            truth: torch.Tensor
            for feature, truth in train_dataloader:
                feature = feature.to(self.__device)
                truth = truth.to(self.__device)
                pred = self.__net(feature)
                loss = self.__lossfunc(pred, truth)
                sumloss += float(loss) / 3.
                self.__optimizer.zero_grad()
                loss.backward()
                self.__optimizer.step()
            self.__scheduler.step()

            meanloss = sumloss / len(train_dataloader)
            loss_history.append(meanloss)

            if not best_loss:
                best_loss = float(meanloss)
                best_state_dict = self.__net.state_dict()
                endure = 0
            elif best_loss >= meanloss:
                best_loss = float(meanloss)
                best_state_dict = self.__net.state_dict()
                endure = 0
            else:
                endure += 1

            if endure > self.max_endure:
                print("Early Stop \n")
                break

        print("complete train!")
        return self.__net, loss_history, best_state_dict
