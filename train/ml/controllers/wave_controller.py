import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from services.application import report
from ml.datasets import wave_dataset
from ml import net


class WaveTrainController:
    def __init__(self, train_config: config.NwfConfig):
        self.__train_config = train_config
        self.__train_dataset = wave_dataset.WaveTrainDataset(
            self.__train_config)
        self.__eval_dataset = wave_dataset.WaveEvalDataset(
            self.__train_config,
            self.__train_dataset.normalizer)

    # TODO このクラスと切り離したい． __enter__で別クラスを渡す
    def train(self):
        train_dataloader = DataLoader(
            self.__train_dataset,
            batch_size=self.__train_config.batch_size)
        eval_dataloader = DataLoader(
            self.__eval_dataset,
            batch_size=self.__train_config.batch_size)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        nwf_net = net.NNWFNet(
            self.__train_dataset.feature_size,
            self.__train_dataset.truth_size).to(device)
        optimizer = torch.optim.Adam(
            nwf_net.parameters(),
            lr=self.__train_config.learning_rate)
        loss_func = torch.nn.MSELoss()

        best_epoch = None
        best_state_dict = None
        best_eval_loss = None
        train_loss_history = []
        eval_loss_history = []
        for epoch in tqdm(range(self.__train_config.epochs)):
            # train
            nwf_net.train()
            train_loss: torch.nn.MSELoss = 0
            for feature, truth in train_dataloader:
                train_loss = loss_func(nwf_net(feature), truth)

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

            train_loss_history.append(train_loss.item())

            # eval
            nwf_net.eval()
            count_batches = len(eval_dataloader)
            eval_loss = 0
            with torch.no_grad():
                for feature, truth in eval_dataloader:
                    eval_loss += loss_func(nwf_net(feature), truth).item()
            eval_loss /= count_batches

            eval_loss_history.append(eval_loss)

            if not best_eval_loss:
                best_epoch = epoch
                best_eval_loss = eval_loss
                best_state_dict = nwf_net.state_dict()

            elif best_eval_loss >= eval_loss:
                best_epoch = epoch
                best_eval_loss = eval_loss
                best_state_dict = nwf_net.state_dict()

            if self.__train_config.earlystop_endure < epoch - eval_loss_history.index(best_eval_loss):
                print("Early Stop \n")
                break

        nwf_net.eval()
        for feature, truth in self.__eval_dataset:
            pred: torch.Tensor = nwf_net(feature)
            pred.tolist()

        report_service = report.ReportWriteService(self.__train_config)
        report_service.config(self.__train_config)
        report_service.state_dict(best_state_dict)
        report_service.loss_history(train_loss_history, eval_loss_history)

        print("done")
        print("best epoch: ", best_epoch + 1)
        print("best eval loss: ", round(best_eval_loss, 5))
        print("best eval RMSE: ", round(best_eval_loss**0.5, 5))

        self.__train_dataset.close()
        self.__eval_dataset.close()
