import os
import json

import torch
from torchvision import transforms
import matplotlib.pyplot as plt

import config
from ml import net
from infrastructure import report_store


class ReportWriteService:
    report_dir: str = report_store.REPORT_DIR
    config_name: str = "config.json"
    state_dict_name: str = "state_dict.pt"

    def __init__(self, nwf_config: config.NwfConfig) -> None:
        target_name: str
        if nwf_config.target.height and nwf_config.target.period:
            target_name = "both"
        elif nwf_config.target.height:
            target_name = "height"
        elif nwf_config.target.period:
            target_name = "period"

        self.report_dir +=\
            f"{nwf_config.report_name}/{target_name+nwf_config.eval_year}/"
        if not os.path.exists(self.report_dir):
            os.makedirs(self.report_dir)

    def config(self, nwf_config: config.NwfConfig) -> None:
        config_dict = vars(nwf_config).copy()
        config_dict[nwf_config.explanatory.explanatory_dict_key] = vars(
            nwf_config.explanatory)
        config_dict[nwf_config.target.target_dict_key] = vars(
            nwf_config.target)

        save_path = self.report_dir + self.config_name
        with open(save_path, mode="w") as f:
            json.dump(config_dict, f)

    def state_dict(self, state_dict):
        save_path = self.report_dir + self.state_dict_name
        torch.save(state_dict, save_path)

    def loss_history(self, train_loss_history: list, eval_loss_history: list) -> None:
        fig, ax = plt.subplots()
        ax.set(
            title=f"best eval epoch: {eval_loss_history.index(min(eval_loss_history))+1}",
            xlabel="Epochs",
            ylabel="MSE Loss")
        ax.plot(
            range(len(train_loss_history)),
            train_loss_history, label="train")
        ax.plot(
            range(len(eval_loss_history)),
            eval_loss_history, label="eval")
        ax.grid()
        ax.legend()
        plt.subplots_adjust()
        plt.savefig(self.report_dir+"loss_history.jpg")

    def train_result(self, net: net.NNWFNet, normalizer: transforms.Lambda, eval_dataset):
        # TODO filename
        with open("hoge.csv", "w") as f:
            for feature, truth in eval_dataset:
                norm_feature = normalizer(feature)
                feature.tolist()
                predict = net(norm_feature)
