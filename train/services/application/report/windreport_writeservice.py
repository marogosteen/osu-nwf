import os
import json

import torch
from torchvision import transforms
import matplotlib.pyplot as plt

import config
from ml import net
from infrastructure import report_store


class WindReportWriteService:
    report_dir: str = report_store.REPORT_DIR
    state_dict_name: str = "state_dict.pt"
    truthfilename: str = "truth.csv"
    predfilename: str = "pred.csv"

    def __init__(self, reportname: str, eval_year: int) -> None:
        self.report_dir += f"{reportname}/{eval_year}/"
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

    def loss_history(self, loss_history: list) -> None:
        fig, ax = plt.subplots()
        ax.set(
            title=f"best train loss: {loss_history.index(min(loss_history))+1}",
            xlabel="Epochs",
            ylabel="MSE Loss")
        ax.plot(
            range(len(loss_history)),
            loss_history, label="train")
        ax.grid()
        ax.legend()
        plt.subplots_adjust()
        plt.savefig(self.report_dir+"loss_history.jpg")

    def train_result(self, net: net.NNWFNet, normalizer: transforms.Lambda, eval_dataset):
        # TODO filename
        with open("hoge.csv", "w") as f:
            for feature, _ in eval_dataset:
                norm_feature = normalizer(feature)
                feature.tolist()
                predict = net(norm_feature)

    def save_truths(self, truths: list) -> None:
        path = self.report_dir + self.truthfilename
        with open(path, mode="w") as f:
            for line in truths:
                line = list(map(str, line))
                line = ",".join(line)+"/"
                f.write(line)

    def save_preds(self, preds: list) -> None:
        path = self.report_dir + self.predfilename
        with open(path, mode="w") as f:
            for line in preds:
                line = list(map(str, line))
                line = ",".join(line)+"/"
                f.write(line)
