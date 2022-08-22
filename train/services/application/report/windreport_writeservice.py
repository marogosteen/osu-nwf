import datetime
import os

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
    lossimagefilename: str = "loss_history.jpg"

    def __init__(self, reportname: str, target_year: int) -> None:
        reportname = os.path.join(reportname, str(target_year))
        self.report_dir = os.path.join(self.report_dir, reportname)
        if not os.path.exists(self.report_dir):
            os.makedirs(self.report_dir)

    def config(self, nwf_config: config.NwfConfig) -> None:
        config_dict = vars(nwf_config).copy()
        config_dict[nwf_config.explanatory.explanatory_dict_key] = vars(
            nwf_config.explanatory)
        config_dict[nwf_config.target.target_dict_key] = vars(
            nwf_config.target)

    def state_dict_path(self) -> str:
        return os.path.join(self.report_dir, self.state_dict_name)

    def state_dict(self, state_dict):
        save_path = os.path.join(self.report_dir, self.state_dict_name)
        torch.save(state_dict, save_path)

    def loss_history(self, loss_history: list) -> None:
        _, ax = plt.subplots()
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
        plt.savefig(os.path.join(self.report_dir, self.lossimagefilename))

    def train_result(self, net: net.NNWFNet, normalizer: transforms.Lambda, eval_dataset):
        # TODO filename
        with open("hoge.csv", "w") as f:
            for feature, _ in eval_dataset:
                norm_feature = normalizer(feature)
                feature.tolist()
                predict = net(norm_feature)

    def save_truths(self, truths: list, datetimes: list[datetime.datetime]) -> None:
        path = os.path.join(self.report_dir, self.truthfilename)
        with open(path, mode="w") as f:
            for line, s in zip(truths, datetimes):
                line = list(map(str, line))
                line.insert(0, s)
                line = ",".join(line)+"\n"
                f.write(line)

    def save_preds(self, preds: list, datetimes: list[datetime.datetime]) -> None:
        path = os.path.join(self.report_dir, self.predfilename)
        with open(path, mode="w") as f:
            for line, s in zip(preds, datetimes):
                line = list(map(str, line))
                line.insert(0, s)
                line = ",".join(line)+"\n"
                f.write(line)
