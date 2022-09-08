import datetime
import os

import torch
import matplotlib.pyplot as plt

from infrastructure import report_store


class TrainReportWriteService:
    report_dir: str = report_store.REPORT_DIR
    state_dict_name: str = "state_dict.pt"
    truthfilename: str = "truth.csv"
    predfilename: str = "pred.csv"
    lossimagefilename: str = "loss_history.jpg"

    def __init__(self, reportname: str, target_year: int) -> None:
        self.report_dir = os.path.join(self.report_dir, reportname)
        if not os.path.exists(self.report_dir):
            os.makedirs(self.report_dir)

    def state_dict_path(self) -> str:
        return os.path.join(self.report_dir, self.state_dict_name)

    def state_dict(self, state_dict):
        save_path = os.path.join(self.report_dir, self.state_dict_name)
        torch.save(state_dict, save_path)

    def loss_history(self, loss_history: list) -> None:
        bestloss = loss_history.index(min(loss_history))+1

        _, ax = plt.subplots()
        ax.set(
            title=f"best train loss: {bestloss}",
            xlabel="Epochs",
            ylabel="MSE Loss")
        ax.plot(
            range(len(loss_history)),
            loss_history, label="train")
        ax.grid()
        ax.legend()
        plt.subplots_adjust()
        plt.savefig(os.path.join(self.report_dir, self.lossimagefilename))

    def save_truths(
        self, truths: list, datetimes: list[datetime.datetime]
    ) -> None:
        path = os.path.join(self.report_dir, self.truthfilename)
        with open(path, mode="w") as f:
            for line, s in zip(truths, datetimes):
                line = list(map(str, line))
                line.insert(0, s)
                line = ",".join(line)+"\n"
                f.write(line)

    def save_preds(
        self, preds: list, datetimes: list[datetime.datetime]
    ) -> None:
        path = os.path.join(self.report_dir, self.predfilename)
        with open(path, mode="w") as f:
            for line, s in zip(preds, datetimes):
                line = list(map(str, line))
                line.insert(0, s)
                line = ",".join(line)+"\n"
                f.write(line)
