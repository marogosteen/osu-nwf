import numpy as np
import shap
import torch
from torchvision import transforms

from nwf.datasets.dataset import EvalDatasetModel, TrainDatasetModel
from nwf.net import NNWF_Net
from services.query import DbQuery
from services.recordService import RecordService

TARGET_YEAR = 2018
TRAIN_HOUR = 1
FORECAST_HOUR = 1

CASE_NAME = "Not_Use_WaveClass2_Period"
CASEDIR = f"result/{CASE_NAME}{TRAIN_HOUR}HTrain{FORECAST_HOUR}HLater{TARGET_YEAR}/"
STATE_DICT_PATH = CASEDIR + "state_dict.pt"
SAVE_CSV_PATH = CASEDIR + "shaplayValue.csv"

print(CASEDIR)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


tds = TrainDatasetModel(
    forecastHour=FORECAST_HOUR,
    trainHour=TRAIN_HOUR,
    recordService=RecordService(
        DbQuery(targetyear=TARGET_YEAR, mode="train")))

eds = EvalDatasetModel(
    forecastHour=FORECAST_HOUR,
    trainHour=TRAIN_HOUR,
    recordService=RecordService(
        DbQuery(targetyear=TARGET_YEAR, mode="eval")))

transform = transforms.Lambda(
    lambda x: (x - tds.mean)/tds.std)

net = NNWF_Net(eds.dataSize).to(device)
net.eval()
net.load_state_dict(torch.load(STATE_DICT_PATH))

inputList = []
for data, label in eds:
    inputList.append(transform(data))

inputTensor = torch.stack(inputList, dim=0)
print("inputTensor shape: ", inputTensor.shape)

explainer = shap.DeepExplainer(net, inputTensor)

shap_values = explainer.shap_values(inputTensor)
np.savetxt(SAVE_CSV_PATH, shap_values, delimiter=",")

print("\nDone\n")
