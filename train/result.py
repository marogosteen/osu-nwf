import matplotlib.pyplot as plt
import torch

from ml import net
import config
from ml.datasets import wave_dataset

# REPORT_DIR = "../report/not_use_press_temp_train1_forecast1/height2018/"
REPORT_DIR = "../report/Base_train1_forecast1/height2018/"
CONFIG_PATH = REPORT_DIR + "config.json"
STATE_DICT_PATH = REPORT_DIR + "state_dict.pt"

nwf_config = config.NwfConfig(CONFIG_PATH)

train_dataset = wave_dataset.WaveTrainDataset(nwf_config)
eval_dataset = wave_dataset.WaveEvalDataset(nwf_config, train_dataset.normalizer)

state_dict = torch.load(STATE_DICT_PATH)
nwf_net = net.NWFNet(
    eval_dataset.feature_size,
    eval_dataset.truth_size)
nwf_net.load_state_dict(state_dict)
nwf_net.eval()

observed_list = []
pred_list = []
eval_loss = 0
for feature, truth in eval_dataset:
    predicted: torch.Tensor = nwf_net(feature)
    eval_loss += torch.square((predicted - truth))

    observed_list.append(truth[0].item())
    pred_list.append(predicted[0].item())
eval_loss /= len(eval_dataset)
print(f"RMSE: {round(eval_loss.item()**0.5, 3)}cm")

fig, ax = plt.subplots()
ax.plot(range(len(observed_list)), observed_list)
ax.plot(range(len(pred_list)), pred_list)
plt.show()
