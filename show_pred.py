import os

import numpy as np
import matplotlib.pyplot as plt

def rmse(o:list, p:list):
    oa = np.array(o)
    pa = np.array(p)
    return np.sqrt(np.mean(np.square(oa - pa)))

ukbcol = 1
kixcol = 2
tomogashimacol = 3
year = 2019
forecast_timedelta = 1
reportdir = f"report/windvelocity_{forecast_timedelta}hourlater/{year}"
truthpath = os.path.join(reportdir, "truth.csv")
predpath = os.path.join(reportdir, "pred.csv")

fig, ax = plt.subplots(3, 1)

ukb_observed = list(map(
    lambda l: float(l.strip().split(",")[ukbcol]),
    open(truthpath).readlines()))
ukb_predicted = list(map(
    lambda l: float(l.strip().split(",")[ukbcol]),
    open(predpath).readlines()))

b = len(ukb_observed) == len(ukb_predicted)
msg = f"obs: {len(ukb_observed)}, prd: {len(ukb_predicted)}"
assert b, msg

ax[0].plot(range(len(ukb_observed)), ukb_observed, label="observed")
ax[0].plot(range(len(ukb_predicted)), ukb_predicted, label="predicted")
ax[0].set_title("ukb")
ax[0].set_xlim([5800, 6400])
ax[0].set_ylim([0, 40])

kix_observed = list(map(
    lambda l: float(l.strip().split(",")[kixcol]),
    open(truthpath).readlines()))
kix_predicted = list(map(
    lambda l: float(l.strip().split(",")[kixcol]),
    open(predpath).readlines()))

b = len(kix_observed) == len(kix_predicted)
msg = f"obs: {len(kix_observed)}, prd: {len(kix_predicted)}"
assert b, msg

ax[1].plot(range(len(kix_observed)), kix_observed, label="observed")
ax[1].plot(range(len(kix_predicted)), kix_predicted, label="predicted")
ax[1].set_title("kix")
ax[1].set_xlim([5800, 6400])
ax[1].set_ylim([0, 40])

tomogashima_observed = list(map(
    lambda l: float(l.strip().split(",")[tomogashimacol]),
    open(truthpath).readlines()))
tomogashima_predicted = list(map(
    lambda l: float(l.strip().split(",")[tomogashimacol]),
    open(predpath).readlines()))

b = len(tomogashima_observed) == len(tomogashima_predicted)
msg = f"obs: {len(tomogashima_observed)}, prd: {len(tomogashima_predicted)}"
assert b, msg

ax[2].plot(range(len(tomogashima_observed)), tomogashima_observed, label="observed")
ax[2].plot(range(len(tomogashima_predicted)), tomogashima_predicted, label="predicted")
ax[2].set_title("tomogashima")
ax[2].set_xlim([5800, 6400])
ax[2].set_ylim([0, 40])

print(f"{year}ukb RMSE:", round(rmse(ukb_observed, ukb_predicted), 4))
print(f"{year}kix RMSE:", round(rmse(kix_observed, kix_predicted), 4))
print(f"{year}tomogashima RMSE:", round(rmse(tomogashima_observed, tomogashima_predicted), 4))

# plt.tight_layout()
# plt.show()