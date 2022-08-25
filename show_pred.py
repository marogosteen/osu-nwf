import datetime
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates


DATETIME_PATTERN = "%Y-%m-%d %H:%M"


def rmse(o: list, p: list):
    oa = np.array(o)
    pa = np.array(p)
    return np.sqrt(np.mean(np.square(oa - pa)))


def cast_trainresult(l: str) -> list:
    l = l.strip().split(",")
    l[datetimecol] = datetime.datetime.strptime(
        l[datetimecol], DATETIME_PATTERN)
    l[ukbcol] = float(l[ukbcol])
    l[kixcol] = float(l[kixcol])
    l[tomogashimacol] = float(l[tomogashimacol])
    return l


datetimecol = 0
ukbcol = 1
kixcol = 2
tomogashimacol = 3
year = 2019
forecast_timedelta = 1
reportdir = f"report/windvelocity_{forecast_timedelta}hourlater/{year}"
truthpath = os.path.join(reportdir, "truth.csv")
predpath = os.path.join(reportdir, "pred.csv")
truths = list(map(cast_trainresult, open(truthpath).readlines()))
preds = list(map(cast_trainresult, open(predpath).readlines()))

ax: list[plt.Axes]
fig, ax = plt.subplots(3, 1)
datetimes: list[datetime.datetime] = list(
    map(lambda l: l[datetimecol], truths))

# ukb
observed: list[float] = list(map(lambda l: l[ukbcol], truths))
predicted: list[float] = list(map(lambda l: l[ukbcol], preds))
msg = f"obs: {len(observed)}, prd: {len(predicted)}"
assert len(observed) == len(predicted), msg

ax[0].plot(datetimes, observed, label="observed")
ax[0].plot(datetimes, predicted, label="predicted")
ax[0].set_title(str(year)+"ukb")
ax[0].set_ylabel("wind velocity")
ax[0].set_xlabel("datetime")
ax[0].set_ylim([0, 40])
ax[0].xaxis.set_major_formatter(dates.DateFormatter('%Y年\n%m月%d日%H時'))
print(f"{year}ukb RMSE:", round(rmse(observed, predicted), 4))

# kix
observed: list[float] = list(map(lambda l: l[kixcol], truths))
predicted: list[float] = list(map(lambda l: l[kixcol], preds))
msg = f"obs: {len(observed)}, prd: {len(predicted)}"
assert len(observed) == len(predicted), msg

ax[1].plot(datetimes, observed, label="observed")
ax[1].plot(datetimes, predicted, label="predicted")
ax[1].set_title(str(year)+"kix")
ax[1].set_ylabel("wind velocity")
ax[1].set_xlabel("datetime")
ax[1].set_ylim([0, 40])
ax[1].xaxis.set_major_formatter(dates.DateFormatter('%Y年\n%m月%d日%H時'))
print(f"{year}kix RMSE:", round(rmse(observed, predicted), 4))

# tomogashima
observed: list[float] = list(map(lambda l: l[tomogashimacol], truths))
predicted: list[float] = list(map(lambda l: l[tomogashimacol], preds))
msg = f"obs: {len(observed)}, prd: {len(predicted)}"
assert len(observed) == len(predicted), msg

ax[2].plot(datetimes, observed, label="observed")
ax[2].plot(datetimes, predicted, label="predicted")
ax[2].set_title(str(year)+"tomogashima")
ax[2].set_ylabel("wind velocity")
ax[2].set_xlabel("datetime")
ax[2].set_ylim([0, 40])
ax[2].xaxis.set_major_formatter(dates.DateFormatter('%Y年\n%m月%d日%H時'))
print(f"{year}tomogashima RMSE:", round(rmse(observed, predicted), 4))

# plt.tight_layout()
# plt.show()
