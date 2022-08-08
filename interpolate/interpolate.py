import datetime
import os
import sys

import sqlite3
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.interpolate import Rbf


NUM_NEED_ARG = 1
UINT8 = 255
GRID_SIZE = 100
MIN_PRESSURE = 955
MAX_PRESSURE = 1060
XX, YY = np.meshgrid(range(GRID_SIZE), range(GRID_SIZE))
DBPATH = "pressure_db.sqlite"
START_TIME = datetime.datetime(year=2016, month=1, day=1, hour=0, minute=0)
END_YEAR = 2018
DBCONNECT = sqlite3.connect(DBPATH)
CURSOR = DBCONNECT.cursor()


def fetch_pointarray() -> np.ndarray:
    query = f"""
SELECT
    coodinate.latitude,
    coodinate.longitude
FROM
    coodinate
ORDER BY
    coodinate.point_name
;"""
    CURSOR.execute(query)
    return np.array(CURSOR.fetchall())


def conv_index(points: np.ndarray) -> np.ndarray:
    mins = np.min(points, axis=0)
    points = points - mins
    maxs = np.max(points, axis=0)
    return np.round(points / maxs * GRID_SIZE).astype(np.int64)


def fetch_corner(points: np.ndarray):
    minpoint = np.min(points, axis=0)
    maxpoint = np.max(points - minpoint, axis=0)

    query = f"""
SELECT
    latitude,
    longitude
FROM
    coodinate
WHERE
    coodinate.point_name == 'matsue'
    or coodinate.point_name == 'shionomisaki'
;"""
    CURSOR.execute(query)
    cornerpoint = np.array(CURSOR.fetchall())
    return np.round((cornerpoint - minpoint) / maxpoint * GRID_SIZE).astype(np.int64)


def trim(a: np.ndarray, corner_indices: np.ndarray):
    bottom = corner_indices[:, 0].min()
    left = corner_indices[:, 1].min()
    top = corner_indices[:, 0].max()
    right = corner_indices[:, 1].max()
    return a[bottom:top+1, left:right+1]


def norm_image(pressure_array) -> np.ndarray:
    return (pressure_array - MIN_PRESSURE) / (1060-955) * UINT8


num_arg = len(sys.argv[1:])
if num_arg != NUM_NEED_ARG:
    raise TypeError(f"takes exactly one argment ({num_arg} given)")
output_dir = sys.argv[1]

pressure_delta = MAX_PRESSURE - MIN_PRESSURE
points = fetch_pointarray()
place_indices = conv_index(points)
corner_indices = fetch_corner(points)

timedelta = datetime.timedelta(hours=1)
pattern = "%Y-%m-%d %H:%M"
time = START_TIME
while True:
    if time.year == END_YEAR:
        break

    str_time = time.strftime(pattern)
    savedir = output_dir+f"{time.year}/{str(time.month).zfill(2)}/"
    savedir = os.path.join(
        output_dir,
        str(time.year),
        str(time.month).zfill(2)
    )
    if not os.path.exists(savedir):
        print(str_time)
        os.makedirs(savedir)

    query = f"""
SELECT
    air_pressure.air_pressure
FROM
    air_pressure
    inner join coodinate on air_pressure.place == coodinate.point_name
WHERE
    air_pressure.datetime = '{str_time}'
ORDER BY
    coodinate.point_name
;"""
    CURSOR.execute(query)
    records = CURSOR.fetchall()
    records = np.hstack((place_indices, records))
    records = records[records[:, 2] != None].astype(np.float64)
    rbf = Rbf(
        records[:, 1],
        records[:, 0],
        records[:, 2],
        function='linear'
    )
    pressure_array: np.ndarray = rbf(XX, YY)
    pressure_array = trim(pressure_array, corner_indices)
    pressure_array = np.round(
        norm_image(pressure_array)
    ).astype(np.uint8)
    pressure_array = pressure_array[-1::-1]

    img = Image.fromarray(pressure_array)
    savepath = os.path.join(savedir, f"{str_time}.jpg")
    img.save(savepath, quolity=100)

    time += timedelta
