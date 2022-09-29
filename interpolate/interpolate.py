import datetime
import os

import sqlite3
import numpy as np
from PIL import Image
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt
from matplotlib import cm


NUM_NEED_ARG = 1
UINT8 = 255
Y_GRID_SIZE = 200
X_GRID_SIZE = 300
MIN_PRESSURE = 955
MAX_PRESSURE = 1060
IMAGE_DIR = "../assets/pressure_images"
DBPATH = "../assets/weather.sqlite"
RECORD_PATTERN = "%Y-%m-%d %H:%M"
FILENAME_PATTERN = "%Y/%m/%d/%H%M"

db = sqlite3.connect(DBPATH)
cursor = db.cursor()
xx, yy = np.meshgrid(range(X_GRID_SIZE), range(Y_GRID_SIZE))


class Geocode:
    def __init__(self, record: tuple) -> None:
        self.latitude = record[0]
        self.longitude = record[1]


class Pixcoode:
    def __init__(self, geocode_record_arr: np.ndarray, geocode: Geocode) -> None:
        min_geocode_record_arr = np.min(geocode_record_arr, axis=0)
        max_geocode = Geocode(
            np.max(geocode_record_arr - min_geocode_record_arr, axis=0)
        )
        min_geocode = Geocode(min_geocode_record_arr)

        # pixcelに正規化
        self.y = int((
            geocode.latitude - min_geocode.latitude
        ) / max_geocode.latitude * Y_GRID_SIZE)
        self.x = int((
            geocode.longitude - min_geocode.longitude
        ) / max_geocode.longitude * X_GRID_SIZE)


class CornerPixIndex:
    def __init__(self, geocode_record_arr) -> None:
        top_left_geocode = (36.2033, 133.3333)
        bottom_right_geocode = (30.8377, 140.0093)

        geocode = Geocode(top_left_geocode)
        self.top = Pixcoode(geocode_record_arr, geocode).y
        self.left = Pixcoode(geocode_record_arr, geocode).x

        geocode = Geocode(bottom_right_geocode)
        self.bottom = Pixcoode(geocode_record_arr, geocode).y
        self.right = Pixcoode(geocode_record_arr, geocode).x


def conv_pix_index(points: np.ndarray) -> np.ndarray:
    # 緯度経度をPixcel座標に変換する
    mins = np.min(points, axis=0)
    maxs = np.max(points - mins, axis=0)
    points = (points - mins) / maxs
    points[:, 0] *= Y_GRID_SIZE
    points[:, 1] *= X_GRID_SIZE
    return np.round(points).astype(int)


def trim(arr: np.ndarray, cornerPixIndex: CornerPixIndex):
    return arr[
        cornerPixIndex.bottom: cornerPixIndex.top+1,
        cornerPixIndex.left: cornerPixIndex.right+1
    ]


def norm_pressure(press2d: np.ndarray) -> np.ndarray:
    press2d = (press2d - MIN_PRESSURE) / (MAX_PRESSURE - MIN_PRESSURE)
    press2d[press2d < 0] = 0
    press2d[press2d > 1] = 1
    return press2d.astype(float)


output_dir = "../assets/pressure_images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

geocode_query = open("./query/station_coodinates.sql").read()
geocode_record_arr = np.array(cursor.execute(geocode_query).fetchall())
geoindices = conv_pix_index(geocode_record_arr)
cornerPixIndex = CornerPixIndex(geocode_record_arr=geocode_record_arr)

press_query = open("./query/pressures_per_hour.sql").read()
cursor = cursor.execute(press_query)
colormap = cm.get_cmap("jet")
while True:
    press_records: list[list[str]] = cursor.fetchmany(5000)
    if not press_records:
        break

    for record in press_records:
        record_time = datetime.datetime.strptime(record[0], RECORD_PATTERN)
        writepath = os.path.join(
            IMAGE_DIR, record_time.strftime(FILENAME_PATTERN)+".jpg")
        dirname = os.path.dirname(writepath)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
            print(dirname)

        press_arr = np.array([
            float(pressure) if pressure else None for pressure in record[1].split(",")
        ]).reshape(-1, 1)
        if np.sum(press_arr == None) > len(press_arr)//2:
            msg = "欠損値が半数以上で、画像化できません。record time: {}".format(
                record_time.strftime(RECORD_PATTERN)
            )
            exit(msg)

        idx_and_press = np.hstack((geoindices, press_arr))
        idx_and_press = np.asarray(
            idx_and_press[idx_and_press[:, 2] != None],
            dtype=float
        )

        rbf = Rbf(
            idx_and_press[:, 1],
            idx_and_press[:, 0],
            idx_and_press[:, 2],
            function='linear'
        )

        press2d: np.ndarray = rbf(xx, yy)
        norm_press2d = norm_pressure(press2d)

        # norm_press2d = np.uint8(colormap(norm_press2d) * UINT8)
        # fig, ax = plt.subplots()
        # ax.imshow(press_img, cmap="jet")
        # ax.plot(idx_and_press[:, 1], idx_and_press[:, 0], "o", fillstyle="none")
        # ax.invert_yaxis()
        # plt.savefig(writepath)
        # plt.close()

        trimed_norm_press2d = trim(norm_press2d, cornerPixIndex)
        press_img = np.uint8(colormap(trimed_norm_press2d) * UINT8)
        press_img = press_img[-1::-1]
        pil_img = Image.fromarray(press_img)
        pil_img = pil_img.convert("RGB")
        pil_img.save(writepath, quolity=100)

    print(writepath)
