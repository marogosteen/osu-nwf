import datetime
import os
import sqlite3

import numpy as np
from matplotlib import cm
from PIL import Image

NUM_NEED_ARG = 1
UINT8 = 255
Y_GRID_SIZE = 500
X_GRID_SIZE = 400
MIN_PRESSURE = 955
MAX_PRESSURE = 1060
WRITE_IMAGE_DIR = "../assets/pressure_images"
DBPATH = "../assets/weather.sqlite"
RECORD_PATTERN = "%Y-%m-%d %H:%M"
FILENAME_PATTERN = "%Y/%m/%d/%H%M"
STATION_POINTS_QUERY_PATH = "./query/station_coodinates.sql"
PRESSURES_QUERY_PATH = "./query/pressures_per_hour.sql"


def conv_pix_index(points: np.ndarray) -> np.ndarray:
    """緯度経度をPixcel座標に変換する"""
    mins = np.min(points, axis=0)
    maxs = np.max(points - mins, axis=0)
    points = (points - mins) / maxs
    points[:, 0] *= Y_GRID_SIZE
    points[:, 1] *= X_GRID_SIZE
    return np.round(points).astype(int)


def get_power_maps():
    """各観測点における圧力の勢力Mapを返す"""
    pix_points = conv_pix_index(geo_points)
    observed_count = len(pix_points)
    yy_ = np.repeat(np.arange(Y_GRID_SIZE).reshape(-1, 1), X_GRID_SIZE, axis=1)

    maps = np.empty((observed_count, Y_GRID_SIZE, X_GRID_SIZE))
    for main_id in range(observed_count):
        x_distance = np.abs(xx - pix_points[main_id, 1])
        y_distance = np.abs(yy - pix_points[main_id, 0])

        map_item = maps[main_id]
        with np.errstate(divide="ignore"):
            map_item = 1 / np.sqrt(
                np.square(x_distance) + np.square(y_distance))
        map_item = np.where(map_item == np.inf, 1, map_item)

        # 勢力が及ぼさない範囲
        other_ids = np.delete(np.arange(observed_count), main_id)
        point_diffs = pix_points[main_id, :2] - pix_points[other_ids, :2]
        with np.errstate(divide="ignore"):
            grads = - 1 / (point_diffs[:, 0] / point_diffs[:, 1])
        slicies = pix_points[other_ids, 0] - grads * pix_points[other_ids, 1]

        is_grad_inf = grads == -np.inf
        grads[is_grad_inf], slicies[is_grad_inf] = 0, 0

        borders = np.repeat(
            np.arange(X_GRID_SIZE).reshape(1, -1), observed_count-1, axis=0
        ) * grads.reshape(-1, 1) + slicies.reshape(-1, 1)

        is_positive = point_diffs[:, 0] >= 0
        if is_positive.sum() > 0:
            under_border = np.repeat(
                np.max(borders[is_positive], axis=0).reshape(1, -1),
                Y_GRID_SIZE,
                axis=0
            )
            map_item[yy_ <= under_border] = 0

        is_negative = (point_diffs[:, 0] < 0)
        if is_negative.sum() > 0:
            over_border = np.repeat(
                np.min(borders[is_negative], axis=0).reshape(1, -1),
                Y_GRID_SIZE,
                axis=0
            )
            map_item[yy_ > over_border] = 0

        maps[main_id] = map_item

    maps /= np.sum(maps, axis=0)
    return maps


db = sqlite3.connect(DBPATH)
cursor = db.cursor()
xx, yy = np.meshgrid(range(X_GRID_SIZE), range(Y_GRID_SIZE))

geo_points = np.array(cursor.execute(
    open(STATION_POINTS_QUERY_PATH).read()
).fetchall())

power_maps = get_power_maps()
cursor = cursor.execute(open(PRESSURES_QUERY_PATH).read())
observed_count = len(geo_points)
err_threshold = observed_count // 2
colormap = cm.get_cmap("turbo")
while True:
    press_records: list[list[str]] = cursor.fetchmany(5000)
    if not press_records:
        break

    for img_num, record in enumerate(press_records):
        press_map_pairts = power_maps.copy()

        record_time = datetime.datetime.strptime(record[0], RECORD_PATTERN)
        write_path = os.path.join(
            WRITE_IMAGE_DIR, record_time.strftime(FILENAME_PATTERN)+".png"
        )
        write_dir = os.path.dirname(write_path)
        if not os.path.exists(write_dir):
            os.makedirs(write_dir)

        pressures = np.array(list(map(
            lambda pressure: float(pressure) if pressure else None,
            record[1].split(",")
        )), dtype=float).reshape(observed_count, 1, 1)

        if np.isnan(pressures).sum() > err_threshold:
            msg = "欠損値が半数以上で、画像化できません。record time: {}".format(
                record_time.strftime(RECORD_PATTERN)
            )
            raise RuntimeError(msg)

        press_map_pairts *= pressures
        press_map = press_map_pairts.sum(axis=0)

        press_map = press_map[-1::-1]
        press_map = 255 * colormap((press_map - MIN_PRESSURE) / MAX_PRESSURE)
        press_map = press_map.astype(np.uint8)
        pil_img = Image.fromarray(press_map)
        pil_img.save(write_path, quolity=100)

        if img_num == 100:
            print("\r"+write_path, end="")
            img_num = 0
