import datetime
import os

import cv2 as cv

write_path = "assets/pressure.mp4"
img_dir = os.path.join("assets", "pressure_images")

img_path = os.path.join(
    img_dir, "2016/01/01/0000.jpg"
)
img_shape = cv.imread(img_path).shape

video = cv.VideoWriter(
    write_path,
    cv.VideoWriter_fourcc('m', 'p', '4', 'v'),
    20.0,
    (img_shape[1], img_shape[0])
)
if not video.isOpened():
    raise ValueError("can't be open.")

time_delta = datetime.timedelta(hours=4)
current_time = datetime.datetime(2016, 1, 1, 0, 0, 0)
while current_time.year < 2020:
    img_path = os.path.join(
        img_dir, current_time.strftime("%Y/%m/%d/%H%M")+".jpg"
    )
    img = cv.imread(img_path)
    if img is None:
        raise ValueError("can't image read.")

    cv.putText(
        img,
        text=current_time.strftime("%Y%m%d"),
        org=(5, 50),
        fontFace=cv.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=(0, 0, 0),
        thickness=1,
        lineType=cv.LINE_4
    )

    # add
    video.write(img)

    current_time += time_delta

video.release()
print("done")
