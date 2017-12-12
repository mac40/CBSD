# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import pandas as pd
import argparse
import imutils
import time
import dlib
import cv2

#fetch arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
                help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="",
                help="path to input video file")
args = vars(ap.parse_args())

#facial detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# start the video stream thread
vs = FileVideoStream(args["video"]).start()
fileStream = True
time.sleep(1.0)

try:
    SHOWCASE_DATA = pd.read_csv("showcase_data/{}.tag".format(args["video"][6:-4]), sep='\t',
                                header=0, names=['frame', 'Manual_Detection', 'Auto_Detection'], index_col="frame")
except FileNotFoundError:
    SHOWCASE_DATA = pd.read_csv("showcase_data/{}.tag".format(args["video"][7:-4]), sep='\t',
                                header=0, names=['frame', 'Manual_Detection', 'Auto_Detection'], index_col="frame")
FRAME = 0

while True:
    if fileStream and not vs.more():
        break

    frame = vs.read()
    frame = imutils.resize(frame, width=900)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for rect in rects:

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        cv2.putText(frame, "Manual Detection: {}".format(SHOWCASE_DATA.Manual_Detection[FRAME]), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Frame: {}".format(FRAME), (10, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Auto Detection: {}".format(SHOWCASE_DATA.Auto_Detection[FRAME]), (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
            break
    FRAME += 1

cv2.destroyAllWindows()
vs.stop()
