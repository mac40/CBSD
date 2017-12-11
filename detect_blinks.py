# USAGE
# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat --video blink_detection_demo.mp4
# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat

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

def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)

	# return the eye aspect ratio
	return ear
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="",
	help="path to input video file")
args = vars(ap.parse_args())

# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
EYE_AR_THRESH = 0.30
EYE_AR_CONSEC_FRAMES = 2

# initialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream thread
print("[INFO] starting video stream thread...")
vs = FileVideoStream(args["video"]).start()
fileStream = True
# vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
# fileStream = False
time.sleep(1.0)

FRAME=0

ear_list=list()

# loop over frames from the video stream
while True:
	# if this is a file video stream, then we need to check if
	# there any more frames left in the buffer to process
	if fileStream and not vs.more():
		break
	# grab the frame from the threaded video file stream, resize
	# it, and convert it to grayscale
	# channels)
	frame = vs.read()
	frame = imutils.resize(frame, width=900)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale frame
	rects = detector(gray, 0)

	# loop over the face detections
	for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# extract the left and right eye coordinates, then use the
		# coordinates to compute the eye aspect ratio for both eyes
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)

		# average the eye aspect ratio together for both eyes
		ear = (leftEAR + rightEAR) / 2.0
		ear_list.append(ear)
		# compute the convex hull for the left and right eye, then
		# visualize each of the eyes
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)

		# check to see if the eye aspect ratio is below the blink
		# threshold, and if so, increment the blink frame counter
		if ear < EYE_AR_THRESH:
			COUNTER += 1

		# otherwise, the eye aspect ratio is not below the blink
		# threshold
		else:
			# if the eyes were closed for a sufficient number of
			# then increment the total number of blinks
			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				TOTAL += 1
			# reset the eye frame counter
			COUNTER = 0

		# draw the total number of blinks on the frame along with
		# the computed eye aspect ratio for the frame
		cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame, "Frame: {}".format(FRAME), (10, 300),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		for (x, y) in shape:
			cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

	# show the frame
	# cv2.imshow("Frame", frame)
	print(FRAME)
	key = cv2.waitKey(1) & 0xFF

	FRAME += 1
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

#moving avareage function
def moving_av(mylist, N):
	cumsum, moving_aves = [0], []
	for i, x in enumerate(mylist, 1):
		cumsum.append(cumsum[i-1] + x)
		if i>=N:
			moving_ave = (cumsum[i] - cumsum[i-N])/N
			#can do stuff with moving_ave here
			moving_aves.append(moving_ave)
	return moving_aves

try:
	users_final = pd.read_csv("tag/{}.tag".format(args["video"][6:-4]), sep='\t', header=None,
	                          names=['frame', 'tag'], index_col="frame")
except FileNotFoundError:
	users_final = pd.read_csv("tag/{}.tag".format(args["video"][7:-4]), sep='\t', header=None,
	                          names=['frame', 'tag'], index_col="frame")

mov_ear_3=moving_av(ear_list,3)
mov_ear_5=moving_av(ear_list,5)
mov_ear_7=moving_av(ear_list,7)

ear_list = pd.Series(ear_list, index=range(0, len(ear_list)))
mov_ear_3=pd.Series(mov_ear_3, index=range(2, len(mov_ear_3)+2))
mov_ear_5=pd.Series(mov_ear_5, index=range(3, len(mov_ear_5)+3))
mov_ear_7=pd.Series(mov_ear_7, index=range(4, len(mov_ear_7)+4))

ear_list = pd.DataFrame(ear_list)
ear_list["tag"] = users_final.tag
ear_list["mov_ear_3"] = mov_ear_3
ear_list["mov_ear_5"] = mov_ear_5
ear_list["mov_ear_7"] = mov_ear_7
ear_list.columns = ["ear", "tag", "mov_ear_3","mov_ear_5","mov_ear_7"]
ear_list = ear_list.fillna(0)
#mask = ear_list.tag == 0
#ear_list.tag = ear_list.tag.where(mask, 1)
ear_list.index.name="frame"
try:
	ear_list.to_csv("raw_data/{}.csv".format(
			args["video"][6:-4]), index=True, header=True)
except FileNotFoundError:
	ear_list.to_csv("raw_data/{}.csv".format(
            args["video"][7:-4]), index=True, header=True)

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
