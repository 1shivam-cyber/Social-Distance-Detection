#importing all the necessary libraries...................................................................................................
from Detection_folder import social_distancing_config as config
from Detection_folder.detection import detect_people
from scipy.spatial import distance as dist
import numpy as np
import argparse
import imutils
import cv2
import os
#........................................................................................................................................




# construct the argument parse and parse the arguments...................................................................................
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="",
	help="path to (optional) input video file")
ap.add_argument("-o", "--output", type=str, default="",
	help="path to (optional) output video file")
ap.add_argument("-d", "--display", type=int, default=1,
	help="whether or not output frame should be displayed")
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join(["yolo-coco", "coco.names"])
classes = open(labelsPath).read().strip().split("\n")

# derive the paths to the YOLO weights and model configuration...........................................................................
weightsPath = os.path.sep.join(["yolo-coco", "yolov3.weights"])
configPath = os.path.sep.join(["yolo-coco", "yolov3.cfg"])
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]




# initialize the video stream and pointer to output video file..........................................................................
print("[INFO] accessing video stream...")
vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
writer = None





# loop over the frames from the video stream............................................................................................
while True:
	# read the next frame from the file
	(flag, frame) = vs.read()
	if not flag:
		break


		

	# resize the frame and then detect people (and only people) in it
	frame = imutils.resize(frame, width=700)
	results = detect_people(frame, net, ln,personIdx=classes.index("person"))
	violate = set()




	if len(results) >= 2:
		# extract all centroids from the results and compute the Euclidean distances between all pairs of the centroids.............
		centroids = np.array([r[2] for r in results])
		D = dist.cdist(centroids, centroids, metric="euclidean")

		for i in range(0, D.shape[0]):
			for j in range(i + 1, D.shape[1]):
				if D[i, j] < 100:
					violate.add(i)
					violate.add(j)






# loop over the results and making binding boxes round the violate and non-violate pairs..............................................
	for (i, (prob, bbox, centroid)) in enumerate(results):
		(startX, startY, endX, endY) = bbox
		(cX, cY) = centroid
		color = (0, 255, 0)
		if i in violate:
			color = (0, 0, 255)

		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
		cv2.circle(frame, (cX, cY), 5, color, 1)




	# draw the total number of social distancing violations on the output frame........................................................
	text = "Social Distancing Violations: {}".format(len(violate))
	cv2.putText(frame, text, (10, frame.shape[0] - 25),
		cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)



	# check to see if the output frame should be displayed to our......................................................................
	# screen
	if args["display"] > 0:
		# show the output frame
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

	# if an output video file path has been supplied and the video writer has not been initialized, do so now...........................
	if args["output"] != "" and writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 25,
			(frame.shape[1], frame.shape[0]), True)


	# if the video writer is not None, write the frame to the output....................................................................
	# video file
	if writer is not None:
		writer.write(frame)
