# Social-Distance-Detection
Social distancing detection using deep learning to evaluate the distance between people to mitigate the impact of this coronavirus pandemic. The detection tool was developed to alert people to maintain a safe distance with each other by evaluating a video feed. The video frame from the camera was used as input, and the open-source object detection pre-trained model based on the YOLOv3 algorithm was employed for pedestrian detection. Later, the video frame was transformed into top-down view for distance measurement from the 2D plane. The distance between people can be estimated and any noncompliant pair of people in the display will be indicated with a red boundary box. The proposed method was validated on a pre-recorded video of University students walking on the street. The result shows that the proposed method is able to determine the social distancing measures between multiple people in the video. The developed technique can be further developed as a detection tool in realtime application.


Github usually doesn't support files larger than 25 Mb.You can find the yolo weights in https://pjreddie.com/darknet/yolo/
Download it & move to yolo-coco folder
