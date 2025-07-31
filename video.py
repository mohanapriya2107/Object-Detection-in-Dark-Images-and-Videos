import cv2
import cvzone
from PIL import Image
#opencv and python imaging library are imported

thres = 0.55
nmsThres = 0.2
#threshold and non maximum suppression threshold values.the image that  has less confidence threshold value images are ignored.
# './Road_traffic_video2.mp4'

cap = cv2.VideoCapture(r"C:\Users\hp\Downloads\human_detection\static\uploads\Road_traffic_video2.mp4")
#variable 'cap' to hold the link of a video

cap.set(3, 640)
cap.set(4, 480)
#set the width (3) and height (4) of the frames to 640x480.


classNames = []
classFile = './coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().split('\n')
#This reads the class names for the objects that the model can detect from the file "./coco.names" and stores them in the list classNames.

configPath = './ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = "./frozen_inference_graph.pb"
#specify the paths to the configuration file (configPath) and
# the pre-trained model weights file (weightsPath) for the object detection model.

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)
#This initializes the object detection model using the specified weights and configuration.
#  It also sets various parameters such as input size, input scale, input mean, and input channel swapping.



while True:
    success, img = cap.read()
    #frames are read from the video file using the cap.read() method.
    # print(type(img))
    #img = cv2.resize(img, dsize=(128, 128))
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nmsThres)
    #This line performs object detection on the current frame (img) using the specified confidence threshold (thres) and non-maximum suppression threshold (nmsThres).
    # It returns the detected class IDs, confidence scores, and bounding boxes.

    for classId, conf, box in zip(classIds.flatten(), confs.flatten(), bbox):
        cv2.rectangle(img,box,color=(0,255,0),thickness=2)
        # cvzone.cornerRect(img, box)
        cv2.putText(img, classNames[classId-1].upper(),(box[0]+10,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        # cv2.imwrite('detected.jpg', img).resize(1000,1000)
        #loop iterates over the detected objects, draws rectangles around them, and
        # annotates the image with the class name and confidence score.

    cv2.imshow("Image", img)
    cv2.waitKey(1000)
    #These lines display the annotated image in a window and wait for a key press.
    #  The loop continues until a key is pressed, and the window is closed