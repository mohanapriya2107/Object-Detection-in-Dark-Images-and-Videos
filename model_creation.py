import cv2
import cvzone
import numpy as np
cap=cv2.VideoCapture(r"C:\Users\hp\Downloads\human_detection\static\uploads\Road_traffic_video2.mp4")
cap.set(3,540)
cap.set(4,480)

classNames=[]
classFile='coco.names'
with open(classFile,'rt') as f:
    classNames=f.read().split('\n')
configPath="ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath="frozen_inference_graph.pb"

net=cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,328)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)



while True:
    sucess,img=cap.read()
    classIds,confs,bbox =net.detect(img,confThreshold=0.6,nmsThreshold=0.2)
    try:
       for classId,conf,box in zip(classIds.flatten(),confs.flatten(),bbox):
           if classId==1:
               cvzone.cornerRect(img, box, rt=5)
               cv2.putText(img,f'{classNames[classId-1].upper()} {conf}',(box[0],box[1]),
                       cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0) ,2)

    except:
        pass


    cv2.imshow("Image",img)
    cv2.waitKey(0)



