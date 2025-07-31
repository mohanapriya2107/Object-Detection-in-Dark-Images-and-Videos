# Object-Detection-in-Dark-Images-and-Videos

Files usage

main.py -> Image cropping

main2.py, main3.py -> Video Cropping

These are for object cropping from images and videos


app.py - > Web app for images, videos, real time detection

video_detection.py -> Bounding boxes for objects in video for app.py

video.py -> Bounding boxes for objects in videos

new.py -> Same as above but does not open a new inner window.

yolo_segmentation.py ->wraps a YOLOv8 segmentation model to detect and segment objects in an image.

model_creation.py -> New model creation

human_detection.py -> Same as new.py but opens in the new frame

yolov8_supervision.py -> Does not bounds boxes

Ex:- 0: 384x640 15 cars, 2 trucks, 1 traffic light, 4396.0ms
Speed: 8.1ms preprocess, 4396.0ms inference, 7.7ms postprocess per image at shape (1, 3, 384, 640)

webapp.py -> This is not a web app but it shows the features for drag and drop for image and video both does not work simulataneously but it only works
1. For images when we remove line no 54
2. For videos when we remove line no 33
