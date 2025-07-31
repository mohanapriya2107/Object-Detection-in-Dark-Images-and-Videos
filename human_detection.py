import cv2


from ultralytics import YOLO
import supervision as sv
import numpy as np

def main():

    cap=cv2.VideoCapture(r"C:\Users\hp\Downloads\human_detection\static\uploads\stock-footage-night-traffic-jam-timelapse-bangkok-thailand-rush-hour-in-downtown-motion-of-car-tail-lights.webm")

    model=YOLO("yolov8l.pt")

    box_annotator=sv.BoxAnnotator()
    while True:
        ret,frame=cap.read()
        #frame =cv2.resize(frame,(1020,500))
        result=model.predict(frame,agnostic_nms=True)[0]
        # Convert YOLOv8 results to supervision Detections manually
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        
        detections = sv.Detections(
            xyxy=boxes,
            confidence=confidences,
            class_id=class_ids
        )
        #detections=detections[detections.class_id==0]
        labels = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for class_id, confidence in zip(detections.class_id, detections.confidence)
        ]


        frame=box_annotator.annotate(scene=frame,
                                     detections=detections)

        cv2.imshow("frame",frame)
        cv2.waitKey(10)


if  __name__ == "__main__":
    main()


