import cv2
import argparse
import pygame


from ultralytics import YOLO
import supervision as sv
import numpy as np

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="yolov8_human_detection")
    parser.add_argument("--webcam-resolution",
                        default=[1280,720],
                        nargs=2,type=int)
    args=parser.parse_args()
    return args
def main():
    args=parse_arguments()
    frame_width,frame_height=args.webcam_resolution

    cap=cv2.VideoCapture(r"C:\Users\hp\Downloads\human_detection\static\uploads\Road_traffic_video2.mp4")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,frame_height)

    model=YOLO("yolov8l.pt")

    box_annotator=sv.BoxAnnotator(
)
    while True:
        ret,frame=cap.read()
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
        detections=detections[detections.class_id==0]
        labels = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for class_id, confidence in zip(detections.class_id, detections.confidence)
        ]


        frame=box_annotator.annotate(scene=frame,
                                     detections=detections)
        '''if detections.class_id.all()==0:
            pygame.mixer.init()
            sound_file = r"C:Users\hp\Downloads\human_detection\security-alarm-80493.mp3"
            pygame.mixer.music.load(sound_file)
            pygame.mixer.music.play()
            pygame.time.delay(5000)'''
        cv2.imshow("frame",frame)
        cv2.waitKey(1)


if  __name__ == "__main__":
    main()


