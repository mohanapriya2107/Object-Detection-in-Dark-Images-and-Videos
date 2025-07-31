import cv2
import numpy as np
import os
from ultralytics import YOLO

# List of class names corresponding to the YOLO model's class IDs
class_names = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]

# Function to crop a frame based on the bounding box
def crop_frame(frame, bbox):
    x1, y1, x2, y2 = bbox
    cropped_frame = frame[y1:y2, x1:x2]
    return cropped_frame

# Load the YOLOv8 model
model = YOLO('yolov8m-seg.pt')

# Load the video
video_path = r"C:\Users\hp\Downloads\human_detection\static\uploads\3555555-hd_1280_720_30fps.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise FileNotFoundError("The video file was not found.")

# Create directory to save cropped images
output_dir = "cropped_images"
os.makedirs(output_dir, exist_ok=True)

# Get the object name to crop from the user
desired_object_name = input("Enter the object name to crop: ").strip().lower()

# Find the class ID corresponding to the object name
if desired_object_name in class_names:
    desired_class_id = class_names.index(desired_object_name)
else:
    raise ValueError(f"Object name '{desired_object_name}' not found in class names.")

frame_count = 0
cropped_image_count = 0  # To keep track of all cropped objects

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 360))  # Resize the frame for better processing speed

    # Perform detection on the current frame
    results = model(frame)[0]

    for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
        cls_id = int(cls)
        bbox = box.cpu().numpy().astype(int)
        label = class_names[cls_id]

        # Draw bounding box and label
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Check if this is the desired object
        if cls_id == desired_class_id:
            cropped_frame = crop_frame(frame, bbox)

            # Save each cropped image with a unique filename
            output_path = os.path.join(output_dir, f"{desired_object_name}_{cropped_image_count}.jpg")
            cv2.imwrite(output_path, cropped_frame)
            cropped_image_count += 1

    # Display the frame with all detected objects
    cv2.imshow("Detected Objects", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()

print(f"Saved {cropped_image_count} cropped images of '{desired_object_name}' to {output_dir}")
