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

# Function to place the object on a white background
def place_on_white_background(img, mask, bbox):
    x1, y1, x2, y2 = bbox
    cropped_object = img[y1:y2, x1:x2]
    mask = mask[y1:y2, x1:x2]

    # Create a white background image of the same size as the cropped object
    white_background = np.ones_like(cropped_object, dtype=np.uint8) * 255

    # Use the mask to place the object on the white background
    white_background[mask > 0] = cropped_object[mask > 0]
    return white_background

# Load the YOLOv8 model
model = YOLO('yolov8m-seg.pt')

# Load the video
video_path = r"C:\Users\hp\Downloads\human_detection\static\uploads\stock-footage-night-traffic-jam-timelapse-bangkok-thailand-rush-hour-in-downtown-motion-of-car-tail-lights.webm"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise FileNotFoundError("The video file was not found or could not be opened.")

# Get the object name to crop from the user
desired_object_name = input("Enter the object name to crop: ").strip().lower()

# Find the class ID corresponding to the object name
if desired_object_name in class_names:
    desired_class_id = class_names.index(desired_object_name)
else:
    raise ValueError(f"Object name '{desired_object_name}' not found in class names.")

# Create directory to save cropped images
output_dir = "cropped_images"
os.makedirs(output_dir, exist_ok=True)

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection
    results = model(frame)[0]  # Get the first result

    found = False
    for box, cls, seg in zip(results.boxes.xyxy, results.boxes.cls, results.masks.data):
        cls_id = int(cls)
        if cls_id != desired_class_id:
            continue

        bbox = box.cpu().numpy().astype(int)
        mask = seg.cpu().numpy().astype(np.uint8) * 255
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

        object_on_white = place_on_white_background(frame, mask, bbox)
        found = True

        # Save the cropped object on white background
        output_path = os.path.join(output_dir, f"{desired_object_name}_frame_{frame_count}.jpg")
        cv2.imwrite(output_path, object_on_white)

        # Draw bounding box and label
        label = class_names[cls_id]
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Detected Objects", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
