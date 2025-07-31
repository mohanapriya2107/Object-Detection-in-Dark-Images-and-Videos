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

# Load the image
img = cv2.imread('images/basket.jpg')
if img is None:
    raise FileNotFoundError("The image file was not found.")

img = cv2.resize(img, None, fx=0.5, fy=0.5)

# Perform detection
results = model(img)[0]  # Get the first result

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

# Process each detected object and visualize the detections
found = False
for box, cls, seg in zip(results.boxes.xyxy, results.boxes.cls, results.masks.data):
    cls_id = int(cls)
    bbox = box.cpu().numpy().astype(int)
    label = class_names[cls_id]

    # Draw bounding box and label
    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
    cv2.putText(img, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Check if this is the desired object
    if cls_id == desired_class_id:
        mask = seg.cpu().numpy().astype(np.uint8) * 255
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        object_on_white = place_on_white_background(img, mask, bbox)
        found = True

        # Save the cropped object on white background
        output_path = os.path.join(output_dir, f"{desired_object_name}.jpg")
        cv2.imwrite(output_path, object_on_white)

# Display the image with all detected objects
cv2.imshow("Detected Objects", img)
cv2.waitKey(0)

# If the desired object is found, display the object on white background
if found:
    cv2.imshow("Object on White Background", object_on_white)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print(f"No object with name '{desired_object_name}' found in the image.")
