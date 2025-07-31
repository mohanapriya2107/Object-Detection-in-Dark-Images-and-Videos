from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # load an official model

results = model.predict(r"C:\Users\hp\Downloads\human_detection\static\uploads\pngtree-traffic-light-night-scene-with-human-silhouette-travel-human-countryside-photo-image_30277144.jpg")
result = results[0]
for box in result.boxes:
  class_id = result.names[box.cls[0].item()]
  cords = box.xyxy[0].tolist()
  cords = [round(x) for x in cords]
  conf = round(box.conf[0].item(), 2)
  print("Object type:", class_id)
  print("Coordinates:", cords)
  print("Probability:", conf)
  print("---")
from PIL import Image
img=Image.fromarray(result.plot()[:,:,::-1])
import matplotlib.pyplot as plt
plt.imshow(img)
plt.show()




