from ultralytics import YOLO
import numpy as np

class YOLOSegmentation:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, img):
        # Get image shape
        height, width, _ = img.shape

        results = self.model.predict(source=img.copy(), save=False, save_txt=False)
        result = results[0]
        segmentation_contours_idx = []
        if result.masks:  # Check if there are masks
            for mask in result.masks.xy:  # Use xy attribute for segments
                # Contours
                mask[:, 0] *= width
                mask[:, 1] *= height
                segment = np.array(mask, dtype=np.int32)
                segmentation_contours_idx.append(segment)

        bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
        # Get class ids
        class_ids = np.array(result.boxes.cls.cpu(), dtype="int")
        # Get scores
        scores = np.array(result.boxes.conf.cpu(), dtype="float").round(2)
        return bboxes, class_ids, segmentation_contours_idx, scores
