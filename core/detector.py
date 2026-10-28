from ultralytics import YOLO


class PersonDetector:
    """
    Pure detection logic — no PyQt, no cv2.imshow, no UI concerns at all.
    This class could be unit-tested, swapped for a different model, or
    reused in a CLI script without touching a single line of GUI code.
    """

    PERSON_CLASS_ID = 0  # COCO dataset: class 0 is "person"

    def __init__(self, model_path: str = "yolo11n.pt"):
        self.model = YOLO(model_path)

    def detect(self, frame):
        """
        Runs detection on a single frame.
        Returns (person_count, annotated_frame).
        """
        results = self.model(frame, classes=[self.PERSON_CLASS_ID], verbose=False)
        person_count = len(results[0].boxes)
        annotated_frame = results[0].plot()
        return person_count, annotated_frame

    def detect_boxes(self, frame):
        """
        Runs detection on a single frame and returns only the person
        bounding boxes, as a list of (x1, y1, x2, y2) tuples in pixels.
        Used by PersonTracker, which draws its own overlay.
        """
        results = self.model(frame, classes=[self.PERSON_CLASS_ID], verbose=False)
        return [
            tuple(float(v) for v in box)
            for box in results[0].boxes.xyxy.cpu().numpy()
        ]