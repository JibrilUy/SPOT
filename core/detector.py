from ultralytics import YOLO


class PersonDetector:
    """
    Person detection + tracking.

    Uses YOLO's built-in ByteTrack tracker to maintain IDs between
    frames so temporary detection losses don't immediately cause
    people to disappear from the count.
    """

    PERSON_CLASS_ID = 0  # COCO: person

    def __init__(self, model_path: str = "yolo11m.pt"):
        self.model = YOLO(model_path)

    def detect(self, frame):
        """
        Returns:
            (person_count, annotated_frame)
        """

        results = self.model.track(
            frame,
            classes=[self.PERSON_CLASS_ID],
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.10,
            iou=0.45,
            verbose=False
        )

        result = results[0]

        active_ids = set()

        if result.boxes is not None:
            for box in result.boxes:
                if box.id is not None:
                    active_ids.add(int(box.id.item()))

        person_count = len(active_ids)
        annotated_frame = result.plot()

        return person_count, annotated_frame