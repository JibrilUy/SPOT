import threading

import cv2
import numpy as np

from core.detector import PersonDetector


class PersonTracker:
    """
    Follows ONE selected person across frames and records the path taken.
    Pure logic: no PyQt. detect(frame) returns (count, annotated_frame),
    the same shape as PersonDetector.detect, so a worker can use either.

    How it works:
      - Every frame, YOLO finds all people. They are drawn as thin boxes.
      - select(...) picks the box under a click as the target.
      - Each following frame, the target's next position is predicted from
        its recent velocity, and the detection closest to that prediction
        (within a gate) becomes the new target box.
      - The bottom-center of the box (roughly where the feet touch the
        floor) is added to the path, and the path is drawn on the frame.
      - If no detection matches, the box coasts along its last velocity.
        After MAX_MISSED_FRAMES it gives up, but the path stays visible
        until clear is requested.

    select/clear are meant to be called from the GUI thread; they only set
    flags, and all tracking state is changed on the worker thread.

    Known limit: matching uses motion only. If two people cross very close
    together the target can jump to the other person. Appearance matching
    (re-identification) is the natural next upgrade.
    """

    BOX_COLOR = (0, 200, 0)        # BGR: other people
    TARGET_COLOR = (0, 140, 255)   # BGR: tracked person and path
    LOST_COLOR = (0, 0, 255)       # BGR: target currently not found

    MAX_MISSED_FRAMES = 60         # about 2 s at 30 fps
    MIN_STEP_PX = 3                # ignore jitter smaller than this
    MAX_PATH_POINTS = 2000

    def __init__(self, model_path="yolo11n.pt", on_event=None, detector=None):
        self._detector = detector if detector is not None else PersonDetector(model_path)
        self.on_event = on_event  # called with a short event name

        self._lock = threading.Lock()
        self._pending_click = None
        self._clear_requested = False

        self._reset()

    # -----------------------------------------------------------------
    # Requests from the GUI thread
    # -----------------------------------------------------------------
    def request_select(self, nx, ny):
        """nx, ny: click position as fractions (0..1) of the video size."""
        with self._lock:
            self._pending_click = (nx, ny)

    def request_clear(self):
        with self._lock:
            self._clear_requested = True

    # -----------------------------------------------------------------
    # Per-frame work (worker thread)
    # -----------------------------------------------------------------
    def detect(self, frame):
        h, w = frame.shape[:2]
        boxes = self._detector.detect_boxes(frame)

        self._apply_requests(boxes, w, h)
        self._update_target(boxes)
        self._draw(frame, boxes)

        return len(boxes), frame

    def _apply_requests(self, boxes, w, h):
        with self._lock:
            click = self._pending_click
            clear = self._clear_requested
            self._pending_click = None
            self._clear_requested = False

        if clear:
            self._reset()
            self._emit("cleared")

        if click is not None:
            chosen = self._box_at(boxes, click[0] * w, click[1] * h)
            if chosen is None:
                self._emit("no_person")
            else:
                self._reset()
                self._target_box = np.array(chosen, dtype=float)
                self._path.append(self._foot_point(self._target_box))
                self._emit("selected")

    def _update_target(self, boxes):
        if self._target_box is None:
            return

        # Where we expect the person to be, given their recent velocity.
        predicted = self._target_box.copy()
        predicted[[0, 2]] += self._velocity[0]
        predicted[[1, 3]] += self._velocity[1]
        predicted_center = self._center(predicted)

        width = self._target_box[2] - self._target_box[0]
        height = self._target_box[3] - self._target_box[1]
        # Search radius grows the longer the person has been missing.
        gate = 0.75 * max(width, height) * (1 + 0.25 * self._missed)

        best, best_dist = None, None
        for box in boxes:
            dist = float(np.hypot(*(self._center(box) - predicted_center)))
            if dist <= gate and (best_dist is None or dist < best_dist):
                best, best_dist = box, dist

        if best is not None:
            new_box = np.array(best, dtype=float)
            moved = self._center(new_box) - self._center(self._target_box)
            self._velocity = 0.6 * self._velocity + 0.4 * moved
            self._target_box = new_box

            if self._missed > 0:
                self._emit("reacquired")
            self._missed = 0

            self._append_path(self._foot_point(new_box))
        else:
            self._missed += 1
            self._target_box = predicted     # coast along the last velocity
            self._velocity = self._velocity * 0.9

            if self._missed == 1:
                self._emit("lost")
            if self._missed > self.MAX_MISSED_FRAMES:
                self._target_box = None      # give up; path stays on screen
                self._emit("ended")

    def _draw(self, frame, boxes):
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(
                frame, (int(x1), int(y1)), (int(x2), int(y2)), self.BOX_COLOR, 1
            )

        if len(self._path) > 1:
            points = np.array(self._path, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(frame, [points], False, self.TARGET_COLOR, 3, cv2.LINE_AA)

        if self._path:
            cv2.circle(frame, self._path[0], 6, (255, 255, 255), -1)      # start
            cv2.circle(frame, self._path[-1], 6, self.TARGET_COLOR, -1)   # now

        if self._target_box is not None:
            x1, y1, x2, y2 = (int(v) for v in self._target_box)
            lost = self._missed > 0
            color = self.LOST_COLOR if lost else self.TARGET_COLOR
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(
                frame, "LOST" if lost else "TRACKING", (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA,
            )

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------
    def _reset(self):
        self._target_box = None
        self._velocity = np.zeros(2)
        self._missed = 0
        self._path = []

    def _emit(self, name):
        if self.on_event is not None:
            self.on_event(name)

    def _append_path(self, point):
        if self._path:
            last = self._path[-1]
            if np.hypot(point[0] - last[0], point[1] - last[1]) < self.MIN_STEP_PX:
                return
        self._path.append(point)
        if len(self._path) > self.MAX_PATH_POINTS:
            self._path.pop(0)

    @staticmethod
    def _center(box):
        return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])

    @staticmethod
    def _foot_point(box):
        return (int((box[0] + box[2]) / 2), int(box[3]))

    @staticmethod
    def _box_at(boxes, px, py):
        """The smallest box containing the point, or None."""
        hits = [b for b in boxes if b[0] <= px <= b[2] and b[1] <= py <= b[3]]
        if not hits:
            return None
        return min(hits, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))