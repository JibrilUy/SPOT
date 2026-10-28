from PyQt6.QtCore import pyqtSignal

from core.camera_worker import CameraWorker
from core.person_tracker import PersonTracker


class TrackingWorker(CameraWorker):
    """
    A CameraWorker that also tracks one selected person.

    It reuses CameraWorker's whole capture loop (webcam or video file,
    looping, FPS pacing, frame_ready / count_updated / error_occurred) and
    only swaps the detector for a PersonTracker. The frames it emits
    already have the boxes, the tracked person and the path drawn on them.

    Extra signal:
      target_event(str): "selected", "no_person", "lost", "reacquired",
                         "ended" or "cleared"
    """

    target_event = pyqtSignal(str)

    def __init__(self, camera_index=0, model_path="yolo11n.pt", loop=False):
        tracker = PersonTracker(model_path)
        super().__init__(camera_index, model_path, loop, detector=tracker)

        self._tracker = tracker
        # Emitted from the worker thread; Qt delivers it to the GUI thread.
        tracker.on_event = self.target_event.emit

    def select_at(self, nx, ny):
        """Pick the person under a click. nx, ny are fractions (0..1) of the video."""
        self._tracker.request_select(nx, ny)

    def clear_target(self):
        """Stop tracking and erase the path."""
        self._tracker.request_clear()