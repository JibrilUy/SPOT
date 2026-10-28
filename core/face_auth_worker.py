import cv2
import threading
from PyQt6.QtCore import QThread, pyqtSignal

from core.face_recognizer import FaceRecognizer


class FaceAuthWorker(QThread):
    """
    Background thread that continuously reads the webcam and runs face
    recognition each frame — same bridge pattern as CameraWorker, just for
    face auth instead of person counting. The UI only ever reacts to the
    signals below; it never imports FaceRecognizer or cv2 directly.
    """

    frame_ready = pyqtSignal(object)   # annotated frame (numpy array, BGR)
    match_updated = pyqtSignal(bool)   # True if the enrolled face is currently visible
    error_occurred = pyqtSignal(str)

    def __init__(self, camera_index=0, model_path="assets/faces/trainer.yml"):
        super().__init__()
        self._camera_index = camera_index
        self._running = False
        self._label_lock = threading.Lock()
        self._expected_label = None

        try:
            self._recognizer = FaceRecognizer(model_path=model_path)
            self._init_error = None
        except FileNotFoundError as e:
            self._recognizer = None
            self._init_error = str(e)

    def reload_model(self):
        """Reload trainer.yml after super-admin enrollment, before restarting camera."""
        if self.isRunning():
            self.stop()
        try:
            self._recognizer = FaceRecognizer()
            self._init_error = None
        except (FileNotFoundError, RuntimeError) as exc:
            self._recognizer = None
            self._init_error = str(exc)

    def set_expected_label(self, face_label):
        """Select the ID to authenticate, or None to disallow matches."""
        with self._label_lock:
            self._expected_label = face_label

    def run(self):
        if self._recognizer is None:
            self.error_occurred.emit(self._init_error)
            return

        if isinstance(self._camera_index, int):
            cap = cv2.VideoCapture(self._camera_index, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(self._camera_index)

        if not cap.isOpened():
            self.error_occurred.emit(f"Could not open camera: {self._camera_index!r}")
            return

        self._running = True
        while self._running and cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            with self._label_lock:
                expected_label = self._expected_label
            annotated, is_match, _confidence = self._recognizer.process(
                frame, expected_label=expected_label
            )

            self.frame_ready.emit(annotated)
            self.match_updated.emit(is_match)

        cap.release()

    def stop(self):
        self._running = False
        self.wait()