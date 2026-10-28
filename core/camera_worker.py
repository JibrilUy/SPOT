import time

import cv2
from PyQt6.QtCore import QThread, pyqtSignal

from core.detector import PersonDetector


class CameraWorker(QThread):
    """
    Runs camera capture + YOLO inference on a background thread so the GUI's
    event loop never freezes waiting for a detection result. This is the
    ONLY place that touches both cv2 and PyQt — it's the bridge, not the
    logic itself. The UI layer only ever connects to the signals below;
    it never imports PersonDetector or calls .detect() directly.

    `camera_index` can be a webcam index (int) or a path to a video file
    (str). For video files:
      - loop=True rewinds to the first frame when the video ends.
      - playback is paced to the file's own FPS, instead of running as
        fast as inference allows.
    """

    frame_ready = pyqtSignal(object)  # annotated frame, numpy array (BGR)
    count_updated = pyqtSignal(int)   # current person count
    error_occurred = pyqtSignal(str)  # human-readable error message

    def __init__(
        self,
        camera_index=0,
        model_path: str = "yolo11n.pt",
        loop: bool = False,
        detector=None,
    ):
        super().__init__()
        self._camera_index = camera_index
        # Anything with a detect(frame) -> (count, annotated_frame) method
        # works here. TrackingWorker injects a PersonTracker so it can reuse
        # this whole capture/loop/pacing loop instead of copying it.
        self._detector = detector if detector is not None else PersonDetector(model_path)
        self._loop = loop
        self._running = False

        # Count smoothing
        self._last_count = 0
        self._missed_frames = 0
        self._max_missed_frames = 30

    def run(self):
        is_file = not isinstance(self._camera_index, int)

        if is_file:
            cap = cv2.VideoCapture(self._camera_index)
        else:
            # DirectShow backend avoids a common Windows issue where
            # OpenCV's default backend selection fails to enumerate
            # webcams correctly, producing "Camera index out of range"
            # or an FFMPEG/libavdevice warning even when a camera IS
            # physically present. File paths don't need this.
            cap = cv2.VideoCapture(self._camera_index, cv2.CAP_DSHOW)

        if not cap.isOpened():
            self.error_occurred.emit(
                f"Could not open video source: {self._camera_index!r}. "
                f"Check the path/camera index is correct and the file exists."
            )
            return

        # Video files only: how long each frame SHOULD take on screen.
        frame_interval = 0.0
        if is_file:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps and fps > 0:
                frame_interval = 1.0 / fps

        self._running = True
        just_rewound = False

        while self._running and cap.isOpened():
            loop_start = time.monotonic()

            success, frame = cap.read()
            if not success:
                if self._loop and is_file and not just_rewound:
                    # End of file: jump back to frame 0 and keep going.
                    # `just_rewound` stops an unreadable/empty file from
                    # rewinding forever — if the very next read also
                    # fails, we fall through to break.
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    if hasattr(self._detector, "reset_tracking"):
                        self._detector.reset_tracking()
                    just_rewound = True
                    continue
                break
            just_rewound = False

            count, annotated = self._detector.detect(frame)

            # Prevent temporary detection losses from making
            # the displayed count instantly drop.
            if count == 0 and self._last_count > 0:
                self._missed_frames += 10

                if self._missed_frames <= self._max_missed_frames:
                    count = self._last_count
                else:
                    self._last_count = count
                    self._missed_frames = 0
            else:
                self._last_count = count
                self._missed_frames = 0

            self.frame_ready.emit(annotated)
            self.count_updated.emit(count)

            if frame_interval:
                remaining = frame_interval - (time.monotonic() - loop_start)
                if remaining > 0:
                    self.msleep(int(remaining * 1000))

        cap.release()

    def stop(self):
        """Call this before closing the app, or the thread leaks on exit."""
        self._running = False
        self.wait()