# controllers/floor_camera_controller.py

from PyQt6 import QtWidgets

from core.camera_worker import CameraWorker
from core.qt_image_utils import cv_frame_to_pixmap


class FloorCameraController:
    """
    Owns everything about ONE floor's camera page:
      - puts the live/video feed inside the page's `frame` widget
      - sets current_camera_indicator to this feed's camera number
      - writes the detected person count to camera_count_indicator and
        floor_count_indicator

    One instance per floor. Adding another floor later is one more
    instantiation in main.py, not more methods on SPOTApp.

    NOT started automatically — call .start() when the feed should begin
    (main.py does this once login succeeds).

    `source` is a webcam index (int) or a video file path (str).
    The widget names are parameters so a floor whose UI names things
    differently can override them without touching this class.
    """

    def __init__(
        self,
        ui,
        floor_num,
        source,
        camera_number=1,
        loop=False,
        model_path="yolo11m.pt",
        video_frame_attr="frame",
        current_camera_attr="current_camera_indicator",
        camera_count_attr="camera_count_indicator",
        floor_count_attr="floor_count_indicator",
    ):
        self.ui = ui
        self.floor_num = floor_num

        self._video_frame = getattr(ui, video_frame_attr)
        self._camera_count_label = getattr(ui, camera_count_attr)
        self._floor_count_label = getattr(ui, floor_count_attr)

        getattr(ui, current_camera_attr).setText(str(camera_number))

        self._build_video_label()

        self.camera_worker = CameraWorker(
            camera_index=source,
            model_path=model_path,
            loop=loop,
        )
        self.camera_worker.frame_ready.connect(self._on_frame_ready)
        self.camera_worker.count_updated.connect(self._on_count_updated)
        self.camera_worker.error_occurred.connect(self._on_error)

    def _build_video_label(self):
        # If Designer already gave `frame` a layout, reuse it — Qt ignores
        # (and warns about) a second layout set on the same widget, which
        # would leave the video label invisible.
        layout = self._video_frame.layout()
        if layout is None:
            layout = QtWidgets.QGridLayout(self._video_frame)
        layout.setContentsMargins(0, 0, 0, 0)

        self.video_label = QtWidgets.QLabel()
        self.video_label.setScaledContents(True)

        if isinstance(layout, QtWidgets.QGridLayout):
            layout.addWidget(self.video_label, 0, 0)
        else:
            layout.addWidget(self.video_label)

    def start(self):
        if not self.camera_worker.isRunning():
            self.camera_worker.start()

    def stop(self):
        if self.camera_worker.isRunning():
            self.camera_worker.stop()

    def _on_frame_ready(self, frame):
        self.video_label.setPixmap(cv_frame_to_pixmap(frame))

    def _on_count_updated(self, count):
        text = str(count)
        self._camera_count_label.setText(text)
        self._floor_count_label.setText(text)

    def _on_error(self, message):
        print(f"[Camera Error - Floor {self.floor_num}] {message}")
