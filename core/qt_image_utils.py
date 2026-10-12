import cv2
from PyQt6.QtGui import QImage, QPixmap


def cv_frame_to_pixmap(frame) -> QPixmap:
    """
    Converts an OpenCV BGR frame (numpy array) into a QPixmap that can be
    set directly on a QLabel via .setPixmap(). Kept separate from
    camera_worker.py so either the detector module or any other part of
    the UI can reuse this conversion without duplicating it.
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_frame.shape
    bytes_per_line = ch * w
    qimage = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimage)
