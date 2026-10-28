from PyQt6.QtWidgets import QLabel
from PyQt6.QtCore import pyqtSignal, Qt


class ClickableVideoLabel(QLabel):
    """
    A QLabel for showing video that tells you where it was clicked.

    clicked_at(nx, ny) gives the click as FRACTIONS of the label's size
    (0..1), not pixels. That works with setScaledContents(True), where the
    frame is stretched to fill the label, so the same fractions point at
    the same spot in the original video frame whatever size the label is.
    """

    clicked_at = pyqtSignal(float, float)

    def mousePressEvent(self, event):
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self.width() > 0
            and self.height() > 0
        ):
            pos = event.position()
            self.clicked_at.emit(pos.x() / self.width(), pos.y() / self.height())
        super().mousePressEvent(event)