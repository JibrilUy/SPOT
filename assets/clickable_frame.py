from PyQt6.QtWidgets import QFrame
from PyQt6.QtCore import pyqtSignal, Qt


class ClickableFrame(QFrame):
    """
    A QFrame that emits a real 'clicked' signal, so it behaves like a
    button for connect() purposes while keeping all the QFrame styling,
    child layouts, and stylesheet-based appearance you already built in
    Designer (border-radius, background-color, etc. — none of that is
    available on QPushButton without extra stylesheet work).
    """
    clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)