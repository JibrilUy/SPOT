"""Display-only Student Dashboard for SPOT; no floor navigation/database calls.

Use install_student_dashboard(owner, dashboard_ui) after dashboard_ui.setupUi(...)
"""
from datetime import datetime
from PyQt6 import QtCore, QtWidgets
from gui.student_dashboard_page_ui import Ui_student_dashboard_page
from controllers.datetime_manager import DateTimeManager


# Ground Floor is floor 1 in the generated student dashboard UI.
# These are deliberately STATIC demo counts from the reference screenshot.
FLOOR_COUNTS = {7: 26, 6: 70, 5: 20, 4: 75, 3: 100, 2: 35, 1: 40}


def floor_colors(count: int) -> tuple[str, str]:
    """More people => darker burgundy background and a redder floor title."""
    value = max(0, min(100, int(count))) / 100.0
    red = round(155 - 120 * value)
    green = round(16 - 14 * value)
    blue = round(19 - 16 * value)
    background = f"#{red:02x}{green:02x}{blue:02x}"

    # Mirror the provided reference: under 60 the title is white;
    # crowded floors (70, 75, 100) turn increasingly strong red.
    if count < 60:
        text_color = "#FFFFFF"
    else:
        degree = max(0.0, min(1.0, (count - 60) / 40.0))
        title_green = round(63 - 36 * degree)
        title_blue = round(19 - 11 * degree)
        text_color = f"#ff{title_green:02x}{title_blue:02x}"
    return background, text_color


class StudentClockManager(DateTimeManager):
    """Use HH:MM for the student window without changing other clocks."""

    def update_datetime(self):
        super().update_datetime()
        self.ui.datetime_current_time.setText(datetime.now().strftime("%I:%M"))


class StudentDashboardWindow(QtWidgets.QMainWindow):
    """An independent Qt window, owned by the main app for safe cleanup."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlag(QtCore.Qt.WindowType.Window, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.ui = Ui_student_dashboard_page()
        self.ui.setupUi(self)
        self.setWindowTitle("SPOT - Student Dashboard")

        for floor, count in FLOOR_COUNTS.items():
            self.set_floor_count(floor, count)
        self.ui.total_count_label.setText(str(sum(FLOOR_COUNTS.values())))

        # A local clock only: this page does NOT query PostgreSQL or cameras.
        self.datetime_manager = StudentClockManager(self.ui)

        # Important: do NOT connect floor_button_N.clicked to anything.

    def set_floor_count(self, floor: int, count: int):
        """Static for now; callable later if real occupancy is connected."""
        if floor not in range(1, 8):
            raise ValueError("floor must be between 1 (ground) and 7")
        frame = getattr(self.ui, f"floor_button_{floor}")
        indicator = getattr(self.ui, f"floor_indicator_{floor}")
        number = getattr(self.ui, f"floor_button_count_indicator_{floor}")

        background, title_color = floor_colors(count)
        object_name = frame.objectName()
        frame.setStyleSheet(
            f"QFrame#{object_name} {{ background-color: {background}; border: none; }}"
            f"QFrame#{object_name}:hover {{ background-color: {background}; }}"
        )
        indicator.setStyleSheet(
            f"QLabel#{indicator.objectName()} {{ color: {title_color}; "
            "background-color: transparent; }"
        )
        number.setText(f" {count}")
        number.setStyleSheet("color: #F5AA2B; background: transparent;")


def install_student_dashboard(owner, dashboard_ui):
    """Connect main dashboard's STUDENT PANEL ClickableFrame to another window.

    Returns the window: save it on owner to prevent garbage collection.
    """
    button = getattr(dashboard_ui, "student_panel_button", None)
    if button is None:
        button = getattr(dashboard_ui, "student_dashboard_button", None)
    if button is None:
        raise AttributeError(
            "Main dashboard UI has no student_panel_button. "
            "Use the updated gui/dashboard_page_ui.py from your uploaded design."
        )

    student_window = StudentDashboardWindow(parent=owner)

    def open_window(*_clicked_args):
        if not student_window.isVisible():
            student_window.showMaximized()
        student_window.raise_()
        student_window.activateWindow()

    button.clicked.connect(open_window)
    return student_window
