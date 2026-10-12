import sys
import os
import glob

from PyQt6 import QtWidgets

from gui.login_page_ui import Ui_login_page
from gui.dashboard_page_ui import Ui_dashboard_page

from gui.floor_camera_pages.floor_nav_buttons_ui import Ui_floor_nav_buttons
from gui.floor_camera_pages.floor_camera_page_ui_1 import Ui_floor_camera_page_1
from gui.floor_camera_pages.floor_camera_page_ui_2 import Ui_floor_camera_page_2

from assets import resources_rc
from assets import clickable_frame

from controllers.login_handler import LoginHandler
from controllers.floor_switcher import FloorSwitcher
from controllers.floor_camera_controller import FloorCameraController
from controllers.datetime_manager import DateTimeManager


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# floor number -> generated UI class. Add a floor here and it gets a
# page, a nav button, a dashboard button and a switcher entry.
FLOOR_PAGES = {
    1: Ui_floor_camera_page_1,
    2: Ui_floor_camera_page_2,
}


def find_video(name):
    """
    Finds assets/videos/<name>.<any extension>, so it doesn't matter
    whether the file is .mp4, .avi, .mov, etc. (Windows hides extensions
    by default, which makes the real one easy to miss.)
    """
    matches = glob.glob(os.path.join(BASE_DIR, "assets", "videos", f"{name}.*"))
    if not matches:
        print(f"[WARNING] No video named '{name}' found in assets/videos")
        return None
    return matches[0]


class SPOTApp(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("SPOT E-Library")
        self.resize(1920, 1080)
        self.showMaximized()

        self.floor_windows = {}
        self.floor_uis = {}
        self.floor_cameras = {}
        self.floor_datetime_managers = []

        self.stack = QtWidgets.QStackedWidget()
        self.setCentralWidget(self.stack)

        self._build_login_page()
        self._build_dashboard_page()
        self._build_floor_pages()
        self._build_floor_cameras()

        self.floor_switcher.show_login(self.login_window)

    # =====================================================
    # PAGE BUILDERS
    # =====================================================

    def _build_login_page(self):
        self.login_window = QtWidgets.QMainWindow()

        self.login_ui = Ui_login_page()
        self.login_ui.setupUi(self.login_window)

        # LoginHandler owns everything login-related: the webcam feed,
        # face verification, the button, the error label. main.py only
        # says what should happen once login succeeds.
        self.login_handler = LoginHandler(
            login_window=self.login_window,
            login_ui=self.login_ui,
            on_login_success=self._on_login_success,
        )

        self.stack.addWidget(self.login_window)

    def _build_dashboard_page(self):
        self.dashboard_window = QtWidgets.QMainWindow()

        self.dashboard_ui = Ui_dashboard_page()
        self.dashboard_ui.setupUi(self.dashboard_window)

        self.dashboard_datetime = DateTimeManager(self.dashboard_ui)

        self.stack.addWidget(self.dashboard_window)

    def _build_floor_pages(self):
        self.floor_switcher = FloorSwitcher(
            stack=self.stack,
            count_window=self.dashboard_window,
        )

        for floor_num, ui_class in FLOOR_PAGES.items():

            floor_window = QtWidgets.QMainWindow()

            floor_ui = ui_class()
            floor_ui.setupUi(floor_window)

            self.floor_windows[floor_num] = floor_window
            self.floor_uis[floor_num] = floor_ui

            self._attach_floor_nav(floor_window, floor_ui)

            # keeps the date/time labels on this floor updated
            self.floor_datetime_managers.append(DateTimeManager(floor_ui))

            self.stack.addWidget(floor_window)
            self.floor_switcher.register_floor(floor_num, floor_window)

            # floor page -> back to dashboard
            floor_ui.dashboard_button.clicked.connect(
                lambda: self.floor_switcher.show_dashboard(self.dashboard_window)
            )

            # dashboard -> this floor
            dashboard_floor_button = getattr(
                self.dashboard_ui, f"floor_button_{floor_num}"
            )
            dashboard_floor_button.clicked.connect(
                lambda f=floor_num: self.floor_switcher.show_floor(f)
            )

    def _attach_floor_nav(self, floor_window, floor_ui):
        """Floor selector bar: toggled by floor_selector, jumps between floors."""
        nav_widget = QtWidgets.QWidget(floor_window)

        nav_ui = Ui_floor_nav_buttons()
        nav_ui.setupUi(nav_widget)

        nav_widget.move(1520, 0)
        nav_widget.hide()
        nav_widget.raise_()

        floor_ui.nav_widget = nav_widget
        floor_ui.nav_ui = nav_ui

        floor_ui.floor_selector.clicked.connect(
            lambda checked=False, w=nav_widget: w.setVisible(not w.isVisible())
        )

        for target_floor in FLOOR_PAGES:
            button = getattr(nav_ui, f"floor_button_{target_floor}")

            # hide the bar, then go to the chosen floor (one connection,
            # so show_floor only fires once per click)
            button.clicked.connect(
                lambda f=target_floor, w=nav_widget: (
                    w.hide(),
                    self.floor_switcher.show_floor(f),
                )
            )

    def _build_floor_cameras(self):
        """
        One FloorCameraController per floor that has a feed. Floors not
        listed here simply have no camera yet. Controllers are created
        here but only STARTED after login (see _on_login_success).
        """
        floor_camera_sources = {}

        video = find_video("sample_floor_video")
        if video:
            floor_camera_sources[1] = {
                "source": video,
                "camera_number": 1,
                "loop": True,
            }

        for floor_num, config in floor_camera_sources.items():
            self.floor_cameras[floor_num] = FloorCameraController(
                ui=self.floor_uis[floor_num],
                floor_num=floor_num,
                **config,
            )

    # =====================================================
    # LOGIN
    # =====================================================

    def _on_login_success(self):
        for camera in self.floor_cameras.values():
            camera.start()

        self.floor_switcher.show_dashboard(self.dashboard_window)

    # =====================================================
    # APP SHUTDOWN
    # =====================================================

    def closeEvent(self, event):
        self.login_handler.shutdown()

        for camera in self.floor_cameras.values():
            camera.stop()

        super().closeEvent(event)


if __name__ == "__main__":

    app = QtWidgets.QApplication(sys.argv)

    window = SPOTApp()

    window.show()

    sys.exit(app.exec())