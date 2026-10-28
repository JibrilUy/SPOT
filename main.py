import sys
import os
import glob

from PyQt6 import QtWidgets, QtCore, QtGui
from core.shared_target_registry import SharedTargetRegistry

from gui.login_page_ui import Ui_login_page
from gui.dashboard_page_ui import Ui_dashboard_page

from gui.floor_camera_pages.floor_nav_buttons_ui import Ui_floor_nav_buttons
from gui.floor_camera_pages.floor_camera_page_ui_1 import Ui_floor_camera_page_1
from gui.floor_camera_pages.floor_camera_page_ui_2 import Ui_floor_camera_page_2
from gui.floor_camera_pages.floor_camera_page_ui_3 import Ui_floor_camera_page_3
from gui.floor_camera_pages.floor_camera_page_ui_4 import Ui_floor_camera_page_4
from gui.floor_camera_pages.floor_camera_page_ui_5 import Ui_floor_camera_page_5
from gui.floor_camera_pages.floor_camera_page_ui_6 import Ui_floor_camera_page_6
from gui.floor_camera_pages.floor_camera_page_ui_7 import Ui_floor_camera_page_7

from assets import resources_rc
from assets import clickable_frame

from controllers.student_dashboard_controller import install_student_dashboard
from controllers.login_handler import LoginHandler
from controllers.floor_switcher import FloorSwitcher
from controllers.floor_camera_controller import FloorCameraController
from controllers.datetime_manager import DateTimeManager
from database.db import Database
from security.usb_key import find_tokens
from controllers.super_admin_dashboard import SuperAdminDashboard
from pathlib import Path

if getattr(sys, "frozen", False):
    os.chdir(Path(sys.executable).resolve().parent)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# floor number -> generated UI class. Add a floor here and it gets a
# page, a nav button, a dashboard button and a switcher entry.
FLOOR_PAGES = {
    1: Ui_floor_camera_page_1,
    2: Ui_floor_camera_page_2,
    3: Ui_floor_camera_page_3,
    4: Ui_floor_camera_page_4,
    5: Ui_floor_camera_page_5,
    6: Ui_floor_camera_page_6,
    7: Ui_floor_camera_page_7,

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
        self.db = Database()  # Startup fails rather than running without audit logs.
        self.current_user = None
        self.current_session_id = None
        self._active_usb_key_id = None
        self._usb_requires_removal = False
        self._rejected_usb_ids = set()

        self.stack = QtWidgets.QStackedWidget()
        self.setCentralWidget(self.stack)

        self._build_login_page()
        self._build_dashboard_page()
        self._build_super_admin_page()
        self._build_floor_pages()
        self.super_admin_page.bind_floor_buttons(set(FLOOR_PAGES), self.floor_switcher.show_floor)
        self.shared_target = SharedTargetRegistry()
        self._build_floor_cameras()

        self.floor_switcher.show_login(self.login_window)

        # Logout works without any Designer GUI modifications.
        session_menu = self.menuBar().addMenu('Session')
        self.logout_action = QtGui.QAction('Log out', self)
        self.logout_action.triggered.connect(
            lambda checked=False: self.logout(reason="manual")
        )        
        self.logout_action.setEnabled(False)
        session_menu.addAction(self.logout_action)

        # The USB is only checked while at login, or to enforce continued
        # physical-key presence during a super-admin session.
        self.usb_timer = QtCore.QTimer(self)
        self.usb_timer.setInterval(1000)
        self.usb_timer.timeout.connect(self._usb_tick)
        self.usb_timer.start()
        self._usb_tick()

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
            db=self.db,
        )

        self.stack.addWidget(self.login_window)

    def _build_dashboard_page(self):
        self.dashboard_window = QtWidgets.QMainWindow()

        self.dashboard_ui = Ui_dashboard_page()
        self.dashboard_ui.setupUi(self.dashboard_window)
        self.student_dashboard_window = install_student_dashboard(self, self.dashboard_ui)
        
        self.dashboard_ui.logout_button.clicked.connect(
            lambda checked=False: self.logout(reason="manual")
        )
        self.dashboard_datetime = DateTimeManager(self.dashboard_ui)

        self.stack.addWidget(self.dashboard_window)

    def _build_super_admin_page(self):
        # Temporary functional UI; replace with your own .ui later.
        self.super_admin_page = SuperAdminDashboard(
            get_session_id=lambda: self.current_session_id,
            db=self.db,
            get_usb_token=self._get_live_superadmin_usb,
            show_monitoring=self._open_monitoring_for_superadmin,
            logout=self.logout,
        )
        self.stack.addWidget(self.super_admin_page)
        self.superadmin_datetime = DateTimeManager(self.super_admin_page.ui)

    def _show_home(self):
        if self.current_user and self.current_user['role'] == 'super_admin':
            if self._require_superadmin_usb():
                self.super_admin_page.reload()
                self.stack.setCurrentWidget(self.super_admin_page)
        elif self.current_user and self.current_user['role'] == 'admin':
            self.floor_switcher.show_dashboard(self.dashboard_window)

    def _start_cameras(self):
        for camera in self.floor_cameras.values():
            camera.start()

    def _open_monitoring_for_superadmin(self):
        if self._require_superadmin_usb():
            self._start_cameras()
            self.floor_switcher.show_dashboard(self.dashboard_window)

    def _get_live_superadmin_usb(self):
        """Fetch a verified credential from the plugged-in USB, NOT memory.

        The privileged DB methods reverify the secret on every operation.
        """
        if not self.current_user or self.current_user['role'] != 'super_admin':
            return None
        try:
            for key_id, secret in find_tokens():
                if (key_id == self._active_usb_key_id and
                        self.db.usb_key_still_authorized(self.current_session_id, key_id, secret)):
                    return key_id, secret
        except Exception as exc:
            print('[USB verify]', exc)
        self.logout(reason='usb_removed')
        return None

    def _require_superadmin_usb(self):
        return self._get_live_superadmin_usb() is not None

    def _usb_tick(self):
        if self.current_user:
            if self.current_user['role'] == 'super_admin':
                self._require_superadmin_usb()   # Auto log out if USB removed
            return
        try:
            detected_tokens = list(find_tokens())
        except Exception as exc:
            print('[USB scan]', exc)
            return
        if not detected_tokens:
            self._usb_requires_removal = False
            self._rejected_usb_ids.clear()
            return
        if self._usb_requires_removal:
            return   # prevents a plugged-in key auto-logging in after logout
        for key_id, secret in detected_tokens:
            try:
                identity = self.db.start_superadmin_session(key_id, secret)
            except Exception as exc:
                print('[USB authentication]', exc)
                return
            if identity:
                account, session_id = identity
                # Log the successful USB authentication under the known username.
                try:
                    attempt = self.db.start_login_attempt(
                        account['username'], auth_method='usb_key', stage='usb',
                        user_id=account['id'])
                    self.db.finish_login_attempt(attempt, 'successful', session_id=session_id)
                except Exception as exc:
                    print('[Database - USB attempt audit]', exc)
                self.login_handler.suspend_for_usb()
                self._active_usb_key_id = key_id
                self._on_login_success(account, session_id)
                return
            # Do not generate one failed attempt per one-second USB poll.
            if key_id not in self._rejected_usb_ids:
                self._rejected_usb_ids.add(key_id)
                try:
                    a = self.db.start_login_attempt('[unknown_usb_key]',
                                                    auth_method='usb_key', stage='usb')
                    self.db.finish_login_attempt(a, 'usb_failed',
                                                 failure_reason='unrecognized_or_revoked_key')
                except Exception as exc:
                    print('[Database - Rejected USB audit]', exc)

    def _build_floor_pages(self):
        self.floor_switcher = FloorSwitcher(
            stack=self.stack,
            count_window=self.dashboard_window,
            before_show_floor=self._authorize_and_log_floor,
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
                lambda checked=False: self._show_home()
            )

            # dashboard -> this floor
            dashboard_floor_button = getattr(
                self.dashboard_ui, f"floor_button_{floor_num}"
            )
            dashboard_floor_button.clicked.connect(
                lambda checked=False, f=floor_num: self.floor_switcher.show_floor(f)
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
                lambda checked=False, f=target_floor, w=nav_widget: (
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
        # --- 1. EASILY CHANGE THE VIDEO NAME HERE ---
        # Change this string once, and it updates across all floors below.
        video_name1 = "secure_spot_sample"
        video_name2 = "tracking_sample_1"
        video_name3 = "tracking_sample_2"
        video_name4 = "tracking_sample_3"
        video_name5 = "sample_floor_video"

        
        video1 = find_video(video_name1)
        if not video1:
            return

        video2 = find_video(video_name2)
        if not video2:
            return
        video3 = find_video(video_name3)
        if not video3:
            return
        video4 = find_video(video_name4)
        if not video4:
            return
        video5 = find_video(video_name5)
        if not video5:
            return


        
        # --- 2. DEFINE WHICH FLOORS GET A CAMERA FEED ---
        # You can list all active floors here. If you want different videos 
        # per floor later, you can swap `video` for individual `find_video("...")` calls.
        floor_camera_sources = {
            1: {"source": video1, "camera_number": 1, "loop": True, "secure_zone_tracking": True, "shared_target": self.shared_target},
            2: {"source": video2, "camera_number": 2, "loop": True, "cross_floor_tracking": True, "shared_target": self.shared_target},
            3: {"source": video3, "camera_number": 3, "loop": True, "cross_floor_tracking": True, "shared_target": self.shared_target},
            4: {"source": video4, "camera_number": 4, "loop": True, "cross_floor_tracking": True, "shared_target": self.shared_target},
            5: {"source": video5, "camera_number": 5, "loop": True},

        }

        for floor_num, config in floor_camera_sources.items():
            # Safety check to ensure the UI for this floor actually exists
            if floor_num in self.floor_uis:
                self.floor_cameras[floor_num] = FloorCameraController(
                    ui=self.floor_uis[floor_num],
                    floor_num=floor_num,
                    **config,
                )

    # =====================================================
    # LOGIN
    # =====================================================

    def _on_login_success(self, account, session_id):
        self.current_user = account
        self.current_session_id = session_id
        self.logout_action.setEnabled(True)
        if account['role'] == 'super_admin':
            self.super_admin_page.reload()
            self.stack.setCurrentWidget(self.super_admin_page)
        elif account['role'] == 'admin':
            self._start_cameras()
            self.floor_switcher.show_dashboard(self.dashboard_window)
        else:
            self.logout()

    def _authorize_and_log_floor(self, floor_num):
        if self.current_session_id is None:
            return False
        if self.current_user['role'] == 'super_admin' and not self._require_superadmin_usb():
            return False
        try:
            allowed = self.db.record_floor_open(self.current_session_id, floor_num)
        except Exception as exc:
            print(f"[Database - Floor {floor_num}] {exc}")
            QtWidgets.QMessageBox.warning(
                self, "Database error", "Could not verify or record floor access."
            )
            return False
        if not allowed:
            QtWidgets.QMessageBox.warning(
                self, "Access denied", "This account cannot open that floor."
            )
        return allowed

    def logout(self, reason='manual'):
        """Stop video, close the DB session, and return to admin/USB login."""
        # Stop feeds first, so no monitoring is left running after logout.
        for camera in self.floor_cameras.values():
            camera.stop()
        if self.current_session_id is not None:
            try:
                self.db.end_session(self.current_session_id, reason=reason)
            except Exception as exc:
                print(f"[Database - Logout] {exc}")
        if self.current_user and self.current_user['role'] == 'super_admin':
            self._usb_requires_removal = True
        self.current_user = None
        self.current_session_id = None
        self._active_usb_key_id = None
        self.logout_action.setEnabled(False)
        self.floor_switcher.show_login(self.login_window)
        self.login_handler.prepare_new_login()

    # =====================================================
    # APP SHUTDOWN
    # =====================================================

    def closeEvent(self, event):
        self.usb_timer.stop()
        self.login_handler.shutdown()
        for camera in self.floor_cameras.values():
            camera.stop()
        if self.current_session_id is not None:
            try:
                self.db.end_session(self.current_session_id, reason='application_exit')
            except Exception as exc:
                print(f"[Database - Shutdown] {exc}")
        self.db.close()
        super().closeEvent(event)


if __name__ == "__main__":

    app = QtWidgets.QApplication(sys.argv)

    window = SPOTApp()
    window.show()

    try:
        exit_code = app.exec()
    finally:
        # Ensure Qt shutdown runs even if the event loop exits
        # without a normal window-close event.
        try:
            window.close()
        finally:
            # Closing an already-closed psycopg connection pool
            # is safe.
            window.db.close()

    sys.exit(exit_code)