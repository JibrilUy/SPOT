# controllers/login_handler.py

from PyQt6 import QtWidgets, QtCore

from core.face_auth_worker import FaceAuthWorker
from core.qt_image_utils import cv_frame_to_pixmap


VERIFICATION_DURATION_MS = 2000  # face must stay matched this long (2 seconds), unbroken
TICK_INTERVAL_MS = 50            # how often the progress bar updates


class LoginHandler:
    """
    Owns ALL of login_page.ui's behavior, so main.py doesn't need any
    login-specific code beyond constructing this once:

      - shows the live webcam feed in camera_layout
      - runs the FaceAuthWorker (starts it, stops it)
      - wires login_button itself
      - username/password check -> wrong_account_label on failure
      - on success, reveals the face-verification widgets and requires
        VERIFICATION_DURATION_MS of CONTINUOUS face match. Any unmatched
        tick resets progress to zero rather than pausing it.
      - when verification completes, releases the webcam and calls
        on_login_success() (main.py decides what happens next).

    Assumes these object names exist on login_page.ui:
      camera_layout, camera_layout_bg, wrong_account_label,
      facial_verification_label_layout, facial_verification_tip_label,
      facial_verification_progress_bar,
      login_username_entry, login_password_entry, login_button
    """

    def __init__(
        self,
        login_window,
        login_ui,
        on_login_success,
        camera_index=0,
    ):
        self.login_window = login_window
        self.login_ui = login_ui
        self.on_login_success = on_login_success

        self.face_matched = False
        self.verifying = False
        self.progress_ms = 0

        self.verification_timer = QtCore.QTimer()
        self.verification_timer.setInterval(TICK_INTERVAL_MS)
        self.verification_timer.timeout.connect(self._on_verification_tick)

        self._build_camera_label()
        self._reset_ui()

        self.face_auth_worker = FaceAuthWorker(camera_index=camera_index)
        self.face_auth_worker.frame_ready.connect(self._on_frame_ready)
        self.face_auth_worker.match_updated.connect(self.set_face_match)
        self.face_auth_worker.error_occurred.connect(self._on_camera_error)
        self.face_auth_worker.start()

        self.login_ui.login_button.clicked.connect(self.attempt_login)

    # -----------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------
    def _build_camera_label(self):
        layout = self.login_ui.camera_layout.layout()
        if layout is None:
            layout = QtWidgets.QGridLayout(self.login_ui.camera_layout)
        layout.setContentsMargins(0, 0, 0, 0)

        self.video_label = QtWidgets.QLabel()
        self.video_label.setScaledContents(True)

        if isinstance(layout, QtWidgets.QGridLayout):
            layout.addWidget(self.video_label, 0, 0)
        else:
            layout.addWidget(self.video_label)

    def _reset_ui(self):
        """Hides the error label and all face-verification widgets."""
        self.login_ui.wrong_account_label.hide()
        self.login_ui.camera_layout_bg.hide()
        self.login_ui.facial_verification_tip_label.hide()
        self.login_ui.facial_verification_label_layout.hide()
        self.login_ui.facial_verification_progress_bar.hide()
        self.login_ui.facial_verification_progress_bar.setValue(0)

    # -----------------------------------------------------------------
    # Face worker signals
    # -----------------------------------------------------------------
    def _on_frame_ready(self, frame):
        self.video_label.setPixmap(cv_frame_to_pixmap(frame))

    def set_face_match(self, matched):
        self.face_matched = matched

    def _on_camera_error(self, message):
        print(f"[Camera Error - Login] {message}")

    # -----------------------------------------------------------------
    # Login button
    # -----------------------------------------------------------------
    def attempt_login(self):
        # Ignore repeat clicks while a verification is already running.
        if self.verifying:
            return

        username = self.login_ui.login_username_entry.text()
        password = self.login_ui.login_password_entry.text()

        credentials_ok = (username == "username" and password == "123")

        if not credentials_ok:
            self.login_ui.wrong_account_label.show()
            self._stop_verification(reset=True)
            return

        self.login_ui.wrong_account_label.hide()
        self._start_verification()

    # -----------------------------------------------------------------
    # Face verification
    # -----------------------------------------------------------------
    def _start_verification(self):
        self.verifying = True
        self.progress_ms = 0

        self.login_ui.camera_layout_bg.show()
        self.login_ui.facial_verification_label_layout.show()
        self.login_ui.facial_verification_tip_label.show()
        self.login_ui.facial_verification_progress_bar.show()
        self.login_ui.facial_verification_progress_bar.setValue(0)

        self.verification_timer.start()

    def _stop_verification(self, reset=False):
        self.verifying = False
        self.verification_timer.stop()

        if reset:
            self.progress_ms = 0
            self._reset_ui()

    def _on_verification_tick(self):
        if self.face_matched:
            self.progress_ms += TICK_INTERVAL_MS
        else:
            # Interrupted — restart from zero.
            self.progress_ms = 0

        percent = min(100, int(self.progress_ms / VERIFICATION_DURATION_MS * 100))
        self.login_ui.facial_verification_progress_bar.setValue(percent)

        if self.progress_ms >= VERIFICATION_DURATION_MS:
            self._on_verification_complete()

    def _on_verification_complete(self):
        self._stop_verification(reset=False)

        # Release the webcam BEFORE anything else tries to open it.
        self.face_auth_worker.stop()

        self.on_login_success()

    # -----------------------------------------------------------------
    # App shutdown
    # -----------------------------------------------------------------
    def shutdown(self):
        self.verification_timer.stop()
        if self.face_auth_worker.isRunning():
            self.face_auth_worker.stop()