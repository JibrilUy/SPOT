# controllers/login_handler.py

from PyQt6 import QtWidgets, QtCore

from core.face_auth_worker import FaceAuthWorker
from core.qt_image_utils import cv_frame_to_pixmap


VERIFICATION_DURATION_MS = 100  # face must stay matched this long (2 seconds), unbroken
TICK_INTERVAL_MS = 50            # how often the progress bar updates
FACE_MAX_WAIT_MS = 20000         # one failure per timed-out attempt, not per frame


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
        db,
        camera_index=0,
    ):
        self.login_window = login_window
        self.login_ui = login_ui
        # Hide typed characters and limit admin passwords to eight characters.
        self.login_ui.login_password_entry.setEchoMode(
            QtWidgets.QLineEdit.EchoMode.Password
        )
        self.login_ui.login_password_entry.setMaxLength(8)
        self.on_login_success = on_login_success
        self.db = db
        self.pending_user = None
        self.pending_attempt_id = None

        self.face_matched = False
        self.verifying = False
        self.progress_ms = 0

        self.verification_timer = QtCore.QTimer()
        self.verification_timer.setInterval(TICK_INTERVAL_MS)
        self.verification_timer.timeout.connect(self._on_verification_tick)
        self.face_timeout_timer = QtCore.QTimer()
        self.face_timeout_timer.setSingleShot(True)
        self.face_timeout_timer.timeout.connect(self._face_timed_out)

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
        if self.verifying:
            self._finish_attempt('system_error', message[:120])
            self.pending_user = None
            self._stop_verification(reset=True)
            self.login_ui.wrong_account_label.setText('Face camera is unavailable')
            self.login_ui.wrong_account_label.show()

    # -----------------------------------------------------------------
    # Login button
    # -----------------------------------------------------------------
    def _finish_attempt(self, result, reason=None, session_id=None):
        attempt_id = self.pending_attempt_id
        self.pending_attempt_id = None
        if attempt_id is not None:
            self.db.finish_login_attempt(attempt_id, result,
                                         failure_reason=reason, session_id=session_id)

    def attempt_login(self):
        if self.verifying:
            return
        username = self.login_ui.login_username_entry.text().strip()
        password = self.login_ui.login_password_entry.text()

        try:
            # A single row is created when the login button is clicked.
            self.pending_attempt_id = self.db.start_login_attempt(username)
            account = self.db.authenticate_admin(username, password)
            if not account:
                self._finish_attempt('password_failed', 'invalid_credentials_or_unenrolled')
                self.pending_user = None
                self.login_ui.wrong_account_label.setText('Wrong account or password')
                self.login_ui.wrong_account_label.show()
                self._stop_verification(reset=True)
                return
            self.db.update_login_attempt(self.pending_attempt_id,
                                         stage='face', user_id=account['id'])
        except Exception as exc:
            print(f'[Database - Login] {exc}')
            if self.pending_attempt_id is not None:
                try:
                    self._finish_attempt('system_error', 'database_error')
                except Exception:
                    self.pending_attempt_id = None
            self.pending_user = None
            # A database failure must never authenticate the user.
            self.login_ui.wrong_account_label.setText('Database unavailable; login denied')
            self.login_ui.wrong_account_label.show()
            self._stop_verification(reset=True)
            return

        self.pending_user = account
        self.face_matched = False
        self.face_auth_worker.set_expected_label(account['face_label'])
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
        self.face_timeout_timer.start(FACE_MAX_WAIT_MS)

    def _stop_verification(self, reset=False):
        self.verifying = False
        self.verification_timer.stop()
        self.face_timeout_timer.stop()

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

    def _face_timed_out(self):
        if not self.verifying:
            return
        try:
            self._finish_attempt('face_failed', 'face_verification_timeout')
        except Exception as exc:
            print('[Database - Face failure logging]', exc)
        self.pending_user = None
        self.face_auth_worker.set_expected_label(None)
        self.face_matched = False
        self._stop_verification(reset=True)
        self.login_ui.wrong_account_label.setText('Face verification timed out; try again')
        self.login_ui.wrong_account_label.show()

    def _on_verification_complete(self):
        self._stop_verification(reset=False)
        if not self.pending_user:
            return

        session_id = None
        try:
            # A login record exists only if password AND face succeeded.
            session_id = self.db.start_admin_session(self.pending_user["id"])
            self._finish_attempt('successful', session_id=session_id)
        except Exception as exc:
            if session_id is not None:
                try:
                    self.db.end_session(session_id, reason='system_error')
                except Exception:
                    pass
            print(f"[Database - Start session] {exc}")
            try:
                self._finish_attempt('system_error', 'session_creation_failed')
            except Exception:
                pass
            self.pending_user = None
            self.face_matched = False
            self.face_auth_worker.set_expected_label(None)
            self._reset_ui()
            self.login_ui.wrong_account_label.show()
            return

        account = self.pending_user
        self.pending_user = None
        self.face_auth_worker.set_expected_label(None)
        self.login_ui.login_password_entry.clear()
        # Release webcam BEFORE other workers attempt to use it.
        self.face_auth_worker.stop()
        self.on_login_success(account, session_id)

    def suspend_for_usb(self):
        """Stop face verification immediately when the USB superadmin takes over."""
        self._stop_verification(reset=True)
        if self.pending_attempt_id is not None:
            try:
                self._finish_attempt('cancelled', 'usb_superadmin_login')
            except Exception as exc:
                print('[Database - Cancel login]', exc)
        self.pending_user = None
        self.face_matched = False
        self.face_auth_worker.set_expected_label(None)
        self.login_ui.login_password_entry.clear()
        if self.face_auth_worker.isRunning():
            self.face_auth_worker.stop()

    def prepare_new_login(self):
        """Call on logout before navigating to the login page."""
        self._stop_verification(reset=True)
        self.pending_user = None
        self.face_matched = False
        self.face_auth_worker.set_expected_label(None)
        self.login_ui.login_password_entry.clear()
        # The LBPH trainer may have changed while a USB super admin
        # created an account. Always load the latest model before login.
        self.face_auth_worker.reload_model()
        if not self.face_auth_worker.isRunning():
            self.face_auth_worker.start()

    # -----------------------------------------------------------------
    # App shutdown
    # -----------------------------------------------------------------
    def shutdown(self):
        self.verification_timer.stop()
        self.face_timeout_timer.stop()
        if self.face_auth_worker.isRunning():
            self.face_auth_worker.stop()