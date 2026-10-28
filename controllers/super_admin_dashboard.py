"""Super-admin dashboard using the user's original librarian account cards.

The first three visible cards are the exact Qt Designer widgets already in
admin_dashboard_page_ui.py. Additional accounts use matching scrollable cards.
All privileged mutations recheck possession of the live USB key in Database.
"""
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

from PyQt6 import QtCore, QtGui, QtWidgets

from gui.admin_dashboard_page_ui import Ui_admin_dashboard_page
from controllers.face_enrollment_dialog import FaceEnrollmentDialog
from core.face_enrollment import MODEL_PATH, build_staged_model, get_next_label, publish_model
from core.face_model_replace import build_replacement_model


CARD_STYLE = '''
QFrame#accountCard {
    background-color: rgba(70, 0, 0, 120);
    border: 2px solid #F5AA2B;
    border-radius: 12px;
}
QFrame#accountCard:hover { background-color: rgba(140, 45, 0, 200); }
'''
SELECTED_STYLE = '''
QFrame#accountCard {
    background-color: rgba(135, 0, 1, 220);
    border: 3px solid #FFD276;
    border-radius: 12px;
}
'''
ACTION_STYLE = '''
QPushButton {
    color: #180800; background-color: #F5AA2B;
    border: 0; border-radius: 9px;
    font-weight: 700; font-size: 12pt;
}
QPushButton:hover { background-color: #FFBA45; }
QPushButton:disabled { background-color: #81643D; color: #D0C5B8; }
'''


def format_recency(moment):
    """Last successful login recency; never confuse account creation with login."""
    if not moment:
        return 'Never logged in'
    if isinstance(moment, str):
        try:
            moment = datetime.fromisoformat(moment)
        except ValueError:
            return 'Last login unavailable'
    if not isinstance(moment, datetime):
        return 'Last login unavailable'
    now = datetime.now(moment.tzinfo) if moment.tzinfo else datetime.now()
    seconds = max(0, int((now - moment).total_seconds()))
    if seconds < 3600:
        return 'Less than 1 hour ago'
    hours = seconds // 3600
    if hours < 24:
        return 'Since 1 hour ago' if hours == 1 else f'Since {hours} hours ago'
    if hours < 48:
        return 'More than 24 hours ago'
    days = seconds // 86400
    return f'{days} days ago'


class SuperAdminDashboard(QtWidgets.QMainWindow):
    def __init__(self, get_session_id, db, get_usb_token, show_monitoring, logout, parent=None):
        super().__init__(parent)
        self.ui = Ui_admin_dashboard_page()
        self.ui.setupUi(self)
        self.db = db
        self.get_session_id = get_session_id
        self.get_usb_token = get_usb_token
        self.show_monitoring = show_monitoring
        self.logout = logout
        self.selected_admin_id = None
        self._account_rows = {}
        self._cards = []
        self._click_ids = {}
        self.ui.dashboard_username.setText('Welcome Super Administrator')
        self.ui.librarian_list_label.setText('ACCOUNTS')
        self.ui.logs.hide()
        self.ui.total_count_label.setText('—')

        self._setup_account_cards()
        self._setup_profile_panel()
        self._setup_activity_table()

        self.ui.create_acc_button.clicked.connect(self.create_admin)
        self.ui.logout_button.clicked.connect(lambda checked=False: self.logout(reason='manual'))
        # Printing/tracking controllers are not supplied yet; do not fake actions.
        self.ui.transcript_button.hide()
        self.ui.tracking_button.hide()

    def _setup_account_cards(self):
        container = self.ui.librarian_list_frame
        outer = QtWidgets.QVBoxLayout(container)
        outer.setContentsMargins(10, 12, 8, 10)
        self.account_scroll = QtWidgets.QScrollArea(container)
        self.account_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.account_scroll.setWidgetResizable(True)
        self.account_scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.account_scroll.setStyleSheet('background: transparent; border: none;')
        self.card_host = QtWidgets.QWidget()
        self.card_host.setStyleSheet('background: transparent;')
        self.card_layout = QtWidgets.QVBoxLayout(self.card_host)
        self.card_layout.setContentsMargins(0, 2, 0, 0)
        self.card_layout.setSpacing(10)
        self.card_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        self.account_scroll.setWidget(self.card_host)
        outer.addWidget(self.account_scroll)

        # Reuse the real named Qt Designer labels, as requested.
        for index, suffix in enumerate(('', '_2', '_3')):
            frame = getattr(self.ui, 'librarian_profile_frame' + suffix)
            username = getattr(self.ui, 'librarian_profile_username' + suffix)
            recency = getattr(self.ui, 'librarian_profile_recency' + suffix)
            logo = getattr(self.ui, 'librarian_profile_logo' + suffix)
            frame.setParent(self.card_host)
            frame.setObjectName('accountCard')
            frame.setFixedHeight(100)
            frame.setMinimumWidth(260)
            frame.setMaximumWidth(16777215)
            frame.setStyleSheet(CARD_STYLE)
            username.setGeometry(99, 9, 224, 42)
            username.setStyleSheet('color: white; font-size: 20pt; font-weight: 700; background: transparent;')
            username.setText('')
            username.setWordWrap(False)
            recency.setGeometry(99, 49, 231, 34)
            recency.setStyleSheet('color: white; font-size: 13pt; background: transparent;')
            recency.setText('')
            for child in (logo, username, recency):
                child.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            frame.installEventFilter(self)
            self.card_layout.addWidget(frame)
            self._cards.append((frame, username, recency))
            frame.hide()
        self.card_layout.addStretch(1)
        # Dynamic cards are inserted before the stretch as accounts grow.

    def _extra_card(self):
        frame = QtWidgets.QFrame(self.card_host)
        frame.setObjectName('accountCard')
        frame.setMinimumWidth(260)
        frame.setFixedHeight(100)
        frame.setStyleSheet(CARD_STYLE)
        pixmap = self.ui.librarian_profile_logo.pixmap()
        icon = QtWidgets.QLabel(frame)
        icon.setGeometry(30, 17, 41, 65)
        if pixmap is not None:
            icon.setPixmap(pixmap)
        icon.setScaledContents(True)
        name = QtWidgets.QLabel(frame)
        name.setGeometry(99, 9, 224, 42)
        name.setStyleSheet('color: white; font-size: 20pt; font-weight: 700;')
        name.setText('')
        recency = QtWidgets.QLabel(frame)
        recency.setGeometry(99, 49, 231, 34)
        recency.setStyleSheet('color: white; font-size: 13pt;')
        for child in (icon, name, recency):
            child.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        frame.installEventFilter(self)
        self.card_layout.insertWidget(self.card_layout.count() - 1, frame)
        self._cards.append((frame, name, recency))

    def eventFilter(self, watched, event):
        if (watched in self._click_ids and
                event.type() == QtCore.QEvent.Type.MouseButtonRelease and
                event.button() == QtCore.Qt.MouseButton.LeftButton):
            self.select_account(self._click_ids[watched])
            return True
        return super().eventFilter(watched, event)

    def _setup_profile_panel(self):
        ui = self.ui
        panel = ui.librarian_personal_profile_frame
        panel.show()
        panel.setObjectName('selectedProfileCard')
        panel.setStyleSheet('QFrame#selectedProfileCard { background: transparent; border: none; }')
        ui.librarian_personal_profile_username.setGeometry(98, 12, 230, 44)
        ui.librarian_personal_profile_username.setStyleSheet(
            'color: white; font-size: 20pt; font-weight: bold;')
        ui.librarian_personal_profile_username.setWordWrap(False)
        ui.librarian_personal_profile_id_number.setGeometry(99, 53, 225, 32)
        ui.librarian_personal_profile_id_number.setStyleSheet('color: white; font-size: 13pt;')
        for name, top in (('label', 90), ('label_2', 123), ('label_3', 157)):
            label = getattr(ui, name)
            label.setGeometry(13, top, 317, 31)
            label.setStyleSheet('color: white; font-size: 12pt;')
            label.setWordWrap(False)
        self.profile_status = QtWidgets.QLabel(panel)
        self.profile_status.setGeometry(13, 192, 317, 34)
        self.profile_status.setStyleSheet('color: #F5AA2B; font-size: 11pt;')
        self.change_password_button = QtWidgets.QPushButton('CHANGE PASSWORD', panel)
        self.change_password_button.setGeometry(13, 240, 311, 38)
        self.change_password_button.setStyleSheet(ACTION_STYLE)
        self.change_password_button.clicked.connect(self.change_password)
        self.update_face_button = QtWidgets.QPushButton('UPDATE FACE ID', panel)
        self.update_face_button.setGeometry(13, 285, 311, 38)
        self.update_face_button.setStyleSheet(ACTION_STYLE)
        self.update_face_button.clicked.connect(self.update_face_id)
        self._clear_profile()

    def _clear_profile(self):
        self.ui.librarian_personal_profile_username.setText('Select an account')
        self.ui.librarian_personal_profile_id_number.setText('')
        for field in (self.ui.label, self.ui.label_2, self.ui.label_3, self.profile_status):
            field.setText('')
        self.change_password_button.setEnabled(False)
        self.update_face_button.setEnabled(False)

    def _setup_activity_table(self):
        log_layout = QtWidgets.QVBoxLayout(self.ui.librarian_log_frame)
        self.refresh_button = QtWidgets.QPushButton('Refresh accounts and logs')
        self.refresh_button.clicked.connect(self.reload)
        log_layout.addWidget(self.refresh_button)
        self.activity = QtWidgets.QTableWidget(0, 4)
        self.activity.setHorizontalHeaderLabels(['Username', 'Action', 'Details', 'Date/time'])
        self.activity.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.activity.horizontalHeader().setStretchLastSection(True)
        self.activity.setAlternatingRowColors(True)
        self.activity.setStyleSheet('background-color: white; color: black; font-size: 12px;')
        log_layout.addWidget(self.activity)

    def bind_floor_buttons(self, available_floor_numbers, callback):
        for floor in range(1, 8):
            button = getattr(self.ui, f'floor_button_{floor}', None)
            if button is None:
                continue
            if floor not in available_floor_numbers:
                button.hide()
            else:
                button.clicked.connect(lambda checked=False, f=floor: callback(f))

    def select_account(self, admin_id):
        info = self._account_rows.get(admin_id)
        if info is None:
            return
        self.selected_admin_id = admin_id
        ui = self.ui
        ui.librarian_personal_profile_username.setText(str(info['username']))
        ui.librarian_personal_profile_username.setToolTip(str(info['username']))
        ui.librarian_personal_profile_id_number.setText(f"Account #{info['id']}")
        ui.label.setText(f"Name: {info['full_name'] or '(not set)'}")
        ui.label.setToolTip(str(info['full_name'] or ''))
        last = info.get('last_successful_login')
        ui.label_2.setText('Last login: ' + format_recency(last))
        ui.label_2.setToolTip(str(last or 'Never'))
        seconds = int(info.get('completed_session_seconds') or 0)
        logins = int(info.get('successful_login_count') or 0)
        ui.label_3.setText(f'Total: {seconds // 3600}h {(seconds % 3600) // 60}m · {logins} logins')
        label = info.get('face_label')
        status = 'Enabled' if info['is_active'] else 'Disabled'
        self.profile_status.setText(f"Face ID: {label if label is not None else 'None'}  |  {status}")
        self.change_password_button.setEnabled(True)
        self.update_face_button.setEnabled(label is not None and bool(info['is_active']))
        for frame, _username, _recency in self._cards:
            frame.setStyleSheet(SELECTED_STYLE if self._click_ids.get(frame) == admin_id else CARD_STYLE)

    def _require_selected_and_usb(self):
        info = self._account_rows.get(self.selected_admin_id)
        if not info:
            QtWidgets.QMessageBox.warning(self, 'Select account', 'Choose an Administrator account first.')
            return None, None
        token = self.get_usb_token()
        if token is None:
            QtWidgets.QMessageBox.warning(self, 'Access denied', 'Super Administrator USB is required.')
            return None, None
        return info, token

    def change_password(self):
        info, token = self._require_selected_and_usb()
        if info is None:
            return
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(f"Change Password — {info['username']}")
        form = QtWidgets.QFormLayout(dlg)
        pwd = QtWidgets.QLineEdit(dlg)
        repeat = QtWidgets.QLineEdit(dlg)
        for field in (pwd, repeat):
            field.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
            field.setMaxLength(8)
        form.addRow('New password (8 characters)', pwd)
        form.addRow('Confirm new password', repeat)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok |
            QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        form.addRow(buttons)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        if len(pwd.text()) != 8 or pwd.text() != repeat.text():
            QtWidgets.QMessageBox.warning(self, 'Invalid password',
                'Enter exactly 8 matching password characters.')
            return
        if QtWidgets.QMessageBox.question(
                self, 'Confirm password change',
                f"Reset the password for {info['username']}?") != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        try:
            # Acquire USB credentials again just before the privileged write.
            token = self.get_usb_token()
            if token is None:
                return
            self.db.change_admin_password(
                self.get_session_id(), *token, info['id'], pwd.text())
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, 'Password unchanged', str(exc))
            return
        QtWidgets.QMessageBox.information(self, 'Password updated',
                                          f"Updated password for {info['username']}.")
        self.reload()

    def update_face_id(self):
        info, token = self._require_selected_and_usb()
        if info is None:
            return
        face_label = info.get('face_label')
        if face_label is None:
            QtWidgets.QMessageBox.warning(self, 'No enrolled face',
                'This account has no existing face ID. Repair enrollment instead.')
            return
        if QtWidgets.QMessageBox.question(
                self, 'Replace enrolled face',
                f"Replace the stored face for {info['username']} (Face ID {face_label})?\n\n"
                'This permanently replaces their old templates. Other faces are preserved.') \
                != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        scan = FaceEnrollmentDialog(self)
        if scan.exec() != QtWidgets.QDialog.DialogCode.Accepted or not scan.samples:
            return
        staged = None
        backup = None
        published = False
        try:
            token = self.get_usb_token()
            if token is None:
                return
            # A fresh admin list also protects against using stale account details.
            current = {row['id']: row for row in
                       self.db.list_admins(self.get_session_id(), *token)}.get(info['id'])
            if not current or current['face_label'] != face_label or not current['is_active']:
                raise ValueError('Account or face ID changed while scanning. Please refresh.')
            staged = build_replacement_model(scan.samples, face_label, MODEL_PATH)
            backup_handle, backup_path = tempfile.mkstemp(
                prefix='.spot-face-backup-', suffix='.yml', dir=str(MODEL_PATH.parent))
            os.close(backup_handle)
            backup = Path(backup_path)
            shutil.copy2(MODEL_PATH, backup)
            publish_model(staged, MODEL_PATH)
            staged = None
            published = True
            # If the audit write fails, roll the original model back.
            self.db.record_admin_face_updated(self.get_session_id(), *token,
                                              info['id'], face_label)
        except Exception as exc:
            if published and backup is not None:
                try:
                    os.replace(backup, MODEL_PATH)
                    backup = None
                except OSError as restore_exc:
                    QtWidgets.QMessageBox.critical(self, 'Manual recovery required',
                        f'Face update failed: {exc}\nBackup restore also failed: {restore_exc}\n'
                        f'Preserve backup file: {backup}')
                    return
            QtWidgets.QMessageBox.warning(self, 'Face ID unchanged', str(exc))
            return
        finally:
            if staged is not None:
                staged.unlink(missing_ok=True)
            if backup is not None and not published:
                backup.unlink(missing_ok=True)
        # Leave a successful old-model backup so administrators can recover
        # safely if the rest of the app did not reload yet.
        if backup is not None:
            try:
                backup.unlink(missing_ok=True)
            except OSError:
                pass
        QtWidgets.QMessageBox.information(self, 'Face updated',
            f"Face ID {face_label} was updated for {info['username']}.\n"
            'Other accounts remain enrolled. Restart/log out the affected account '
            'so the live recognizer reloads the updated model.')
        self.reload()

    def create_admin(self):
        """Create an Administrator only after 24 webcam samples pass validation."""
        token = self.get_usb_token()
        if token is None:
            return
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle('Create Administrator — Step 1 of 2')
        form = QtWidgets.QFormLayout(dlg)
        username = QtWidgets.QLineEdit()
        username.setPlaceholderText('e.g. librarian01')
        full_name = QtWidgets.QLineEdit()
        password = QtWidgets.QLineEdit()
        confirm = QtWidgets.QLineEdit()
        for field in (password, confirm):
            field.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
            field.setMaxLength(8)
        form.addRow('Username', username)
        form.addRow('Full name', full_name)
        form.addRow('Password (exactly 8 characters)', password)
        form.addRow('Confirm password', confirm)
        form.addRow(QtWidgets.QLabel(
            'Next, SPOT will scan and verify the administrator’s face using the webcam.'))
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok |
            QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setText('Next: Scan Face')
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        form.addRow(buttons)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        if len(password.text()) != 8 or password.text() != confirm.text():
            QtWidgets.QMessageBox.warning(self, 'Check password',
                'Password must contain exactly 8 matching characters.')
            return
        entered_username = username.text().strip()
        if not entered_username:
            QtWidgets.QMessageBox.warning(self, 'Missing username', 'Enter an administrator username.')
            return
        scan = FaceEnrollmentDialog(self)
        if scan.exec() != QtWidgets.QDialog.DialogCode.Accepted or not scan.samples:
            return
        staged = None
        admin_id = None
        try:
            token = self.get_usb_token()
            if token is None:
                return
            admins = self.db.list_admins(self.get_session_id(), *token)
            next_label = get_next_label([u['face_label'] for u in admins], MODEL_PATH)
            staged = build_staged_model(scan.samples, next_label, MODEL_PATH)
            admin_id = self.db.create_admin(
                self.get_session_id(), *token, entered_username, password.text(),
                full_name=full_name.text().strip() or None,
                face_label=next_label, is_active=False)
            publish_model(staged, MODEL_PATH)
            staged = None
            self.db.activate_enrolled_admin(self.get_session_id(), *token, admin_id)
        except Exception as exc:
            message = str(exc)
            if admin_id is not None:
                message += ('\n\nAn INACTIVE account may have been created. '
                            'Review it before trying this username again.')
            QtWidgets.QMessageBox.warning(self, 'Enrollment not completed', message)
            return
        finally:
            if staged is not None:
                staged.unlink(missing_ok=True)
        QtWidgets.QMessageBox.information(self, 'Administrator Enrolled',
            f'Created administrator #{admin_id} with face label {next_label}.')
        self.reload()
        self.select_account(admin_id)

    def reload(self):
        token = self.get_usb_token()
        if token is None:
            return
        try:
            admins = self.db.list_admins(self.get_session_id(), *token)
            events = self.db.get_all_activity(self.get_session_id(), *token)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, 'PostgreSQL error', str(exc))
            return
        selected = self.selected_admin_id
        self._account_rows = {int(u['id']): u for u in admins}
        while len(self._cards) < len(admins):
            self._extra_card()
        self._click_ids.clear()
        for index, (frame, name_label, recency_label) in enumerate(self._cards):
            if index >= len(admins):
                frame.hide()
                continue
            user = admins[index]
            admin_id = int(user['id'])
            name_label.setText(str(user['username']))
            name_label.setToolTip(str(user['username']))
            recency_label.setText(format_recency(user.get('last_successful_login')))
            recency_label.setToolTip('Last successful login: ' +
                str(user.get('last_successful_login') or 'Never'))
            self._click_ids[frame] = admin_id
            frame.show()
        self.activity.setRowCount(len(events))
        for index, event in enumerate(events):
            for column, key in enumerate(('actor', 'event_type', 'detail', 'occurred_at')):
                self.activity.setItem(index, column,
                    QtWidgets.QTableWidgetItem(str(event[key] or '')))
        self.activity.resizeColumnsToContents()
        if selected in self._account_rows:
            self.select_account(selected)
        elif admins:
            self.select_account(int(admins[0]['id']))
        else:
            self.selected_admin_id = None
            self._clear_profile()
