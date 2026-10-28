"""Live webcam enrollment dialog used only by USB-authenticated super admins.

No Qt Designer changes required. Keeps all camera and face-quality work out
of the database and normal CCTV camera/tracking workers.
"""
import os

import cv2
from PyQt6 import QtCore, QtGui, QtWidgets

from core.face_enrollment import QualityFaceCapture, PROMPTS
from core.qt_image_utils import cv_frame_to_pixmap


class FaceEnrollmentDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, camera_index=None):
        super().__init__(parent)
        self.setWindowTitle('SPOT — Enroll Administrator Face')
        self.resize(740, 690)
        self.setModal(True)
        self.setMinimumWidth(620)
        self.samples = None
        self.capture = QualityFaceCapture()
        self.camera_index = int(os.getenv('SPOT_ENROLL_CAMERA_INDEX', '0')) if camera_index is None else camera_index
        self.cap = None
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(80)
        self.timer.timeout.connect(self._on_frame)

        layout = QtWidgets.QVBoxLayout(self)
        title = QtWidgets.QLabel('Administrator Face Enrollment')
        title.setStyleSheet('font-size: 20pt; font-weight: 700;')
        layout.addWidget(title)
        layout.addWidget(QtWidgets.QLabel('Only the person receiving this account should be in the camera. '
                                          'Use even lighting and remove anything covering the face.'))
        self.preview = QtWidgets.QLabel('Camera will start after you click Start Scan')
        self.preview.setMinimumSize(580, 360)
        self.preview.setStyleSheet('background: #161616; color: white; border-radius: 8px;')
        self.preview.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.preview)
        self.pose = QtWidgets.QLabel(PROMPTS[0])
        self.pose.setStyleSheet('font-size: 13pt; font-weight: bold;')
        layout.addWidget(self.pose)
        self.feedback = QtWidgets.QLabel('No samples collected.')
        self.feedback.setWordWrap(True)
        layout.addWidget(self.feedback)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, self.capture.required)
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        controls = QtWidgets.QHBoxLayout()
        self.start_button = QtWidgets.QPushButton('Start Scan')
        self.start_button.clicked.connect(self.start_scan)
        controls.addWidget(self.start_button)
        self.cancel_button = QtWidgets.QPushButton('Cancel')
        self.cancel_button.clicked.connect(self.reject)
        controls.addWidget(self.cancel_button)
        self.save_button = QtWidgets.QPushButton('Use This Face Scan')
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self._accept_scan)
        controls.addWidget(self.save_button)
        layout.addLayout(controls)

    def _open_camera(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if isinstance(self.camera_index, int) and os.name == 'nt':
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            self._release_camera()
            QtWidgets.QMessageBox.warning(self, 'Webcam unavailable',
                'Could not open the face enrollment webcam. Close other apps using it, '
                'or change SPOT_ENROLL_CAMERA_INDEX in .env.')
            return False
        return True

    def _release_camera(self):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def start_scan(self):
        self._release_camera()
        self.capture.reset()
        self.samples = None
        self.progress.setValue(0)
        self.pose.setText(self.capture.prompt)
        self.feedback.setText('Checking image quality. You can adjust position while scanning.')
        self.save_button.setEnabled(False)
        if not self._open_camera():
            return
        self.start_button.setText('Restart Scan')
        self.timer.start()

    def _on_frame(self):
        if self.cap is None:
            return
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.feedback.setText('Camera stopped delivering frames. Restart the scan.')
            self._release_camera()
            return
        box, feedback, accepted = self.capture.process_frame(frame)
        if box:
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 190, 0) if 'Good' in feedback or accepted else (0, 0, 255), 2)
        preview_pixmap = cv_frame_to_pixmap(frame)
        self.preview.setPixmap(preview_pixmap.scaled(
            self.preview.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation))
        self.feedback.setText(feedback)
        self.progress.setValue(len(self.capture.samples))
        self.pose.setText(self.capture.prompt)
        if self.capture.complete:
            self._release_camera()
            self.feedback.setText('Captured 24 quality-checked samples. Click Use This Face Scan to validate recognition.')
            self.save_button.setEnabled(True)

    def _accept_scan(self):
        from core.face_enrollment import validate_capture
        try:
            outcome = validate_capture(self.capture.samples)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, 'Scan quality insufficient',
                f'{exc}\n\nClick Restart Scan and try again.')
            self.save_button.setEnabled(False)
            return
        self.samples = list(self.capture.samples)
        self.feedback.setText(f'Validation passed: {outcome["passing"]}/{outcome["held_out"]} held-out face images recognized.')
        self.accept()

    def reject(self):
        self.samples = None
        self._release_camera()
        super().reject()

    def closeEvent(self, event):
        self._release_camera()
        super().closeEvent(event)
