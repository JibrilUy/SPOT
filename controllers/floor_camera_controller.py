# controllers/floor_camera_controller.py
"""SPOT camera-page controller, with optional Floor 1 secure-zone tracking.

Existing manual person tracking and non-tracking modes are preserved.
Secure-zone mode adds a toolbar and a transparent click surface to define a polygon.
"""
import json
import os
from pathlib import Path

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt

from assets.clickable_video_label import ClickableVideoLabel
from core.camera_worker import CameraWorker
from core.tracking_worker import TrackingWorker
from core.secure_zone_worker import SecureZoneWorker
from core.qt_image_utils import cv_frame_to_pixmap


class _ZoneClickSurface(QtWidgets.QWidget):
    """Transparent mouse receiver above the video *only while drawing*.

    Some generated Floor 1 pages contain sibling widgets above the clickable
    QLabel in Qt's stacking order. In that situation QLabel.clicked_at never
    fires. A raised receiver on the Floor 1 page intercepts ONLY drawing
    clicks, leaving the actual rendered video and toolbar untouched.
    """
    point_clicked = QtCore.pyqtSignal(QtCore.QPoint)

    def __init__(self, parent):
        super().__init__(parent)
        self.setObjectName("spot_zone_click_surface")
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAutoFillBackground(False)
        # The outline identifies the true clickable video rectangle.
        self.setStyleSheet(
            "QWidget#spot_zone_click_surface {"
            "background: transparent; border: 2px dashed #F5AA2B; }"
        )
        self.hide()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.point_clicked.emit(event.position().toPoint())
            event.accept()
            return
        super().mousePressEvent(event)


class FloorCameraController(QtCore.QObject):
    def __init__(
        self,
        ui,
        floor_num,
        source,
        camera_number=1,
        loop=False,
        model_path="yolo11n.pt",
        enable_tracking=False,
        secure_zone_tracking=False,
        cross_floor_tracking=False,
        shared_target=None,
        secure_zone_file=None,
        video_frame_attr="frame",
        current_camera_attr="current_camera_indicator",
        camera_count_attr="camera_count_indicator",
        floor_count_attr="floor_count_indicator",
        track_button_attr="tracking_button",
        status_label_attr=None,
    ):
        super().__init__()
        if sum((bool(enable_tracking), bool(secure_zone_tracking), bool(cross_floor_tracking))) > 1:
            raise ValueError("Choose only one tracking mode per floor.")
        self.ui = ui
        self.floor_num = floor_num
        self._mode = "idle"  # Existing manual tracking state.
        self._secure_zone_tracking = secure_zone_tracking
        self._cross_floor_tracking = cross_floor_tracking
        self._drawing_zone = False
        self._draft_zone = []
        self._zone_path = (
            Path(secure_zone_file) if secure_zone_file else
            Path(__file__).resolve().parents[1] / "config" / f"floor_{floor_num}_secure_zone.json"
        )
        self._video_frame = getattr(ui, video_frame_attr)
        self._camera_count_label = getattr(ui, camera_count_attr)
        self._floor_count_label = getattr(ui, floor_count_attr)
        self._status_label = getattr(ui, status_label_attr, None) if status_label_attr else None
        getattr(ui, current_camera_attr).setText(str(camera_number))
        self._build_video_label()
        if secure_zone_tracking:
            self._build_zone_toolbar()
            self._build_zone_click_surface()
            self._install_toolbar_anchors()
            self._install_zone_shortcuts()

        if secure_zone_tracking:
            try:
                zone = self._load_zone()
            except (OSError, ValueError, TypeError) as exc:
                print(f"[Secure Zone - Floor {floor_num}] Invalid saved zone: {exc}")
                zone = None
            self.camera_worker = SecureZoneWorker(
                camera_index=source, model_path=model_path, loop=loop, zone=zone,
                shared_target=shared_target, floor_num=floor_num
            )
            self.video_label.clicked_at.connect(self._on_zone_video_click)
            self.draw_zone_button.clicked.connect(lambda checked=False: self._begin_zone())
            self.save_zone_button.clicked.connect(lambda checked=False: self._finish_zone())
            self.clear_zone_button.clicked.connect(lambda checked=False: self._clear_zone())
            self._zone_message("Secure spot loaded. Drawing trails on entry." if zone
                               else "No secure spot saved. Click DRAW ZONE, then click 3+ corners in the video.")
        elif cross_floor_tracking:
            # No drawing controls or video overlays. Floor 1 supplies the
            # white-shirt appearance reference; this feed owns its own trail.
            self.camera_worker = SecureZoneWorker(
                camera_index=source, model_path=model_path, loop=loop, zone=None,
                shared_target=shared_target, floor_num=floor_num,
                follow_reference=True,
            )
            print(f"[SPOT Cross-floor] Floor {floor_num} waiting for Floor 1 white-shirt target")
        else:
            track_button = getattr(ui, track_button_attr, None) if enable_tracking else None
            if enable_tracking and track_button is None:
                print(f"[Tracking - Floor {floor_num}] Missing widget '{track_button_attr}'; tracking off.")
            if track_button is not None:
                self.camera_worker = TrackingWorker(
                    camera_index=source, model_path=model_path, loop=loop
                )
                self.camera_worker.target_event.connect(self._on_target_event)
                self.video_label.clicked_at.connect(self._on_video_clicked)
                track_button.clicked.connect(self._on_track_button)
            else:
                self.camera_worker = CameraWorker(
                    camera_index=source, model_path=model_path, loop=loop
                )
        self.camera_worker.frame_ready.connect(self._on_frame_ready)
        self.camera_worker.count_updated.connect(self._on_count_updated)
        self.camera_worker.error_occurred.connect(self._on_error)

    def _build_video_label(self):
        """Only the video goes inside the Designer-provided frame layout.

        Putting the toolbar in the same layout can leave it beneath existing
        Designer widgets, resulting in visible but unclickable controls.
        """
        layout = self._video_frame.layout()
        if layout is None:
            layout = QtWidgets.QGridLayout(self._video_frame)
        layout.setSpacing(0)
        # Reserve a strip at the top for the separate, raised toolbar.
        layout.setContentsMargins(0, 52 if self._secure_zone_tracking else 0, 0, 0)
        self.video_label = ClickableVideoLabel()
        self.video_label.setScaledContents(True)
        self.video_label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        if isinstance(layout, QtWidgets.QGridLayout):
            layout.addWidget(self.video_label, 0, 0)
        else:
            layout.addWidget(self.video_label)

    def _find_floor_page(self):
        """Find the NESTED Floor 1 QMainWindow, not SPOTApp's main window.

        floor_window is itself a QMainWindow stored in the application's
        QStackedWidget, so QWidget.window() may unexpectedly return SPOTApp.
        """
        widget = self._video_frame.parentWidget()
        while widget is not None:
            if isinstance(widget, QtWidgets.QMainWindow):
                return widget
            widget = widget.parentWidget()
        return None

    def _build_zone_toolbar(self):
        self._floor_page = self._find_floor_page()
        if self._floor_page is None:
            raise RuntimeError("Could not locate Floor 1 page for secure-zone toolbar")

        # A direct child of Floor 1's central widget is above the Designer's
        # other nested frames when raised. Unlike a top-level Qt window, it
        # follows the floor page when changing dashboard/login pages.
        self._zone_host = self._floor_page.centralWidget() or self._floor_page
        self._zone_toolbar = QtWidgets.QFrame(self._zone_host)
        self._zone_toolbar.setObjectName("spot_secure_zone_toolbar")
        self._zone_toolbar.setStyleSheet(
            "QFrame#spot_secure_zone_toolbar {"
            "background-color: rgba(47,0,0,235);"
            "border: 1px solid #F5AA2B; border-radius: 6px; }"
        )
        bar = QtWidgets.QHBoxLayout(self._zone_toolbar)
        bar.setContentsMargins(5, 4, 5, 4)
        bar.setSpacing(5)

        self.draw_zone_button = QtWidgets.QPushButton("DRAW ZONE", self._zone_toolbar)
        self.save_zone_button = QtWidgets.QPushButton("FINISH ZONE", self._zone_toolbar)
        self.clear_zone_button = QtWidgets.QPushButton("CLEAR ZONE", self._zone_toolbar)
        self._zone_status = QtWidgets.QLabel("", self._zone_toolbar)
        self._zone_status.setWordWrap(False)
        self._zone_status.setStyleSheet("color:white; font-size:12px; padding-left:6px;")
        self._zone_status.setMinimumWidth(0)
        for button in (self.draw_zone_button, self.save_zone_button, self.clear_zone_button):
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setMinimumHeight(35)
            button.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            button.setStyleSheet(
                "QPushButton {background:#F5AA2B; color:black; border-radius:5px;"
                "padding:5px 8px; font-weight:bold; border:none;}"
                "QPushButton:hover {background:#FFC35B;}"
                "QPushButton:disabled {background:#92733B; color:#393939;}"
            )
            bar.addWidget(button)
        bar.addWidget(self._zone_status, stretch=1)
        self.save_zone_button.setEnabled(False)  # Enabled after DRAW ZONE.
        self.save_zone_button.setToolTip("Choose DRAW ZONE first, then 3+ corners")
        self._zone_toolbar.setFixedHeight(50)
        self._zone_toolbar.show()
        self._zone_toolbar.raise_()
        QtCore.QTimer.singleShot(0, self._position_zone_toolbar)

    def _build_zone_click_surface(self):
        # Parent and stacking context match the toolbar, but the surface
        # covers the *video* instead of covering the buttons.
        self._zone_click_surface = _ZoneClickSurface(self._zone_host)
        self._zone_click_surface.point_clicked.connect(self._on_zone_surface_click)
        self._zone_click_surface.hide()

    def _on_zone_surface_click(self, point):
        # Map through Qt coordinates, not guessed widget sizes: reliable
        # even when the Designer page has nested/resized frames.
        # IMPORTANT: these widgets are siblings under _zone_host, not
        # ancestor/descendant widgets. QWidget.mapTo(other_widget, point)
        # only accepts an ancestor, and logs
        # "QWidget::mapTo(): parent must be in parent hierarchy" otherwise.
        # Mapping via the global coordinate system supports siblings and
        # arbitrary nesting (including Designer-generated floor pages).
        global_point = self._zone_click_surface.mapToGlobal(point)
        video_point = self.video_label.mapFromGlobal(global_point)
        width = self.video_label.width()
        height = self.video_label.height()
        if width <= 0 or height <= 0:
            self._zone_message("Video is not laid out yet; try again once it appears.")
            return
        if not self.video_label.rect().contains(video_point):
            return
        nx = max(0.0, min(1.0, video_point.x() / width))
        ny = max(0.0, min(1.0, video_point.y() / height))
        self._on_zone_video_click(nx, ny)

    def _install_toolbar_anchors(self):
        # Moving/resizing the video or a surrounding layout repositions the
        # toolbar, but no expensive work is performed per video frame.
        w = self.video_label
        while w is not None:
            w.installEventFilter(self)
            if w is self._zone_host:
                break
            w = w.parentWidget()
        self._position_zone_toolbar()

    def _install_zone_shortcuts(self):
        # Alternative while troubleshooting any other Designer overlay.
        self._zone_shortcuts = []
        for keys, callback in (
            ("Ctrl+Alt+D", self._begin_zone),
            ("Ctrl+Alt+F", self._finish_zone),
            ("Ctrl+Alt+C", self._clear_zone),
        ):
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(keys), self._floor_page)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(callback)
            self._zone_shortcuts.append(shortcut)

    def eventFilter(self, watched, event):
        if self._secure_zone_tracking and event.type() in (
            QtCore.QEvent.Type.Show,
            QtCore.QEvent.Type.Resize,
            QtCore.QEvent.Type.Move,
            QtCore.QEvent.Type.LayoutRequest,
        ):
            # Defer until Qt has finalized the layout for this event.
            QtCore.QTimer.singleShot(0, self._position_zone_toolbar)
        return super().eventFilter(watched, event)

    def _position_zone_toolbar(self):
        if not getattr(self, "_zone_toolbar", None):
            return
        # mapTo is important: the video frame may be nested in multiple
        # layouts, and Floor 1 itself lives inside a QStackedWidget.
        top_left = self._video_frame.mapTo(self._zone_host, QtCore.QPoint(0, 0))
        width = max(1, self._video_frame.width())
        rect = QtCore.QRect(top_left.x(), top_left.y(), width, 50)
        if self._zone_toolbar.geometry() != rect:
            self._zone_toolbar.setGeometry(rect)
        self._zone_status.setVisible(width >= 590)

        # The QLabel (not the full camera frame) defines the actual rendered
        # video rectangle. A drawing receiver is positioned above any other
        # overlapping Designer widgets, but below the toolbar. Reserve the
        # upper strip explicitly so controls never lose their clicks.
        if hasattr(self, "_zone_click_surface"):
            origin = self.video_label.mapTo(self._zone_host, QtCore.QPoint(0, 0))
            video_rect = QtCore.QRect(origin, self.video_label.size())
            frame_origin = self._video_frame.mapTo(self._zone_host, QtCore.QPoint(0, 0))
            frame_rect = QtCore.QRect(frame_origin, self._video_frame.size())
            safe_area = frame_rect.adjusted(0, 52, 0, 0)
            active_rect = video_rect.intersected(safe_area)
            if active_rect.isValid() and not active_rect.isEmpty():
                if self._zone_click_surface.geometry() != active_rect:
                    self._zone_click_surface.setGeometry(active_rect)
                if self._drawing_zone:
                    self._zone_click_surface.show()
                    self._zone_click_surface.raise_()
            else:
                self._zone_click_surface.hide()
        self._zone_toolbar.raise_()

    def _load_zone(self):
        if not self._zone_path.exists():
            return None
        data = json.loads(self._zone_path.read_text(encoding="utf-8"))
        return data["points"]

    def _save_zone(self, points):
        self._zone_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._zone_path.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps({"version": 1, "points": points}, indent=2),
                           encoding="utf-8")
            os.replace(tmp, self._zone_path)
        finally:
            if tmp.exists():
                tmp.unlink()

    def _zone_message(self, text):
        print(f"[Secure Zone - Floor {self.floor_num}] {text}")
        self._zone_status.setText(text)
        if self._status_label is not None:
            self._status_label.setText(text)

    def _begin_zone(self):
        self._drawing_zone = True
        self._draft_zone = []
        self.camera_worker.set_preview([])
        self.video_label.setCursor(Qt.CursorShape.CrossCursor)
        self._position_zone_toolbar()
        self.save_zone_button.setEnabled(True)
        self._zone_message("Click 3–20 corners around the secure spot, then FINISH ZONE.")

    def _on_zone_video_click(self, nx, ny):
        if not self._drawing_zone:
            return
        if len(self._draft_zone) >= 20:
            self._zone_message("Maximum 20 corners. Press FINISH ZONE.")
            return
        self._draft_zone.append([max(0.0, min(1.0, float(nx))),
                                 max(0.0, min(1.0, float(ny)))])
        self.camera_worker.set_preview(self._draft_zone)
        self._zone_message(f"{len(self._draft_zone)} corners. Need 3+; press FINISH ZONE to save.")

    def _finish_zone(self):
        if len(self._draft_zone) < 3:
            QtWidgets.QMessageBox.warning(self._video_frame, "Secure spot",
                                          "Click at least three corners on the video.")
            return
        try:
            # Validation occurs in the tracker; the existing zone is unaffected if invalid.
            self.camera_worker.set_zone(self._draft_zone)
            self._save_zone(self._draft_zone)
        except (ValueError, OSError) as exc:
            QtWidgets.QMessageBox.warning(self._video_frame, "Invalid secure spot", str(exc))
            return
        self._drawing_zone = False
        self._draft_zone = []
        self.camera_worker.set_preview([])
        self.video_label.unsetCursor()
        self._zone_click_surface.hide()
        self.save_zone_button.setEnabled(False)
        self._zone_message("Secure spot saved. Entering people are tracked automatically.")

    def _clear_zone(self):
        choice = QtWidgets.QMessageBox.question(
            self._video_frame, "Clear secure spot?",
            "Remove the secure spot and all active tracking trails for this floor?"
        )
        if choice != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        try:
            if self._zone_path.exists():
                self._zone_path.unlink()
        except OSError as exc:
            QtWidgets.QMessageBox.warning(self._video_frame, "Can't clear zone", str(exc))
            return
        self.camera_worker.set_zone(None)
        self.camera_worker.set_preview([])
        self._draft_zone = []
        self._drawing_zone = False
        self.save_zone_button.setEnabled(False)
        self.video_label.unsetCursor()
        self._zone_click_surface.hide()
        self._zone_message("Secure spot cleared. Click DRAW ZONE to configure a new one.")

    def start(self):
        if not self.camera_worker.isRunning():
            self.camera_worker.start()

    def stop(self):
        if self.camera_worker.isRunning():
            self.camera_worker.stop()

    def _on_frame_ready(self, frame):
        self.video_label.setPixmap(cv_frame_to_pixmap(frame))

    def _on_count_updated(self, count):
        txt = str(count)
        self._camera_count_label.setText(txt)
        self._floor_count_label.setText(txt)

    def _on_error(self, message):
        print(f"[Camera Error - Floor {self.floor_num}] {message}")
        if self._secure_zone_tracking:
            self._zone_message(message)

    # Existing click-to-track behavior for other floors remains available.
    def _on_track_button(self, *_):
        if self._mode == "idle":
            self._mode = "selecting"
            self.video_label.setCursor(Qt.CursorShape.CrossCursor)
            self._set_status("Click a person in the video to track them.")
        elif self._mode == "selecting":
            self._mode = "idle"
            self.video_label.unsetCursor()
            self._set_status("Selection cancelled.")
        else:
            self.camera_worker.clear_target()
            self._mode = "idle"
            self._set_status("Tracking cleared.")

    def _on_video_clicked(self, nx, ny):
        if self._mode == "selecting":
            self.camera_worker.select_at(nx, ny)

    def _on_target_event(self, name):
        if name == "selected":
            self._mode = "tracking"
            self.video_label.unsetCursor()
            self._set_status("Tracking the selected person.")
        elif name == "no_person":
            self._set_status("No person at that spot. Click directly on a person.")
        elif name == "lost":
            self._set_status("Person lost, searching...")
        elif name == "reacquired":
            self._set_status("Tracking the selected person.")
        elif name == "ended":
            self._set_status("Person not found. Press the button to clear the path.")

    def _set_status(self, text):
        print(f"[Tracking - Floor {self.floor_num}] {text}")
        if self._status_label is not None:
            self._status_label.setText(text)
