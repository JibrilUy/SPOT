"""Real-time video-file pacing for SPOT tracked floor feeds.

When YOLO is slower than the source FPS, stale video frames are skipped via
VideoCapture.grab() rather than making the entire recording play in slow motion.
The output can still look choppy on an overloaded CPU; skipping frames cannot
make slow inference generate more annotated frames per second.
"""
import os
import time

import cv2

from core.camera_worker import CameraWorker
from core.secure_zone_tracker import SecureZoneTracker


class SecureZoneWorker(CameraWorker):
    MAX_CATCHUP_FRAMES = 90

    def __init__(self, camera_index=0, model_path="yolo11n.pt", loop=False, zone=None,
                 shared_target=None, floor_num=1, follow_reference=False):
        # Change without touching Python code:
        # PowerShell: $env:SPOT_FLOOR1_IMGSZ = "416"
        try:
            imgsz = int(os.getenv("SPOT_CROSSFLOOR_IMGSZ" if follow_reference
                                  else "SPOT_FLOOR1_IMGSZ",
                                  "416" if follow_reference else "512"))
        except ValueError:
            imgsz = 512
        if imgsz not in (320, 384, 416, 480, 512, 576, 640):
            imgsz = 512
        self._floor_num = floor_num
        self._shared_target = shared_target
        self._secure_tracker = SecureZoneTracker(
            model_path=model_path, zone=zone, imgsz=imgsz,
            shared_target=shared_target, floor_num=floor_num,
            follow_reference=follow_reference)

        super().__init__(camera_index=camera_index, model_path=model_path,
                         loop=loop, detector=self._secure_tracker)

    def set_zone(self, vertices):
        self._secure_tracker.set_zone(vertices)

    def set_preview(self, vertices):
        self._secure_tracker.set_preview(vertices)

    def get_zone(self):
        return self._secure_tracker.get_zone()

    def run(self):
        # Each new playback session begins with empty local history. Floor 1
        # is authoritative for the current shared clothing reference.
        if self._floor_num == 1 and self._shared_target is not None:
            self._shared_target.clear()
        self._secure_tracker.reset_tracking()
        is_file = not isinstance(self._camera_index, int)
        cap = (cv2.VideoCapture(self._camera_index) if is_file else
               cv2.VideoCapture(self._camera_index, cv2.CAP_DSHOW))
        if not cap.isOpened():
            self.error_occurred.emit(f"Could not open video source: {self._camera_index!r}")
            return

        fps = float(cap.get(cv2.CAP_PROP_FPS)) if is_file else 0.0
        if not (0.1 <= fps <= 240.0):
            fps = 0.0
        frame_index = 0
        started_at = time.monotonic()
        self._running = True
        self._last_count = 0
        self._missed_frames = 0
        just_rewound = False

        def rewind():
            nonlocal frame_index, started_at, just_rewound
            if not (is_file and self._loop) or just_rewound:
                return False
            # The model's ByteTrack IDs and all prior trails must reset as
            # soon as the source video starts over.
            if not cap.set(cv2.CAP_PROP_POS_FRAMES, 0):
                self.error_occurred.emit(f"Couldn't rewind Floor {self._floor_num} video.")
                return False
            self._secure_tracker.reset_tracking()
            self._last_count = 0
            self._missed_frames = 0
            frame_index = 0
            started_at = time.monotonic()
            just_rewound = True
            return True

        try:
            while self._running and cap.isOpened():
                if fps:
                    elapsed = time.monotonic() - started_at
                    due = int(elapsed * fps)
                    if due < frame_index:
                        wait_ms = int(1000.0 * (frame_index / fps - elapsed))
                        self.msleep(max(1, min(25, wait_ms)))
                        continue
                    skip = min(max(0, due - frame_index), self.MAX_CATCHUP_FRAMES)
                    missed_end = False
                    for _ in range(skip):
                        if not cap.grab():
                            missed_end = True
                            break
                        frame_index += 1
                    if missed_end:
                        if rewind():
                            continue
                        break

                ok, frame = cap.read()
                if not ok:
                    if rewind():
                        continue
                    break
                just_rewound = False
                frame_index += 1

                try:
                    count, annotated = self._secure_tracker.detect(frame)
                except Exception as exc:
                    self.error_occurred.emit(f"Floor {self._secure_tracker._floor_num} tracking failed: {exc}")
                    break

                # Keep the original SPOT count smoothing behavior.
                if count == 0 and self._last_count > 0:
                    self._missed_frames += 10
                    if self._missed_frames <= self._max_missed_frames:
                        count = self._last_count
                    else:
                        self._last_count = count
                        self._missed_frames = 0
                else:
                    self._last_count = count
                    self._missed_frames = 0
                self.frame_ready.emit(annotated)
                self.count_updated.emit(count)
        finally:
            cap.release()
            self._running = False
