"""SPOT secure-zone / cross-floor tracker: persistent trails and occlusion recovery.

Compatible with SPOT's CameraWorker.detect(frame) contract.  This is a
camera-local *motion* tracker, not facial recognition.  It cannot observe
someone while they are fully hidden: orange boxes/paths are predictions.
"""
from pathlib import Path
from threading import Lock
import time

import cv2
import numpy as np


class SecureZoneTracker:
    MIN_AREA_FRACTION = 0.0002
    MIN_STEP_PIXELS = 3.0
    MAX_RECENT_PATH_POINTS = 2000  # Old pixels remain on persistent trail canvas.
    MAX_OCCLUSION_SECONDS = 6.0   # Estimate only briefly while out of sight.
    REACQUIRE_SECONDS = 12.0       # Try to reconnect a changed ByteTrack ID.
    MOTION_DECAY_SECONDS = 3.0     # Cap drift when last velocity is unreliable.
    HISTOGRAM_DISTANCE_LIMIT = 0.56
    CROSS_FLOOR_APPEARANCE_LIMIT = 0.63
    CROSS_FLOOR_AMBIGUITY_MARGIN = 0.10
    ZONE_COLOR = (0, 210, 255)    # yellow (BGR)
    TRACK_COLOR = (0, 245, 90)    # fallback green
    TRACK_COLORS = (
        (0, 245, 90), (255, 175, 0), (220, 90, 245),
        (65, 215, 255), (255, 120, 75), (85, 155, 255),
    )  # distinct BGR colors for different people
    DETECT_COLOR = (140, 140, 140)
    ESTIMATE_COLOR = (0, 165, 255)  # orange for guessed position
    PREVIEW_COLOR = (255, 150, 0)
    LAST_SEEN_COLOR = (180, 180, 180)

    def __init__(self, model_path="yolo11n.pt", model=None, zone=None,
                 imgsz=512, clock=None, shared_target=None, floor_num=1,
                 follow_reference=False):
        if model is None:
            from ultralytics import YOLO
            model = YOLO(model_path)
        self.model = model
        self.imgsz = int(imgsz)
        self._clock = clock or time.monotonic
        self._shared_target = shared_target
        self._floor_num = int(floor_num)
        self._follow_reference = bool(follow_reference)
        if self._follow_reference and self._shared_target is None:
            raise ValueError("Cross-floor tracking requires a SharedTargetRegistry")
        tracker_config = Path(__file__).resolve().parents[1] / "config" / "spot_bytetrack_floor1.yaml"
        self._tracker_config = str(tracker_config) if tracker_config.is_file() else "bytetrack.yaml"

        self._lock = Lock()
        self._zone = None
        self._preview = []
        self._tracked = {}        # stable SPOT ID -> track, NEVER pruned until rewind/reset
        self._external_ids = {}   # ByteTrack ID -> stable SPOT ID
        self._unarmed = {}        # id -> (last foot, timestamp) before zone entry
        self._next_id = 1
        self._frame_number = 0
        self._trail_canvas = None # rasterized paths avoid redrawing years of history
        self._fps_last = None
        self._fps_ema = None
        self.set_zone(zone)

    @staticmethod
    def _validated_polygon(vertices):
        if vertices is None or vertices == []:
            return None
        if not isinstance(vertices, (list, tuple)) or not 3 <= len(vertices) <= 20:
            raise ValueError("Secure spot needs 3–20 corner points.")
        pts = []
        for item in vertices:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError("Each corner needs two coordinates.")
            x, y = float(item[0]), float(item[1])
            if not np.isfinite(x) or not np.isfinite(y) or not (0 <= x <= 1 and 0 <= y <= 1):
                raise ValueError("Corner coordinates must be between 0 and 1.")
            pts.append((x, y))
        area = abs(sum(pts[i][0] * pts[(i + 1) % len(pts)][1] -
                       pts[(i + 1) % len(pts)][0] * pts[i][1]
                       for i in range(len(pts)))) / 2
        if area < SecureZoneTracker.MIN_AREA_FRACTION:
            raise ValueError("Secure spot is too small. Draw a larger region.")

        def orientation(a, b, c):
            return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

        def intersects(a, b, c, d):
            return (orientation(a, b, c) * orientation(a, b, d) < 0 and
                    orientation(c, d, a) * orientation(c, d, b) < 0)

        n = len(pts)
        for i in range(n):
            for j in range(i + 1, n):
                if j == i + 1 or (i == 0 and j == n - 1):
                    continue
                if intersects(pts[i], pts[(i + 1) % n], pts[j], pts[(j + 1) % n]):
                    raise ValueError("Zone outline crosses itself. Try again.")
        return tuple(pts)

    def set_zone(self, vertices):
        polygon = self._validated_polygon(vertices)
        with self._lock:
            # Editing the secure zone must NEVER delete previously recorded trails.
            # The only automatic history reset is an actual source-video rewind.
            self._zone = polygon

    def get_zone(self):
        with self._lock:
            return [list(p) for p in self._zone] if self._zone else None

    def set_preview(self, vertices):
        preview = [(max(0.0, min(1.0, float(x))),
                    max(0.0, min(1.0, float(y)))) for x, y in vertices]
        with self._lock:
            self._preview = preview

    def _clear_tracks(self):
        self._tracked.clear()
        self._external_ids.clear()
        self._unarmed.clear()
        self._next_id = 1
        self._frame_number = 0
        self._trail_canvas = None
        self._fps_last = None
        self._fps_ema = None

    def reset_tracking(self):
        """Called on video-file rewind, not after a temporary detection loss."""
        with self._lock:
            self._clear_tracks()
            predictor = getattr(self.model, "predictor", None)
        for tracker in getattr(predictor, "trackers", ()) if predictor is not None else ():
            if hasattr(tracker, "reset"):
                tracker.reset()

    @staticmethod
    def _pixels(vertices, width, height):
        return np.asarray([(round(x * (width - 1)), round(y * (height - 1)))
                           for x, y in vertices], dtype=np.int32)

    @staticmethod
    def _foot(box):
        x1, y1, x2, y2 = box
        return (int((x1 + x2) / 2), int(y2))

    @staticmethod
    def _extract(result):
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return [], []
        coords = boxes.xyxy.cpu().numpy().tolist()
        ids_obj = getattr(boxes, "id", None)
        if ids_obj is None:
            return coords, [None] * len(coords)
        ids = ids_obj.int().cpu().tolist()
        if len(ids) != len(coords):
            return coords, [None] * len(coords)
        return coords, [int(i) for i in ids]

    @staticmethod
    def _crossed_zone(before, after, polygon):
        """A sparse detection stream may step right across a narrow zone."""
        poly = polygon.reshape(-1, 2).astype(float)
        p = np.asarray(before, dtype=float)
        q = np.asarray(after, dtype=float)
        if cv2.pointPolygonTest(polygon.astype(np.float32), tuple(p), False) >= 0:
            return True
        if cv2.pointPolygonTest(polygon.astype(np.float32), tuple(q), False) >= 0:
            return True
        r = q - p
        for i in range(len(poly)):
            a, b = poly[i], poly[(i + 1) % len(poly)]
            s = b - a
            denom = r[0] * s[1] - r[1] * s[0]
            if abs(denom) < 1e-8:
                continue
            ap = a - p
            t = (ap[0] * s[1] - ap[1] * s[0]) / denom
            u = (ap[0] * r[1] - ap[1] * r[0]) / denom
            if 0 <= t <= 1 and 0 <= u <= 1:
                return True
        return False

    @staticmethod
    def _histogram(frame, box):
        """Coarse clothing appearance signature to help reconnect occluded tracks."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        left = max(0, min(w, int(x1 + width * 0.18)))
        right = max(0, min(w, int(x2 - width * 0.18)))
        top = max(0, min(h, int(y1 + height * 0.23)))
        bottom = max(0, min(h, int(y1 + height * 0.73)))
        if right - left < 10 or bottom - top < 10:
            return None
        roi = frame[top:bottom, left:right]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 1.0, 0, cv2.NORM_L1)
        return hist

    @staticmethod
    def _clothing_descriptor(frame, box):
        """A rough upper-body + trousers HSV signature; NOT identity recognition.

        We test the torso for a low-saturation bright shirt (white/gray), and
        use the lower-body colors as a second cue when matching floor videos.
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = (float(v) for v in box)
        bw, bh = x2 - x1, y2 - y1
        if bw < 20 or bh < 50:
            return None

        def region(left, right, top, bottom):
            a = max(0, min(w, int(x1 + bw * left)))
            b = max(0, min(w, int(x1 + bw * right)))
            c = max(0, min(h, int(y1 + bh * top)))
            d = max(0, min(h, int(y1 + bh * bottom)))
            if b-a < 8 or d-c < 8:
                return None
            return cv2.cvtColor(frame[c:d, a:b], cv2.COLOR_BGR2HSV)

        torso = region(.23, .77, .24, .59)
        pants = region(.25, .75, .62, .91)
        if torso is None or pants is None:
            return None
        sat = torso[:, :, 1]
        val = torso[:, :, 2]
        white_fraction = float(np.mean((sat <= 85) & (val >= 130)))

        def svhist(roi):
            hist = cv2.calcHist([roi], [1, 2], None, [8, 8], [0, 256, 0, 256])
            return cv2.normalize(hist, None, alpha=1.0, norm_type=cv2.NORM_L1)

        return {"shirt": svhist(torso), "pants": svhist(pants),
                "white_fraction": white_fraction}

    @staticmethod
    def _clothing_distance(a, b):
        if a is None or b is None:
            return float("inf")
        shirt = cv2.compareHist(a["shirt"], b["shirt"], cv2.HISTCMP_BHATTACHARYYA)
        pants = cv2.compareHist(a["pants"], b["pants"], cv2.HISTCMP_BHATTACHARYYA)
        return (0.60 * shirt + 0.35 * pants +
                0.18 * abs(a["white_fraction"] - b["white_fraction"]))

    def _best_reference_candidate(self, descriptors, reference):
        """Select ONE convincing person, or refuse ambiguous white-shirt matches."""
        if reference is None:
            return None
        qualified = []
        for index, sample in enumerate(descriptors):
            if sample is None or sample["white_fraction"] < 0.22:
                continue
            score = self._clothing_distance(sample, reference["descriptor"])
            if score <= self.CROSS_FLOOR_APPEARANCE_LIMIT:
                qualified.append((score, index))
        qualified.sort()
        if not qualified:
            return None
        if (len(qualified) > 1 and
                qualified[1][0] - qualified[0][0] < self.CROSS_FLOOR_AMBIGUITY_MARGIN):
            return None
        return qualified[0][1]

    def _display_id(self, stable_id):
        if self._follow_reference and self._shared_target is not None:
            ref = self._shared_target.snapshot()
            if ref is not None:
                return ref["track_id"]
        return stable_id

    def _prediction(self, item, now):
        dt = max(0.0, now - item["last_seen_time"])
        # Integrate decaying velocity, so the estimated person stops drifting
        # indefinitely once a pillar fully hides them. This is NOT evidence.
        dt = min(dt, self.MAX_OCCLUSION_SECONDS)
        travel = self.MOTION_DECAY_SECONDS * (1 - np.exp(-dt / self.MOTION_DECAY_SECONDS))
        dx, dy = item["velocity"] * travel
        return (item["foot"][0] + dx, item["foot"][1] + dy)

    def _possible_reacquisition(self, foot, hist, now, used):
        """Match a *new* ByteTrack ID only to a temporarily missing track."""
        best_id, best_score = None, float("inf")
        for stable_id, item in self._tracked.items():
            if stable_id in used:
                continue
            dt = now - item["last_seen_time"]
            if dt <= 0 or dt > self.REACQUIRE_SECONDS:
                continue
            # Only orphaned tracks: an already-matched ID cannot be stolen.
            if item["seen_frame"] == self._frame_number:
                continue
            predicted = self._prediction(item, now)
            distance = float(np.hypot(foot[0] - predicted[0],
                                      foot[1] - predicted[1]))
            box = item["box"]
            height = box[3] - box[1]
            gate = max(65.0, height * (1.1 + 0.15 * dt))
            if distance > gate:
                continue
            old_hist = item["hist"]
            # A new tracker ID must have usable, similar appearance so it
            # cannot easily steal a different person's old path.
            if hist is None or old_hist is None:
                continue
            similarity_distance = cv2.compareHist(
                old_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
            if similarity_distance > self.HISTOGRAM_DISTANCE_LIMIT:
                continue
            score = distance / gate + similarity_distance
            if score < best_score:
                best_id, best_score = stable_id, score
        return best_id

    def _mark_seen(self, stable_id, external_id, box, foot, hist, now):
        item = self._tracked[stable_id]
        gap_seconds = max(0.0, now - item["last_seen_time"])
        dt = max(0.03, gap_seconds)
        v = np.array([(foot[0] - item["foot"][0]) / dt,
                      (foot[1] - item["foot"][1]) / dt], dtype=float)
        # Clamp outliers from tracker ID switches or skipped video frames.
        speed_limit = max(100.0, (box[3] - box[1]) * 5.0)
        speed = float(np.linalg.norm(v))
        if speed > speed_limit:
            v *= speed_limit / speed
        item["velocity"] = 0.5 * item["velocity"] + 0.5 * v
        item["box"] = tuple(box)
        item["foot"] = foot
        item["hist"] = hist if hist is not None else item["hist"]
        item["last_seen_time"] = now
        item["prior_seen_frame"] = item["seen_frame"]
        item["seen_frame"] = self._frame_number
        item["yolo_id"] = external_id
        if external_id is not None:
            self._external_ids[external_id] = stable_id
        # This segment was not actually observed: show a dashed estimated
        # connection if consecutive analyzed frames are far apart.
        self._append_path(item, foot, inferred=(gap_seconds > 0.6 or
                                                    self._frame_number - item.get("prior_seen_frame", self._frame_number) > 1))

    def _append_path(self, item, point, inferred=False):
        path = item["path"]
        if path and np.hypot(point[0] - path[-1][0], point[1] - path[-1][1]) < self.MIN_STEP_PIXELS:
            return
        if path and self._trail_canvas is not None:
            if inferred:
                self._dashed_line(self._trail_canvas, path[-1], point,
                                  self.ESTIMATE_COLOR, thickness=2)
            else:
                cv2.line(self._trail_canvas, path[-1], point, item.get("color", self.TRACK_COLOR),
                         3, cv2.LINE_AA)
        path.append(point)
        # All historic line pixels are already in _trail_canvas. Retain only
        # recent points in RAM: otherwise long recordings grow indefinitely.
        if len(path) > self.MAX_RECENT_PATH_POINTS:
            del path[:len(path) - self.MAX_RECENT_PATH_POINTS]

    @staticmethod
    def _dashed_line(frame, a, b, color, thickness=2, dash=10):
        distance = int(np.hypot(b[0] - a[0], b[1] - a[1]))
        if distance <= 0:
            return
        for start in range(0, distance, dash * 2):
            end = min(distance, start + dash)
            p = (int(a[0] + (b[0] - a[0]) * start / distance),
                 int(a[1] + (b[1] - a[1]) * start / distance))
            q = (int(a[0] + (b[0] - a[0]) * end / distance),
                 int(a[1] + (b[1] - a[1]) * end / distance))
            cv2.line(frame, p, q, color, thickness, cv2.LINE_AA)

    @staticmethod
    def _dashed_rectangle(frame, box, color, thickness=2, dash=12):
        x1, y1, x2, y2 = (int(v) for v in box)
        h, w = frame.shape[:2]
        x1, x2 = max(0, min(w - 1, x1)), max(0, min(w - 1, x2))
        y1, y2 = max(0, min(h - 1, y1)), max(0, min(h - 1, y2))
        for a, b in (((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)),
                     ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))):
            distance = int(np.hypot(b[0] - a[0], b[1] - a[1]))
            if distance == 0:
                continue
            for start in range(0, distance, dash * 2):
                end = min(distance, start + dash)
                p = (int(a[0] + (b[0]-a[0]) * start / distance),
                     int(a[1] + (b[1]-a[1]) * start / distance))
                q = (int(a[0] + (b[0]-a[0]) * end / distance),
                     int(a[1] + (b[1]-a[1]) * end / distance))
                cv2.line(frame, p, q, color, thickness, cv2.LINE_AA)

    def detect(self, frame):
        height, width = frame.shape[:2]
        now = float(self._clock())
        with self._lock:
            zone = self._zone
            preview = tuple(self._preview)

        # Image size 512 trades some very small-person accuracy for speed.
        # persist=True is essential for ByteTrack IDs across video frames.
        results = self.model.track(frame, persist=True, classes=[0],
                                   tracker=self._tracker_config,
                                   conf=0.15, imgsz=self.imgsz, verbose=False)
        boxes, ids = self._extract(results[0])
        self._frame_number += 1
        if self._trail_canvas is None or self._trail_canvas.shape != frame.shape:
            self._trail_canvas = np.zeros_like(frame)

        region = self._pixels(zone, width, height) if zone else None
        # All histograms must be read from the original video BEFORE drawing.
        signatures = [self._histogram(frame, b) for b in boxes]
        foots = [self._foot(box) for box in boxes]
        descriptors = ([self._clothing_descriptor(frame, b) for b in boxes]
                       if self._shared_target is not None else [None] * len(boxes))
        reference = (self._shared_target.snapshot()
                     if self._shared_target is not None else None)
        best_ref_index = (self._best_reference_candidate(descriptors, reference)
                          if self._follow_reference else None)
        matches = {}
        used = set()

        # First pass: preserve stable IDs already tied to ByteTrack IDs.
        for index, external_id in enumerate(ids):
            if external_id is None:
                continue
            # Other similar-looking people must not steal a shared track.
            if self._follow_reference and best_ref_index != index:
                # Keep an already-confirmed continuous ByteTrack ID only
                # when this detection has at least SOME clothing evidence.
                if (self._external_ids.get(external_id) is None or
                    descriptors[index] is None or
                    reference is None or
                    self._clothing_distance(descriptors[index],
                                             reference["descriptor"]) > 0.78):
                    continue
            stable_id = self._external_ids.get(external_id)
            if stable_id is None or stable_id in used:
                continue
            item = self._tracked.get(stable_id)
            if item is None or now - item["last_seen_time"] > self.REACQUIRE_SECONDS:
                self._external_ids.pop(external_id, None)
                continue
            # ByteTrack may recycle an ID or swap identities across a long
            # occlusion. Only trust its old mapping on a long gap if both
            # appearance and geometry are still plausible.
            gap = now - item["last_seen_time"]
            if gap > 0.8:
                signature = self._histogram(frame, boxes[index])
                old_signature = item["hist"]
                if signature is None or old_signature is None:
                    self._external_ids.pop(external_id, None)
                    continue
                appearance = cv2.compareHist(old_signature, signature, cv2.HISTCMP_BHATTACHARYYA)
                prediction = self._prediction(item, now)
                distance = float(np.hypot(foots[index][0] - prediction[0], foots[index][1] - prediction[1]))
                gate = max(65.0, (item["box"][3] - item["box"][1]) * (1.1 + 0.15 * gap))
                if appearance > self.HISTOGRAM_DISTANCE_LIMIT or distance > gate:
                    self._external_ids.pop(external_id, None)
                    continue
            matches[index] = stable_id
            used.add(stable_id)

        # Second pass: try to reunite a changed ByteTrack ID with an old
        # path, otherwise create a brand-new stable ID when inside the zone.
        for index, (box, external_id) in enumerate(zip(boxes, ids)):
            if external_id is None or index in matches:
                continue
            if self._follow_reference and index != best_ref_index:
                continue
            stable_id = self._possible_reacquisition(
                foots[index], signatures[index], now, used)
            if (stable_id is None and self._follow_reference and self._tracked
                    and best_ref_index == index and not used):
                # Allow reacquisition beyond the 12-second motion gate
                # if there is only one shared target and its clothing
                # still matches. The gap is shown as an ESTIMATE.
                stable_id = next(iter(self._tracked))
            if stable_id is not None:
                # Forget the stale external-ID association for that track.
                for old_external, old_stable in list(self._external_ids.items()):
                    if old_stable == stable_id:
                        self._external_ids.pop(old_external, None)
            else:
                previous = self._unarmed.get(external_id)
                entered = region is not None and (
                    cv2.pointPolygonTest(region.astype(np.float32), foots[index], False) >= 0 or
                    (previous is not None and now - previous[1] <= 3.0 and
                     self._crossed_zone(previous[0], foots[index], region)))
                if self._follow_reference:
                    entered = bool(reference is not None and best_ref_index == index)
                stable_id = self._next_id if entered else None
            if stable_id is not None and stable_id == self._next_id:
                self._next_id += 1
                self._tracked[stable_id] = {
                    "path": [], "box": tuple(box), "foot": foots[index],
                    "hist": signatures[index], "velocity": np.zeros(2, dtype=float),
                    "last_seen_time": now, "seen_frame": self._frame_number,
                    "yolo_id": external_id,
                    "color": self.TRACK_COLORS[(stable_id - 1) % len(self.TRACK_COLORS)],
                }
                self._append_path(self._tracked[stable_id], foots[index])
                self._external_ids[external_id] = stable_id
                if self._follow_reference and reference is not None:
                    print(f"[SPOT Cross-floor] Floor {self._floor_num}: "
                          f"appearance candidate linked to TRACK #{reference['track_id']}")
            if stable_id is not None:
                self._unarmed.pop(external_id, None)
                matches[index] = stable_id
                used.add(stable_id)
            else:
                self._unarmed[external_id] = (foots[index], now)

        # Avoid unbounded buildup of IDs that never entered the zone.
        self._unarmed = {key: value for key, value in self._unarmed.items()
                         if now - value[1] <= 3.0}

        for index, stable_id in matches.items():

            self._mark_seen(stable_id, ids[index], boxes[index], foots[index],
                            signatures[index], now)

        # Floor 1 shares an appearance reference only after the person has
        # crossed the user-defined secure spot and has a clearly bright shirt.
        # It is deliberately NOT a face ID or an identity guarantee.
        if (self._shared_target is not None and not self._follow_reference
                and region is not None and self._shared_target.snapshot() is None):
            for index, stable_id in matches.items():
                sample = descriptors[index]
                if sample is not None and self._shared_target.register_if_empty(
                        stable_id, sample):
                    print(f"[SPOT Cross-floor] Floor 1 registered white-shirt "
                          f"TRACK #{stable_id} for Floors 2–4")
                    break

        # Track everything indefinitely in the raster canvas. Only the
        # *predicted* box expires after MAX_OCCLUSION_SECONDS.
        if region is not None:
            overlay = frame.copy()
            cv2.fillPoly(overlay, [region], self.ZONE_COLOR)
            cv2.addWeighted(overlay, 0.14, frame, 0.86, 0, frame)
            cv2.polylines(frame, [region], True, self.ZONE_COLOR, 2, cv2.LINE_AA)
            px, py = map(int, region[0])
            cv2.putText(frame, "SECURE SPOT", (px, max(22, py - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, self.ZONE_COLOR, 2, cv2.LINE_AA)
        if self._trail_canvas is not None:
            canvas_mask = cv2.cvtColor(self._trail_canvas, cv2.COLOR_BGR2GRAY)
            cv2.copyTo(self._trail_canvas, canvas_mask, frame)

        for index, box in enumerate(boxes):
            x1, y1, x2, y2 = (int(v) for v in box)
            stable_id = matches.get(index)
            if stable_id is None:
                cv2.rectangle(frame, (x1, y1), (x2, y2), self.DETECT_COLOR, 1)
                continue
            color = self._tracked[stable_id].get("color", self.TRACK_COLOR)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            label = (f"TRACK #{self._display_id(stable_id)} (appearance)"
                     if self._follow_reference else
                     f"TRACK #{self._display_id(stable_id)}")
            cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color,
                        2, cv2.LINE_AA)
            cv2.circle(frame, foots[index], 5, color, -1, cv2.LINE_AA)

        hidden_count = 0
        last_seen_count = 0
        for stable_id, item in self._tracked.items():
            if item["seen_frame"] == self._frame_number:
                continue
            elapsed = now - item["last_seen_time"]
            if elapsed < 0:
                continue
            if elapsed > self.MAX_OCCLUSION_SECONDS:
                # Keep a last-known marker indefinitely without pretending
                # to know where the hidden person went. Trail also remains.
                last_seen_count += 1
                last_foot = item["foot"]
                cv2.circle(frame, last_foot, 6, self.LAST_SEEN_COLOR, 2, cv2.LINE_AA)
                cv2.putText(frame, f"#{self._display_id(stable_id)} LAST SEEN",
                            (max(0, last_foot[0] - 25), max(20, last_foot[1] - 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.46, self.LAST_SEEN_COLOR,
                            1, cv2.LINE_AA)
                continue
            hidden_count += 1
            x1, y1, x2, y2 = item["box"]
            predicted = self._prediction(item, now)
            dx, dy = predicted[0] - item["foot"][0], predicted[1] - item["foot"][1]
            predicted_box = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)
            self._dashed_rectangle(frame, predicted_box, self.ESTIMATE_COLOR)
            # Dashed orange trail is a *prediction*, not a confirmed route.
            self._dashed_line(frame, item["foot"], predicted,
                              self.ESTIMATE_COLOR, thickness=2)
            cv2.putText(frame, f"#{self._display_id(stable_id)} ESTIMATED (hidden)",
                        (max(0, int(x1 + dx)), max(22, int(y1 + dy) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.53,
                        self.ESTIMATE_COLOR, 2, cv2.LINE_AA)

        if preview:
            arr = self._pixels(preview, width, height)
            if len(arr) > 1:
                cv2.polylines(frame, [arr], False, self.PREVIEW_COLOR,
                              2, cv2.LINE_AA)
            for x, y in arr:
                cv2.circle(frame, (int(x), int(y)), 5, self.PREVIEW_COLOR, -1)

        if self._fps_last is not None:
            dt = now - self._fps_last
            if dt > 0.005:
                fps = 1.0 / dt
                self._fps_ema = (fps if self._fps_ema is None else
                                 0.8 * self._fps_ema + 0.2 * fps)
        self._fps_last = now
        speed = f" | processed {self._fps_ema:.1f} FPS" if self._fps_ema else ""
        if self._follow_reference:
            state = (f"FLOOR {self._floor_num} | shared TRACK #{reference['track_id']}"
                     f" | trails {len(self._tracked)} | estimated {hidden_count}{speed}"
                     if reference is not None else
                     f"FLOOR {self._floor_num} | waiting for white-shirt entry on Floor 1{speed}")
        elif zone is not None:
            state = (f"People {len(boxes)} | trails {len(self._tracked)} | "
                     f"estimated {hidden_count} | last seen {last_seen_count}{speed}")
        else:
            state = "NO SECURE SPOT - click DRAW ZONE"
        cv2.rectangle(frame, (0, 0), (min(width, 780), 31), (22, 22, 22), -1)
        cv2.putText(frame, state, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.50, (255, 255, 255), 1, cv2.LINE_AA)
        return len(boxes), frame
