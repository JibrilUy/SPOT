"""Run: python -m unittest discover -s tests -v (inside this ZIP's folder).
No webcam, Qt or Ultralytics required; uses a synthetic detection source.
"""
import importlib.util
from pathlib import Path
import unittest

import cv2
import numpy as np

PATH = Path(__file__).resolve().parents[1] / "core" / "secure_zone_tracker.py"
spec = importlib.util.spec_from_file_location("secure_zone_tracker_v2", PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
SecureZoneTracker = module.SecureZoneTracker


class ValueArray:
    def __init__(self, value):
        self.value = np.asarray(value)

    def cpu(self):
        return self

    def numpy(self):
        return self.value

    def int(self):
        return ValueArray(self.value.astype(int))

    def tolist(self):
        return self.value.tolist()


class FakeBoxes:
    def __init__(self, boxes, ids):
        self.xyxy = ValueArray(np.asarray(boxes).reshape(-1, 4))
        self.id = ValueArray(ids) if ids is not None else None
        self._boxes = boxes

    def __len__(self):
        return len(self._boxes)


class FakeResult:
    def __init__(self, boxes, ids):
        self.boxes = FakeBoxes(boxes, ids)


class FakeTrackerState:
    def __init__(self):
        self.reset_called = 0

    def reset(self):
        self.reset_called += 1


class FakeModel:
    def __init__(self):
        self._next = ([], [])
        self.reset_state = FakeTrackerState()
        self.predictor = type("Predictor", (), {"trackers": [self.reset_state]})()

    def set(self, boxes, ids):
        self._next = (boxes, ids)

    def track(self, frame, **kwargs):
        assert kwargs.get("persist") is True
        assert kwargs.get("imgsz") == 512
        return [FakeResult(*self._next)]


class Floor1Tests(unittest.TestCase):
    def setUp(self):
        self.model = FakeModel()
        self.timestamp = 0.0
        self.tracker = SecureZoneTracker(model=self.model,
                                         zone=[[0.05, 0.1], [0.5, 0.1],
                                               [0.5, 0.95], [0.05, 0.95]],
                                         clock=lambda: self.timestamp)

    def step(self, t, boxes, ids):
        self.timestamp = float(t)
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        # Make a turquoise body region for histogram appearance matching.
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 160, 30), -1)
        self.model.set(boxes, ids)
        return self.tracker.detect(frame)[1]

    def test_trail_stays_after_over_90_missing_frames(self):
        self.step(0, [(60, 90, 110, 220)], [10])
        self.step(0.2, [(80, 90, 130, 220)], [10])
        self.assertEqual(len(self.tracker._tracked), 1)
        for t in np.arange(.4, 21, .2):
            self.step(t, [], [])
        self.assertEqual(len(self.tracker._tracked), 1)
        self.assertGreaterEqual(len(self.tracker._tracked[1]["path"]), 2)
        self.assertTrue(np.any(self.tracker._trail_canvas))

    def test_pillar_occlusion_estimate_and_new_id_reconnection(self):
        self.step(0, [(80, 70, 135, 210)], [42])
        self.step(.2, [(90, 70, 145, 210)], [42])
        self.step(.4, [], [])
        self.assertEqual(len(self.tracker._tracked), 1)
        self.step(.6, [], [])
        self.step(.8, [(110, 70, 165, 210)], [99])
        self.assertEqual(len(self.tracker._tracked), 1)
        self.assertEqual(self.tracker._tracked[1]["yolo_id"], 99)
        self.assertEqual(self.tracker._external_ids[99], 1)
        self.assertGreaterEqual(len(self.tracker._tracked[1]["path"]), 3)

    def test_fast_crossing_between_processed_frames_triggers_zone(self):
        # Person appears on opposite sides of the secure zone, but frame
        # skipping hid the intermediate positions. Segment entry still works.
        self.tracker.set_zone([[.40, .20], [.50, .20],
                               [.50, .90], [.40, .90]])
        self.step(0, [(80, 90, 120, 220)], [7])
        self.assertFalse(self.tracker._tracked)
        self.step(.2, [(390, 90, 430, 220)], [7])
        self.assertEqual(len(self.tracker._tracked), 1)

    def test_reset_on_video_loop(self):
        self.step(0, [(60, 90, 110, 220)], [1])
        self.step(.3, [(80, 90, 130, 220)], [1])
        self.assertTrue(np.any(self.tracker._trail_canvas))
        self.tracker.reset_tracking()
        self.assertEqual(len(self.tracker._tracked), 0)
        self.assertIsNone(self.tracker._trail_canvas)
        self.assertEqual(self.model.reset_state.reset_called, 1)
        self.step(1, [], [])
        self.assertFalse(np.any(self.tracker._trail_canvas))

    def test_zone_edits_do_not_clear_previous_tracks(self):
        self.step(0, [(60, 90, 110, 220)], [10])
        self.step(0.2, [(80, 90, 130, 220)], [10])
        self.assertTrue(np.any(self.tracker._trail_canvas))
        self.tracker.set_zone(None)
        self.step(0.4, [], [])
        self.assertEqual(len(self.tracker._tracked), 1)
        self.assertTrue(np.any(self.tracker._trail_canvas))
        self.tracker.set_zone([[0.1, 0.1], [0.6, 0.1], [0.6, 0.9], [0.1, 0.9]])
        self.step(0.6, [], [])
        self.assertEqual(len(self.tracker._tracked), 1)
        self.assertTrue(np.any(self.tracker._trail_canvas))

    def test_long_pillar_disappearance_keeps_last_seen_and_full_trail(self):
        self.step(0, [(60, 90, 110, 220)], [10])
        self.step(0.2, [(80, 90, 130, 220)], [10])
        for t in np.arange(.4, 32, .2):
            self.step(t, [], [])
        self.assertEqual(len(self.tracker._tracked), 1)
        self.assertTrue(np.any(self.tracker._trail_canvas))
        self.assertEqual(self.tracker._tracked[1]["path"][0], (85, 220))

    def test_reconnect_after_five_second_occlusion(self):
        self.step(0, [(80, 70, 135, 210)], [42])
        self.step(.2, [(90, 70, 145, 210)], [42])
        for t in np.arange(.4, 6, .2):
            self.step(t, [], [])
        self.step(6.0, [(115, 70, 170, 210)], [99])
        self.assertEqual(len(self.tracker._tracked), 1)
        self.assertEqual(self.tracker._tracked[1]["yolo_id"], 99)


    def test_multiple_people_keep_distinct_ids_and_colors(self):
        self.step(0, [(60, 90, 110, 220), (190, 90, 245, 220)], [1, 2])
        self.step(.2, [(70, 90, 120, 220), (205, 90, 260, 220)], [1, 2])
        self.assertEqual(set(self.tracker._tracked), {1, 2})
        self.assertNotEqual(self.tracker._tracked[1]["color"],
                            self.tracker._tracked[2]["color"])
        for t in np.arange(.4, 10.0, .2):
            self.step(t, [], [])
        self.assertTrue(np.any(self.tracker._trail_canvas))
        self.assertEqual(set(self.tracker._tracked), {1, 2})

    def test_recent_path_pruning_does_not_erase_old_drawn_lines(self):
        self.tracker.MAX_RECENT_PATH_POINTS = 5
        for i in range(15):
            self.step(i * .1, [(60 + 5*i, 90, 110 + 5*i, 220)], [10])
        self.assertLessEqual(len(self.tracker._tracked[1]["path"]), 5)
        # The line from the earliest frames should remain on the canvas.
        self.assertTrue(np.any(self.tracker._trail_canvas[219:222, 85:100]))

    def test_wrong_person_does_not_steal_old_track_after_occlusion(self):
        self.step(0, [(80, 70, 135, 210)], [42])
        self.step(.2, [(90, 70, 145, 210)], [42])
        for t in np.arange(.4, 2, .2):
            self.step(t, [], [])
        # New detection is spatially plausible but its clothing differs.
        t = 2.0
        self.timestamp = t
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.rectangle(frame, (110, 70), (165, 210), (0, 0, 255), -1)
        self.model.set([(110, 70, 165, 210)], [99])
        self.tracker.detect(frame)
        self.assertEqual(len(self.tracker._tracked), 2)
        self.assertEqual(self.tracker._tracked[1]["yolo_id"], 42)


if __name__ == "__main__":
    unittest.main()
