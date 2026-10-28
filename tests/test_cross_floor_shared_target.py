"""Simulated multi-floor test: no cameras, Ultralytics or Qt required.

Run from the project root: python -m unittest discover -s tests -v
"""
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.secure_zone_tracker import SecureZoneTracker
from core.shared_target_registry import SharedTargetRegistry


class Array:
    def __init__(self, a):
        self.a = np.asarray(a)
    def cpu(self): return self
    def numpy(self): return self.a
    def int(self): return Array(self.a.astype(int))
    def tolist(self): return self.a.tolist()

class Boxes:
    def __init__(self, boxes, ids):
        self.xyxy = Array(np.asarray(boxes, dtype=float).reshape(-1, 4))
        self.id = Array(ids) if ids is not None else None
        self._boxes = boxes
    def __len__(self): return len(self._boxes)

class Result:
    def __init__(self, boxes, ids): self.boxes = Boxes(boxes, ids)

class FakeModel:
    def __init__(self):
        self.boxes = []
        self.ids = []
        self.predictor = None
    def set(self, boxes, ids):
        self.boxes = boxes
        self.ids = ids
    def track(self, frame, **kwargs):
        assert kwargs.get('persist') is True
        return [Result(self.boxes, self.ids)]


def scene(boxes, shirts, pants=None):
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    if pants is None: pants = [(30, 55, 80)] * len(boxes)
    for box, shirt, trouser in zip(boxes, shirts, pants):
        x1, y1, x2, y2 = box
        w, h = x2-x1, y2-y1
        a,b = int(x1+.1*w),int(x2-.1*w)
        cv2.rectangle(frame,(a,int(y1+.2*h)),(b,int(y1+.65*h)),shirt,-1)
        cv2.rectangle(frame,(a,int(y1+.63*h)),(b,int(y1+.95*h)),trouser,-1)
    return frame

WHITE = (230,230,230)
RED = (50,20,220)
BLUISH = (220,225,230)

class CrossFloorTests(unittest.TestCase):
    def setUp(self):
        self.ref = SharedTargetRegistry()
        self.clocks = [0.0] * 4
        self.models = [FakeModel() for _ in range(4)]
        self.zone = [[0,0],[0.95,0],[0.95,0.99],[0,0.99]]
        self.trackers = [SecureZoneTracker(model=self.models[i],
                            zone=self.zone if i==0 else None,
                            shared_target=self.ref,floor_num=i+1,
                            follow_reference=i>0,
                            imgsz=512,
                            clock=lambda i=i:self.clocks[i]) for i in range(4)]
    def step(self, floor, t, boxes, shirts, ids, pants=None):
        i=floor-1
        self.clocks[i]=t
        self.models[i].set(boxes,ids)
        return self.trackers[i].detect(scene(boxes,shirts,pants))[1]
    def test_other_floors_wait_for_floor_one(self):
        self.step(2, 0, [(100,40,170,290)], [WHITE], [2])
        self.assertEqual(len(self.trackers[1]._tracked),0)
    def test_one_white_shirt_shared_across_2_3_4(self):
        person=(120,40,185,290)
        # A red-shirted person enters Floor 1 first, so its stable ID is 1.
        self.step(1,0,[(20,50,90,290)],[RED],[20])
        self.step(1,.2,[(20,50,90,290), person],[RED,WHITE],[20,40])
        self.assertIsNotNone(self.ref.snapshot())
        self.assertEqual(self.ref.snapshot()['track_id'],2)
        for floor in (2,3,4):
            self.step(floor,1,[(90,40,170,280)],[WHITE],[floor*10])
            self.assertEqual(len(self.trackers[floor-1]._tracked),1)
            self.assertEqual(self.trackers[floor-1]._display_id(1),2)
    def test_red_not_selected_followers(self):
        self.step(1,0,[(100,40,170,280)],[WHITE],[4])
        self.step(3,1,[(80,45,160,290)],[RED],[3])
        self.assertEqual(len(self.trackers[2]._tracked),0)
    def test_ambiguous_white_shirts_not_selected(self):
        self.step(1,0,[(100,40,170,280)],[WHITE],[4])
        self.step(2,1,[(80,45,160,290),(250,45,330,290)],[WHITE,WHITE],[3,5])
        self.assertEqual(len(self.trackers[1]._tracked),0)
    def test_reacquire_after_hidden_and_changed_id(self):
        self.step(1,0,[(100,40,170,280)],[WHITE],[4])
        self.step(2,1,[(80,45,160,290)],[WHITE],[3])
        self.step(2,1.3,[(95,45,175,290)],[WHITE],[3])
        self.assertTrue(np.any(self.trackers[1]._trail_canvas))
        for t in np.arange(2, 25, .5):
            self.step(2,float(t),[],[],[])
        self.assertEqual(len(self.trackers[1]._tracked),1)
        self.assertTrue(np.any(self.trackers[1]._trail_canvas))
        self.step(2,25,[(115,45,195,290)],[WHITE],[99])
        self.assertEqual(len(self.trackers[1]._tracked),1)
        self.assertEqual(self.trackers[1]._tracked[1]['yolo_id'],99)
    def test_reset_local_trails_on_rewind(self):
        self.step(1,0,[(100,40,170,280)],[WHITE],[4])
        self.step(2,1,[(80,45,160,290)],[WHITE],[3])
        self.step(2,2,[(95,45,175,290)],[WHITE],[3])
        self.trackers[1].reset_tracking()
        self.assertEqual(len(self.trackers[1]._tracked),0)
        self.assertIsNone(self.trackers[1]._trail_canvas)
        self.assertIsNotNone(self.ref.snapshot())

if __name__ == '__main__':unittest.main()
