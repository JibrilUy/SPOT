import sys
from pathlib import Path
import unittest
from unittest.mock import Mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.secure_zone_tracker import SecureZoneTracker


class TensorStub:
    def __init__(self, arr):
        self.arr = np.asarray(arr)
    def cpu(self):
        return self
    def numpy(self):
        return self.arr
    def int(self):
        return TensorStub(self.arr.astype(int))
    def tolist(self):
        return self.arr.tolist()


class BoxesStub:
    def __init__(self, boxes, ids):
        self.xyxy = TensorStub(boxes)
        self.id = TensorStub(ids) if ids is not None else None
        self._n = len(boxes)
    def __len__(self):
        return self._n


class ResultStub:
    def __init__(self, boxes, ids):
        self.boxes = BoxesStub(boxes, ids)


class SequencedModel:
    def __init__(self, frames):
        self.frames = iter(frames)
        self.calls = []
    def track(self, frame, **kwargs):
        self.calls.append(kwargs)
        b, ids = next(self.frames)
        return [ResultStub(b, ids)]


class TrackingTests(unittest.TestCase):
    def test_start_on_entry_continue_after_exit_and_multiple_people(self):
        # Zone: 30%-70% of 200x200, person 1 starts outside then walks in/out.
        # Person 2 enters independently. Person 3 stays outside.
        model = SequencedModel([
            ([[3,20,23,40],[160,20,180,40]], [1,3]),
            ([[65,20,85,90],[75,70,95,100]], [1,2]),
            ([[160,20,180,100],[90,100,110,160]], [1,2]),
        ])
        t = SecureZoneTracker(model=model, zone=[[.3,.3],[.7,.3],[.7,.7],[.3,.7]])
        frame = np.zeros((200,200,3), dtype=np.uint8)
        n,_ = t.detect(frame.copy())
        self.assertEqual(n,2)
        self.assertEqual(len(t._tracked),0)
        t.detect(frame.copy())
        self.assertEqual(set(t._tracked),{1,2})
        t.detect(frame.copy())
        self.assertEqual(set(t._tracked),{1,2})
        self.assertGreaterEqual(len(t._tracked[1]['path']),2)
        self.assertEqual(model.calls[0]['tracker'], 'bytetrack.yaml')
        self.assertTrue(model.calls[0]['persist'])

    def test_no_zone_no_active_trails(self):
        model = SequencedModel([([[50,50,100,110]],[55])])
        t = SecureZoneTracker(model=model)
        t.detect(np.zeros((200,200,3),dtype=np.uint8))
        self.assertEqual(len(t._tracked), 0)

    def test_change_zone_clears_trails_next_frame(self):
        model = SequencedModel([([[65,65,90,95]],[1]), ([],None)])
        t = SecureZoneTracker(model=model, zone=[[.2,.2],[.8,.2],[.8,.8],[.2,.8]])
        t.detect(np.zeros((200,200,3),dtype=np.uint8))
        self.assertIn(1,t._tracked)
        t.set_zone(None)
        t.detect(np.zeros((200,200,3),dtype=np.uint8))
        self.assertFalse(t._tracked)

    def test_polygon_validation(self):
        valid = [[.1,.1],[.9,.1],[.9,.9],[.1,.9]]
        self.assertEqual(len(SecureZoneTracker._validated_polygon(valid)), 4)
        with self.assertRaises(ValueError):
            SecureZoneTracker._validated_polygon([[0,0],[.1,.1]])
        with self.assertRaises(ValueError):
            SecureZoneTracker._validated_polygon([[0.1,0.1],[.9,.9],[.1,.9],[.9,.1]])

if __name__ == '__main__':
    unittest.main()
