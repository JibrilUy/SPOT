"""Thread-safe shared Floor 1 appearance reference for SPOT multi-floor video.

Not facial recognition or proof of identity. Each camera has a distinct
ByteTrack process; this shares a rough clothing appearance and display ID.
"""
from threading import Lock
import os


class SharedTargetRegistry:
    def __init__(self):
        self._lock = Lock()
        self._reference = None
        try:
            self.white_minimum = float(os.getenv("SPOT_WHITE_SHIRT_MIN", "0.30"))
        except ValueError:
            self.white_minimum = 0.30
        self.white_minimum = max(0.05, min(0.9, self.white_minimum))

    def register_if_empty(self, track_id, descriptor):
        """Only a previously zone-armed white-shirt Floor 1 track can register."""
        if descriptor is None or descriptor["white_fraction"] < self.white_minimum:
            return False
        with self._lock:
            if self._reference is not None:
                return False
            self._reference = {
                "track_id": int(track_id),
                "descriptor": {
                    "shirt": descriptor["shirt"].copy(),
                    "pants": descriptor["pants"].copy(),
                    "white_fraction": float(descriptor["white_fraction"]),
                },
            }
            return True

    def snapshot(self):
        with self._lock:
            # Arrays are never mutated after publication. Consumers must
            # treat the snapshot as read-only.
            return self._reference

    def clear(self):
        """Manual reset between unrelated recordings/sessions, if needed."""
        with self._lock:
            self._reference = None
