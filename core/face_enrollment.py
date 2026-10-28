"""Quality-gated multi-sample enrollment for SPOT administrators.

Only a single face is accepted. This is an image quality gate, *not* a
liveness/anti-spoof check: photos/videos of a person can still pass LBPH.
No raw face images are written to disk; the OpenCV model is the stored template.
"""
from pathlib import Path
import os
import tempfile
import time

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / 'assets' / 'faces' / 'trainer.yml'
CASCADE_PATH = PROJECT_ROOT / 'assets' / 'haarcascade_frontalface_default.xml'
SAMPLES_PER_POSE = 8
PROMPTS = (
    'Look straight at the camera. Keep your face still.',
    'Turn your head slightly LEFT and look at the camera.',
    'Turn your head slightly RIGHT and look at the camera.',
)
SAMPLE_SIZE = (160, 160)


def need_cv2_contrib():
    if not hasattr(cv2, 'face') or not hasattr(cv2.face, 'LBPHFaceRecognizer_create'):
        raise RuntimeError('OpenCV LBPH is missing. Install opencv-contrib-python (not only opencv-python).')


class QualityFaceCapture:
    """Returns a new sample only when its image quality meets baseline criteria."""

    def __init__(self, cascade_path=CASCADE_PATH, now=None):
        self.now = now or time.monotonic
        self.cascade = cv2.CascadeClassifier(str(cascade_path))
        if self.cascade.empty():
            raise FileNotFoundError(f'Could not load face detector: {cascade_path}')
        self.samples = []
        self.last_sample_at = float('-inf')
        self.last_sample = None

    @property
    def required(self):
        return SAMPLES_PER_POSE * len(PROMPTS)

    @property
    def complete(self):
        return len(self.samples) >= self.required

    @property
    def prompt(self):
        index = min(len(self.samples) // SAMPLES_PER_POSE, len(PROMPTS) - 1)
        return PROMPTS[index]

    def reset(self):
        self.samples.clear()
        self.last_sample_at = float('-inf')
        self.last_sample = None

    def assess(self, frame):
        """Return (sample|None, bounding_box|None, user_feedback)."""
        if frame is None or getattr(frame, 'ndim', 0) != 3:
            return None, None, 'Camera frame is unavailable.'
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        faces = self.cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=6,
                                              minSize=(85, 85))
        if len(faces) == 0:
            return None, None, 'No face detected. Face the webcam.'
        if len(faces) > 1:
            return None, None, 'More than one face detected. Only one person should be visible.'
        x, y, w, h = (int(n) for n in faces[0])
        box = (x, y, w, h)
        if w < 110 or h < 110 or w < 0.17 * width:
            return None, box, 'Move closer: your face is too small.'
        if w > 0.70 * width or h > 0.78 * height:
            return None, box, 'Move back: your face is too close.'
        cx, cy = (x + w / 2) / width, (y + h / 2) / height
        if not 0.23 < cx < 0.77 or not 0.20 < cy < 0.82:
            return None, box, 'Center your face in the preview.'
        face = gray[y:y + h, x:x + w]
        light = float(np.mean(face))
        if light < 55:
            return None, box, 'Too dark. Face a light source.'
        if light > 210:
            return None, box, 'Too bright. Reduce direct light.'
        if float(np.std(face)) < 27:
            return None, box, 'Low contrast. Improve lighting.'
        blur_score = float(cv2.Laplacian(face, cv2.CV_64F).var())
        if blur_score < 65:
            return None, box, 'Blurry image. Hold still or improve focus.'
        normalized = cv2.resize(face, SAMPLE_SIZE, interpolation=cv2.INTER_AREA)
        return normalized, box, 'Good quality — capturing samples.'

    def process_frame(self, frame):
        """Return (bounding_box, feedback, accepted_new_sample)."""
        if self.complete:
            return None, 'Scan completed. Review and save.', False
        sample, box, reason = self.assess(frame)
        if sample is None:
            return box, reason, False
        now = self.now()
        if now - self.last_sample_at < 0.25:
            return box, reason, False
        if self.last_sample is not None:
            difference = float(np.mean(cv2.absdiff(sample, self.last_sample)))
            if difference < 1.5:
                return box, 'Move your head slightly for a different sample.', False
        self.samples.append(sample)
        self.last_sample = sample
        self.last_sample_at = now
        return box, f'Sample {len(self.samples)}/{self.required} accepted.', True


def validation_distances(samples):
    """Test whether unseen captured frames match a model trained on other frames."""
    need_cv2_contrib()
    if len(samples) < SAMPLES_PER_POSE * len(PROMPTS):
        raise ValueError('Not enough accepted face samples')
    # Every fourth image is reserved for verification, not training.
    train_idx = [i for i in range(len(samples)) if i % 4 != 0]
    test_idx = [i for i in range(len(samples)) if i % 4 == 0]
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train([samples[i] for i in train_idx],
                     np.zeros(len(train_idx), dtype=np.int32))
    return [float(recognizer.predict(samples[i])[1]) for i in test_idx]


def validate_capture(samples):
    """Require held-out frames to be consistently recognizable."""
    distances = validation_distances(samples)
    # OpenCV LBPH uses a DISTANCE, lower is better. These are conservative
    # starting points; calibrate against your own webcam and enrolled users.
    passing = sum(distance < 85 for distance in distances)
    if passing < len(distances) - 1:
        raise ValueError('Face scan not consistent across images. Improve lighting and rescan.')
    return {'held_out': len(distances), 'passing': passing,
            'median_lbph_distance': float(np.median(distances))}


def get_next_label(used_db_labels, model_path=MODEL_PATH):
    """Allocate a label not already used by the database OR existing trainer."""
    need_cv2_contrib()
    labels = {int(x) for x in used_db_labels if x is not None}
    model_path = Path(model_path)
    if model_path.exists():
        rec = cv2.face.LBPHFaceRecognizer_create()
        rec.read(str(model_path))
        labels.update(int(x) for x in rec.getLabels().reshape(-1))
    return max(labels, default=-1) + 1


def build_staged_model(samples, new_label, model_path=MODEL_PATH):
    """Create a candidate model file without changing the live trainer.

    Existing labels are preserved using LBPH.update(); the model is only
    installed after a database account has been created in inactive state.
    """
    need_cv2_contrib()
    validate_capture(samples)
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    rec = cv2.face.LBPHFaceRecognizer_create()
    if model_path.exists():
        rec.read(str(model_path))
        if new_label in {int(x) for x in rec.getLabels().reshape(-1)}:
            raise ValueError('Face label is already used in trainer.yml')
        # Best-effort duplicate prevention. Needs real-world calibration.
        predictions = [rec.predict(face) for face in samples[::3]]
        if any(float(distance) < 55 for _, distance in predictions):
            raise ValueError('This face may already be enrolled. Review existing accounts.')
        rec.update(samples, np.full(len(samples), new_label, dtype=np.int32))
    else:
        rec.train(samples, np.full(len(samples), new_label, dtype=np.int32))
    # LBPH update should not erase previously learned identities.
    labels = {int(x) for x in rec.getLabels().reshape(-1)}
    if new_label not in labels:
        raise RuntimeError('New face label missing after training')
    handle, temporary = tempfile.mkstemp(prefix='.spot-trainer-', suffix='.yml',
                                         dir=str(model_path.parent))
    os.close(handle)
    try:
        rec.write(temporary)
        check = cv2.face.LBPHFaceRecognizer_create()
        check.read(temporary)
        test_label, test_distance = check.predict(samples[-1])
        if test_label != new_label or test_distance >= 100:
            raise RuntimeError('Saved face model failed a recognition self-test')
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise
    return Path(temporary)


def publish_model(staged_path, model_path=MODEL_PATH):
    """Atomic same-directory replacement of the face template model."""
    os.replace(str(staged_path), str(model_path))
