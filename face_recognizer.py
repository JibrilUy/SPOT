import os
import cv2


class FaceRecognizer:
    """
    Single-user face recognizer using OpenCV's built-in LBPH algorithm.
    Requires a model already trained via enroll_face.py — this class only
    loads and runs it, it doesn't do enrollment itself.
    """

    # LBPH confidence is actually a DISTANCE — lower means a closer match,
    # 0 would be identical. Tune this if you get too many false
    # accepts/rejects: lower = stricter, higher = more lenient.
    CONFIDENCE_THRESHOLD = 100

    def __init__(self, model_path: str = "assets/faces/trainer.yml", cascade_path: str = None):
        # Defaults to a local copy in assets/ (see download_cascade.py)
        # instead of cv2.data.haarcascades, since that bundled path has
        # been unreliable on some OpenCV builds (e.g. missing data files
        # on very new Python versions).
        cascade_path = cascade_path or "assets/haarcascade_frontalface_default.xml"

        if not os.path.exists(cascade_path):
            raise FileNotFoundError(
                f"Cascade file not found at '{cascade_path}'. "
                f"Run download_cascade.py first."
            )

        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise RuntimeError(
                f"Cascade file at '{cascade_path}' failed to load — "
                f"it may be corrupted or incomplete. Try re-running "
                f"download_cascade.py."
            )

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No trained face model found at '{model_path}'. "
                f"Run enroll_face.py first to capture and train your face."
            )

        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read(model_path)

    def process(self, frame):
        """
        Detects faces in the frame and attempts recognition against the
        enrolled model. Draws a box around each detected face (green if
        matched, red if not) directly onto the frame.

        Returns (annotated_frame, is_match, confidence).
        is_match is True if ANY detected face matches the enrolled user
        within CONFIDENCE_THRESHOLD. confidence is the best (lowest)
        distance found, or None if no face was detected at all.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

        is_match = False
        best_confidence = None

        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            label, confidence = self.recognizer.predict(face_roi)

            matched = confidence < self.CONFIDENCE_THRESHOLD
            if matched:
                is_match = True
                if best_confidence is None or confidence < best_confidence:
                    best_confidence = confidence

            color = (0, 255, 0) if matched else (0, 0, 255)
            label_text = "Recognized" if matched else "Unknown"

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame, label_text, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
            )

        return frame, is_match, best_confidence