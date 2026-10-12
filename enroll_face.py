"""
Run this ONCE, by itself (not part of the main app), before using
face-recognition login.

Look at the camera and press 'c' to capture a sample — do this ~20-30
times, moving your head slightly and varying your expression each time
for better accuracy. Press 'q' when done to train and save the model.
"""

import cv2
import os
import numpy as np

SAVE_DIR = "assets/faces/samples"
MODEL_PATH = "assets/faces/trainer.yml"
USER_LABEL = 0  # single-user system — the one enrolled face is always label 0

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

CASCADE_PATH = "assets/haarcascade_frontalface_default.xml"

if not os.path.exists(CASCADE_PATH):
    raise FileNotFoundError(
        f"Cascade file not found at '{CASCADE_PATH}'. "
        f"Run download_cascade.py first."
    )

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
if face_cascade.empty():
    raise RuntimeError(
        f"Cascade file at '{CASCADE_PATH}' failed to load — it may be "
        f"corrupted. Try re-running download_cascade.py."
    )
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

sample_count = 0
print("Press 'c' to capture a face sample, 'q' to finish and train.")

while True:
    success, frame = cap.read()
    if not success:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.putText(
        frame, f"Samples captured: {sample_count}", (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
    )
    cv2.imshow("Face Enrollment - press 'c' to capture, 'q' to finish", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("c") and len(faces) > 0:
        (x, y, w, h) = faces[0]
        face_img = gray[y:y + h, x:x + w]
        sample_count += 1
        cv2.imwrite(f"{SAVE_DIR}/{sample_count}.jpg", face_img)
        print(f"Captured sample {sample_count}")
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

if sample_count == 0:
    print("No samples captured — nothing to train.")
else:
    print("Training model...")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    images, labels = [], []

    for filename in os.listdir(SAVE_DIR):
        path = os.path.join(SAVE_DIR, filename)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        images.append(img)
        labels.append(USER_LABEL)

    recognizer.train(images, np.array(labels))
    recognizer.save(MODEL_PATH)
    print(f"Model trained and saved to {MODEL_PATH}")