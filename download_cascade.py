"""
Run this once. Downloads the face-detection cascade file directly from
OpenCV's GitHub repo into your project's assets folder, bypassing whatever
is wrong with the bundled copy in your pip-installed opencv package.
"""

import os
import urllib.request

URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
SAVE_PATH = "assets/haarcascade_frontalface_default.xml"

os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

print(f"Downloading cascade file to {SAVE_PATH} ...")
urllib.request.urlretrieve(URL, SAVE_PATH)
print("Done.")