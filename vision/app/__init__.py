"""The vision app: one window for everything (python -m vision.app).

``server``   One local server. The app page runs on http://127.0.0.1:8765/
             and the phone camera page on https://<LAN-IP>:8443/ (you
             connect the phone by scanning a QR code).
``bench``    The Photos tab: drop photos in, re-run them with other
             settings, and save labelled photos to the dataset.
``quality``  Checks the phone's live preview (is the board lined up and
             straight, exposure, glare, motion, is the MCU sharp and big
             enough), decides when to take the photo automatically, and
             checks the final full-resolution photo.
``net``      LAN address, self-signed certificate, a small WebSocket server.
``qr``       Makes the QR code for connecting the phone.
``web/``     index.html (the app) and phone.html (the guided camera).

A scan has two steps: (1) the whole board, (2) move the phone close to the MCU.
"""

from .bench import State, build_response, parse_options  # noqa: F401
from .quality import (CaptureGate, FrameAnalyzer, FrameMeta, FrameReport,  # noqa: F401
                      analyze_photo, detect_board)
