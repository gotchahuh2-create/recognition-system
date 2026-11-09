"""Smart camera acquisition: a phone as a guided, quality-gated camera.

    python -m vision.camera

``quality``  live assessment of preview frames (board detection + alignment,
             perspective, exposure, glare, motion, focus; MCU-marking size,
             focus and glare from native-resolution crops), the auto-capture
             gate and the check of the full-resolution still
``server``   local endpoint: https + WebSocket for the phone, loopback page
             for the PC, QR pairing, original-file transfer, hand-off to
             ``vision.pipeline`` (board shot, then an MCU close-up if needed)
``net``      LAN address, self-signed certificate, minimal WebSocket
``qr``       QR encoder for the pairing code
"""

from .quality import (CaptureGate, FrameAnalyzer, FrameMeta, FrameReport,  # noqa: F401
                      analyze_photo, detect_board)
