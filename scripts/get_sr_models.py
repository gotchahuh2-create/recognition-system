"""Download super-resolution models for enlarging text crops (vision/enhance.py).

    python scripts/get_sr_models.py                 # ESPCN + FSRCNN x2/x3/x4 (small, fast)
    python scripts/get_sr_models.py --all           # also EDSR (~38 MB each, slow on CPU)

You need ``pip install opencv-contrib-python-headless`` (for cv2.dnn_superres).
Install it *instead of* opencv-python-headless, because the two packages
conflict. These are the models linked from the OpenCV dnn_superres
tutorial. The files go into ``models/`` (change this with the VISION_MODELS
environment variable).

Then choose them with ``--upscaler espcn`` (command line), the Upscaler menu
in ``python -m vision.app``, or ``eval/ocr_eval.py --configs full,sr-espcn``.
Judge super-resolution by OCR accuracy, not by image quality scores: compare
with the eval before changing the default away from bicubic.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vision.enhance import MODELS_DIR  # noqa: E402

URLS = {
    "ESPCN": "https://raw.githubusercontent.com/fannymonori/TF-ESPCN/master/export/ESPCN_x{s}.pb",
    "FSRCNN": "https://raw.githubusercontent.com/Saafke/FSRCNN_Tensorflow/master/models/FSRCNN_x{s}.pb",
    "EDSR": "https://raw.githubusercontent.com/Saafke/EDSR_Tensorflow/master/models/EDSR_x{s}.pb",
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="also EDSR (large)")
    ap.add_argument("--scales", default="2,3,4")
    a = ap.parse_args(argv)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    names = ["ESPCN", "FSRCNN"] + (["EDSR"] if a.all else [])
    for n in names:
        for s in a.scales.split(","):
            dst = MODELS_DIR / f"{n}_x{s}.pb"
            if dst.is_file():
                print(f"have {dst.name}")
                continue
            url = URLS[n].format(s=s)
            print(f"get  {dst.name} <- {url}")
            try:
                urllib.request.urlretrieve(url, dst)
            except Exception as e:
                print(f"     failed: {e}")
                dst.unlink(missing_ok=True)
    try:
        import cv2
        ok = hasattr(cv2, "dnn_superres")
    except ImportError:
        ok = False
    if not ok:
        print("\nNOTE: cv2.dnn_superres missing -> pip install opencv-contrib-python-headless")
    return 0


if __name__ == "__main__":
    sys.exit(main())
