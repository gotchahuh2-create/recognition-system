"""Experimental clean-up for very faint chip markings.

The steps (one module each; every step can save debug images through
``debug.StageSaver``):

    1 region     find the dark area in the middle (the MCU), straighten
                 and enlarge it
    2 features   colour values per pixel in RGB / HSV / LAB, with uneven
                 lighting removed, normalised and lightly smoothed. Glare
                 is marked, not removed.
    4 cluster    group pixels by colour in each colour space, let the
                 spaces vote, find the best text/surface split, then refine
                 it locally
    5 threshold  many different thresholds: fixed levels, Otsu,
                 hysteresis, and local ones (Gaussian / Niblack / Sauvola /
                 Wolf) at two window sizes
    6 preserve   careful clean-up that only removes blobs that can't be
                 letters
    7 variants   gray and black-and-white images for OCR
    8 fusion     run OCR on every image, score each by how much it agrees
                 with the others, vote letter by letter per line, and add
                 the combined result next to the raw reads

(There is no separate step 3: the colour features are part of step 2.)

Entry points: ``FaintChipReader``, ``read_faint`` and ``python -m vision.faint``.
Used by the card step with ``--chip-mode faint|both`` (the default is classic).
"""

from .reader import FaintChipReader, FaintConfig, FaintReading, read_faint  # noqa: F401
from .region import ChipCrop, extract_chip  # noqa: F401
