---
id: H-003
title: ESPCN/EDSR super-resolution reads more chip markings than bicubic at 150-250 px chips
status: blocked
weakness: W-SMALL-CARD
experiment: none
---

Blocked: the SR models need `opencv-contrib-python-headless` and a download (`scripts/get_sr_models.py`);
neither is available in the build environment. Run on the owner's PC:
`python -m phoenix experiment E-003` after creating its spec with `upscaler: espcn`.
