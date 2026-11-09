---
id: H-008
title: Layout anchors read by the engine and resolution in use make geometry-first localisation possible
status: supported
weakness: W-NO-LAYOUT
experiment: E-005
---

## Claim

With Tesseract the UNI-DS v8 layout is placed on 1 of 13 real photos (3 inliers): the 73 anchors were
calibrated with RapidOCR on a 960x1280 photo, so the text Tesseract reads on 2160x3840 captures rarely
matches them. Without placement there are no socket areas and no geometry-first card search, which the
brief requires. Adding anchors read by Tesseract on one 4K capture (`data/layouts_v2.json`, 204 anchors)
places the layout on most held-out photos, with no change in card / MCU answers.

## Test

E-005 on real3 + captures. Placement rate and inliers come from the traces (`layout` field).
`20261105_153735_board_01` is the calibration photo and is reported separately.

## Result

E-005: layout placed on 11/13 real photos (was 1/13), held-out 10/12; identical decisions; real-photo time -23 % (no failed-fit re-reads). Proposal P-003.
