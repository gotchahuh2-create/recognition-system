# ADR-0071: Smart camera acquisition (`vision/camera`): phone page streams a ~960 px preview PLUS a na

- Date: 2026-11-05
- Status: Accepted
- Source: DECISIONS.md row 71 (migrated)

## Decision

Smart camera acquisition (`vision/camera`): phone page streams a ~960 px preview PLUS a native-resolution crop of the MCU card; capture is gated on the chip's predicted size in the full photo, its focus (no-reference re-blur, at the recent peak), chip glare, motion and board alignment

## Context and reason

Focus, glare and marking size are invisible in a downscaled preview; 0.8 showed the chip must be >= ~250 px in the photo

## Evidence

tests/test_camera.py, headless-browser run with a fake camera
