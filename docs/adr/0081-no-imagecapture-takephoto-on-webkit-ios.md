# ADR-0081: No `ImageCapture.takePhoto()` on WebKit (iOS / iPadOS Safari)

- Date: 2026-11-05
- Status: Accepted
- Source: DECISIONS.md row 81 (migrated)

## Decision

No `ImageCapture.takePhoto()` on WebKit (iOS / iPadOS Safari); video-frame still + camera watchdog (reopen when frames stall > 2.5 s or the track ends)

## Context and reason

User captures from iPhone Safari 26: takePhoto failed (source video-frame) and the feed went black before the close-up; photo_w 4032 from getPhotoCapabilities also overstated the chip size of the actual 2160x3840 still

## Evidence

photos/captures/20261105_1446*/1448* sidecars; e2e: track stopped -> reopened
