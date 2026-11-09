# ADR-0042: Three read passes on the card at 0/90/180/270°: page, label boxes only (rest blanked), chi

- Date: 2026-11-04
- Status: Accepted
- Source: DECISIONS.md row 42 (migrated)

## Decision

Three read passes on the card at 0/90/180/270°: page, label boxes only (rest blanked), chip close-up (largest dark low-texture square, rectified, enlarged, Otsu/Sauvola)

## Context and reason

MikroE prints the mikroBUS 3 column horizontally and the other rows vertically; the chip marking is the one thing that separates cards of one layout and is ~1/18 of the chip side

## Evidence

synthetic close / mid: chip read on every card where it was rendered legibly
