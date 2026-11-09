# Architecture decision records

ADR-0001..%04d were migrated from `DECISIONS.md` (frozen, 2026-11-09). New decisions: copy `TEMPLATE.md` to the next number.

| ADR | Date | Status | Decision |
|---|---|---|---|
| [0001](0001-ocr-behind-a-backend-interface-paddle.md) | 2026-11-03 | Accepted | OCR behind a backend interface: paddle > rapidocr > tesseract |
| [0002](0002-board-names-must-match-within-one.md) | 2026-11-03 | Accepted | Board names must match within one text line (or two stacked lines) |
| [0003](0003-idf-counted-per-product-missing-family.md) | 2026-11-03 | Accepted | IDF counted per product, missing family word ×0.6 |
| [0004](0004-mcu-accept-threshold-0-90-abstain.md) | 2026-11-03 | Accepted | MCU accept threshold 0.90, abstain on margin < 0.05 |
| [0005](0005-add-two-stage-crop-refinement-detect.md) | 2026-11-03 | Accepted | Add two-stage crop refinement (detect → crop from full-res photo → enlarge to ~44 px glyphs → denoise/deblur/b |
| [0006](0006-global-denoise-gamma-lift-sharpen-implemented.md) | 2026-11-03 | Accepted | Global denoise / gamma lift / sharpen implemented but **off by default** (UI toggle "global enhance", CLI --gl |
| [0007](0007-crop-reads-are-not-filtered-by.md) | 2026-11-03 | Accepted | Crop reads are not filtered by Tesseract confidence |
| [0008](0008-treat-8-s-as-an-ocr.md) | 2026-11-03 | Accepted | Treat 8↔S as an OCR-confusable pair |
| [0009](0009-3-3-median-before-tesseract-s.md) | 2026-11-03 | Accepted | 3×3 median before Tesseract's whole-image pass on noisy photos (σ > 6) |
| [0010](0010-hi-res-tile-tokens-mapped-into.md) | 2026-11-03 | Accepted | Hi-res tile tokens mapped into the processed frame |
| [0011](0011-super-resolution-espcn-fsrcnn-edsr-and.md) | 2026-11-03 | Accepted | Super-resolution (ESPCN/FSRCNN/EDSR) and TrOCR are optional, not default |
| [0012](0012-test-window-is-a-local-web.md) | 2026-11-03 | Accepted | Test window is a local web page (stdlib http.server), not a Tk window |
| [0013](0013-rapidocr-detector-resize-limit-type-max.md) | 2026-11-03 | Accepted | RapidOCR detector resize `limit_type=max, 2016` instead of PP-OCR's `min, 736` |
| [0014](0014-engines-with-an-angle-classifier-rapidocr.md) | 2026-11-03 | Accepted | Engines with an angle classifier (RapidOCR, Paddle) read one rotation per 0/180 and 90/270 pair |
| [0015](0015-considered-and-rejected-recognise-refine-crops.md) | 2026-11-03 | Rejected | Considered and rejected: recognise refine crops with RapidOCR's recogniser only (no detector) |
| [0016](0016-click-recogniser-frozen-dinov2-vit-s.md) | 2026-11-03 | Accepted | Click recogniser = frozen DINOv2 ViT-S/14 + registers, CLS pooling by default, ONNX Runtime CPU |
| [0017](0017-one-preprocessing-function-for-gallery-and.md) | 2026-11-03 | Accepted | One preprocessing function for gallery and query (pad to square, no stretch), versioned with the model; galler |
| [0018](0018-augmentations-in-opencv-numpy-instead-of.md) | 2026-11-03 | Accepted | Augmentations in OpenCV/NumPy instead of Albumentations |
| [0019](0019-margin-of-the-abstain-rule-measured.md) | 2026-11-03 | Accepted | Margin of the abstain rule measured on re-ranked scores; strong OCR (≥90, number exact, unique) can replace th |
| [0020](0020-ocr-of-a-portrait-click-crop.md) | 2026-11-03 | Accepted | OCR of a portrait Click crop is rotated 90° first, then read at 0°/180° |
| [0021](0021-local-feature-verification-with-opencv-akaze.md) | 2026-11-03 | Accepted | Local feature verification with OpenCV AKAZE + RANSAC (off by default) |
| [0022](0022-thresholds-t-accept-t-unknown-m.md) | 2026-11-03 | Accepted | Thresholds T_accept / T_unknown / M are placeholders (0.70 / 0.45 / 0.05) until `click_eval.py --write-thresho |
| [0023](0023-two-stage-mode-per-board-layout.md) | 2026-11-03 | Accepted | Two-stage mode: per-board layout (`data/layouts.json`, board-relative 0–1, board upright); MCU read only from  |
| [0024](0024-layout-placed-from-silkscreen-text-anchors.md) | 2026-11-03 | Accepted | Layout placed from silkscreen text anchors already read by stage 1 (RANSAC homography ≥ 6 / similarity ≥ 3), o |
| [0025](0025-anchors-inside-mcu-card-socket-areas.md) | 2026-11-03 | Accepted | Anchors inside MCU-card / socket areas are not stored |
| [0026](0026-board-identified-by-layout-when-the.md) | 2026-11-03 | Accepted | Board identified by layout when the name OCR abstains (≥ 8 consistent anchors, ≥ 50 % of matched, 2× the runne |
| [0027](0027-no-whole-photo-fallback-when-the.md) | 2026-11-03 | Accepted | No whole-photo fallback when the MCU-card ROI abstains (option `--roi-fallback`) |
| [0028](0028-calibrate-anchors-from-the-photo-as.md) | 2026-11-03 | Accepted | Calibrate anchors from the photo as stage 1 reads it, in addition to the rectified board |
| [0029](0029-weak-layout-fit-6-inliers-read.md) | 2026-11-03 | Accepted | Weak layout fit (< 6 inliers) → read the skipped rotation once at ≤ 1400 px and refit |
| [0030](0030-board-names-may-match-glued-uni.md) | 2026-11-03 | Accepted | Board names may match glued ("UNI-DSV8"), strict words verbatim in the joined line |
| [0031](0031-mcu-abstains-with-a-photo-too.md) | 2026-11-03 | Accepted | MCU abstains with a "photo too small" reason when the card crop < 1000 px |
| [0032](0032-card-title-identifies-a-layout-group.md) | 2026-11-04 | Accepted | Card title identifies a layout group; the card is decided by chip marking, PID label, or a single-card group — |
| [0033](0033-card-titles-parsed-structurally-mcu-card.md) | 2026-11-04 | Accepted | Card titles parsed structurally (`MCU CARD [N] for FAMILY`, `SiBRAIN for FAMILY`), number and family exact |
| [0034](0034-a-read-title-restricts-part-number.md) | 2026-11-04 | Accepted | A read title restricts part-number matching to the group's MCUs; a clear chip outside the group still gives th |
| [0035](0035-pin-labels-at-mikrobus-1-groups.md) | 2026-11-04 | Accepted | Pin labels at mikroBUS 1: groups whose distinctive pins were not seen lose what the best group gained |
| [0036](0036-card-mcus-incl-compatible-lists-of.md) | 2026-11-04 | Accepted | Card MCUs (incl. compatible lists of empty cards) added to the MCU catalogue; ≤3-char ordering suffixes fold i |
| [0037](0037-considered-and-rejected-lower-mcu-accept.md) | 2026-11-04 | Rejected | Considered and rejected: lower MCU accept threshold inside a known group |
| [0038](0038-0-6-after-the-board-locate.md) | 2026-11-04 | Accepted | 0.6: after the board, locate the MCU card itself, rectify it out of the ORIGINAL photo and zoom it (long side  |
| [0039](0039-card-located-primarily-by-its-closed.md) | 2026-11-04 | Accepted | Card located primarily by its closed yellow frame (component fill + morphological opening + rectangle fit); la |
| [0040](0040-frame-test-bridges-gaps-closing-3.md) | 2026-11-04 | Accepted | Frame test bridges gaps (closing 3 % of size) and strips attachments (opening 6 %) instead of using contour hi |
| [0041](0041-a-frame-candidate-that-encloses-another.md) | 2026-11-04 | Accepted | A frame candidate that encloses another candidate is scored ×0.6 |
| [0042](0042-three-read-passes-on-the-card.md) | 2026-11-04 | Accepted | Three read passes on the card at 0/90/180/270°: page, label boxes only (rest blanked), chip close-up (largest  |
| [0043](0043-chip-tokens-mapped-into-the-card.md) | 2026-11-04 | Accepted | Chip tokens mapped into the card frame axis-preserving (scaled into the chip's box), not through the chip's 45 |
| [0044](0044-full-text-match-signal-pin-pairs.md) | 2026-11-04 | Accepted | Full-text match: signal:pin pairs (1.0), pins (0.5), other strings (0.8), each × IDF over layouts; pins / pair |
| [0045](0045-labels-alone-may-name-a-layout.md) | 2026-11-04 | Accepted | Labels alone may name a layout only with net evidence ≥ 6 and ≥ 1.5× the runner-up; a few common labels never  |
| [0046](0046-a-title-read-without-a-number.md) | 2026-11-04 | Accepted | A title read without a number ("CARDFORKINETIS") also admits the numbered layouts of that family at 0.9; one o |
| [0047](0047-chip-reads-also-matched-against-each.md) | 2026-11-04 | Accepted | Chip reads also matched against each card's shop-photo marking (dataset `chip_marking_observed` → joined patte |
| [0048](0048-no-roi-whole-photo-fallback-when.md) | 2026-11-04 | Accepted | No ROI / whole-photo fallback when the card was located with score ≥ 0.8 and its crop gave nothing |
| [0049](0049-considered-and-rejected-reading-the-label.md) | 2026-11-04 | Rejected | Considered and rejected: reading the label boxes one by one (connected components → per-box OCR) |
| [0050](0050-faint-chip-markings-new-modular-pipeline.md) | 2026-11-05 | Accepted | Faint chip markings: new modular pipeline `vision/faint` (region → RGB/HSV/LAB features → multi-stage clusteri |
| [0051](0051-chip-body-found-by-low-texture.md) | 2026-11-05 | Accepted | Chip body found by LOW TEXTURE (local std of L), not darkness alone |
| [0052](0052-lighting-flattened-with-a-1-6.md) | 2026-11-05 | Accepted | Lighting flattened with a 1.6-glyph median per channel (not a Gaussian) |
| [0053](0053-robust-z-scores-get-a-floor.md) | 2026-11-05 | Accepted | Robust z-scores get a floor (1 level raw, 0.05 smoothed) |
| [0054](0054-text-likeness-area-of-glyph-sized.md) | 2026-11-05 | Accepted | "Text-likeness" = area of glyph-sized components chained into rows (similar height, small gap) or word-shaped; |
| [0055](0055-fisher-discriminant-over-all-10-colour.md) | 2026-11-05 | Accepted | Fisher discriminant over all 10 colour features, seeded by whichever of {cross-space vote, ±lightness} looks m |
| [0056](0056-multi-scale-variants-un-smoothed-projection.md) | 2026-11-05 | Accepted | Multi-scale variants: un-smoothed projection kept only inside the zone of the smoothed text |
| [0057](0057-reading-direction-from-geometry-glyph-chains.md) | 2026-11-05 | Accepted | Reading direction from geometry (glyph chains along x or y), OCR only at that axis' two turns |
| [0058](0058-reflection-streaks-solid-fill-0-72.md) | 2026-11-05 | Accepted | Reflection streaks: solid (fill > 0.72) line-length components dropped / blanked |
| [0059](0059-fused-consensus-lines-are-added-next.md) | 2026-11-05 | Accepted | Fused consensus lines are ADDED next to raw tokens, never replace them |
| [0060](0060-card-locate-order-quad-falls-back.md) | 2026-11-05 | Accepted | `card_locate.order_quad` falls back to angle ordering when the sum/diff rule returns a corner twice |
| [0061](0061-the-three-real-uni-ds-v8.md) | 2026-11-05 | Accepted | The three real UNI-DS v8 photos (960x1280, no EXIF, progressive JPEG q~70 = messenger copies) are NOT asked to |
| [0062](0062-card-frame-snapped-to-the-first.md) | 2026-11-05 | Accepted | Card frame snapped to the first CONTINUOUS yellow line inside the frame candidate (inner one when every side s |
| [0063](0063-chip-body-dark-threshold-relative-to.md) | 2026-11-05 | Accepted | Chip body "dark" threshold relative to the card (max(90, median+15)) |
| [0064](0064-package-check-chip-body-card-side.md) | 2026-11-05 | Accepted | Package check: chip body / card side -> QFP body -> pin count; cards with an impossible pin count leave the gr |
| [0065](0065-m-h-ocr-look-alike-title.md) | 2026-11-05 | Accepted | M<->H OCR look-alike; title tokens also joined in reverse order |
| [0066](0066-stacked-chip-lines-joined-up-to.md) | 2026-11-05 | Accepted | Stacked chip lines joined up to three when the first is short (logo line) |
| [0067](0067-resolution-model-marking-verifier-vision-faint.md) | 2026-11-05 | Accepted | Resolution-model marking verifier (`vision/faint/synth.py`: render candidates in a stroke font, blur + sample  |
| [0068](0068-super-resolution-model-plugged-into-the.md) | 2026-11-05 | Accepted | Super-resolution model plugged into the faint chip stage (`region.rectify(..., upscaler=)`: warp at native siz |
| [0069](0069-photo-too-small-threshold-1000-800.md) | 2026-11-05 | Accepted | "Photo too small" threshold 1000 -> 800 px card, message asks for the original file |
| [0070](0070-0-9-cleanup-removed-the-trocr.md) | 2026-11-05 | Accepted | 0.9 cleanup: removed the TrOCR crop reader, the `--global-enhance` CLI flag / UI toggle, `recognise_board` / ` |
| [0071](0071-smart-camera-acquisition-vision-camera-phone.md) | 2026-11-05 | Accepted | Smart camera acquisition (`vision/camera`): phone page streams a ~960 px preview PLUS a native-resolution crop |
| [0072](0072-phone-side-is-https-with-a.md) | 2026-11-05 | Accepted | Phone side is https with a self-signed certificate made by `openssl`; PC page on 127.0.0.1 only; QR token on e |
| [0073](0073-the-original-camera-file-is-transferred.md) | 2026-11-05 | Accepted | The ORIGINAL camera file is transferred (chunked, SHA-256) and saved byte for byte with a JSON sidecar of all  |
| [0074](0074-board-shot-first-mcu-close-up.md) | 2026-11-05 | Accepted | Board shot first; MCU close-up only while the MCU or the card is still open; close-up recognised with `Pipelin |
| [0075](0075-board-perspective-rectification-before-recognition-is.md) | 2026-11-05 | Accepted | Board perspective rectification before recognition is opt-in (`--rectify`) |
| [0076](0076-card-reader-find-chip-accepts-a.md) | 2026-11-05 | Accepted | `card_reader.find_chip` accepts a square chip body whose convex hull is a clean rectangle (hull/rect >= 0.9, a |
| [0077](0077-0-10-vision-camera-merged-into.md) | 2026-11-05 | Accepted | 0.10: `vision.camera` merged into `vision.app` (one asyncio server on 8765 + phone on 8443; bench routes serve |
| [0078](0078-scan-always-takes-two-photos-board.md) | 2026-11-05 | Accepted | Scan always takes two photos: board, then MCU close-up; auto-capture held between the steps until the user tap |
| [0079](0079-ui-restyled-after-necto-studio-7.md) | 2026-11-05 | Accepted | UI restyled after NECTO Studio 7 (flat near-black panels, amber outlined / filled buttons, setup card with ico |
| [0080](0080-scan-processing-starts-only-when-both.md) | 2026-11-05 | Accepted | Scan processing starts only when both photos are in; board photo: fast board-only pass (2000 px, no MCU / card |
| [0081](0081-no-imagecapture-takephoto-on-webkit-ios.md) | 2026-11-05 | Accepted | No `ImageCapture.takePhoto()` on WebKit (iOS / iPadOS Safari); video-frame still + camera watchdog (reopen whe |

## PHOENIX decisions (0100+)

| ADR | Title |
|---|---|
| [0100](0100-phoenix-layered-architecture.md) | ADR-0100: PHOENIX is a layer around `vision/`, which stays protected |
| [0101](0101-legacy-camera-and-app-module.md) | ADR-0101: `vision/camera/` and `vision/app.py` are legacy |
| [0102](0102-parameter-registry.md) | ADR-0102: One parameter registry, generated from the code, with binding detection |
| [0103](0103-benchmark-datasets-and-scoring.md) | ADR-0103: Benchmarks are manifests; abstaining is never wrong; unverified data only measures agreement |
| [0104](0104-socket-geometry-single-source.md) | ADR-0104: Socket geometry comes from `data/layouts.json`; `data/sockets.csv` is derived |
| [0105](0105-human-approval-for-promotion.md) | ADR-0105: No autonomous code modification; promotions need a named human approval |
| [0106](0106-version-control.md) | ADR-0106: Git from v0.10, photos and models outside history |

New decisions: copy `TEMPLATE.md` to the next free number from 0100 up.
