# MIKROE recognition — vision steps 1–3b + Click retrieval (T7)

> 0.9 (5 Nov 2026): **smart camera acquisition** — the phone becomes a guided camera for
> the PC over the local Wi-Fi (QR pairing, no cloud); live quality checks around the MCU
> marking, auto-capture of the full-resolution original, hand-off to the pipeline. Code
> cleanup. See "0.9" below.

> 0.8 (5 Nov 2026): low-resolution real photos — card frame snapping, package-size check,
> PIC32 three-line markings, resolution sweep, opt-in resolution-model verifier and SR in the
> faint chip stage — see "0.8" below.
> 0.7 (5 Nov 2026, experimental): faint chip markings — colour clustering + many thresholds +
> OCR of every variant + fusion, opt-in `--chip-mode faint|both` — see "0.7" below.
> 0.6 (4 Nov 2026): MCU card located on the photo, cropped + zoomed, ALL its text read and
> matched against the card dataset — see "0.6" below.
> 0.5 (4 Nov 2026): MCU cards identified from the scraped card dataset — see "0.5" below.

Implements the first part of the prototype plan (2 Nov 2026), updated on
3 Nov 2026 with the research brief *"OCR of small text on circuit boards and
MCUs"* (see "What changed in 0.2" below).

| Plan step | Module | What it does |
|---|---|---|
| 1. Photo in | `vision/loader.py` | Loads files / folders / uploads / video frames; applies EXIF rotation; parses `MIKROE-1234_phoneA_lamp_20deg_01.jpg` labels; content hashes for frozen splits; optional `labels.csv` |
| 2. Preprocess | `vision/preprocess.py` | Quality check (blur, glare, exposure → retake warnings), resize, glare inpainting, small-angle deskew, CLAHE, 4-click perspective warp, OCR variants, tiles. Tracks a 3×3 matrix back to the original photo |
| 3a. Dev board | `vision/board_ocr.py` | OCR the silkscreen → IDF-weighted fuzzy match of board names **within one text line**; tries 0/180° then 90/270°; abstains below threshold or on small top-1/top-2 margin; hook for the T7 embedding fallback |
| 3b. MCU card stage | `vision/card_locate.py`, `vision/card_reader.py`, `vision/card_text.py` | **0.6** find the card (yellow frame / layout ROI / title), rectify from the original photo, zoom ×≤4, read page + label boxes + chip close-up at 4 turns, match every label to the layout's full silkscreen profile |
| 3b. MCU / card | `vision/mcu_ocr.py`, `vision/card_ocr.py` | OCR chip marking → OCR-aware part-number match (accepts ordering suffixes, repairs `5TM32`→`STM32`); MCU-card label → its MCU; tiled hi-res pass for tiny markings; unlisted part numbers reported, never forced |
| glue | `vision/pipeline.py`, `python -m vision` | photo → partial manifest `{board, mcu, mcu_card, clicks: [], needs_user}` |
| 2+. Enhance | `vision/enhance.py` | **0.2** Text enhancement: bicubic ×N or super-resolution (ESPCN/FSRCNN/EDSR), non-local-means denoise, Richardson–Lucy deblur, CLAHE, gamma lift, Otsu / Sauvola binarization |
| 3b+. Refine | `vision/refine.py` | **0.2** Two-stage *detect → crop → enhance → recognise* pass for tiny / blurred / dark markings |
| App | `vision/app/` | **0.10** One window: **Scan** (phone camera, two steps: board, then MCU close-up) and **Photos** (drop photos: boxes, crops, candidates, 4-click perspective, save labelled photos) |
| T8 eval | `eval/ocr_eval.py` | Writes `eval/ocr_eval.md`: per-kind accuracy, precision when answering, top-3, **CER / WER**, **config ablation**, **degradation robustness**, by distance, glyph size |
| T7 Clicks | `vision/clicks/` | **0.3** DINOv2 ViT-S/14 + registers (ONNX Runtime) → exact cosine search over a PID-tagged gallery → TTA / OCR / size-prior / local-feature re-rank → accept / ask / unknown. See "Click recognition" below |
| T7 eval | `eval/click_eval.py` | **0.3** Session-based splits, threshold calibration on val, top-1/top-3/macro recall/hard pairs/open set/buckets/p95 → `eval/results.csv`, contact sheet of the 20 worst failures |

## Run

```bash
pip install -r requirements.txt          # + one OCR engine, see below
python -m vision.app                     # the app -> http://127.0.0.1:8765 (Scan with the phone + Photos)
python -m vision photos/ --summary       # one line per photo
python -m vision photo.jpg               # full JSON (candidates, evidence boxes, timings)
python -m vision photos/ --debug dbg/    # annotated images for failure review
python eval/ocr_eval.py photos/          # needs photos/labels.csv: filename,kind,label[,distance_cm]
python eval/ocr_eval.py photos/ --configs baseline,preproc,full          # what each change buys
python eval/ocr_eval.py photos/ --degrade none,defocus4,motion9,dark,jpeg25,half
python scripts/make_synthetic.py syn/ --hard   # synthetic boards (+ hard small/blurred/dark chips)
python scripts/get_sr_models.py          # optional super-resolution models (needs opencv-contrib)
python -m pytest -q
```

## 0.10 — one app, two-step scan (5 Nov 2026)

`vision.camera` is merged into `vision.app`: one command, one server, one window.

```bash
python -m vision.app                     # opens http://127.0.0.1:8765; scan the QR with the phone
python -m vision.app --no-recognize      # scan captures only -> photos/captures/
python -m vision.app --chip-mode both --upscaler espcn --gallery gallery/
```

* **Scan** (left rail): QR pairing, the phone's live view with the quality overlay, a
  step bar, the setup card (board / MCU / MCU card / Clicks, with a chooser where the
  pipeline asks the user), the captured photos, Activity / Manifest console. Status bar:
  phone state, step, *Active setup*.
* **Photos**: the former drag-and-drop test window (same features), settings in the header.
* **Both photos first, then processing.** Nothing is recognised until the close-up is
  in. Then the board photo gets a fast board-only OCR pass (`State.board_pipeline`:
  silkscreen name at 2000 px, no MCU / card / refinement / layout fit — real 4K board
  photo with Tesseract 5 s instead of 20 s) and the close-up the full MCU reading with
  the board as a hint. Only if the close-up gives no MCU (or the fast pass misses the
  board) does the board photo get the full pass as a fallback.
* iPhone / iPad Safari: `ImageCapture.takePhoto()` is not used (it failed there and left
  the camera feed black — the "black screen at the MCU step"); the 4K video frame is the
  still. A watchdog reopens the camera whenever the feed stalls or ends (also after the
  lock screen / another app).
* The scan **always** has two steps: (1) the whole board, (2) the phone moved close to
  the MCU card. After step 1 the phone shows the recognised board and how to hold the
  phone; auto-capture is held until *Continue* (or 4 s). Step 2 finishes the scan; the
  phone shows the result. *Rescan board*, *Retake MCU*, *Take photo now*, *New scan*
  on the PC; *Retake the close-up* / *Scan another board* on the phone.
* Layout: `vision/app/server.py` (both listeners, bench routes `/api/recognize|rerun|label`,
  recognition on one worker thread), `bench.py` (former `app.py`: state, pipelines,
  response building, labelling), `quality.py` / `net.py` / `qr.py` (moved from
  `vision/camera/`), `web/index.html` (app), `web/phone.html` (camera).
* Ports: app 8765 (`--port`), phone 8443 (`--phone-port`). The old PC page on 8766 is gone.

## 0.9 — smart camera acquisition (5 Nov 2026)

(0.10: the commands below are now `python -m vision.app`; files moved from
`vision/camera/` to `vision/app/`; the close-up is no longer optional.)

Every real photo so far was a 960x1280 messenger copy: chip body ~75 px, marking ~3 px,
unreadable by any OCR (0.8). The camera system makes the phone take the photo the OCR
needs, and checks it before recognition.

```bash
python -m vision.camera                  # PC page opens; scan its QR code with the phone
python -m vision.camera --no-recognize   # capture only -> photos/captures/
python -m vision.camera --chip-mode both --upscaler espcn     # pipeline options as in python -m vision
```

```
phone (browser, https)                          PC (python -m vision.camera)
  camera stream 4K/1080p ── preview 960 px + ──►  quality.FrameAnalyzer  ── report ──► phone overlay
                            native crop of the        board / card / chip,           + PC live view
                            MCU card (WebSocket)      10 metrics, instruction
                         ◄── "capture" ──────────  CaptureGate (stable N frames + focus at peak)
  takePhoto() full res ──── original file, chunked, sha256 ──► analyze_photo (retake?) ──► Pipeline.run
                         ◄── result / "now the MCU close-up" ◄── session: board shot + close-up
```

| | module | what it does |
|---|---|---|
| live checks | `vision/camera/quality.py` | **board**: outline from three proposals (dark PCB region, hull of the yellow silkscreen, edge contour) -> found / framing (cut off, too small, turn to landscape) / centred / in-plane rotation / **perspective** (opposite-side ratio, corner angles -> "lower the top edge") · **exposure** (p98 of the board, clipped share), **glare** on the board, **noise** · **steadiness** (phase correlation between frames + phone gyroscope) · **MCU**: the card is found with `card_locate`; the phone then streams that area at NATIVE stream resolution, where `find_chip` measures the chip: **size it will have in the full photo** (stream px x still/stream ratio; >= 250 px needed, see 0.8), **focus** (no-reference re-blur metric, also must be within 8 % of the best focus of the last 4 s, i.e. autofocus has settled), **motion blur** (horizontal vs vertical blur), **glare on the chip** (also when a reflection hides the chip body), marking contrast |
| capture gate | `quality.CaptureGate` | auto-capture after 3 passing frames over >= 450 ms; never before the MCU focus was measured on native pixels |
| photo check | `quality.analyze_photo` | the full-resolution still is re-checked (board, chip size, chip blur, chip glare); blurred / glare -> automatic retake (2x, then the best photo is used and flagged); board shot with a chip < 250 px -> accepted, close-up requested |
| server | `vision/camera/server.py`, `net.py`, `qr.py` | asyncio, standard library only. Phone side: https on the LAN (self-signed certificate made once with `openssl`, `~/.cache/mikroe-vision/camera-tls/`; the camera API refuses plain http), token from the QR code on every request, one phone at a time. PC side: 127.0.0.1 only. WebSocket (RFC 6455) for preview frames, analysis and the photo; the photo is the camera's ORIGINAL file, sent in 256 kB chunks with a SHA-256 check and saved byte for byte to `photos/captures/` with a JSON sidecar (preview metrics, photo check, phone camera info, recognition result) |
| phone page | `vision/camera/web/phone.html` | guide frame (board) / square (MCU close-up), detected board / card / chip outlines, one big instruction + hints, metric chips, progress ring; focus + exposure point set on the chip (`pointsOfInterest`), tap to focus, torch, flash forced off (glare), wake lock; still via `ImageCapture.takePhoto()` at the sensor's full size, else the native video frame; "Camera app" button = the phone's own camera app (file input), checked the same way |
| PC page | `vision/camera/web/pc.html` | QR code(s), live view with overlays and all metrics, captures with their check, result (board / MCU / card / what is still open), zoomed card + chip crop |
| two-shot session | `server._advance`, `Pipeline.run(board_hint=)` | board shot first; if the MCU or the card is still open, the phone switches to the MCU close-up; the close-up is recognised with the board from shot 1 (`board_hint`: used only when the close-up's own board read abstains; no layout placement on a close-up) and the manifests are merged |

Measured here (headless Chromium with a fake 1920x1440 camera showing a synthetic UNI-DS
board + MCU CARD 29 for STM32 / STM32F215RE; Tesseract):

| shot | chip in photo | capture | result |
|---|---|---|---|
| whole board | 195 px (predicted 195) | auto, ~1.5 s after start | MCU STM32F215RE, card decided (board name too small in this synthetic scene) |
| MCU close-up | 319 px | auto, ~1.5 s | before the `find_chip` fix below: chip body not found, MCU abstains; after: STM32F215RE, card decided |

Degraded streams never trigger a capture (tests/test_camera.py): Gaussian blur 2.2 px,
camera moving 60 px/frame, gyroscope 30°/s, exposure x0.25, 12 % keystone (-> "lower its
top edge"). A 30 px reflection on the chip makes the full-photo check ask for a retake.

`card_reader.find_chip` fix: on a sharp close-up the legible marking is high-texture and
cut a notch into the chip's "dark + smooth" mask (fill 0.7499 < 0.75) — a BETTER photo lost
the chip. A square body whose convex hull is a clean rectangle (hull / rect >= 0.9, area /
hull >= 0.6) is now accepted. Unchanged on the three real photos; synthetic card set: see
DECISIONS.md.

Thresholds in `quality.py` are [ASSUMPTION] values checked on synthetic blur / glare and
the three real photos; every capture's sidecar JSON holds the metrics needed to tune them
on real phones.

Phone notes: same Wi-Fi as the PC (guest networks often isolate clients); accept the
certificate warning once (Advanced -> Proceed); Linux firewall: `sudo ufw allow 8443/tcp`.
Chrome on Android gives the full sensor resolution through `ImageCapture`; Safari on iOS
has no `ImageCapture`, so the still is the largest video frame (often 1920 or 3840 px) —
use the close-up step, or the "Camera app" button for the native camera's full resolution.

**Cleanup in 0.9.** Removed: the TrOCR crop reader (`--crop-recognizer`, UI select, the
`trocr` eval config — never runnable or measured here); the `--global-enhance` CLI flag and
UI toggle (measured harmful, DECISIONS 3 Nov; the preprocess options stay for
`eval/ocr_eval.py --configs preproc`); two unused one-shot wrappers
(`recognise_board`, `recognise_mcu`) and unused imports. Everything else was kept: the
board OCR, MCU / card readers, the faint-marking colour clustering, enhancement / SR,
card / chip / socket locators, layouts, Clicks (not runnable until the DINOv2 model and a
gallery exist, but tested end to end with a toy model) and the evaluation scripts.

## 0.8 — low-resolution real photos (5 Nov 2026)

Three real UNI-DS v8 photos (`tests/real`): SiBRAIN for PIC32MZ2048EFH144 and two of
MCU CARD for STM32 (STM32F429ZIT6). All are 960x1280 messenger copies (no EXIF,
progressive JPEG q~70): the card is 230-260 px, the chip body 71-77 px, a marking glyph
~3.5 px high and ~2 px wide. The faint pipeline still separates the PIC32 logo and three
rows of blobs on p1, but only the ST / ARM logos on p2/p3 — **the part number is not in
these pixels**, so no upscaler (bicubic or SR) and no matcher can name the exact card.

| photo | before | 0.8 |
|---|---|---|
| p1 PIC32MZ2048EFH144 | layout 'MCU CARD 3 for PIC32', 10 cards (two layouts share "SiBRAIN for PIC32") | same layout, **4 cards** (package check drops the six 100-pin MX cards); MCU abstains |
| p2 STM32F429ZI | card crop = board socket outline, **no title read** | card snapped to its frame, title read -> 'MCU CARD for STM32', 19 cards; MCU abstains |
| p3 STM32F429ZI | 'MCU CARD for STM32', 19 cards | same; MCU abstains |

No photo gives a wrong card or MCU. The abstain reason now says the card is too small and
asks for the original file (messengers shrink photos to ~1280 px).

What changed:

* `card_locate._snap_to_frame` — the card frame touching the board's yellow socket outline
  made one component; each side now snaps to the first *continuous* yellow line (the inner
  one when outline + gap + frame are all closed).
* `card_reader.find_chip` — "dark" is relative to the card (over-exposed p2: body at 102).
* `vision/package.py` — chip body / card side -> QFP body -> pin count; cards with an
  impossible pin count leave the layout group (144-pin ratio ~0.31, 100-pin ~0.22).
* titles: M<->H look-alike, title tokens also joined in reverse order ('for STH32' + 'MCU CARD').
* chip markings: up to three stacked lines joined after a short logo line
  ('PIC32' / 'MZ2048EFH' / '144-I/PH'); PIC32 markings were never accepted before, even
  when every line was read correctly.
* `vision/faint/synth.py` (opt-in `--verify-marking`) — the "resolution model": render each
  candidate marking of the layout in a stroke font, blur + sample it at the photo's native
  pixel size, cubic up like the chip crop, and correlate the *differing* character cells with
  the faint pipeline's lightness residual; decides only with a margin.
* `faint.region.rectify(..., upscaler=...)` — with `--upscaler espcn|fsrcnn|edsr` the chip is
  warped at native size and the SR network enlarges it before the colour features.
  Models: `python scripts/get_sr_models.py` (needs opencv-contrib). Not measured yet: the
  build environment could not download the models.

Resolution sweep (`eval/synth_eval.py`, synthetic chips drawn in a different font than the
verifier renders, 45°, blur, JPEG q70; `eval/synth_eval_synthetic.md`):

| layout (cards) | chip px in photo | OCR (faint) ✓/✗/– | verifier ✓/✗/– |
|---|---|---|---|
| MCU CARD 3 for PIC32 (4) | 70 (= real photos) | 0/0/3 | 0/0/3 |
| | 140 | 0/0/3 | 1/0/2 |
| | 200-400 | 1/0/2 each (EFE vs EFF: E/F look-alike, abstains by design) | 1/0/2 each |
| MCU CARD for STM32 (19) | 70 / 140 / 200 | 0/0/3 | 0/0/3 |
| | 250 / 400 | **3/0/0** | 0/0/3 |

Nothing wrong at any size. The verifier never errs but rarely decides — the real laser font
is unknown, so the rendered candidates only approximately match; that is why it is opt-in.

Take-away: **a chip body of ~250 px (card ~800 px) is needed** — a phone's original photo of
the board from ~30 cm, or any close-up of the card. Send the photo as a file, not as a
compressed messenger image.

```bash
python -m vision photo.jpg --summary                                  # default
python -m vision photo.jpg --chip-mode faint --upscaler espcn --verify-marking   # all of it
python eval/synth_eval.py --ocr --out eval/synth_eval_synthetic.md    # resolution sweep
```

## 0.7 — experimental: extremely faint chip markings (5 Nov 2026)

`vision/faint/` — one module per stage, each writes debug images:

| stage | module | what it does |
|---|---|---|
| 1 region | `region.py` | dark central region = MCU body: low-TEXTURE detector (local std of L — on black solder mask the board is as dark as the chip) + darkness, scored by area / squareness / centrality; card stage's blob detector as a second candidate; rectified + enlarged (~44 px glyphs); body mask without pins / rim |
| 2–3 features | `features.py` | per pixel RGB, HSV (hue as S·cos/S·sin), LAB; lighting removed per channel by a 1.6-glyph median (follows glare and gradients, ignores strokes); robust z-scores with a floor (JPEG chroma blocks); smoothing at stroke scale; glare marked, never deleted |
| 4 cluster | `cluster.py` | A: k-means (k=4) per colour space, text clusters picked by "text-likeness" → B: cross-space vote 0–3 → C: Fisher discriminant over all 10 features, seeded by the most text-like of {vote, ±lightness}, oriented so marking = high → D: local 2-class re-split inside the possible-text zone |
| 5 threshold | `threshold.py` | global z > 1.5 / 2.5 / 4, Otsu, hysteresis, adaptive Gaussian / Niblack / Sauvola / Wolf at 1× and 2.5× glyph — on the Fisher map and on lightness alone |
| 6 preserve | `preserve.py` | gentle cleanup: only specks, solid glare patches and solid reflection streaks go; hysteresis keeps faint pixels attached to clear ones; "text-likeness" = glyph-sized components chained into rows |
| 7 variants | `variants.py` | soft gray (Fisher score, flattened L), multi-scale (un-smoothed projection inside the smoothed text zone), vote, local split, best N thresholded masks |
| OCR | `reader.py` | reading axis from geometry → each variant at 2 turns; `read_crop` (whitelist, no dictionaries) or `read_text` |
| 8 fusion | `fusion.py` | lines clustered across variants, variant reliability = agreement with the others, character vote → consensus tokens added next to the raw ones |

```bash
python -m vision.faint chip.jpg --debug dbg_faint/ --compare   # close-up: stages + classic vs faint
python -m vision photo.jpg --chip-mode both --faint-debug dbg/   # whole pipeline, card stage chip pass
python scripts/make_faint_chips.py syn_faint/ --n 6             # synthetic faint chips with labels
python eval/faint_eval.py syn_faint/ --ocr crop,page --out eval/faint_eval.md
```

Synthetic faint chips (24: contrast 30 / 14 / 8 / 5 levels, glare, gradient, noise, 45°;
`eval/faint_eval_synthetic.md`), same rectified chip for both, OCR call `crop`:

| contrast | classic ✓/✗/– | faint ✓/✗/– | truth in top-3 classic / faint | mean CER classic / faint |
|---|---|---|---|---|
| 30 | 0/0/6 | **5/0/1** | 0 / 6 | 0.70 / 0.12 |
| 14 | 0/0/6 | 1/0/5 | 0 / 5 | 0.89 / 0.23 |
| 8 | 0/0/6 | 0/0/6 | 0 / 1 | 0.91 / 0.51 |
| 5 | 0/0/6 | 0/0/6 | 0 / 0 | 0.88 / 0.77 |
| all | 0/0/24 | **6/0/18** | 0 / 12 | 0.85 / 0.41 |

No wrong answer from either. Easy chips (contrast 140, no glare): both 2/2. With the
`page` OCR call faint is 5/0/19. Fusion adds no decision on this set (ablation column
in the report); ~3.3 s per chip vs ~1 s. Real photos (`tests/real`): the faint pass runs
on the located chip (~60 px, marking ~3 px) and, like classic, finds nothing to read —
same "send a close-up" abstain, +1 s. **Default stays `classic`** until it is measured on
real close-ups with faint markings (label them `kind=mcu`, run `eval/faint_eval.py`).

Also fixed: `card_locate.order_quad` returned one corner twice for a square at exactly
45° (a diagonally soldered QFP), so the chip close-up was warped from a degenerate quad;
it now falls back to angle ordering in that case only.

## 0.6 — MCU-card stage: locate → crop + zoom → read everything → match (4 Nov 2026)

```
stage 1    board (unchanged)
stage 3b-1 locate the card  vision/card_locate.py
             frame   the card's closed yellow frame (146/207 cards: black PCB, gold frame);
                     the board's yellow lines are open — rectangularity 0.02 vs 0.91–0.98
             layout  the calibrated MCU-card ROI (data/layouts.json), if placed
             text    "MCU CARD …" / "SiBRAIN …" / port box read by stage 1
           → best quad (agreement raises the score), rectified out of the ORIGINAL photo,
             zoomed until the long side is 1800 px (×4 max; --upscaler espcn/edsr if installed)
stage 3b-2 read ALL text    vision/card_reader.py, each at 0/90/180/270°
             page    the whole card
             labels  only the white pin boxes / yellow signal boxes, everything else blanked
             chip    largest dark square = MCU body (often at 45°), rectified, enlarged,
                     Otsu / Sauvola — the marking is what separates cards of one layout
stage 3b-3 match            vision/card_text.py + card_ocr.py + mcu_ocr.py
             title + port box           → layout group (as 0.5)
             mikroBUS signal→pin labels → group profile, IDF-weighted ("INT:PD14" rare,
                                          "SCK:PA5" common); names the layout even
                                          without a title read
             chip marking               → part number, or the card's shop-photo marking
                                          ('MK64 / FN1M0V12', 'M20A7V', '32F215RET6')
             PID label                  → card
```

Run: nothing new to pass — `python -m vision photo.jpg` runs it. `--no-card` = 0.5
behaviour, `--card-zoom 2400` for a larger crop, `--debug dbg/` also writes
`<photo>_card.jpg` (zoomed card, chip outlined) and `<photo>_card_chip.jpg`. JSON
output gets a `card` block (location method / score / size / zoom, chip box, text
lines read per pass); the test window draws the located card (pink) and shows the
zoomed card, the chip close-up and everything read under "MCU card crop".

Dataset import (re-run `scripts/import_mcu_cards.py` on the folder): each group now
carries `text` — full mikroBUS pin map (11 groups), all port-pin labels (incl. the
37 MB1 fingerprints), other strings, legacy pin-header rows (6 groups), PCB / frame
colour — and each card `markings`, the shop-photo chip marking joined the way OCR
sees it (WRONG_CHIP and blank photos excluded).

Evaluation: `python scripts/make_synthetic_cards.py syn_cards/ --n 12 --presets close,mid,far`
renders cards FROM the dataset (real titles, every pin-map label, port box, chip with
the shop-photo marking, chip at 45° on every other card) into a board scene with
board port labels as distractors; `python eval/card_eval.py syn_cards/ --compare`
runs the card stage and the 0.5 whole-photo pass on the same photos.

Synthetic set (12 cards from 9 layouts × close / mid / far, Tesseract, 2-core CPU;
`--compare` = the 0.5 whole-photo pass on the same photos, no layout calibrated):

| distance (card long side) | n | card located | layout correct | card correct / wrong | MCU correct / wrong | median s | 0.5: layout / card ok / wrong | 0.5 median s |
|---|---|---|---|---|---|---|---|---|
| close (~1450 px) | 12 | 12 | 12 | **9 / 0** | 9 / 0 | 8.1 | 6 / 2 / 0 | 28.8 |
| mid (~920 px) | 12 | 12 | 11 | **10 / 0** | 10 / 0 | 7.8 | 10 / 2 / 0 | 26.8 |
| far (~480 px) | 12 | 11 | 11 | 1 / 0 | 1 / 0 | 14.0 | 9 / 1 / 0 | 20.1 |
| all | 36 | 35 | 34 | **20 / 0** | 20 / 0 | 8.8 | 25 / 5 / 0 | 25.9 |

(0.5 columns over 33 photos: the 3 STM32F401RB scenes were first rendered with the
dataset's WRONG_CHIP marking — the shop photo shows an STM32F205RB — and both paths read
"F205RB"; the renderer was fixed and the card stage re-run on them: abstain ×3.)
Abstentions: far — the chip marking is ~5 px, the layout is still named (the
manifest then offers the layout's cards); close/mid — one-character neighbours
(L432KC / L442KC, F401RB) and one unread chip. No wrong card or MCU anywhere.

Real photos (`tests/real`, 960×1280, Tesseract): the card is found on both (frame;
photo 1 also by the stage-1 title read and the layout ROI, all three agree), cropped
234×231 / 270×286 px and zoomed ×4 — but its labels are ~5 px and the chip marking
~3 px in these photos, so nothing on it is readable and the MCU abstains with the
"send the original photo or a close-up" reason, as in 0.5. Time 31 / 26 s (0.5:
25 / 26 s). **Card-level accuracy on real photos is still unmeasured** — it needs
close-ups or full-resolution photos (≥ ~900 px card), labelled `kind=mcu_card`.

## 0.5 — MCU-card dataset: title → layout group, chip → card (4 Nov 2026)

Source: `mikroe_mcu_cards_ocr/` (scraped from mikroe.com 4 Nov 2026: 207 cards, 62
silkscreen layout groups, specs, PIDs, supported dev boards, port-pin labels read from
the shop photos). Imported once into `data/mcu_cards.json`:

```bash
python scripts/import_mcu_cards.py ~/Downloads/mikroe_mcu_cards_ocr   # re-run when the dataset changes
```

What the data changed: a card title names a **layout group**, not a card. "MCU CARD for
STM32" is printed on 19 cards, "MCU CARD 29 for STM32" on 10, "EasyMx PRO v7" on 21;
inside a group only the soldered chip (and the PID sticker) differs. The old rule
"card label → its MCU" would have guessed one of 19. Now (`vision/card_ocr.py`):

| step | evidence | effect |
|---|---|---|
| group | title parsed as `MCU CARD [N] for <FAMILY>` / `SiBRAIN for <FAMILY>` (OCR look-alikes, glued words, two stacked lines) | number and family must match exactly: `for STM32` ≠ `2 for STM32`, `STM32` ≠ `STM32L4` |
| | port-width box `16-BIT PORT` / `8-BIT PORT` | disagreeing layouts ×0.7 |
| | port-pin labels at mikroBUS 1 (`PD2 PD3…` vs `PE0 PE1…`) | separates layouts with the same title (SiBRAIN for ATmega, PIC24/dsPIC33) |
| | legacy logo text (EasyPIC PRO v7, EasyMx PRO v7, LV18F v6, BIGAVR6 …) | name match, as boards |
| | identified dev board | cards whose supported-boards list excludes it drop out |
| card | PID label `MIKROE-1234` read anywhere | that card (0.99) |
| | group has one card | that card, and its MCU is implied (as before) |
| | chip marking, **matched only against the group's MCUs** | the card carrying the decided MCU; populated beats empty |
| | chip reads clearly as a part no card of the group carries | MCU from the chip, card left open ("conflict") |
| | nothing else | abstain; `needs_user` lists the group's cards, `manifest.mcu_card_layout` names the layout |

Catalog: all 224 card MCUs (incl. the compatible-MCU lists of empty cards) are added to
`mcus`; ordering suffixes fold into one part (`STM32F407VGT6` → `STM32F407VG`). The two
`[ASSUMPTION]` rows of `data/mcu_cards.csv` were replaced by the dataset (the file stays
for cards the dataset does not have). `fix_prefix` no longer "repairs" `MSP432…` to
`MSP430…`; prefixes AT90, MKV, LM3S/LM4F, MSP432, CEC, FT9 added.

Synthetic set (`make_synthetic.py --hard`, Tesseract, `eval/ocr_eval.py`):

| | board | MCU correct | MCU wrong | MCU in top-3 | median |
|---|---|---|---|---|---|
| 0.4.1 (19 MCUs) | 6/6 | 8/10 | 0 | 10/10 | 4.1 s |
| 0.5 (224 MCUs) | 6/6 | 8/10 | 0 | 9/10 | 3.8 s |

Margins fell from ~0.25 to ~0.09: the catalogue now holds the real one-character
neighbours (F429ZI/F439ZI, 65K40/66K40). hard_defocus7 (read `STM32PAISZITG`, abstains
in both) lost the true part from its top-3 to those neighbours. Real photos (Tesseract):
unchanged — board found, MCU abstains (3 px marking). Card-level accuracy on real photos
is **not measured yet**: needs close-ups of cards, labelled `kind=mcu_card` with the card
key (e.g. `mcu-card-for-stm32-stm32f429zi`). `tests/test_cards.py`: 14 token-level tests.

Data caveats (from the dataset's own notes): 40 of 62 group titles are inferred from the
naming rule, not read from a photo; SiBRAIN STM32H7 / PIC32MZ1024 listings reuse the
"MCU CARD for STM32" / "MCU CARD 3 for PIC32" photos, the physical card may read
"SiBRAIN for …" — both titles are accepted. Legacy cards have no supported-boards data.

## 0.4.1 — UNI-DS v8 calibrated, real-photo test, NECTO-style window (3 Nov 2026)

**Test photos** (`tests/real/`, phone photos 960×1280 of UNI-DS v8 + MCU CARD for STM32 /
STM32F429ZIT6; layout calibrated from photo 1, photo 2 held out — board cut off at two edges):

| RapidOCR | board | layout placement | MCU read from | time (2-core CPU) |
|---|---|---|---|---|
| before (no layout, 0.4) | **? both** (0.71 < 0.72) | — | whole photo | 21–23 s |
| after, photo 1 | **UNI-DS v8** 0.97 | homography 8/8 labels, rms 1.1 px | MCU-card crop only | 6.2 s |
| after, photo 2 (held out) | **UNI-DS v8** 0.97 | homography 13/13 labels, rms 2.3 px | MCU-card crop only | 8.8 s |

Tesseract identifies the board too (layout fit weaker: the anchors were calibrated with RapidOCR).
The MCU abstains on both photos, correctly: the MCU card is only ~260–290 px in these
photos, the chip marking is ~3 px high. The reason now says so and asks for the original
photo or a close-up (`Pipeline.MCU_ROI_MIN_PX = 1000`, [ASSUMPTION]).

Changes:
* `textmatch.NameIndex`: glued names ("UNI-DSV8" for "UNI-DS v8") match through the
  compacted name; version / number words must still appear verbatim (EASYPICV8 ≠ EasyPIC v7).
* `layout.calibrate` also reads the photo the way stage 1 does at run time
  (`anchors_from_photo`) — 47 → 73 anchors, and they are the texts the pipeline actually reads.
* `Pipeline.run`: if the layout fit has < 6 inliers, the rotation stage 1 skipped is read once
  at ≤ 1400 px and the layout re-fitted (photo 2: 3 → 13 labels for +2.7 s).
* `data/layouts.json`: UNI-DS v8 (MCU card + mikroBUS 1–5); `data/mcu_cards.csv`:
  MCU CARD for STM32 STM32F429ZI [ASSUMPTION: name/PID from memory, check against the product DB].
* Test window restyled after NECTO Studio 7.5 (dark chrome, icon rail, explorer tree, Issues
  panel); it now shows the stage-1b crops (MCU card + every socket) that stage 2 will use.
* `tests/test_real_photos.py` (needs `rapidocr_onnxruntime`).

## Two-stage mode: board → stored layout → only the MCU-card area — 0.4

```
stage 1   OCR the whole photo once → board name → board (unchanged)
          name unreadable? → board identified from its layout's labels (RANSAC fit)
stage 1b  data/layouts.json: place the board's layout on the photo from the text
          stage 1 already read (no extra OCR) → rectified, full-resolution crops of
          the MCU card and every mikroBUS socket
          MCU step reads ONLY the MCU-card crop (chip marking + "MCU CARD" label)
stage 2   (next) DINOv2 Click recognition on each socket crop, slot name known
```

Boards without a layout (or photos where it cannot be placed) use the old
whole-photo MCU pass, so nothing gets worse for them.

**Placing a layout** (`vision/layout.py`): calibration stores every unique
silkscreen text on the board ("USB UART", "mikroBUS 1", "CODEGRIP", …) with its
board-relative position, as *anchors*. On a new photo the tokens stage 1 read are
matched to the anchors and RANSAC fits board → photo (homography with ≥ 6
inliers, similarity with ≥ 3). Text inside the MCU-card / socket areas is never an
anchor (it belongs to the card or Click that is plugged in). Fallback: board
outline (4-click quad or auto outline) with the anchors choosing the orientation.
ROIs are grown 6 % (card) / 4 % (sockets) to absorb placement error.

**Calibrating a board** (once per board, one top-down photo; a second adds anchors):

```bash
python -m vision.layout rectify photo.jpg --corners TLx,TLy,TRx,TRy,BRx,BRy,BLx,BLy
#   corners in board-upright order (as the board name reads) -> photo_board.jpg
#   measure x,y,w,h of the MCU card and each socket in photo_board.jpg, then:
python -m vision.layout calibrate photo.jpg --board uni-ds-v8 --corners ... \
    --mcu x,y,w,h --socket "mikroBUS 1=x,y,w,h" --socket "mikroBUS 2=x,y,w,h"
python -m vision.layout calibrate photo2.jpg --board uni-ds-v8 --corners ... --add
python -m vision.layout show other.jpg       # -> other_layout.jpg (outline, ROIs, anchors)
python -m vision.layout info
```

Run: `python -m vision photos/ --summary` (adds `layout=… mcu from roi`),
`--debug dbg/` also writes each ROI crop, `--no-layout` = old behaviour,
`--roi-fallback` = read the whole photo if the card crop gives no answer.
JSON output gets a `layout` block (method, anchors matched / inliers, rms, ROI
quads in original and processed coordinates); the test window draws the ROIs
(cyan, dashed).

## Click recognition (DINOv2) — 0.3

Implements the report *"Click Board Recognition with DINOv2 — Local
Implementation Report"* (3 Nov 2026), build-order steps 1–4.

```bash
# --- once, on the dev machine (needs torch + transformers + onnx + huggingface_hub)
pip install -r requirements-dev.txt
python scripts/get_dinov2.py                       # -> models/hf/dinov2-with-registers-small
python scripts/export_dinov2_onnx.py --pooling cls --size 224 --verify-images catalog_images/
#   -> models/dinov2s_reg_224.onnx + .json (model_version); checks PyTorch vs ONNX cosine >= 0.9999
#   --pooling cls_gem for CLS ⊕ GeM, --size 336/448, --int8 for a quantised copy

# --- gallery (runtime needs only onnxruntime + numpy + opencv)
python -m vision.clicks build --images catalog_images/ --out gallery/ --augment 12
python -m vision.clicks add --gallery gallery/ --pid MIKROE-5678 --name "New Click" img.jpg
python -m vision.clicks info --gallery gallery/

# --- recognise
python -m vision.clicks query --gallery gallery/ photo.jpg [--box x,y,w,h ...] [--no-ocr]
python -m vision photo.jpg --gallery gallery/ [--click-box x,y,w,h]   # full manifest incl. clicks
python -m vision.app --gallery gallery/        # test window: "Click boxes: drag" on the photo

# --- evaluate + calibrate (report §9 experiment grid, one row per run in eval/results.csv)
python eval/click_eval.py photos/clicks/ --gallery gallery/ --exp E1 --sources catalog
python eval/click_eval.py photos/clicks/ --gallery gallery/ --exp E5 --sources catalog,augmented
python eval/click_eval.py photos/clicks/ --gallery gallery/ --exp E6 --add-train
python eval/click_eval.py photos/clicks/ --gallery gallery/ --exp E7 --add-train --tta --ocr --write-thresholds
python scripts/make_synthetic_clicks.py syn_clicks/   # plumbing check only, fake PIDs
```

Catalog image layout: `catalog_images/MIKROE-1234/*.jpg` or `MIKROE-1234_*.jpg`.
Click names / sizes / confusion groups: `data/clicks.csv` (`pid,name,size,group,aliases`,
or `products.db` rows with `type='click'`). Mikrobus socket positions for slot
assignment: `data/sockets.csv` (`board,slot,u,v`, board-relative 0–1). Both are
empty seeds — fill them from the product DB.

| Report section | Where |
|---|---|
| §1 ViT-S/14 + registers, CLS or CLS ⊕ GeM (patches `[5:]`, registers never pooled), 224/336/448, ImageNet norm, local files only | `clicks/torch_model.py`, `clicks/embedder.py`, `scripts/get_dinov2.py` |
| §2 gallery: catalog + 8–20 augmented copies + real train photos; best-vector scoring; `embeddings.npy` / `meta.parquet` (csv without pyarrow) / `index.faiss`; versioned, mismatch fails loudly | `clicks/gallery.py`, `clicks/build.py`, `clicks/augment.py` |
| §2 preprocessing: trim white margin, pad to square (never stretch), optional CLAHE on L, resize, same for gallery and query | `clicks/preprocess.py` |
| §3 exact search (FAISS `IndexFlatIP` or NumPy), top-50 → group by PID (max) → top-5; TTA, OCR bonus, size prior, local-feature check; accept / ask / unknown | `clicks/gallery.py`, `clicks/recognizer.py` |
| §4 v0 localisation: user drags a box per Click (or one Click per photo); boxes grown 5–10 %; rule-based slot assignment from the socket map | `vision/app.py` + UI, `pipeline.run_clicks`, `assign_slots` |
| §5 OCR re-rank: upscale to ~1000 px, 0°/180°, normalise, `token_set_ratio` vs top-5, ≥90 promote / 75–89 +0.10, numbered-series rule, full-catalogue fallback below T_unknown | `clicks/rerank.py` |
| §6 ONNX export (opset 17, dynamic batch, fixed size, checker, simplifier, 0.9999 check), CPU / DirectML / CUDA providers, INT8 option | `scripts/export_dinov2_onnx.py`, `OnnxEmbedder(providers=...)` |
| §9 evaluation protocol | `eval/click_eval.py` |

Not done yet (report: later stages, only when evaluation asks for them):
RF-DETR / zero-shot detector (v0.5/v1 localisation), LoRA metric-learning
fine-tune (§7), DINOv3. **No accuracy numbers exist yet**: this environment
could not download DINOv2 (Hugging Face blocked), so the code was tested end to
end with a hand-built toy ONNX model of the same interface
(`tests/onnx_toy.py`). Run E1–E7 on real photos first.

### OCR engines (all Apache-2.0)
`--backend auto` picks the first available: **paddle** → **rapidocr** → **tesseract**.

* `pip install paddleocr paddlepaddle` — the plan's choice (2.x and 3.x APIs handled)
* `pip install rapidocr_onnxruntime` — same PP-OCR models on ONNX Runtime, lighter
* Tesseract CLI — fallback; this is what the code was **tested** with

The RapidOCR adapter is tested with `rapidocr_onnxruntime` 1.4.4 (0.3); the Paddle adapter and the newer `rapidocr` ≥ 2 package are still **untested**. Run `python -m vision syn/ --summary --backend paddle` once after installing to confirm.

## Photos (drag-and-drop test window)

`python -m vision.app` (the **Photos** view) starts a local page (127.0.0.1 only, nothing leaves the
computer) and opens it in the browser. Drop one or many photos anywhere on the
window, pick them with **Add photos**, or paste with Ctrl+V. For each photo:

* **Board / MCU / MCU card** cards: decision or "Not sure", reason, top-3 with scores and what was read;
* the photo with boxes — green board text, orange MCU marking, dashed magenta refined crops — on the original or the **Processed** image;
* **Refined crops**: the source crop, the enhanced image the recogniser saw, and every read;
* quality warnings, preprocessing steps, timings, raw JSON;
* **Perspective: 4 clicks** — click the board corners, it re-runs with perspective correction;
* change OCR engine / refinement / upscaler / auto-perspective in the top bar and **Re-run**;
* **Save to dataset** copies the photo into `photos/` (or `--photos-dir`) and writes `photos/labels.csv`, so the real-photo set for `eval/ocr_eval.py` grows as you test.

Keys: ↑/↓ (or j/k) move through the photo list, Esc cancels corner picking.

## 0.3 OCR backend fix (RapidOCR)

RapidOCR was the backend `auto` picks once `rapidocr_onnxruntime` is installed,
and its adapter had never been run. Measured on the synthetic set
(`make_synthetic.py --hard`, 16 photos, 2-core CPU):

| `eval/ocr_eval.py --backend rapidocr` | board correct | MCU correct | wrong | median / photo | worst photo | test suite |
|---|---|---|---|---|---|---|
| before | 6/6 | 10/10 | 0 | 5.3 s | **72 s** (unknown_board) | 189 s |
| after | 6/6 | 10/10 | 0 | **2.6 s** | 15 s | 61 s |

Causes and fixes (`vision/ocr.py`, `board_ocr.py`, `mcu_ocr.py`, `refine.py`):

1. **Detector resize.** PP-OCR's default `limit_type=min, limit_side_len=736`
   scaled every small crop so its *short* side was 736 px: a 120×500 refine
   crop became 736×3070 and took ~1.9 s, ×8 reads per crop. Now
   `limit_type=max, limit_side_len=2016`: pages (≤2000 px) and hi-res tiles are
   detected exactly as before, crops at native size (~0.15 s).
2. **Duplicate rotations.** `handles_rotation` was declared but never used:
   RapidOCR/Paddle flip upside-down lines with their angle classifier, yet every
   page, hi-res tile and refine crop was read at 0° *and* 180° (and 90° *and*
   270°). `effective_rotations()` now runs one pass per 0/180 and 90/270 pair
   for those engines; Tesseract is unchanged.
3. **Re-reading empty rotations.** The MCU step re-OCR'd rotations the board
   step had already read when they produced no text; the board step now reports
   `rotations_read`.
4. Robust result unpacking (`[box, text, score, …]`) and the `rapidocr` ≥ 2
   package gets the same detector setting via `params`.

Tesseract results and timings are unchanged.
Note: the project `.venv` has both `opencv-python` and `opencv-python-headless`
installed; they overwrite each other's `cv2` — keep only the headless one
(`pip uninstall -y opencv-python opencv-python-headless && pip install opencv-python-headless`).

## What changed in 0.2 (research brief, 3 Nov 2026)

| Brief recommendation | Implemented as |
|---|---|
| Enlarge small text before OCR (×6 bicubic or super-resolution) | `enhance.upscale` / `scale_for_glyph`: crops are enlarged so glyphs reach ~44 px (max ×6); `--upscaler espcn/fsrcnn/edsr` if models are installed, else bicubic |
| Denoise (NLM), deblur | `enhance.denoise` (strength from a measured noise σ), `richardson_lucy`, `unsharp`; global auto-denoise + auto-sharpen in `preprocess` |
| Contrast enhancement (CLAHE) | kept; plus `gamma_lift` for dark photos *before* CLAHE (CLAHE alone amplified sensor noise into fake text) |
| Binarization (Otsu) — Tesseract prefers binarized input | crop variants `otsu` and `sauvola` (local threshold copes with glare gradients) |
| Perspective correction | 4-click warp existed; added `auto_perspective` (board outline) as an option and the 4-click tool in the UI |
| Two-stage detection → recognition | `vision/refine.py`: regions from fragments the first pass read + a morphological text-line detector; rectified crops from the full-resolution photo; Tesseract single-block mode with a part-number whitelist and English dictionaries off |
| Dictionary / template post-processing | already the core of matching (catalog fuzzy match); refined reads are gated by the same strict MCU thresholds |
| Transformer OCR (TrOCR) | was an optional crop reader; **removed in 0.9** (never runnable here, never measured) |
| CER / WER, ablations, noise/blur robustness | `eval/ocr_eval.py --configs … --degrade …` |
| Synthetic data with realistic degradations | `make_synthetic.py --hard`: defocus, motion blur, low light + noise, small glyphs |

Not taken from the brief (yet): training or fine-tuning a CRNN/attention model, INT8 quantisation/pruning. Those need the real-photo set first; the Save-to-dataset button is how it gets collected.

## Status (3 Nov 2026, v0.2 OCR numbers; 0.3 adds Clicks — see above)

Synthetic set (`make_synthetic.py --hard`: 9 smoke scenes + 7 hard chip scenes), Tesseract, 2-core CPU:

| config | MCU correct (n=10) | MCU wrong | MCU CER | board correct (n=6) | median ms |
|---|---|---|---|---|---|
| v0.1 code | 6 | 0 | — | 6 | 2100 |
| 0.2 `baseline` (refine off) | 5 | 0 | 44% | 6 | 1840 |
| **0.2 `full` (default)** | **8** | **0** | **10%** | **6** | 2840 |
| 0.2 `full`, photos degraded `defocus4` | 7 | 0 | 34% | 5 | 2350 |
| 0.2 `full`, photos degraded `dark` | 5 | 0 | 39% | 6 | 3300 |
| 0.2 `baseline`, `dark` | 2 | 0 | 80% | 6 | 1780 |

Still unsolved: 15 px motion blur and 7 px defocus — both abstain with the right part as top-1 option (no wrong answers anywhere). The slowest dark photo dropped from 55 s to 14 s.

Earlier notes (v0.1):

* 23 tests pass; 9/9 synthetic scenes correct with Tesseract (incl. rotated, tilted, glare, upside-down, and an unknown board that correctly abstains). Median ~1.5 s per photo on CPU, worst ~5 s when all four rotations are needed.
* Synthetic images prove the code path only. **Real photos are the real test** (plan block 6).
* `data/*.csv` are seed lists [ASSUMPTION]: board names and MCU parts typed from memory, PIDs blank. Replace with the product DB (`Catalog.from_sqlite("products.db")` reads the plan's `products`/`compat` schema).
* Thresholds (`ACCEPT`, `MARGIN` in board_ocr.py / mcu_ocr.py, quality limits in preprocess.py) are [ASSUMPTION] starting values; tune on real photos.

## Known limits / next
* Two boards whose printed names are identical (revisions) cannot be separated by OCR → embedding fallback (T7) or ask the user.
* Chip markings below ~30 px glyph height are unreliable; the hi-res tiled pass helps only if the original photo has the pixels. The eval reports glyph size so the UI can say "move closer".
* MCU candidates are not yet filtered by board compatibility; `McuRecognizer.recognise(..., allowed_parts=...)` is ready for the `compat` table.
