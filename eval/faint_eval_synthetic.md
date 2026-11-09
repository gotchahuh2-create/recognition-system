# Faint chip marking: classic chip pass vs vision.faint

Folder `syn`, backend tesseract, 24 photos. Same rectified chip for every method. correct / wrong / abstain = MCU decided by the pipeline's accept rule over the whole catalogue; top3 = truth among 3 best; CER = best read vs printed part-number line (lower is better).

## OCR call: `crop`

| contrast (levels) | n | classic ✓/✗/– | faint ✓/✗/– | both ✓/✗/– | faint w/o fusion ✓/✗/– | top-3 c/f/b | mean CER classic | mean CER faint | median s classic | median s faint |
|---|---|---|---|---|---|---|---|---|---|---|
| 30 | 6 | 0/0/6 | 5/0/1 | 5/0/1 | 5/0/1 | 0/6/6 | 0.70 | 0.12 | 1.6 | 3.2 |
| 14 | 6 | 0/0/6 | 1/0/5 | 1/0/5 | 1/0/5 | 0/5/5 | 0.89 | 0.23 | 0.8 | 3.3 |
| 8 | 6 | 0/0/6 | 0/0/6 | 0/0/6 | 0/0/6 | 0/1/1 | 0.91 | 0.51 | 0.8 | 3.5 |
| 5 | 6 | 0/0/6 | 0/0/6 | 0/0/6 | 0/0/6 | 0/0/0 | 0.88 | 0.77 | 1.1 | 5.1 |
| all | 24 | 0/0/24 | 6/0/18 | 6/0/18 | 6/0/18 | 0/12/12 | 0.85 | 0.41 | 1.1 | 3.4 |

## OCR call: `page`

| contrast (levels) | n | classic ✓/✗/– | faint ✓/✗/– | both ✓/✗/– | faint w/o fusion ✓/✗/– | top-3 c/f/b | mean CER classic | mean CER faint | median s classic | median s faint |
|---|---|---|---|---|---|---|---|---|---|---|
| 30 | 6 | 0/0/6 | 4/0/2 | 4/0/2 | 4/0/2 | 0/6/6 | 0.82 | 0.12 | 1.2 | 3.2 |
| 14 | 6 | 0/0/6 | 1/0/5 | 1/0/5 | 1/0/5 | 0/4/4 | 0.89 | 0.22 | 0.9 | 3.3 |
| 8 | 6 | 0/0/6 | 0/0/6 | 0/0/6 | 0/0/6 | 0/2/2 | 0.95 | 0.56 | 0.8 | 3.4 |
| 5 | 6 | 0/0/6 | 0/0/6 | 0/0/6 | 0/0/6 | 0/0/0 | 0.89 | 0.78 | 1.0 | 4.7 |
| all | 24 | 0/0/24 | 5/0/19 | 5/0/19 | 5/0/19 | 0/12/12 | 0.89 | 0.42 | 1.0 | 3.3 |

## Set and conditions

`python scripts/make_faint_chips.py syn_faint/ --n 6` (seed 0): 6 MCUs with their
shop-photo markings from `data/mcu_cards.json` (MK64FN1M0VDC12, STM32F405RG,
MKV58F1M0VLQ24, STM32L442KC, STM32F423RH, STM32F415RG) × marking contrast 30 / 14 / 8 / 5
gray levels over a ~33-level package. Every chip: worn strokes, ±35 % lighting gradient
+ vignetting, defocus σ 0.9, noise σ 4, JPEG 82, body 300 px in a 484 px photo;
every other chip a specular reflection over the marking; every third chip at 45°.
Tesseract 5.3.4, 2-core CPU. Same rectified chip for every method (`vision.faint.region`).

Sanity check (not in the table): on two easy chips (contrast 140, no glare, flat light)
classic and faint both read the part correctly with both OCR calls — the harness does
not handicap the classic pass; it fails above on glare + gradient + low contrast.

## Reading the table

* **No wrong answer from any method anywhere** — the 0.90 / 0.05 catalogue gate holds.
* faint turns 6 of 24 abstains into correct answers (contrast 30: 5/6, 14: 1/6) and puts
  the truth in the top 3 for 12/24 (classic 0/24) — the `needs_user` list would offer it.
* Between top-3 and correct are 1-character misreads (`STM32F35RGT6`, `STM32F41RGT6`):
  the DejaVu Mono dotted zero breaks apart when faint.
* Contrast 8 and 5 levels: the marking is found (stage images show it) but not read.
* **Fusion adds no decision** (column "faint w/o fusion" is identical): the variants are
  computed from the same pixels and tend to make the same mistake, so the character
  vote has nothing to correct. Kept because it is cheap (< 0.3 s) and only adds tokens.
* Cost: ~3.3 s per chip vs ~1 s (2–4 OCR turns × 6–9 variants instead of 4 × 3).

Example of all stages: `eval/faint_example_sheet.jpg` (contrast 14, read correctly).
