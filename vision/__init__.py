"""MIKROE hardware recognition: identify the dev board, MCU, MCU card and Click boards from
photos.

How a photo flows through the package:

    1. Load the photo       vision.loader
    2. Clean it up          vision.preprocess
    3. Identify the board   vision.board_ocr   (read its printed name, fuzzy match)
    4. Identify the MCU     vision.mcu_ocr     (read the chip's part number / card label)

Helper modules:

    vision.enhance      image clean-up for small text: enlarging (incl.
                        super-resolution), denoising, deblurring, contrast,
                        black-and-white thresholds
    vision.refine       second-chance pass: find small text, cut it out,
                        clean it up and read it again
    vision.layout       stored board layouts: where the MCU card and the
                        mikroBUS sockets are on each board, so only the MCU
                        card area is read
    vision.card_ocr     card title -> layout group (207 cards in 62 groups in
                        data/mcu_cards.json); chip marking or PID label -> card
    vision.card_locate  find the MCU card on the photo (yellow frame, layout,
                        or title), cut it out, straighten and zoom in
    vision.card_reader  read all the text on the card: labels and chip close-up
    vision.card_text    match every label against what each layout should
                        have printed on it (mikroBUS pin labels, port pins,
                        other text)
    vision.clicks       recognise Click boards by DINOv2 image search, with
                        an OCR re-rank and a "not sure" rule
    vision.faint        experimental reader for very faint chip markings
                        (colour clustering, many thresholds, combining OCR
                        results). Opt-in: --chip-mode faint|both, or
                        python -m vision.faint
    vision.app          the app: one window to scan a board with your phone
                        (whole board, then a close-up of the MCU) or test
                        photos (python -m vision.app)
"""

__version__ = "0.10.0"
