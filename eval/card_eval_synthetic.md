# MCU-card stage evaluation

36 photos from `/tmp/claude-0/syn36`, backend `tesseract`.

> Note: the 3 STM32F401RB rows below were rendered with the dataset's WRONG_CHIP marking (an STM32F205RB chip); with the renderer fixed they abstain. See README 0.6.

| set | n | located | layout ok | card ok / wrong | MCU ok / wrong | median s | 0.5 layout ok | 0.5 card ok / wrong | 0.5 MCU ok / wrong | 0.5 median s |
|---|---|---|---|---|---|---|---|---|---|---|
| close | 12 | 12 | 12 | 9 / 1 | 9 / 1 | 8.1 | 6 | 2 / 1 | 2 / 1 | 28.8 |
| far | 12 | 11 | 11 | 1 / 1 | 1 / 1 | 14.0 | 10 | 1 / 0 | 1 / 0 | 20.1 |
| mid | 12 | 12 | 11 | 10 / 1 | 10 / 1 | 7.8 | 11 | 2 / 1 | 2 / 1 | 26.8 |
| all | 36 | 35 | 34 | 20 / 3 | 20 / 3 | 8.8 | 27 | 5 / 2 | 5 / 2 | 25.9 |

| photo | card px | layout | card | MCU | reason |
|---|---|---|---|---|---|
| card_close_mcu-card-29-for-stm32-stm32f411re.jpg | 1492 | correct | correct | correct | MCU CARD 29 for STM32 + chip STM32F411RE -> MCU CARD 29 for STM32 STM32F411RE |
| card_mid_mcu-card-29-for-stm32-stm32f411re.jpg | 937 | correct | correct | correct | MCU CARD 29 for STM32 + chip STM32F411RE -> MCU CARD 29 for STM32 STM32F411RE |
| card_far_mcu-card-29-for-stm32-stm32f411re.jpg | 475 | correct | abstain | abstain | title 'MCU CARD 29 for STM32' is printed on 10 cards; the chip marking decides which one (unclear) |
| card_close_mcu-card-for-tiva-tm4c129enczad.jpg | 1430 | correct | correct | correct | MCU CARD for Tiva + chip TM4C129ENCZAD -> MCU CARD for Tiva TM4C129ENCZAD |
| card_mid_mcu-card-for-tiva-tm4c129enczad.jpg | 907 | correct | correct | correct | MCU CARD for Tiva + chip TM4C129ENCZAD -> MCU CARD for Tiva TM4C129ENCZAD |
| card_far_mcu-card-for-tiva-tm4c129enczad.jpg | 469 | correct | abstain | abstain | title 'MCU CARD for Tiva' is printed on 7 cards; the chip marking decides which one (unclear) |
| card_close_mcu-card-2-for-kinetis-mk20dx128vfm5.jpg | 1428 | correct | correct | correct | MCU CARD 2 for Kinetis + chip MK20DX128VFM5 -> MCU CARD 2 for Kinetis MK20DX128VFM5 |
| card_mid_mcu-card-2-for-kinetis-mk20dx128vfm5.jpg | 900 | correct | correct | correct | MCU CARD 2 for Kinetis + chip MK20DX128VFM5 -> MCU CARD 2 for Kinetis MK20DX128VFM5 |
| card_far_mcu-card-2-for-kinetis-mk20dx128vfm5.jpg | 469 | correct | abstain | abstain | title 'MCU CARD 2 for Kinetis' is printed on 2 cards; the chip marking decides which one (not read) |
| card_close_mcu-card-29-for-stm32-stm32f401rb.jpg | 1445 | correct | WRONG | WRONG | MCU CARD 29 for STM32 + chip STM32F205RB -> MCU CARD 29 for STM32 STM32F205RB |
| card_mid_mcu-card-29-for-stm32-stm32f401rb.jpg | 924 | correct | WRONG | WRONG | MCU CARD 29 for STM32 + chip STM32F205RB -> MCU CARD 29 for STM32 STM32F205RB |
| card_far_mcu-card-29-for-stm32-stm32f401rb.jpg | 482 | correct | WRONG | WRONG | MCU CARD 29 for STM32 + chip STM32F205RB -> MCU CARD 29 for STM32 STM32F205RB |
| card_close_mcu-card-for-kinetis-mk24fn256vdc12.jpg | 1456 | correct | correct | correct | MCU CARD for Kinetis + chip MK24FN256VDC12 -> MCU CARD for Kinetis MK24FN256VDC12 |
| card_mid_mcu-card-for-kinetis-mk24fn256vdc12.jpg | 917 | correct | correct | correct | MCU CARD for Kinetis + chip MK24FN256VDC12 -> MCU CARD for Kinetis MK24FN256VDC12 |
| card_far_mcu-card-for-kinetis-mk24fn256vdc12.jpg | 606 | correct | abstain | abstain | title 'MCU CARD for Kinetis' is printed on 4 cards; the chip marking decides which one (not read) |
| card_close_mcu-card-for-tiva-tm4c1299nczad.jpg | 1436 | correct | correct | correct | MCU CARD for Tiva + chip TM4C1299NCZAD -> MCU CARD for Tiva TM4C1299NCZAD |
| card_mid_mcu-card-for-tiva-tm4c1299nczad.jpg | 909 | correct | correct | correct | MCU CARD for Tiva + chip TM4C1299NCZAD -> MCU CARD for Tiva TM4C1299NCZAD |
| card_far_mcu-card-for-tiva-tm4c1299nczad.jpg | 553 | correct | abstain | abstain | title 'MCU CARD for Tiva' is printed on 7 cards; the chip marking decides which one (unclear) |
| card_close_mcu-card-for-tiva-tm4c129lnczad.jpg | 1451 | correct | correct | correct | MCU CARD for Tiva + chip TM4C129LNCZAD -> MCU CARD for Tiva TM4C129LNCZAD |
| card_mid_mcu-card-for-tiva-tm4c129lnczad.jpg | 911 | correct | correct | correct | MCU CARD for Tiva + chip TM4C129LNCZAD -> MCU CARD for Tiva TM4C129LNCZAD |
| card_far_mcu-card-for-tiva-tm4c129lnczad.jpg | 469 | correct | abstain | abstain | title 'MCU CARD for Tiva' is printed on 7 cards; the chip marking decides which one (unclear) |
| card_close_mcu-card-22-for-stm32-stm32f373vc.jpg | 1440 | correct | correct | correct | MCU CARD 22 for STM32: only card with this layout |
| card_mid_mcu-card-22-for-stm32-stm32f373vc.jpg | 907 | correct | correct | correct | MCU CARD 22 for STM32: only card with this layout |
| card_far_mcu-card-22-for-stm32-stm32f373vc.jpg | 707 | correct | correct | correct | MCU CARD 22 for STM32: only card with this layout |
| card_close_mcu-card-for-tiva-tm4c129xnczad.jpg | 1425 | correct | abstain | abstain | title 'MCU CARD for Tiva' is printed on 21 cards; the chip marking decides which one (not read) |
| card_mid_mcu-card-for-tiva-tm4c129xnczad.jpg | 902 | correct | correct | correct | MCU CARD for Tiva + chip TM4C129XNCZAD -> MCU CARD for Tiva TM4C129XNCZAD |
| card_far_mcu-card-for-tiva-tm4c129xnczad.jpg | 639 | correct | abstain | abstain | title 'MCU CARD for Tiva' is printed on 7 cards; the chip marking decides which one (not read) |
| card_close_mcu-card-16-for-stm32-stm32l432kc.jpg | 1493 | correct | abstain | abstain | title 'MCU CARD 16 for STM32' is printed on 2 cards; the chip marking decides which one (unclear) |
| card_mid_mcu-card-16-for-stm32-stm32l432kc.jpg | 949 | abstain | abstain | abstain | no MCU card title read |
| card_far_mcu-card-16-for-stm32-stm32l432kc.jpg | 514 | correct | abstain | abstain | title 'MCU CARD 16 for STM32' is printed on 2 cards; the chip marking decides which one (unclear) |
| card_close_mcu-card-3-for-stm32-stm32l073rz.jpg | 1497 | correct | correct | correct | MCU CARD 3 for STM32 + chip STM32L073RZ -> SiBRAIN for STM32L073RZ |
| card_mid_mcu-card-3-for-stm32-stm32l073rz.jpg | 949 | correct | correct | correct | MCU CARD 3 for STM32 + chip STM32L073RZ -> SiBRAIN for STM32L073RZ |
| card_far_mcu-card-3-for-stm32-stm32l073rz.jpg | 487 | correct | abstain | abstain | title 'MCU CARD 3 for STM32' is printed on 5 cards; the chip marking decides which one (unclear) |
| card_close_mcu-card-3-for-kinetis-mkv58f1m0vlq24.jpg | 1435 | correct | correct | correct | MCU CARD 3 for Kinetis: only card with this layout |
| card_mid_mcu-card-3-for-kinetis-mkv58f1m0vlq24.jpg | 911 | correct | correct | correct | MCU CARD 3 for Kinetis: only card with this layout |
| card_far_mcu-card-3-for-kinetis-mkv58f1m0vlq24.jpg | 0 | abstain | abstain | abstain | card not found |
