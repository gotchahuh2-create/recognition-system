# Resolution sweep — marking verifier (synthetic)

Verifier: MIN_SCORE 0.45, MARGIN 0.06, PSF 0.7 native px. Chip body side = px in the photo (real UNI-DS v8 test photos: 71-77 px).

| group | cards | chip px | verifier ✓ | ✗ | – | median margin true-vs-best-other | min margin | OCR ✓/✗/– | s/chip |
|---|---|---|---|---|---|---|---|---|---|
| mcu-card-3-for-pic32 | 4 | 70 | 0 | 0 | 3 | -0.004 | -0.028 | 0/0/3 | 16.9 |
| mcu-card-3-for-pic32 | 4 | 140 | 1 | 0 | 2 | -0.002 | -0.007 | 0/0/3 | 29.6 |
| mcu-card-3-for-pic32 | 4 | 200 | 1 | 0 | 2 | +0.026 | +0.010 | 1/0/2 | 27.1 |
| mcu-card-3-for-pic32 | 4 | 250 | 1 | 0 | 2 | +0.030 | -0.040 | 1/0/2 | 23.7 |
| mcu-card-3-for-pic32 | 4 | 400 | 1 | 0 | 2 | +0.024 | +0.018 | 1/0/2 | 28.4 |
| mcu-card-for-stm32 | 19 | 70 | 0 | 0 | 3 | -0.029 | -0.059 | 0/0/3 | 21.9 |
| mcu-card-for-stm32 | 19 | 140 | 0 | 0 | 3 | -0.024 | -0.097 | 0/0/3 | 49.0 |
| mcu-card-for-stm32 | 19 | 200 | 0 | 0 | 3 | +0.021 | +0.004 | 0/0/3 | 48.3 |
| mcu-card-for-stm32 | 19 | 250 | 0 | 0 | 3 | +0.026 | -0.009 | 3/0/0 | 44.6 |
| mcu-card-for-stm32 | 19 | 400 | 0 | 0 | 3 | +0.035 | -0.006 | 3/0/0 | 47.8 |

Per chip:

| group | px | truth | decision | margin |
|---|---|---|---|---|
| mcu-card-3-for-pic32 | 70 | mcu-card-3-for-pic32-pic32mz2048efh144 | – | -0.028 |
| mcu-card-3-for-pic32 | 70 | sibrain-for-pic32mz1024eff144 | – | +0.007 |
| mcu-card-3-for-pic32 | 70 | sibrain-for-pic32mz1024efe144 | – | -0.004 |
| mcu-card-3-for-pic32 | 140 | mcu-card-3-for-pic32-pic32mz2048efh144 | mcu-card-3-for-pic32-pic32mz2048efh144 | +0.120 |
| mcu-card-3-for-pic32 | 140 | sibrain-for-pic32mz1024eff144 | – | -0.007 |
| mcu-card-3-for-pic32 | 140 | sibrain-for-pic32mz1024efe144 | – | -0.002 |
| mcu-card-3-for-pic32 | 200 | mcu-card-3-for-pic32-pic32mz2048efh144 | mcu-card-3-for-pic32-pic32mz2048efh144 | +0.214 |
| mcu-card-3-for-pic32 | 200 | sibrain-for-pic32mz1024eff144 | – | +0.010 |
| mcu-card-3-for-pic32 | 200 | sibrain-for-pic32mz1024efe144 | – | +0.026 |
| mcu-card-3-for-pic32 | 250 | mcu-card-3-for-pic32-pic32mz2048efh144 | mcu-card-3-for-pic32-pic32mz2048efh144 | +0.207 |
| mcu-card-3-for-pic32 | 250 | sibrain-for-pic32mz1024eff144 | – | -0.040 |
| mcu-card-3-for-pic32 | 250 | sibrain-for-pic32mz1024efe144 | – | +0.030 |
| mcu-card-3-for-pic32 | 400 | mcu-card-3-for-pic32-pic32mz2048efh144 | mcu-card-3-for-pic32-pic32mz2048efh144 | +0.241 |
| mcu-card-3-for-pic32 | 400 | sibrain-for-pic32mz1024eff144 | – | +0.018 |
| mcu-card-3-for-pic32 | 400 | sibrain-for-pic32mz1024efe144 | – | +0.024 |
| mcu-card-for-stm32 | 70 | mcu-card-for-stm32-stm32f439zg | – | -0.027 |
| mcu-card-for-stm32 | 70 | mcu-card-for-stm32-stm32f746zg | – | -0.029 |
| mcu-card-for-stm32 | 70 | mcu-card-for-stm32-stm32f767zg | – | -0.059 |
| mcu-card-for-stm32 | 140 | mcu-card-for-stm32-stm32f439zg | – | -0.097 |
| mcu-card-for-stm32 | 140 | mcu-card-for-stm32-stm32f746zg | – | -0.024 |
| mcu-card-for-stm32 | 140 | mcu-card-for-stm32-stm32f767zg | – | +0.029 |
| mcu-card-for-stm32 | 200 | mcu-card-for-stm32-stm32f439zg | – | +0.021 |
| mcu-card-for-stm32 | 200 | mcu-card-for-stm32-stm32f746zg | – | +0.004 |
| mcu-card-for-stm32 | 200 | mcu-card-for-stm32-stm32f767zg | – | +0.034 |
| mcu-card-for-stm32 | 250 | mcu-card-for-stm32-stm32f439zg | – | +0.026 |
| mcu-card-for-stm32 | 250 | mcu-card-for-stm32-stm32f746zg | – | -0.009 |
| mcu-card-for-stm32 | 250 | mcu-card-for-stm32-stm32f767zg | – | +0.038 |
| mcu-card-for-stm32 | 400 | mcu-card-for-stm32-stm32f439zg | – | +0.048 |
| mcu-card-for-stm32 | 400 | mcu-card-for-stm32-stm32f746zg | – | -0.006 |
| mcu-card-for-stm32 | 400 | mcu-card-for-stm32-stm32f767zg | – | +0.035 |
