# HybMem-SNN results

Network 196-64-10 LIF, T = 16 time steps, 8-bit weights, evaluated on 10000 MNIST test images (trained on all 60,000 training images).

### 1. Accuracy (full test set, no memory errors)

| model | accuracy % |
|---|---|
| float weights (same input spikes) | 96.92 |
| 8-bit integer weights = hardware | 96.96 |
| quantization loss | -0.04 |

### 2. Event-driven savings (full test set)

| quantity | event-driven (this design) | dense (every row fetched) | reduction |
|---|---|---|---|
| input spikes / image | 407 of 3136 | 3136 | 87.0% |
| hidden spikes / image | 312 of 1024 | 1024 | 69.6% |
| weight rows fetched / image | 718 | 4160 | 82.7% |
| weight additions / image | 31023 | 217088 | 85.7% |
| cycles / image | 11965 | 58481 | 79.5% |

*Cycle counts come from an exact cycle model of the RTL (checked against simulation by the testbench). At an assumed 100 MHz: 119.6 us per image, 8358 images/s.*

### 3. Per-digit behaviour

| digit | images | accuracy % | input spikes | hidden spikes | rows fetched | cycles |
|---|---|---|---|---|---|---|
| 0 | 980 | 98.7 | 533 | 323 | 856 | 14029 |
| 1 | 1135 | 98.7 | 233 | 307 | 540 | 9162 |
| 2 | 1032 | 96.9 | 462 | 329 | 791 | 12920 |
| 3 | 1010 | 96.9 | 440 | 344 | 783 | 12617 |
| 4 | 982 | 97.1 | 376 | 289 | 665 | 11386 |
| 5 | 892 | 96.2 | 403 | 318 | 721 | 11935 |
| 6 | 958 | 97.8 | 442 | 286 | 728 | 12427 |
| 7 | 1028 | 97.5 | 354 | 304 | 658 | 11090 |
| 8 | 974 | 96.0 | 470 | 314 | 784 | 12990 |
| 9 | 1009 | 93.6 | 382 | 303 | 685 | 11536 |

*Digits with more ink produce more input spikes and take more cycles: here digit 0 is the slowest (14029 cycles) and digit 1 the fastest (9162 cycles).*

### 4. Accuracy vs weight precision (post-training truncation of the 8-bit weights)

| weight bits | accuracy % |
|---|---|
| 8 | 96.96 |
| 7 | 96.90 |
| 6 | 96.62 |
| 5 | 96.20 |
| 4 | 89.76 |
| 3 | 42.27 |
| 2 | 9.55 |

*No retraining: the top b bits of each trained 8-bit weight are kept. Quantization-aware training would do better at low bit widths.*

### 5. Accuracy vs number of time steps (weights trained with T = 16)

| time steps T | accuracy % | cycles / image | latency us @ 100 MHz |
|---|---|---|---|
| 4 | 94.08 | 2936 | 29.4 |
| 6 | 95.77 | 4492 | 44.9 |
| 8 | 96.23 | 5975 | 59.7 |
| 12 | 96.83 | 8954 | 89.5 |
| 16 | 96.96 | 11965 | 119.6 |
| 24 | 97.11 | 18015 | 180.1 |
| 32 | 97.22 | 24110 | 241.1 |

*Inference-only change: the same weights are run for fewer/more steps. The hardware has T = 16.*

### 6. Accuracy % vs fraction of weight bits in the unreliable memory and bit-error probability (first 1000 test images, mean +- std over 3 trials; error-free accuracy on these images = 96.2%)

| noisy bits \ bit-error prob | 1/256 | 1/128 | 1/64 | 1/32 | 1/16 | 1/8 |
|---|---|---|---|---|---|---|
| 2 LSBs | 96.1 +- 0.0 | 96.1 +- 0.2 | 96.2 +- 0.1 | 96.2 +- 0.1 | 96.2 +- 0.0 | 96.3 +- 0.1 |
| 4 LSBs | 96.3 +- 0.1 | 96.1 +- 0.0 | 96.3 +- 0.0 | 96.4 +- 0.1 | 96.3 +- 0.2 | 96.2 +- 0.1 |
| 6 LSBs | 96.1 +- 0.3 | 96.2 +- 0.3 | 95.8 +- 0.2 | 95.0 +- 0.5 | 91.9 +- 0.8 | 83.3 +- 0.2 |
| all 8 bits | 94.1 +- 0.6 | 90.6 +- 0.5 | 78.5 +- 0.6 | 55.4 +- 1.1 | 30.2 +- 1.0 | 18.7 +- 0.9 |
| 4 MSBs | 94.4 +- 0.3 | 90.2 +- 0.5 | 79.1 +- 0.4 | 55.2 +- 1.0 | 30.6 +- 1.0 | 20.0 +- 1.1 |

*'n LSBs' = the n least significant bits of every weight are stored in the unreliable memory, the other bits in reliable memory. '4 MSBs' is the opposite split and shows why the split matters.*

### 7. Headline noise results on all 10000 test images (mean +- std over 2 trials)

| noisy bits | weight bits in unreliable memory | bit-error prob | accuracy % | change vs error-free |
|---|---|---|---|---|
| all 8 bits | 8/8 = 100.0% | 1/256 | 94.98 +- 0.07 | -1.98 |
| all 8 bits | 8/8 = 100.0% | 1/32 | 55.64 +- 0.39 | -41.32 |
| 4 LSBs | 4/8 = 50.0% | 1/32 | 96.91 +- 0.02 | -0.05 |
| 4 LSBs | 4/8 = 50.0% | 1/8 | 96.67 +- 0.02 | -0.29 |
| 4 MSBs | 4/8 = 50.0% | 1/32 | 55.27 +- 0.09 | -41.70 |
