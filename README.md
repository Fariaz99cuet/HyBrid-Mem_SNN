# HybMem-SNN

An event-driven spiking neural network (SNN) accelerator for MNIST, written in Verilog, with a **hybrid reliable/unreliable weight memory** model. It is trained in PyTorch, verified bit-exactly against a Python model in the Vivado simulator, and synthesised and routed for a Zynq-7020.

## Highlights

- **96.96%** test accuracy on all 10,000 MNIST images with 8-bit hardware weights.
- **Event-driven:** weight rows are fetched only for inputs that spike, giving **82.7% fewer weight-row reads** than a dense schedule of the same FSM.
- **Hybrid memory angle:** each weight bit can be marked reliable or unreliable, and unreliable bits are flipped on read with probability 2^-k. Errors in the four least significant weight bits cost almost no accuracy, so those bits could sit in cheaper, less reliable memory.
- **Exact verification:** spike counts and cycle counts match the Python model on 100 test images in simulation. A closed-form cycle model reproduces the hardware exactly.
- **Small:** 7,948 LUTs, 2,009 FFs, 5.5 BRAM tiles, 0 DSPs. Timing is met at 100 MHz (119.6 µs per image, about 8,358 images/s).

## Architecture

| Item | Value |
|---|---|
| Network | 196 inputs, 64 hidden, 10 outputs (output layer padded to 16) |
| Input | MNIST downsampled 14x14 (2x2 average), rate coded over T = 16 steps |
| Spike encoder | Pixel spikes if it exceeds a xorshift32 random byte (identical in Python and Verilog) |
| Neuron | LIF, leak v - (v >> 4), threshold 64, soft reset, 20-bit membrane |
| Weights | 8-bit signed, quantised as round(w x 64), clipped to +/-127, in block ROM, eight per word |
| Schedule | Per step: leak, scan input spikes, then fetch and accumulate the weight row of each spike; scan hidden spikes the same way |

Cycles per image = 1 + 263 T + 16 N_in + 4 N_hid, where N_in and N_hid are the numbers of input and hidden spikes. The mean over the test set is 11,965 cycles.

### Hybrid weight memory

Two registers configure the error injector in the RTL:

- `noisy_mask`: which of the eight weight bits are unreliable.
- `ber_sel` (k): the bit-flip probability per read is 2^-k.

The testbench runs six configurations (clean, all bits noisy at three rates, and two LSB-only settings).

## Repository layout

| File | Purpose |
|---|---|
| `train_mnist_export.py` | Downloads MNIST, trains with surrogate gradients, quantises, and exports `weights_init.vh`, `test_data.vh` and `model_weights.npz` |
| `analysis_report.py` | Accuracy, event statistics, per-digit results, weight-precision, time-step and noise sweeps; writes tables, CSVs and figures to `results/` |
| `snn_accel.v` | The accelerator |
| `tb_snn_accel.v` | Self-checking testbench (prints `RESULT: PASS` when every count and cycle matches) |
| `run_sim.tcl` | Creates a Vivado project and runs the simulation |
| `synth_report.tcl` | Out-of-context synthesis and place-and-route with utilisation, timing and power reports in `synth_out/` |

## Requirements

- Python 3 with `numpy` and `torch`
- Vivado (the free WebPACK edition is enough), tested with 2025.2.1

```bash
pip install numpy torch
```

## How to run

1. **Train and export** (downloads MNIST automatically):

   ```bash
   python train_mnist_export.py
   ```

2. **Analysis and figures** (add `--quick` for a fast run):

   ```bash
   python analysis_report.py
   ```

3. **Simulate the RTL.** In the Vivado Tcl Shell, from the project folder:

   ```tcl
   source run_sim.tcl
   ```

   Expect `RESULT: PASS`.

4. **Synthesise and route:**

   ```tcl
   source synth_report.tcl
   ```

   Set the part and clock with the `PART` and `CLK_NS` environment variables. The default is `xc7z020clg400-1` at 10 ns.

## Results

Routed on the xc7z020clg400-1 with a 10 ns clock constraint (out of context, no I/O constraints):

| Metric | Value |
|---|---|
| LUTs | 7,948 of 53,200 (14.94%) |
| Flip-flops | 2,009 of 106,400 (1.89%) |
| Block RAM | 5.5 of 140 tiles (3.93%) |
| DSP blocks | 0 |
| Worst setup slack at 100 MHz | +0.536 ns |
| Worst hold slack | +0.143 ns |
| Estimated maximum clock | about 105.7 MHz |
| Power (vectorless estimate) | 0.228 W (0.124 W dynamic) |

## Limitations

- The longest path goes through the error injector, which only exists to model unreliable cells, so the clock figure reflects the experiment and not a production design.
- Power uses default switching activity and cannot show the benefit of skipping silent pixels.
- The Python noise model flips bits independently while the RTL injector uses rotated copies of one random word, so noise results agree in trend but not bit for bit. Accuracy figures come from the Python model.
- Results come from a single training run (seed 0). The precision sweep quantises the trained network after training and does not retrain.
- The dense reference used for the read-saving figure is a hypothetical variant of the same FSM, not a separate design.


