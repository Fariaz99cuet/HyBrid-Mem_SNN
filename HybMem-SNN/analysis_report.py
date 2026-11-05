"""
HybMem-SNN : analysis for the report
------------------------------------
Needs model_weights.npz  (made by:  python train_mnist_export.py)

Run :  python analysis_report.py            full run (all 10,000 MNIST test images where possible)
       python analysis_report.py --quick    small/fast version for a first look

Makes (folder results/):
  report_tables.md   all tables, ready to paste into the report
  *.csv              the same numbers as CSV
  fig_*.png          figures (only if matplotlib is installed:  pip install matplotlib)
"""
import os
import sys
import csv
import time
import numpy as np
import torch

from train_mnist_export import (NPIX, NH, NOUT, NOP, TH, T, BETA, OUT_DIR, DENSE_ROWS_PER_STEP,
                                load_mnist, make_R, run_int, run_noisy, cycles_model)

# ---------------- settings you may tune ----------------
QUICK        = "--quick" in sys.argv
CLOCK_MHZ    = 100.0                      # ASSUMPTION for latency/throughput - replace with your Fmax from synthesis
GRID_IMAGES  = 500 if QUICK else 1000     # test images used for the noise grid (first N of the 10,000)
GRID_TRIALS  = 1 if QUICK else 3          # random repetitions per noise setting (mean +- std)
FULL_TRIALS  = 1 if QUICK else 2          # repetitions for the headline noise settings on ALL test images
LSB_COUNTS   = [2, 4, 6, 8]               # how many weight LSBs live in the unreliable memory (8 = all bits)
KS           = [8, 7, 6, 5, 4, 3]         # bit-error probability = 2^-k
BITS_LIST    = [8, 7, 6, 5, 4, 3, 2]      # weight precision sweep
T_LIST       = [4, 6, 8, 12, 16, 24, 32]  # time-step sweep (same weights, different inference length)
HEADLINE     = [("all 8 bits", 0xFF, 8), ("all 8 bits", 0xFF, 5),
                ("4 LSBs", 0x0F, 5), ("4 LSBs", 0x0F, 3), ("4 MSBs", 0xF0, 5)]
# --------------------------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)
md = []                                     # lines of report_tables.md


def acc_of(cnt, y):
    return float((cnt[:, :NOUT].argmax(1) == y).mean() * 100)


def save_csv(name, header, rows):
    with open(os.path.join(OUT_DIR, name), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def table(title, header, rows, note=None):
    md.append(f"### {title}\n")
    md.append("| " + " | ".join(header) + " |")
    md.append("|" + "|".join(["---"] * len(header)) + "|")
    for r in rows:
        md.append("| " + " | ".join(str(c) for c in r) + " |")
    if note:
        md.append(f"\n*{note}*")
    md.append("")
    print(f"\n{title}")
    print("  " + " | ".join(header))
    for r in rows:
        print("  " + " | ".join(str(c) for c in r))


def run_float(pix, W1f, W2f, R):
    """Float weights, same input spikes, threshold 1.0 (no quantization)."""
    N = len(pix)
    v1 = np.zeros((N, NH))
    v2 = np.zeros((N, NOUT))
    cnt = np.zeros((N, NOUT))
    for t in range(len(R)):
        s_in = (pix > R[t][None, :]).astype(np.float64)
        v1 = v1 * BETA + s_in @ W1f.T
        s1 = v1 >= 1.0
        v1 = v1 - s1
        v2 = v2 * BETA + s1.astype(np.float64) @ W2f.T
        s2 = v2 >= 1.0
        v2 = v2 - s2
        cnt += s2
    return cnt


def quant_bits(W, b):
    """Keep only the top b bits of the 8-bit weights (post-training), same scale as the hardware."""
    if b >= 8:
        return W.copy()
    s = 8 - b
    lim = 2 ** (b - 1) - 1
    return (np.clip(np.rint(W / 2 ** s), -lim, lim) * 2 ** s).astype(np.int64)


def mask_name(mask):
    if mask == 0xFF:
        return "all 8 bits"
    if mask == 0xF0:
        return "4 MSBs"
    return f"{bin(mask).count('1')} LSBs"


def main():
    t_start = time.time()
    if not os.path.exists("model_weights.npz"):
        sys.exit("model_weights.npz not found - run  python train_mnist_export.py  first")
    d = np.load("model_weights.npz")
    W1, W2, W1f, W2f = d["W1"], d["W2"], d["W1f"], d["W2f"]
    _, _, xte, yte = load_mnist()
    NT = len(xte)
    R = make_R(T)
    print(f"analysis on {NT} MNIST test images, T={T} time steps"
          f"{'  (QUICK mode)' if QUICK else ''}")
    md.append("# HybMem-SNN results\n")
    md.append(f"Network 196-64-10 LIF, T = {T} time steps, 8-bit weights, evaluated on {NT} MNIST test images "
              f"(trained on all 60,000 training images).\n")

    # ---------------------------------------------------------------- 1. accuracy / quantization
    cnt, n_in, n_h = run_int(xte, W1, W2, R)
    a_int = acc_of(cnt, yte)
    a_flt = acc_of(run_float(xte, W1f, W2f, R), yte)
    table("1. Accuracy (full test set, no memory errors)",
          ["model", "accuracy %"],
          [["float weights (same input spikes)", f"{a_flt:.2f}"],
           ["8-bit integer weights = hardware", f"{a_int:.2f}"],
           ["quantization loss", f"{a_flt - a_int:+.2f}"]])
    save_csv("accuracy.csv", ["model", "accuracy_percent"], [["float", a_flt], ["int8_hardware", a_int]])

    # ---------------------------------------------------------------- 2. event-driven statistics
    rows = n_in + n_h
    dense_rows = T * DENSE_ROWS_PER_STEP
    sops = n_in * 64 + n_h * 16                       # weight additions actually performed (8 lanes x chunks)
    dense_sops = T * (NPIX * 64 + NH * 16)
    cyc = cycles_model(n_in, n_h)
    dense_cyc = 1 + T * (1 + NPIX + 16 * NPIX + 1 + NH + 4 * NH + 1)    # every pixel/neuron treated as a spike
    lat_us = cyc.mean() / CLOCK_MHZ
    table("2. Event-driven savings (full test set)",
          ["quantity", "event-driven (this design)", "dense (every row fetched)", "reduction"],
          [["input spikes / image", f"{n_in.mean():.0f} of {T * NPIX}", f"{T * NPIX}", f"{(1 - n_in.mean() / (T * NPIX)) * 100:.1f}%"],
           ["hidden spikes / image", f"{n_h.mean():.0f} of {T * NH}", f"{T * NH}", f"{(1 - n_h.mean() / (T * NH)) * 100:.1f}%"],
           ["weight rows fetched / image", f"{rows.mean():.0f}", f"{dense_rows}", f"{(1 - rows.mean() / dense_rows) * 100:.1f}%"],
           ["weight additions / image", f"{sops.mean():.0f}", f"{dense_sops}", f"{(1 - sops.mean() / dense_sops) * 100:.1f}%"],
           ["cycles / image", f"{cyc.mean():.0f}", f"{dense_cyc}", f"{(1 - cyc.mean() / dense_cyc) * 100:.1f}%"]],
          note=f"Cycle counts come from an exact cycle model of the RTL (checked against simulation by the testbench). "
               f"At an assumed {CLOCK_MHZ:.0f} MHz: {lat_us:.1f} us per image, {1e6 / lat_us:.0f} images/s.")
    save_csv("event_stats.csv", ["quantity", "event_driven", "dense"],
             [["input_spikes", n_in.mean(), T * NPIX], ["hidden_spikes", n_h.mean(), T * NH],
              ["weight_rows", rows.mean(), dense_rows], ["weight_adds", sops.mean(), dense_sops],
              ["cycles", cyc.mean(), dense_cyc]])

    pc = []
    pred = cnt[:, :NOUT].argmax(1)
    for c in range(10):
        m = yte == c
        pc.append([c, int(m.sum()), f"{(pred[m] == c).mean() * 100:.1f}", f"{n_in[m].mean():.0f}",
                   f"{n_h[m].mean():.0f}", f"{rows[m].mean():.0f}", f"{cyc[m].mean():.0f}"])
    cmean = [cyc[yte == c].mean() for c in range(10)]
    table("3. Per-digit behaviour", ["digit", "images", "accuracy %", "input spikes", "hidden spikes",
                                     "rows fetched", "cycles"], pc,
          note=f"Digits with more ink produce more input spikes and take more cycles: here digit {int(np.argmax(cmean))} "
               f"is the slowest ({max(cmean):.0f} cycles) and digit {int(np.argmin(cmean))} the fastest ({min(cmean):.0f} cycles).")
    save_csv("per_class.csv", ["digit", "images", "accuracy", "input_spikes", "hidden_spikes", "rows", "cycles"], pc)

    # ---------------------------------------------------------------- 4. weight precision
    wb = []
    for b in BITS_LIST:
        c, _, _ = run_int(xte, quant_bits(W1, b), quant_bits(W2, b), R)
        wb.append([b, f"{acc_of(c, yte):.2f}"])
    table("4. Accuracy vs weight precision (post-training truncation of the 8-bit weights)",
          ["weight bits", "accuracy %"], wb,
          note="No retraining: the top b bits of each trained 8-bit weight are kept. "
               "Quantization-aware training would do better at low bit widths.")
    save_csv("weight_bits.csv", ["bits", "accuracy_percent"], wb)

    # ---------------------------------------------------------------- 5. time steps
    ts = []
    for Tn in T_LIST:
        Rn = make_R(Tn)
        c, ni, nh = run_int(xte, W1, W2, Rn)
        cy = cycles_model(ni, nh, Tn)
        ts.append([Tn, f"{acc_of(c, yte):.2f}", f"{cy.mean():.0f}", f"{cy.mean() / CLOCK_MHZ:.1f}"])
    table("5. Accuracy vs number of time steps (weights trained with T = %d)" % T,
          ["time steps T", "accuracy %", "cycles / image", f"latency us @ {CLOCK_MHZ:.0f} MHz"], ts,
          note="Inference-only change: the same weights are run for fewer/more steps. The hardware has T = 16.")
    save_csv("timesteps.csv", ["T", "accuracy_percent", "cycles", "latency_us"], ts)

    # ---------------------------------------------------------------- 6. hybrid memory noise
    print(f"\n6. noise grid on the first {GRID_IMAGES} test images, {GRID_TRIALS} trial(s) each ...")
    xg, yg = xte[:GRID_IMAGES], yte[:GRID_IMAGES]
    base_g = acc_of(cnt[:GRID_IMAGES], yg)
    cfgs = [(f"{n} LSBs" if n < 8 else "all 8 bits", (1 << n) - 1) for n in LSB_COUNTS] + [("4 MSBs", 0xF0)]
    grid = {}
    t0 = time.time()
    done_runs, total_runs = 0, len(cfgs) * len(KS) * GRID_TRIALS
    for name, mask in cfgs:
        for k in KS:
            a = []
            for tr in range(GRID_TRIALS):
                torch.manual_seed(1000 + tr)
                a.append(acc_of(run_noisy(xg, W1, W2, R, mask, k), yg))
                done_runs += 1
            grid[(name, k)] = (float(np.mean(a)), float(np.std(a)))
        el = time.time() - t0
        print(f"   {name:10s} done   ({done_runs}/{total_runs} runs, {el:.0f}s elapsed, "
              f"~{el / done_runs * (total_runs - done_runs):.0f}s left)")
    hdr = ["noisy bits \\ bit-error prob"] + [f"1/{2 ** k}" for k in KS]
    body = []
    for name, _ in cfgs:
        body.append([name] + [f"{grid[(name, k)][0]:.1f} +- {grid[(name, k)][1]:.1f}" for k in KS])
    table(f"6. Accuracy % vs fraction of weight bits in the unreliable memory and bit-error probability "
          f"(first {GRID_IMAGES} test images, mean +- std over {GRID_TRIALS} trials; error-free accuracy on these "
          f"images = {base_g:.1f}%)", hdr, body,
          note="'n LSBs' = the n least significant bits of every weight are stored in the unreliable memory, "
               "the other bits in reliable memory. '4 MSBs' is the opposite split and shows why the split matters.")
    save_csv("noise_grid.csv", ["noisy_bits", "bit_error_prob", "acc_mean", "acc_std"],
             [[n, f"1/{2 ** k}", grid[(n, k)][0], grid[(n, k)][1]] for n, _ in cfgs for k in KS])

    print(f"\n7. headline noise settings on ALL {NT} test images, {FULL_TRIALS} trial(s) each ...")
    hl = []
    for name, mask, k in HEADLINE:
        a = []
        for tr in range(FULL_TRIALS):
            torch.manual_seed(2000 + tr)
            a.append(acc_of(run_noisy(xte, W1, W2, R, mask, k), yte))
        hl.append([name, f"{bin(mask).count('1')}/8 = {bin(mask).count('1') * 12.5:.1f}%", f"1/{2 ** k}",
                   f"{np.mean(a):.2f} +- {np.std(a):.2f}", f"{np.mean(a) - a_int:+.2f}"])
        print("   ", hl[-1])
    table(f"7. Headline noise results on all {NT} test images (mean +- std over {FULL_TRIALS} trials)",
          ["noisy bits", "weight bits in unreliable memory", "bit-error prob", "accuracy %", "change vs error-free"], hl)
    save_csv("noise_headline.csv", ["noisy_bits", "fraction", "ber", "accuracy", "delta"], hl)

    # ---------------------------------------------------------------- figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        plt = None
        print("\n(matplotlib not installed - skipping figures; pip install matplotlib)")
    if plt is not None:
        bers = [2.0 ** -k for k in KS]
        fig, ax = plt.subplots(figsize=(6.2, 4))
        for name, _ in cfgs:
            m = [grid[(name, k)][0] for k in KS]
            s = [grid[(name, k)][1] for k in KS]
            ax.errorbar(bers, m, yerr=s, marker="o", capsize=2, label=name,
                        linestyle="--" if name == "4 MSBs" else "-")
        ax.axhline(base_g, color="gray", linestyle=":", label="error-free")
        ax.set_xscale("log")
        ax.set_xlabel("bit-error probability per read")
        ax.set_ylabel("accuracy (%)")
        ax.set_title("Hybrid weight memory: accuracy vs unreliable bits")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "fig_noise.png"), dpi=200)

        fig, ax = plt.subplots(figsize=(5, 3.6))
        ax.plot([r[0] for r in wb], [float(r[1]) for r in wb], marker="o")
        ax.set_xlabel("weight bits")
        ax.set_ylabel("accuracy (%)")
        ax.set_title("Accuracy vs weight precision")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "fig_weight_bits.png"), dpi=200)

        fig, ax = plt.subplots(figsize=(5.4, 3.6))
        ax.plot([r[0] for r in ts], [float(r[1]) for r in ts], marker="o", color="C0")
        ax.set_xlabel("time steps T")
        ax.set_ylabel("accuracy (%)", color="C0")
        ax2 = ax.twinx()
        ax2.plot([r[0] for r in ts], [float(r[2]) for r in ts], marker="s", color="C1")
        ax2.set_ylabel("cycles / image", color="C1")
        ax.set_title("Accuracy and latency vs time steps")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "fig_timesteps.png"), dpi=200)

        fig, ax = plt.subplots(figsize=(5.4, 3.6))
        ax.hist(rows, bins=50)
        ax.axvline(rows.mean(), color="r", linestyle="--", label=f"mean {rows.mean():.0f}")
        ax.set_xlabel(f"weight rows fetched per image (dense design: {dense_rows})")
        ax.set_ylabel("images")
        ax.set_title("Event-driven memory traffic")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "fig_events.png"), dpi=200)
        print("\nfigures written to results/fig_*.png")

    with open(os.path.join(OUT_DIR, "report_tables.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"\nwrote results/report_tables.md and CSV files  ({time.time() - t_start:.0f}s total)")


if __name__ == "__main__":
    main()
