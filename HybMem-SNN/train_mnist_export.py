"""
HybMem-SNN : event-driven SNN accelerator for MNIST with a hybrid (reliable + unreliable) weight memory
-------------------------------------------------------------------------------------------------------
MNIST is downloaded automatically (no torchvision needed).
Trains on ALL 60,000 training images and tests on ALL 10,000 test images.

Network   : 196 (14x14 pixels, rate coded) -> 64 LIF -> 10 LIF, shift-based leak, soft reset
Hardware  : event-driven, weights in BRAM, 8 weights fetched per cycle (8 "lanes"),
            weight bits can be marked "unreliable" (emerging-memory model) and get random read errors.

Run :  python train_mnist_export.py          (training + export, a few minutes)
Then:  python analysis_report.py             (all sweeps / tables / figures for the report)

Makes: weights_init.vh     weight memory contents + T + threshold for snn_accel.v
       test_data.vh        test images, expected output counts and cycles, noise configs for tb_snn_accel.v
       model_weights.npz   trained weights (float + int8) used by analysis_report.py
       results/train_log.csv
"""
import os
import csv
import gzip
import urllib.request
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- settings you may tune ----------------
EPOCHS   = 10
BATCH    = 128
LR       = 2e-3
T        = 16            # time steps per image
NTEST    = 100           # images exported for the Verilog testbench (keep = tb_snn_accel.v)
SEED     = 0
# (noisy_bit_mask, k):  bits set in the mask live in the UNRELIABLE memory and flip with prob 2^-k
#   0xFF = all 8 weight bits unreliable,  0x0F = only the 4 LSBs unreliable (MSBs in reliable memory)
CONFIGS  = [(0x00, 0), (0xFF, 8), (0xFF, 5), (0xFF, 3), (0x0F, 5), (0x0F, 3)]
# --------------------------------------------------------
DATA_DIR = "mnist_data"
OUT_DIR  = "results"
NPIX, NH, NOUT, NOP = 196, 64, 10, 16      # NOP = outputs padded to 16 (2 chunks of 8 lanes)
TH       = 64            # integer firing threshold  (float threshold 1.0 * 64)
BETA     = 15.0 / 16.0   # float equivalent of  v - (v >> 4)
PRNG_SEED = 1            # seed of the input-spike generator (xorshift32), same as the Verilog
DENSE_ROWS_PER_STEP = NPIX + NH            # weight rows a non event-driven design would fetch per step

torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------- MNIST download ----------------
MIRRORS = ["https://ossci-datasets.s3.amazonaws.com/mnist/",
           "https://storage.googleapis.com/cvdf-datasets/mnist/",
           "https://yann.lecun.com/exdb/mnist/"]
FILES = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
         "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]


def fetch(name):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, name)
    if os.path.exists(path) and os.path.getsize(path) > 100:
        return path
    for base in MIRRORS:
        try:
            print(f"downloading {base}{name}")
            urllib.request.urlretrieve(base + name, path)
            return path
        except Exception as e:
            print("  failed:", e)
    raise RuntimeError(f"could not download {name}; download MNIST manually into '{DATA_DIR}/'")


def read_idx(path, images):
    with gzip.open(path, "rb") as f:
        data = f.read()
    if images:
        x = np.frombuffer(data, np.uint8, offset=16).reshape(-1, 28, 28).astype(np.int64)
        return x.reshape(-1, 14, 2, 14, 2).sum((2, 4)) // 4        # 2x2 average -> 14x14, 0..255
    return np.frombuffer(data, np.uint8, offset=8).astype(np.int64)


def load_mnist():
    xtr = read_idx(fetch(FILES[0]), True).reshape(-1, NPIX)
    ytr = read_idx(fetch(FILES[1]), False)
    xte = read_idx(fetch(FILES[2]), True).reshape(-1, NPIX)
    yte = read_idx(fetch(FILES[3]), False)
    return xtr, ytr, xte, yte


# ---------------- input spike generator (identical in Verilog) ----------------
def xs32(x):
    x ^= (x << 13) & 0xFFFFFFFF
    x ^= x >> 17
    x ^= (x << 5) & 0xFFFFFFFF
    return x


def make_R(steps=T):
    """R[t, p] = random byte compared with pixel p at time step t.  spike = pixel > R."""
    s = PRNG_SEED
    R = np.zeros((steps, NPIX), np.int64)
    for t in range(steps):
        for p in range(NPIX):
            R[t, p] = s & 0xFF
            s = xs32(s)
    return R


# ---------------- float model (training) ----------------
class Spike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x >= 0).float()

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g / (1 + 10 * x.abs()) ** 2


class SNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(NPIX, NH, bias=False)
        self.fc2 = nn.Linear(NH, NOUT, bias=False)
        with torch.no_grad():
            self.fc1.weight.mul_(3.0)

    def forward(self, prob):                         # prob: [B, 196] spike probability = pixel/256
        B = prob.shape[0]
        v1 = torch.zeros(B, NH)
        v2 = torch.zeros(B, NOUT)
        cnt = torch.zeros(B, NOUT)
        for _ in range(T):
            x = (torch.rand_like(prob) < prob).float()
            v1 = v1 * BETA + self.fc1(x)
            s1 = Spike.apply(v1 - 1.0)
            v1 = v1 - s1
            v2 = v2 * BETA + self.fc2(s1)
            s2 = Spike.apply(v2 - 1.0)
            v2 = v2 - s2
            cnt = cnt + s2
        return cnt


# ---------------- integer model (bit-accurate copy of the Verilog, no memory errors) ----------------
def run_int(pix, W1, W2, R):
    """Returns output spike counts [N,16], total input spikes per image, total hidden spikes per image."""
    N = len(pix)
    v1 = np.zeros((N, NH), np.int64)
    v2 = np.zeros((N, NOP), np.int64)
    cnt = np.zeros((N, NOP), np.int64)
    n_in = np.zeros(N, np.int64)
    n_h = np.zeros(N, np.int64)
    W1f, W2f = W1.T.astype(np.float64), W2.T.astype(np.float64)       # exact for these sizes
    for t in range(len(R)):
        s_in = pix > R[t][None, :]
        v1 = v1 - (v1 >> 4) + np.rint(s_in.astype(np.float64) @ W1f).astype(np.int64)
        s1 = v1 >= TH
        v1 = v1 - s1 * TH
        v2 = v2 - (v2 >> 4) + np.rint(s1.astype(np.float64) @ W2f).astype(np.int64)
        s2 = v2 >= TH
        v2 = v2 - s2 * TH
        cnt += s2
        n_in += s_in.sum(1)
        n_h += s1.sum(1)
    return cnt, n_in, n_h


def cycles_model(n_in, n_h, steps=T):
    """Exact cycle count of snn_accel.v.
    per time step: 1 leak + (196 pixel scans + 16 per input spike) + 1 + (64 hidden scans + 4 per hidden spike) + 1,
    plus 1 final cycle (S_DONE)."""
    return 1 + steps * (1 + NPIX + 1 + NH + 1) + 16 * n_in + 4 * n_h


# ---------------- same model, but every weight-row read goes through the noisy memory ----------------
def err_bytes(shape, k, mask):
    """Random error byte: every bit is 1 with probability 2^-k (AND of k random bytes), only on bits in `mask`."""
    e = torch.randint(0, 256, shape, dtype=torch.uint8)
    for _ in range(k - 1):
        e &= torch.randint(0, 256, shape, dtype=torch.uint8)
    return e & mask


def flip_read(rows_u8, mask, k):
    """rows_u8: uint8 tensor of stored weight bytes. Bits in `mask` flip with probability 2^-k on each read."""
    if k > 0 and mask != 0:
        rows_u8 = rows_u8 ^ err_bytes(rows_u8.shape, k, mask)
    return rows_u8.view(torch.int8).long()


def _noisy_chunk(pix, W1u, W2u, R, mask, k):
    N = len(pix)
    v1 = torch.zeros(N, NH, dtype=torch.int64)
    v2 = torch.zeros(N, NOP, dtype=torch.int64)
    cnt = torch.zeros(N, NOP, dtype=torch.int64)
    pix_t = torch.tensor(pix)
    for t in range(len(R)):
        s_in = pix_t > torch.tensor(R[t])[None, :]
        b, p = s_in.nonzero(as_tuple=True)
        i1 = torch.zeros(N, NH, dtype=torch.int64).index_add_(0, b, flip_read(W1u[p], mask, k))
        v1 = v1 - (v1 >> 4) + i1
        s1 = v1 >= TH
        v1 = v1 - s1.long() * TH
        b, h = s1.nonzero(as_tuple=True)
        i2 = torch.zeros(N, NOP, dtype=torch.int64).index_add_(0, b, flip_read(W2u[h], mask, k))
        v2 = v2 - (v2 >> 4) + i2
        s2 = v2 >= TH
        v2 = v2 - s2.long() * TH
        cnt += s2.long()
    return cnt.numpy()


def run_noisy(pix, W1, W2, R, mask, k, chunk=1000):
    W1u = torch.tensor((W1.T & 0xFF).astype(np.uint8))            # [196, 64] one row per input pixel
    W2u = torch.tensor((W2.T & 0xFF).astype(np.uint8))            # [64, 16]  one row per hidden neuron
    return np.concatenate([_noisy_chunk(pix[i:i + chunk], W1u, W2u, R, mask, k)
                           for i in range(0, len(pix), chunk)])


# ---------------- export ----------------
def write_weights(W1, W2):
    w1_words = []
    for p in range(NPIX):
        for c in range(NH // 8):
            w1_words.append(sum((int(W1[c * 8 + l, p]) & 0xFF) << (8 * l) for l in range(8)))
    w2_words = []
    for h in range(NH):
        for c in range(NOP // 8):
            w2_words.append(sum((int(W2[c * 8 + l, h]) & 0xFF) << (8 * l) for l in range(8)))
    with open("weights_init.vh", "w") as f:
        f.write("// auto-generated by train_mnist_export.py - do not edit\n")
        f.write(f"localparam TSTEPS = {T};\n")
        f.write(f"localparam signed [31:0] TH = {TH};\n")
        f.write(f'(* rom_style = "block" *) reg [63:0] w1_mem [0:{len(w1_words) - 1}];\n')
        f.write(f'(* rom_style = "block" *) reg [63:0] w2_mem [0:{len(w2_words) - 1}];\n')
        f.write("initial begin\n")
        for i, w in enumerate(w1_words):
            f.write(f"  w1_mem[{i}] = 64'h{w:016x};\n")
        for i, w in enumerate(w2_words):
            f.write(f"  w2_mem[{i}] = 64'h{w:016x};\n")
        f.write("end\n")


def write_test_data(pix, labels, cnt, cyc):
    with open("test_data.vh", "w") as f:
        f.write("// auto-generated by train_mnist_export.py - do not edit\n")
        f.write(f"localparam NTEST_PY = {len(pix)};\n")
        f.write(f"localparam TB_TSTEPS = {T};\n")
        f.write(f"localparam NCFG = {len(CONFIGS)};\n")
        f.write(f"reg [7:0] cfg_mask [0:{len(CONFIGS) - 1}];\n")
        f.write(f"reg [3:0] cfg_ber  [0:{len(CONFIGS) - 1}];\n")
        f.write("initial begin\n")
        for i, (m, k) in enumerate(CONFIGS):
            f.write(f"  cfg_mask[{i}] = 8'h{m:02x}; cfg_ber[{i}] = 4'd{k};\n")
        for n in range(len(pix)):
            for p in range(NPIX):
                f.write(f"  timg[{n * NPIX + p}] = 8'd{int(pix[n, p])};\n")
            for k in range(NOUT):
                f.write(f"  exp_cnt[{n * NOUT + k}] = 8'd{int(cnt[n, k])};\n")
            f.write(f"  lab[{n}] = 4'd{int(labels[n])};  exp_cyc[{n}] = 32'd{int(cyc[n])};\n")
        f.write("end\n")


# ---------------- main ----------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    xtr, ytr, xte, yte = load_mnist()
    print(f"MNIST ready: {len(xtr)} train / {len(xte)} test images (downsampled to 14x14)")
    R = make_R()

    x_t = torch.tensor(xtr / 256.0, dtype=torch.float32)
    y_t = torch.tensor(ytr)
    model = SNN()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    N = len(y_t)
    log = []
    for ep in range(EPOCHS):
        perm = torch.randperm(N)
        tot, correct = 0.0, 0
        for i in range(0, N, BATCH):
            b = perm[i:i + BATCH]
            out = model(x_t[b])
            loss = F.cross_entropy(out * 0.5, y_t[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                for p in model.parameters():
                    p.clamp_(-1.98, 1.98)            # |w| < 2*threshold  -> fits signed 8 bit
            tot += loss.item() * len(b)
            correct += (out.argmax(1) == y_t[b]).sum().item()
        log.append((ep + 1, tot / N, correct / N * 100))
        print(f"epoch {ep + 1}/{EPOCHS}  loss {tot / N:.3f}  train acc {correct / N * 100:.1f}%")
    with open(os.path.join(OUT_DIR, "train_log.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "loss", "train_acc_percent"])
        w.writerows(log)

    # quantize: w -> round(w * 64) (threshold 1.0 -> 64), pad output layer from 10 to 16 neurons
    W1f = model.fc1.weight.detach().numpy()
    W2f = model.fc2.weight.detach().numpy()
    W1 = np.clip(np.round(W1f * TH), -127, 127).astype(np.int64)             # [64,196]
    W2s = np.clip(np.round(W2f * TH), -127, 127).astype(np.int64)            # [10,64]
    W2 = np.zeros((NOP, NH), np.int64)
    W2[:NOUT] = W2s
    np.savez("model_weights.npz", W1=W1, W2=W2, W1f=W1f, W2f=W2f)

    cnt, n_in, n_h = run_int(xte, W1, W2, R)
    pred = cnt[:, :NOUT].argmax(1)                    # ties -> lowest index (same as Verilog)
    acc = (pred == yte).mean() * 100
    rows = n_in + n_h
    dense = T * DENSE_ROWS_PER_STEP
    cyc = cycles_model(n_in, n_h)
    print(f"\n=== hardware-equivalent integer model, full MNIST test set ({len(xte)} images, no memory errors) ===")
    print(f"accuracy                      : {acc:.2f}%")
    print(f"weight-row reads per image    : {rows.mean():.0f}  (dense, non event-driven: {dense})")
    print(f"skipped by event-driven design: {(1 - rows.mean() / dense) * 100:.1f}%")
    print(f"cycles per image              : {cyc.mean():.0f}  ({cyc.mean() / 100:.1f} us at 100 MHz)")

    write_weights(W1, W2)
    write_test_data(xte[:NTEST], yte[:NTEST], cnt[:NTEST], cyc[:NTEST])
    print(f"\nwrote weights_init.vh, test_data.vh ({NTEST} test images), model_weights.npz")
    print("next:  python analysis_report.py")


if __name__ == "__main__":
    main()
