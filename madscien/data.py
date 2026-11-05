"""
Data tools: byte-level tokenizer, synthetic corpus generator, memmap dataset.

Build a dataset:
    python -m madscien.data --source synth --n_docs 50000 --out data/synth
    python -m madscien.data --source text  --path my_synthetic.txt --out data/mine

Documents in a text file are separated by blank lines. Output: train.bin, val.bin (uint16
token ids) and meta.json (vocab_size) - same layout as nanoGPT.
"""
import argparse
import json
import math
import os

import numpy as np

BOS, EOS, PAD = 256, 257, 258
SYSTEM, USER, ASSISTANT, END = 259, 260, 261, 262     # chat roles + end-of-turn
VOCAB_SIZE = 263


class ByteTokenizer:
    """Dependency-free: ids 0-255 are raw bytes (UTF-8), then BOS / EOS / PAD and the chat tokens."""
    vocab_size = VOCAB_SIZE
    bos_id, eos_id, pad_id = BOS, EOS, PAD
    system_id, user_id, assistant_id, end_id = SYSTEM, USER, ASSISTANT, END

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, ids):
        return bytes(int(i) for i in ids if 0 <= int(i) < 256).decode("utf-8", errors="replace")


# ------------------------------------------------------------------ synthetic corpus
def _fmt(x):
    return f"{x:.3f}".rstrip("0").rstrip(".")


def _sig(x, n=3):
    """n significant figures; for small answers where 3 decimals would round to 0."""
    return f"{x:.{n}g}"


def synth_doc(rng):
    """One synthetic worked problem. Topics mirror the algorithm: electrostatics, EMI,
    angular momentum, MOT bond order, GOC stability, King property, P&C."""
    kind = int(rng.integers(0, 7))
    if kind == 0:
        q1, q2, r = (int(rng.integers(1, 20)), int(rng.integers(1, 20)), int(rng.integers(1, 10)))
        F = 9e9 * (q1 * 1e-6) * (q2 * 1e-6) / (r * r)
        return (f"Problem: Two charges {q1} uC and {q2} uC are {r} m apart. Find the force.\n"
                f"Solution: F = k q1 q2 / r^2 = 9e9 x {q1}e-6 x {q2}e-6 / {r}^2 = {_sig(F)} N, repulsive.\n")
    if kind == 1:
        n, dphi, dt = int(rng.integers(10, 100)), int(rng.integers(1, 50)) / 100, int(rng.integers(1, 6))
        return (f"Problem: A coil of {n} turns sees its flux change by {_fmt(dphi)} Wb in {dt} s. Find the emf.\n"
                f"Solution: e = -N d(phi)/dt = -{n} x {_fmt(dphi)} / {dt} = {_fmt(-n * dphi / dt)} V.\n")
    if kind == 2:
        i1, w1, i2 = int(rng.integers(1, 10)), int(rng.integers(1, 20)), int(rng.integers(1, 10))
        return (f"Problem: A skater has I = {i1} kg m^2 and spins at {w1} rad/s. She changes I to {i2} kg m^2. "
                f"Find the new spin rate.\n"
                f"Solution: No external torque, so L = I w is conserved. w2 = I1 w1 / I2 = "
                f"{i1} x {w1} / {i2} = {_fmt(i1 * w1 / i2)} rad/s.\n")
    if kind == 3:
        nb = int(rng.integers(2, 13))
        na = int(rng.integers(0, nb + 3))
        bo = (nb - na) / 2
        verdict = "is stable" if bo > 0 else ("does not exist" if bo == 0 else "is unstable")
        return (f"Problem: A diatomic species has {nb} bonding and {na} antibonding electrons. Find the bond order.\n"
                f"Solution: bond order = (Nb - Na) / 2 = ({nb} - {na}) / 2 = {_fmt(bo)}, so the species {verdict}.\n")
    if kind == 4:
        names = ["primary", "secondary", "tertiary"]
        rank = {"primary": 1, "secondary": 2, "tertiary": 3}
        opts = [names[i] for i in rng.permutation(3)[: int(rng.integers(2, 4))]]
        best = max(opts, key=lambda s: rank[s])
        return (f"Problem: Which carbocation is most stable: {', '.join(opts)}?\n"
                f"Solution: Stability follows 3 > 2 > 1 because of hyperconjugation and the inductive effect, "
                f"so the {best} carbocation is the most stable.\n")
    if kind == 5:
        a = int(rng.integers(0, 5))
        b = a + int(rng.integers(1, 8))
        f = ["e^x", "x^2 + 1", "2^x"][int(rng.integers(0, 3))]
        return (f"Problem: Evaluate the integral from {a} to {b} of f(x) / (f(x) + f({a + b} - x)) dx, f(x) = {f}.\n"
                f"Solution: By the King property I = integral of f({a + b} - x) / (f({a + b} - x) + f(x)) dx. "
                f"Adding both gives 2I = integral of 1 dx = {b - a}, so I = {_fmt((b - a) / 2)}.\n")
    n = int(rng.integers(5, 16))
    r = int(rng.integers(1, n + 1))
    return (f"Problem: In how many ways can {r} student(s) be chosen from {n}?\n"
            f"Solution: Order does not matter, so use combinations: C({n},{r}) = {math.comb(n, r)}.\n")


def synthetic_documents(n_docs, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(n_docs):
        yield synth_doc(rng)


# ------------------------------------------------------------------ build / load
def build_bins(docs, out_dir, val_frac=0.01, seed=0):
    tok = ByteTokenizer()
    ids = []
    for doc in docs:
        ids.append(BOS)
        ids.extend(tok.encode(doc))
        ids.append(EOS)
    arr = np.array(ids, dtype=np.uint16)
    n_val = max(1, int(len(arr) * val_frac))
    os.makedirs(out_dir, exist_ok=True)
    arr[:-n_val].tofile(os.path.join(out_dir, "train.bin"))
    arr[-n_val:].tofile(os.path.join(out_dir, "val.bin"))
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({"vocab_size": VOCAB_SIZE, "tokenizer": "byte", "n_train": len(arr) - n_val,
                   "n_val": n_val}, f)
    return len(arr) - n_val, n_val


class BinDataset:
    def __init__(self, data_dir):
        self.data = {s: np.memmap(os.path.join(data_dir, f"{s}.bin"), dtype=np.uint16, mode="r")
                     for s in ("train", "val")}
        with open(os.path.join(data_dir, "meta.json")) as f:
            self.meta = json.load(f)

    def get_batch(self, split, batch_size, block_size, device, generator=None):
        import torch
        d = self.data[split]
        hi = len(d) - block_size - 1
        assert hi > 0, f"{split}.bin has only {len(d)} tokens, need > block_size+1"
        ix = torch.randint(hi, (batch_size,), generator=generator)
        x = torch.stack([torch.from_numpy(d[i:i + block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(d[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
        if str(device).startswith("cuda"):
            return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x.to(device), y.to(device)


def slice_xy(tokens, mask, i, T):
    """x = tokens[i:i+T]; y = next tokens, with -100 wherever mask says 'do not train on this token'."""
    x = np.asarray(tokens[i:i + T], dtype=np.int64)
    y = np.asarray(tokens[i + 1:i + 1 + T], dtype=np.int64).copy()
    y[np.asarray(mask[i + 1:i + 1 + T]) == 0] = -100
    return x, y


class ChatBinDataset:
    """SFT data: <split>.bin (uint16 tokens) + <split>_mask.bin (uint8, 1 = train on this token)."""

    def __init__(self, data_dir):
        self.tokens = {s: np.memmap(os.path.join(data_dir, f"{s}.bin"), dtype=np.uint16, mode="r")
                       for s in ("train", "val")}
        self.mask = {s: np.memmap(os.path.join(data_dir, f"{s}_mask.bin"), dtype=np.uint8, mode="r")
                     for s in ("train", "val")}
        with open(os.path.join(data_dir, "meta.json")) as f:
            self.meta = json.load(f)

    def get_batch(self, split, batch_size, block_size, device, generator=None):
        import torch
        t, m = self.tokens[split], self.mask[split]
        hi = len(t) - block_size - 1
        assert hi > 0, f"{split}.bin has only {len(t)} tokens, need > block_size+1"
        xs, ys = [], []
        for _ in range(batch_size):
            for _try in range(50):                         # skip windows with nothing to learn from
                i = int(torch.randint(hi, (1,), generator=generator))
                x, y = slice_xy(t, m, i, block_size)
                if (y != -100).any():
                    break
            xs.append(x)
            ys.append(y)
        return (torch.from_numpy(np.stack(xs)).to(device), torch.from_numpy(np.stack(ys)).to(device))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["synth", "text"], default="synth")
    ap.add_argument("--path", default=None, help="text file (documents separated by blank lines)")
    ap.add_argument("--n_docs", type=int, default=50000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val_frac", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.source == "synth":
        docs = synthetic_documents(a.n_docs, a.seed)
    else:
        assert a.path, "--path is required for --source text"
        docs = [d.strip() + "\n" for d in open(a.path, encoding="utf-8").read().split("\n\n") if d.strip()]
    n_tr, n_va = build_bins(docs, a.out, a.val_frac)
    print(f"wrote {a.out}: train {n_tr} tokens, val {n_va} tokens")
