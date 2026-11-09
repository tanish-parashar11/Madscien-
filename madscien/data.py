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
import zlib

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
# Solutions work the arithmetic out digit by digit, so getting a new answer right means applying the
# method to the numbers in front of the model rather than recalling a finished result.
HELD_OUT_PER_10 = 1          # share of numeric problems never trained on; they form the validation set


def _fmt(x):
    return f"{x:.3f}".rstrip("0").rstrip(".")


def _mul_steps(a, b):
    """'a x b' split over the digits of b, ending in the product."""
    parts = [int(d) * 10 ** i for i, d in enumerate(reversed(str(b))) if d != "0"][::-1]
    if len(parts) < 2:
        return f"{a} x {b} = {a * b}"
    terms = " + ".join(f"{a} x {p}" for p in parts)
    values = " + ".join(str(a * p) for p in parts)
    return f"{a} x {b} = {terms} = {values} = {a * b}"


def _long_division(num, den, places=3):
    """Digit-by-digit division. Returns (steps, value); a value cut off at `places` ends in '...'."""
    steps, whole, rem = [], "", 0
    for digit in str(num):
        cur = rem * 10 + int(digit)
        rem = cur % den
        if whole or cur >= den:
            whole += str(cur // den)
            steps.append(f"{cur} / {den} = {cur // den} r {rem}")
    decimals = ""
    while rem and len(decimals) < places:
        cur = rem * 10
        rem = cur % den
        decimals += str(cur // den)
        steps.append(f"{cur} / {den} = {cur // den} r {rem}")
    value = (whole or "0") + (f".{decimals}" if decimals else "") + ("..." if rem else "")
    return ", ".join(steps), value


def coulomb_problem(q1, q2, r):
    product = 9 * q1
    steps, force = _long_division(product * q2, r * r)
    return (f"Problem: Two charges {q1} uC and {q2} uC are {r} m apart. Find the force.",
            f" F = k q1 q2 / r^2 = 9e9 x {q1}e-6 x {q2}e-6 / {r}^2 N = 9 x {q1} x {q2} / {r}^2 mN. "
            f"{_mul_steps(9, q1)}. {_mul_steps(product, q2)}. {_mul_steps(r, r)}. {steps}. "
            f"F = {force} mN, repulsive.\n")


def emf_problem(n, hundredths, dt):
    dphi = _fmt(hundredths / 100)
    steps, emf = _long_division(n * hundredths, 100 * dt)
    return (f"Problem: A coil of {n} turns sees its flux change by {dphi} Wb in {dt} s. Find the emf.",
            f" e = -N d(phi)/dt = -{n} x {dphi} / {dt} = -{n} x {hundredths} / {100 * dt}. "
            f"{_mul_steps(n, hundredths)}. {steps}. e = -{emf} V.\n")


def angular_problem(i1, w1, i2):
    steps, w2 = _long_division(i1 * w1, i2)
    return (f"Problem: A skater has I = {i1} kg m^2 and spins at {w1} rad/s. She changes I to {i2} kg m^2. "
            f"Find the new spin rate.",
            f" No external torque, so L = I w is conserved. w2 = I1 w1 / I2 = {i1} x {w1} / {i2}. "
            f"{_mul_steps(i1, w1)}. {steps}. w2 = {w2} rad/s.\n")


def bond_problem(nb, na):
    diff = nb - na
    verdict = "is stable" if diff > 0 else ("does not exist" if diff == 0 else "is unstable")
    return (f"Problem: A diatomic species has {nb} bonding and {na} antibonding electrons. Find the bond order.",
            f" bond order = (Nb - Na) / 2 = ({nb} - {na}) / 2 = {diff} / 2 = {_fmt(diff / 2)}, "
            f"so the species {verdict}.\n")


def goc_problem(options):
    rank = {"primary": 1, "secondary": 2, "tertiary": 3}
    best = max(options, key=lambda s: rank[s])
    return (f"Problem: Which carbocation is most stable: {', '.join(options)}?",
            f" Stability follows 3 > 2 > 1 because of hyperconjugation and the inductive effect, "
            f"so the {best} carbocation is the most stable.\n")


def king_problem(a, b, f):
    return (f"Problem: Evaluate the integral from {a} to {b} of f(x) / (f(x) + f({a + b} - x)) dx, f(x) = {f}.",
            f" By the King property I = integral of f({a + b} - x) / (f({a + b} - x) + f(x)) dx. "
            f"Adding both gives 2I = integral of 1 dx = {b} - {a} = {b - a}, "
            f"so I = {b - a} / 2 = {_fmt((b - a) / 2)}.\n")


def combinations_problem(n, r):
    k = min(r, n - r)
    work = f"C({n},{r}) = C({n},{k}). " if k != r else ""
    value = n
    for i in range(2, k + 1):
        product = value * (n - i + 1)
        work += f"{value} x {n - i + 1} = {product}, {product} / {i} = {product // i}. "
        value = product // i
    answer = math.comb(n, r)
    assert k == 0 or value == answer
    return (f"Problem: In how many ways can {r} student(s) be chosen from {n}?",
            f" Order does not matter, so use combinations: {work}C({n},{r}) = {answer}.\n")


def _int(rng, lo, hi):
    return int(rng.integers(lo, hi + 1))


def _goc_options(rng):
    names = ["primary", "secondary", "tertiary"]
    return [names[i] for i in rng.permutation(3)[: _int(rng, 2, 3)]]


def _draw_king(rng):
    a = _int(rng, 0, 60)
    return king_problem(a, a + _int(rng, 1, 60), ["e^x", "x^2 + 1", "2^x"][_int(rng, 0, 2)])


def _draw_combinations(rng):
    n = _int(rng, 5, 20)
    return combinations_problem(n, _int(rng, 1, n))


# Training number ranges. Topics mirror the algorithm: electrostatics, EMI, angular momentum,
# MOT bond order, GOC stability, King property, P&C.
SYNTH_KINDS = {
    "coulomb": lambda rng: coulomb_problem(_int(rng, 1, 99), _int(rng, 1, 99), _int(rng, 1, 19)),
    "emf": lambda rng: emf_problem(_int(rng, 10, 999), _int(rng, 1, 99), _int(rng, 1, 9)),
    "angular": lambda rng: angular_problem(_int(rng, 1, 99), _int(rng, 1, 99), _int(rng, 1, 19)),
    "bond": lambda rng: bond_problem(_int(rng, 2, 60), _int(rng, 0, 60)),
    "goc": lambda rng: goc_problem(_goc_options(rng)),
    "king": _draw_king,
    "combinations": _draw_combinations,
}


def is_held_out(problem):
    """Fixed per problem text, so train and validation can never share a problem. Problems with no
    numbers are never held out: there are too few of them, and they test nothing new."""
    return any(c.isdigit() for c in problem) and zlib.crc32(problem.encode("utf-8")) % 10 < HELD_OUT_PER_10


def synth_problem(rng, kind=None, split="train"):
    """(problem, solution) from the 'train' part of the problem space or the held-out 'val' part."""
    if kind is None:
        kinds = [k for k in SYNTH_KINDS if split == "train" or k != "goc"]
        kind = kinds[int(rng.integers(0, len(kinds)))]
    assert split == "train" or kind != "goc", "goc problems have no numbers, so none are held out"
    while True:
        problem, solution = SYNTH_KINDS[kind](rng)
        if is_held_out(problem) == (split == "val"):
            return problem, solution


def synth_doc(rng, split="train"):
    problem, solution = synth_problem(rng, split=split)
    return f"{problem}\nSolution:{solution}"


def synthetic_documents(n_docs, seed=0, split="train"):
    rng = np.random.default_rng(seed)
    for _ in range(n_docs):
        yield synth_doc(rng, split)


# ------------------------------------------------------------------ build / load
def build_bins(docs, out_dir, val_frac=0.01, val_docs=None):
    """Without val_docs, the last val_frac of the tokens become the validation set."""
    tok = ByteTokenizer()

    def encode(documents):
        ids = []
        for doc in documents:
            ids.append(BOS)
            ids.extend(tok.encode(doc))
            ids.append(EOS)
        return np.array(ids, dtype=np.uint16)

    arr = encode(docs)
    if val_docs is None:
        n_val = max(1, int(len(arr) * val_frac))
        train, val = arr[:-n_val], arr[-n_val:]
    else:
        train, val = arr, encode(val_docs)
    os.makedirs(out_dir, exist_ok=True)
    train.tofile(os.path.join(out_dir, "train.bin"))
    val.tofile(os.path.join(out_dir, "val.bin"))
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({"vocab_size": VOCAB_SIZE, "tokenizer": "byte", "n_train": len(train),
                   "n_val": len(val)}, f)
    return len(train), len(val)


class BinDataset:
    """align_docs=True starts every window at a document's BOS, so the model always sees the start of the
    document it is predicting (for example the problem it is solving). Tokens more than block_size into a
    document are then never trained on, so use a block_size that fits whole documents."""

    def __init__(self, data_dir, align_docs=False):
        self.data = {s: np.memmap(os.path.join(data_dir, f"{s}.bin"), dtype=np.uint16, mode="r")
                     for s in ("train", "val")}
        with open(os.path.join(data_dir, "meta.json")) as f:
            self.meta = json.load(f)
        self.align_docs = align_docs
        self._starts = {}

    def doc_starts(self, split, block_size):
        """BOS positions that leave room for a full window."""
        if split not in self._starts:
            self._starts[split] = np.flatnonzero(np.asarray(self.data[split]) == BOS)
        starts = self._starts[split]
        return starts[starts < len(self.data[split]) - block_size - 1]

    def get_batch(self, split, batch_size, block_size, device, generator=None):
        import torch
        d = self.data[split]
        hi = len(d) - block_size - 1
        assert hi > 0, f"{split}.bin has only {len(d)} tokens, need > block_size+1"
        if self.align_docs:
            starts = self.doc_starts(split, block_size)
            assert len(starts), f"{split}.bin has no document start with a full window after it"
            ix = starts[torch.randint(len(starts), (batch_size,), generator=generator).numpy()]
        else:
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
        n_val = max(10, int(a.n_docs * a.val_frac))      # enough tokens for one training context
        n_tr, n_va = build_bins(synthetic_documents(a.n_docs - n_val, a.seed), a.out,
                                val_docs=synthetic_documents(n_val, a.seed + 1, split="val"))
    else:
        assert a.path, "--path is required for --source text"
        docs = [d.strip() + "\n" for d in open(a.path, encoding="utf-8").read().split("\n\n") if d.strip()]
        n_tr, n_va = build_bins(docs, a.out, a.val_frac)
    print(f"wrote {a.out}: train {n_tr} tokens, val {n_va} tokens")
