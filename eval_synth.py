"""
Score a checkpoint on the synthetic worked problems (madscien/data.py), by exact full solution.

    python eval_synth.py --ckpt out/ckpt.pt --data_dir data/synth

Loss alone cannot separate recall from skill, so the report splits three cases:
  seen          the problem text is in the training set (recall)
  not seen      in the training number ranges, but not in the training set
  out of range  numbers outside the training ranges, the only real test of copying and arithmetic
"""
import argparse
import math
import os
from collections import Counter

import numpy as np

from madscien.data import BOS, EOS, _fmt, _sig, synth_doc

TYPES = {"Two charges": "coulomb", "A coil": "emf", "A skater": "angular", "A diatomic": "bond",
         "Which carbocation": "goc", "Evaluate the integral": "king", "In how many": "combinations"}


def problem_type(problem):
    return next(name for marker, name in TYPES.items() if marker in problem)


def in_range_problem(rng):
    problem, solution = synth_doc(rng).split("\nSolution:", 1)
    return problem, solution


# ---- same text as synth_doc, with numbers outside its ranges
def _coulomb(rng):
    a, b, d = int(rng.integers(20, 60)), int(rng.integers(20, 60)), int(rng.integers(10, 20))
    force = 9e9 * a * 1e-6 * b * 1e-6 / (d * d)
    return (f"Problem: Two charges {a} uC and {b} uC are {d} m apart. Find the force.",
            f" F = k q1 q2 / r^2 = 9e9 x {a}e-6 x {b}e-6 / {d}^2 = {_sig(force)} N, repulsive.\n")


def _emf(rng):
    n, dphi, dt = int(rng.integers(100, 400)), int(rng.integers(50, 99)) / 100, int(rng.integers(6, 12))
    return (f"Problem: A coil of {n} turns sees its flux change by {_fmt(dphi)} Wb in {dt} s. Find the emf.",
            f" e = -N d(phi)/dt = -{n} x {_fmt(dphi)} / {dt} = {_fmt(-n * dphi / dt)} V.\n")


def _bond(rng):
    nb, na = int(rng.integers(14, 30)), int(rng.integers(14, 30))
    bo = (nb - na) / 2
    verdict = "is stable" if bo > 0 else ("does not exist" if bo == 0 else "is unstable")
    return (f"Problem: A diatomic species has {nb} bonding and {na} antibonding electrons. Find the bond order.",
            f" bond order = (Nb - Na) / 2 = ({nb} - {na}) / 2 = {_fmt(bo)}, so the species {verdict}.\n")


def _combinations(rng):
    n = int(rng.integers(16, 30))
    r = int(rng.integers(1, n + 1))
    return (f"Problem: In how many ways can {r} student(s) be chosen from {n}?",
            f" Order does not matter, so use combinations: C({n},{r}) = {math.comb(n, r)}.\n")


OUT_OF_RANGE = {"coulomb": _coulomb, "emf": _emf, "bond": _bond, "combinations": _combinations}


def training_stats(data_dir):
    """Problems in the training set, and the lowest loss the data allows: everything except the problem's
    random numbers is determined, so its entropy / document length estimates the floor (nats/token)."""
    tokens = np.fromfile(os.path.join(data_dir, "train.bin"), dtype=np.uint16)
    text = tokens[tokens < 256].astype(np.uint8).tobytes().decode("utf-8", "replace")
    problems = Counter(line for line in text.split("\n") if line.startswith("Problem:"))
    total = sum(problems.values())
    entropy = -sum(c / total * math.log(c / total) for c in problems.values())
    floor = entropy * total / len(tokens) if total else float("nan")
    return set(problems), floor


def evaluate(solve, known, n_per_type=30, n_out_of_range=20, seed=987654):
    """solve(problem) -> text after 'Solution:'. Returns counts as {group: {type: [correct, total]}}."""
    results = {"seen": {}, "not seen": {}, "out of range": {}}
    wrong = {}

    def score(group, name, problem, solution):
        got = solve(problem)
        row = results[group].setdefault(name, [0, 0])
        row[0] += got == solution
        row[1] += 1
        if got != solution:
            wrong.setdefault((group, name), (problem, solution.strip(), got.strip()))

    rng = np.random.default_rng(seed)
    done = Counter()
    while min(done[name] for name in TYPES.values()) < n_per_type:
        problem, solution = in_range_problem(rng)
        name = problem_type(problem)
        if done[name] < n_per_type:
            done[name] += 1
            score("seen" if problem in known else "not seen", name, problem, solution)
    rng = np.random.default_rng(seed + 1)
    for name, make in OUT_OF_RANGE.items():
        for _ in range(n_out_of_range):
            score("out of range", name, *make(rng))
    return results, wrong


def print_report(results, wrong):
    for group, rows in results.items():
        correct, total = sum(r[0] for r in rows.values()), sum(r[1] for r in rows.values())
        print(f"\n{group}: {correct}/{total}")
        for name, (c, t) in rows.items():
            print(f"  {name:13s} {c}/{t}")
    if wrong:
        print("\nfirst wrong answer per group and type:")
    for (group, name), (problem, expected, got) in wrong.items():
        print(f"  [{group} / {name}] {problem}\n    expected: {expected}\n    got:      {got}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_dir", required=True, help="the dataset the checkpoint was trained on")
    ap.add_argument("--n_per_type", type=int, default=30, help="in-range problems per type")
    ap.add_argument("--n_out_of_range", type=int, default=20, help="out-of-range problems per type")
    ap.add_argument("--seed", type=int, default=987654)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    import torch

    from madscien.config import X5Config
    from madscien.model import MadScienLM

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location=device, weights_only=True)
    model = MadScienLM(X5Config(**ck["cfg"])).to(device).eval()
    model.load_state_dict(ck["model"])

    def solve(problem):
        ids = [BOS] + list((problem + "\nSolution:").encode("utf-8"))
        out = model.generate(torch.tensor([ids], device=device), 260, temperature=0, eos_id=EOS)
        return bytes(t for t in out[0].tolist()[len(ids):] if t < 256).decode("utf-8", "replace")

    known, floor = training_stats(a.data_dir)
    print(f"{a.ckpt}: iter {ck['iter']} | best val {ck['best_val']:.4f} | "
          f"estimated lowest possible loss {floor:.4f}")
    print_report(*evaluate(solve, known, a.n_per_type, a.n_out_of_range, a.seed))


if __name__ == "__main__":
    main()
