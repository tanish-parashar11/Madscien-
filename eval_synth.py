"""
Score a checkpoint on the synthetic worked problems (madscien/data.py), by exact full solution.

    python eval_synth.py --ckpt out/ckpt.pt --data_dir data/synth

Loss alone cannot separate recall from skill, so the report splits three cases:
  seen          the problem text is in the training set (recall)
  not seen      in the training number ranges, but not in the training set
  out of range  numbers outside the training ranges, the hardest test of copying and arithmetic

A checkpoint must be scored on the same solution format it was trained on: older checkpoints fail
every problem type whose solution now has worked steps.
"""
import argparse
import math
import os
from collections import Counter

import numpy as np

from madscien.data import (BOS, EOS, SYNTH_KINDS, _int, angular_problem, bond_problem, combinations_problem,
                           coulomb_problem, emf_problem, king_problem, synth_problem)

MAX_SOLUTION_TOKENS = 800       # longer than any solution in or out of the training ranges

OUT_OF_RANGE = {
    "coulomb": lambda rng: coulomb_problem(_int(rng, 100, 300), _int(rng, 100, 300), _int(rng, 20, 40)),
    "emf": lambda rng: emf_problem(_int(rng, 1000, 3000), _int(rng, 100, 300), _int(rng, 10, 20)),
    "angular": lambda rng: angular_problem(_int(rng, 100, 300), _int(rng, 100, 300), _int(rng, 20, 60)),
    "bond": lambda rng: bond_problem(_int(rng, 100, 400), _int(rng, 100, 400)),
    "king": lambda rng: king_problem(_int(rng, 100, 300), _int(rng, 400, 600), "e^x"),
    "combinations": lambda rng: combinations_problem(_int(rng, 21, 26), _int(rng, 2, 19)),
}


def training_stats(data_dir):
    """Problems in the training set, and a rough lowest loss: everything except the problem's random
    numbers is determined, so its entropy / document length estimates the floor (nats/token)."""
    tokens = np.fromfile(os.path.join(data_dir, "train.bin"), dtype=np.uint16)
    text = tokens[tokens < 256].astype(np.uint8).tobytes().decode("utf-8", "replace")
    problems = Counter(line for line in text.split("\n") if line.startswith("Problem:"))
    total = sum(problems.values())
    entropy = -sum(c / total * math.log(c / total) for c in problems.values())
    floor = entropy * total / len(tokens) if total else float("nan")
    return set(problems), floor


def scoring_problems(n_per_type=30, n_out_of_range=20, seed=987654):
    """[(in_range, type, problem, solution)]: n_per_type from the training part of each type's problem
    space, n_per_type from the held-out part, and n_out_of_range outside the ranges."""
    rng = np.random.default_rng(seed)
    problems = []
    for name in SYNTH_KINDS:
        for split in ("train", "val"):
            if split == "train" or name != "goc":        # goc has no numbers, so nothing is held out
                problems += [(True, name, *synth_problem(rng, name, split)) for _ in range(n_per_type)]
    for name, make in OUT_OF_RANGE.items():
        problems += [(False, name, *make(rng)) for _ in range(n_out_of_range)]
    return problems


def evaluate(solve, known, problems):
    """solve(problem) -> text after 'Solution:'. Returns counts as {group: {type: [correct, total]}}."""
    results = {"seen": {}, "not seen": {}, "out of range": {}}
    wrong = {}
    for in_range, name, problem, solution in problems:
        group = "out of range" if not in_range else ("seen" if problem in known else "not seen")
        got = solve(problem)
        row = results[group].setdefault(name, [0, 0])
        row[0] += got == solution
        row[1] += 1
        if got != solution:
            wrong.setdefault((group, name), (problem, solution.strip(), got.strip()))
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
    ap.add_argument("--n_per_type", type=int, default=30, help="in-range problems per type and split")
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
        out = model.generate(torch.tensor([ids], device=device), MAX_SOLUTION_TOKENS, temperature=0, eos_id=EOS)
        return bytes(t for t in out[0].tolist()[len(ids):] if t < 256).decode("utf-8", "replace")

    known, floor = training_stats(a.data_dir)
    print(f"{a.ckpt}: iter {ck['iter']} | best val {ck['best_val']:.4f} | "
          f"estimated lowest possible loss {floor:.4f}")
    problems = scoring_problems(a.n_per_type, a.n_out_of_range, a.seed)
    print_report(*evaluate(solve, known, problems))


if __name__ == "__main__":
    main()
