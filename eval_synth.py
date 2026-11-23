"""
Score a checkpoint on the synthetic worked problems (madscien/data.py), by exact full solution.

    python eval_synth.py --ckpt out/ckpt.pt --data_dir data/synth

Loss alone cannot separate recall from skill, so the report splits three cases:
  seen          the problem text is in the training set (recall)
  not seen      in the training number ranges, but not in the training set
  out of range  numbers outside the training ranges, the hardest test of copying and arithmetic

A whole solution must match exactly, so one slip fails a problem. Two partial scores show progress
that the exact score hides:
  numbers copied   every number in the problem appears whole somewhere in the solution
  steps correct    written steps (multiplications with their sums, subtractions, long-division steps)
                   that are right given their own inputs, even when an earlier step was wrong

--calculator lets madscien/calculator.py fix every result the model gets wrong as it writes, so the
scores show how well the model sets problems up and copies their numbers, with arithmetic taken out.
The report also counts the corrections it made.

A checkpoint must be scored on the same solution format it was trained on: older checkpoints fail
every problem type whose solution now has worked steps.
"""
import argparse
import math
import os
import re
import sys
import time
from collections import Counter

import numpy as np

from madscien.data import (BOS, EOS, MAX_COMBINATIONS_N, SYNTH_KINDS, _int, angular_problem, bond_problem,
                           combinations_problem, coulomb_problem, emf_problem, king_problem, synth_problem)

MAX_SOLUTION_TOKENS = 1600      # longer than any solution in or out of the training ranges


def _digits(rng, n):
    return _int(rng, 10 ** (n - 1), 10 ** n - 1)


# Every number is longer than training ever uses: the large numbers, which training sometimes draws one
# digit long, by two digits; the divisors by one (combinations: a larger n).
OUT_OF_RANGE = {
    "coulomb": lambda rng: coulomb_problem(_digits(rng, 5), _digits(rng, 5), _digits(rng, 3)),
    "emf": lambda rng: emf_problem(_digits(rng, 6), _digits(rng, 5), _digits(rng, 3)),
    "angular": lambda rng: angular_problem(_digits(rng, 5), _digits(rng, 5), _digits(rng, 3)),
    "bond": lambda rng: bond_problem(_digits(rng, 5), _digits(rng, 5)),
    "king": lambda rng: king_problem(*sorted((_digits(rng, 5), _digits(rng, 5))), "e^x"),
    "combinations": lambda rng: (lambda n: combinations_problem(n, _int(rng, 2, n - 2)))(
        _int(rng, MAX_COMBINATIONS_N + 1, 30)),
}


DIVISION_STEP = re.compile(r"(\d+) / (\d+) = (\d+), (\d+) - (\d+) = (?:\d+ - \d+ - \d+ = \d+ - \d+ = )?(\d+)")
SUBTRACTION_CHAIN = re.compile(r"\d+(?: - \d+)+(?: = \d+(?: - \d+)*)+")
MULTIPLICATION_CHAIN = re.compile(r"\d+ x \d+(?: = \d+(?: [x+] \d+)*)+")    # through the sums and product
NUMBER = re.compile(r"\d+(?:\.\d+)?")


def step_checks(solution):
    """One bool per written step: is it right given the numbers it starts from?"""
    checks = []
    for a, b, q, a2, p, r in DIVISION_STEP.findall(solution):
        a, b, q, a2, p, r = map(int, (a, b, q, a2, p, r))
        checks.append(a == a2 and p == q * b and r == a - p and 0 <= r < b)
    for chain in SUBTRACTION_CHAIN.findall(solution):
        values = [int(t[0]) - sum(map(int, t[1:])) for t in (part.split(" - ") for part in chain.split(" = "))]
        checks.append(len(set(values)) == 1)
    for chain in MULTIPLICATION_CHAIN.findall(solution):
        values = [sum(math.prod(map(int, term.split(" x "))) for term in part.split(" + "))
                  for part in chain.split(" = ")]
        checks.append(len(set(values)) == 1)
    return checks


def numbers_copied(problem, solution):
    """Every number in the problem appears whole in the solution (not cut short, not run on)."""
    return all(re.search(rf"(?<![\d.]){re.escape(n)}(?![\d]|\.\d)", solution) for n in NUMBER.findall(problem))


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


def evaluate(solve, known, problems, progress=None):
    """solve(problem) -> text after 'Solution:'. Returns counts as {group: {type: [correct, total, copied,
    steps right, steps written]}}, and the first wrong answer per group and type.
    progress(done, total) is called after every problem."""
    results = {"seen": {}, "not seen": {}, "out of range": {}}
    wrong = {}
    for done, (in_range, name, problem, solution) in enumerate(problems, 1):
        if progress:
            progress(done - 1, len(problems))
        group = "out of range" if not in_range else ("seen" if problem in known else "not seen")
        got = solve(problem)
        row = results[group].setdefault(name, [0, 0, 0, 0, 0])
        checks = step_checks(got)
        row[0] += got == solution
        row[1] += 1
        row[2] += numbers_copied(problem, got)
        row[3] += sum(checks)
        row[4] += len(checks)
        if got != solution:
            wrong.setdefault((group, name), (problem, solution.strip(), got.strip()))
    return results, wrong


def print_report(results, wrong):
    def pct(a, b):
        return f"{100 * a / b:3.0f}%" if b else "   -"

    print("\nexact = whole solution right | copied = problem's numbers all appear whole | "
          "steps = written steps right given their own inputs")
    for group, rows in results.items():
        c, t, cp, sr, sw = (sum(r[i] for r in rows.values()) for i in range(5))
        print(f"\n{group}: exact {c}/{t} | copied {cp}/{t} | steps {sr}/{sw} {pct(sr, sw)}")
        for name, (c, t, cp, sr, sw) in rows.items():
            print(f"  {name:13s} exact {c:3d}/{t:<3d} | copied {cp:3d}/{t:<3d} | steps {sr:4d}/{sw:<4d} {pct(sr, sw)}")
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
    ap.add_argument("--calculator", action="store_true", help="fix arithmetic slips while generating")
    a = ap.parse_args()

    import torch

    from madscien import calculator
    from madscien.config import X5Config
    from madscien.model import MadScienLM

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location=device, weights_only=True)
    model = MadScienLM(X5Config(**ck["cfg"])).to(device).eval()
    model.load_state_dict(ck["model"])

    fixes = []

    def solve(problem):
        ids = [BOS] + list((problem + "\nSolution:").encode("utf-8"))
        if a.calculator:
            new, n = calculator.generate(model, ids, MAX_SOLUTION_TOKENS, EOS)
            fixes.append(n)
        else:
            out = model.generate(torch.tensor([ids], device=device), MAX_SOLUTION_TOKENS, temperature=0,
                                 eos_id=EOS)
            new = out[0].tolist()[len(ids):]
        return bytes(t for t in new if t < 256).decode("utf-8", "replace")

    known, floor = training_stats(a.data_dir)
    print(f"{a.ckpt}: iter {ck['iter']} | best val {ck['best_val']:.4f} | "
          f"estimated lowest possible loss {floor:.4f}")
    problems = scoring_problems(a.n_per_type, a.n_out_of_range, a.seed)
    start = time.time()

    def progress(done, total):                     # on stderr, so a stall shows within minutes
        if done and done % 10 == 0:
            spent = time.time() - start
            print(f"  scored {done}/{total} problems | {spent / 60:.1f} min | about "
                  f"{spent / done * (total - done) / 60:.0f} min left", file=sys.stderr, flush=True)

    print_report(*evaluate(solve, known, problems, progress))
    if a.calculator:
        print(f"\ncalculator: {sum(fixes)} corrections in {len(fixes)} problems "
              f"({sum(n > 0 for n in fixes)} problems needed at least one)")


if __name__ == "__main__":
    main()
