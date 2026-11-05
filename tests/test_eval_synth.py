"""Scorer checks with mock solvers: no model or GPU needed."""
import math

import numpy as np

from eval_synth import OUT_OF_RANGE, TYPES, evaluate, in_range_problem, problem_type


def answer_key(n_per_type, n_out_of_range, seed):
    """Replays the problems evaluate() will ask, as {problem: solution}."""
    key = {}
    rng = np.random.default_rng(seed)
    # far more draws than the rarest type needs, so every problem asked is covered
    for _ in range(50 * n_per_type * len(TYPES)):
        problem, solution = in_range_problem(rng)
        key[problem] = solution
    rng = np.random.default_rng(seed + 1)
    for make in OUT_OF_RANGE.values():
        for _ in range(n_out_of_range):
            problem, solution = make(rng)
            key[problem] = solution
    return key


def totals(rows):
    return sum(r[0] for r in rows.values()), sum(r[1] for r in rows.values())


def test_perfect_solver_scores_everything_and_wrong_solver_nothing():
    key = answer_key(3, 2, seed=1)
    results, wrong = evaluate(key.__getitem__, known=set(), n_per_type=3, n_out_of_range=2, seed=1)
    assert not wrong
    assert totals(results["not seen"]) == (3 * len(TYPES), 3 * len(TYPES))
    assert totals(results["out of range"]) == (2 * len(OUT_OF_RANGE), 2 * len(OUT_OF_RANGE))

    results, wrong = evaluate(lambda p: "nonsense", known=set(), n_per_type=3, n_out_of_range=2, seed=1)
    assert totals(results["not seen"])[0] == 0 and totals(results["out of range"])[0] == 0
    assert wrong


def test_known_problems_are_counted_as_seen():
    key = answer_key(2, 1, seed=2)
    results, _ = evaluate(key.__getitem__, known=set(key), n_per_type=2, n_out_of_range=1, seed=2)
    assert totals(results["seen"])[1] == 2 * len(TYPES)
    assert not results["not seen"]


def test_out_of_range_problems_are_well_formed():
    rng = np.random.default_rng(0)
    for name, make in OUT_OF_RANGE.items():
        for _ in range(20):
            problem, solution = make(rng)
            assert problem_type(problem) == name and problem.startswith("Problem: ")
            assert solution.startswith(" ") and solution.endswith("\n")
    problem, solution = OUT_OF_RANGE["combinations"](np.random.default_rng(3))
    n = int(problem.split("from ")[1].rstrip("?"))
    r = int(problem.split("can ")[1].split(" student")[0])
    assert n >= 16 and solution.rstrip().endswith(f"= {math.comb(n, r)}.")
