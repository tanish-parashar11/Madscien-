"""Scorer checks with mock solvers: no model or GPU needed."""
import numpy as np

from eval_synth import MAX_SOLUTION_TOKENS, OUT_OF_RANGE, evaluate, scoring_problems
from madscien.data import SYNTH_KINDS, is_held_out


def totals(rows):
    return sum(r[0] for r in rows.values()), sum(r[1] for r in rows.values())


def test_perfect_solver_scores_everything_and_wrong_solver_nothing():
    problems = scoring_problems(3, 2, seed=1)
    key = {p: s for _, _, p, s in problems}
    results, wrong = evaluate(key.__getitem__, set(), problems)
    assert not wrong
    in_range = 3 * len(SYNTH_KINDS) + 3 * (len(SYNTH_KINDS) - 1)      # goc has no held-out problems
    assert totals(results["not seen"]) == (in_range, in_range)
    assert totals(results["out of range"]) == (2 * len(OUT_OF_RANGE), 2 * len(OUT_OF_RANGE))

    results, wrong = evaluate(lambda p: "nonsense", set(), problems)
    assert totals(results["not seen"])[0] == 0 and totals(results["out of range"])[0] == 0
    assert wrong


def test_known_problems_are_counted_as_seen():
    problems = scoring_problems(2, 1, seed=2)
    key = {p: s for _, _, p, s in problems}
    results, _ = evaluate(key.__getitem__, set(key), problems)
    assert totals(results["seen"])[1] == 2 * len(SYNTH_KINDS) + 2 * (len(SYNTH_KINDS) - 1)
    assert not results["not seen"]


def test_problem_groups_keep_their_promises():
    problems = scoring_problems(20, 20, seed=3)
    assert {name for _, name, _, _ in problems} == set(SYNTH_KINDS)
    assert sum(is_held_out(p) for in_range, name, p, _ in problems if in_range) == 20 * (len(SYNTH_KINDS) - 1)


def test_generation_limit_covers_the_longest_solutions():
    rng = np.random.default_rng(0)
    for make in OUT_OF_RANGE.values():
        for _ in range(500):
            _, solution = make(rng)
            assert len(solution) + 1 < MAX_SOLUTION_TOKENS
