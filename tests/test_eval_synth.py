"""Scorer checks with mock solvers: no model or GPU needed."""
import re
from collections import Counter

import numpy as np

from eval_synth import (MAX_SOLUTION_TOKENS, OUT_OF_RANGE, evaluate, numbers_copied, scoring_problems,
                        step_checks)
from madscien.data import MAX_DOC_TOKENS, SYNTH_KINDS, is_held_out


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


def test_generation_limit_and_context_cover_the_longest_solutions():
    rng = np.random.default_rng(0)
    for make in OUT_OF_RANGE.values():
        for _ in range(500):
            problem, solution = make(rng)
            assert len(solution) + 1 < MAX_SOLUTION_TOKENS
            assert doc_tokens(problem, solution) <= MAX_DOC_TOKENS     # the problem stays in view


def doc_tokens(problem, solution):
    return len(f"{problem}\nSolution:{solution}".encode()) + 2           # + BOS and EOS


def test_training_documents_fit_the_recommended_context():
    rng = np.random.default_rng(1)
    for make in SYNTH_KINDS.values():
        for _ in range(2000):
            assert doc_tokens(*make(rng)) <= MAX_DOC_TOKENS


def test_out_of_range_numbers_are_longer_than_any_in_training():
    rng = np.random.default_rng(2)

    def longest(make):
        return max(len(n) for _ in range(3000) for n in re.findall(r"\d+", make(rng)[0]))

    for name, make in OUT_OF_RANGE.items():
        if name != "combinations":                           # combinations: larger n, more steps
            assert longest(make) > longest(SYNTH_KINDS[name]), name
    assert longest(SYNTH_KINDS["coulomb"]) == 4 and longest(SYNTH_KINDS["emf"]) == 5    # with the long mix
    for name in ("bond", "king"):                            # every length up to the longest is common
        lengths = Counter(len(n) for _ in range(3000) for n in re.findall(r"\d+", SYNTH_KINDS[name](rng)[0])[:2])
        assert min(lengths[d] for d in (1, 2, 3)) > 0.2 * sum(lengths.values()), (name, lengths)


def test_partial_scores_on_real_model_mistakes():
    # from scored checkpoints: a 4-digit charge cut to 3 digits, and a wrong long-division step
    problem = "Problem: Two charges 5858 uC and 2453 uC are 590 m apart. Find the force."
    assert numbers_copied(problem, " F = 9e9 x 5858e-6 x 2453e-6 / 590^2 N = 9 x 5858 x 2453 / 590^2 mN.")
    assert not numbers_copied(problem, " F = 9e9 x 585e-6 x 245e-6 / 590^2 N.")
    assert not numbers_copied(problem, " F = 9e9 x 5858e-6 x 2453e-6 / 59^2 N.")
    assert not numbers_copied(problem, " F = 9e9 x 58580e-6 x 2453e-6 / 590^2 N.")      # run on
    assert numbers_copied("Problem: flux change by 2.89 Wb in 2 s.", " -9997 x 2.89 / 2")
    assert not numbers_copied("Problem: flux change by 2.89 Wb in 2 s.", " -9997 x 2.8 / 2")
    got = ("1156 x 9 = 10404. 10040 / 1156 = 9, 10040 - 10404 = 936, 9360 / 1156 = 8, 9360 - 9248 = 112. "
           "7200 + 560 + 40 = 7700. 8 x 975 = 8 x 900 + 8 x 70 + 8 x 5 = 7200 + 560 + 40 = 7700. 12 x = 5")
    # division and subtraction steps: one wrong, one right each; "1156 x 9" right; "8 x 975" wrong in its sum
    assert sorted(step_checks(got)) == [False, False, False, True, True, True]


def test_scorer_counts_partial_scores():
    problems = scoring_problems(2, 2, seed=4)
    key = {p: s for _, _, p, s in problems}
    results, _ = evaluate(key.__getitem__, set(), problems)
    for rows in results.values():
        for c, t, copied, right, written in rows.values():
            assert c == t == copied and right == written
    assert sum(r[4] for rows in results.values() for r in rows.values()) > 100
    results, _ = evaluate(lambda p: "no steps here", set(), problems)
    assert all(r[2] == r[3] == r[4] == 0 for name, r in results["out of range"].items())


def test_a_few_training_numbers_run_one_digit_long():
    rng = np.random.default_rng(5)
    charges = [len(n) for _ in range(4000)
               for n in re.findall(r"(\d+) uC", SYNTH_KINDS["coulomb"](rng)[0])]
    distances = [len(re.findall(r"(\d+) m apart", SYNTH_KINDS["coulomb"](rng)[0])[0]) for _ in range(2000)]
    share = charges.count(4) / len(charges)
    assert 0.03 < share < 0.07 and max(charges) == 4, share          # LONG_NUMBER_SHARE, one digit only
    assert max(distances) == 2                                        # divisors keep their range
