"""Scorer checks with mock solvers: no model or GPU needed."""
import re
from collections import Counter

import numpy as np

from eval_synth import MAX_SOLUTION_TOKENS, OUT_OF_RANGE, evaluate, scoring_problems
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
    assert longest(SYNTH_KINDS["coulomb"]) == 3 and longest(SYNTH_KINDS["emf"]) == 4
    for name in ("bond", "king"):                            # every length up to the longest is common
        lengths = Counter(len(n) for _ in range(3000) for n in re.findall(r"\d+", SYNTH_KINDS[name](rng)[0])[:2])
        assert min(lengths[d] for d in (1, 2, 3)) > 0.2 * sum(lengths.values()), (name, lengths)
