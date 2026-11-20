"""The synthetic solutions must be right, step by step: a model trained on them learns every mistake."""
import math
import re
from fractions import Fraction

import numpy as np
import pytest

from eval_synth import OUT_OF_RANGE, step_checks
from madscien.data import (BinDataset, SYNTH_KINDS, _long_division, _mul_steps, _sub_steps, build_bins,
                           is_held_out, synth_doc, synth_problem, synthetic_documents)


def truncated(value, places=3):
    """The text _long_division must give for an exact Fraction."""
    scaled = value * 10 ** places
    whole, frac = divmod(int(scaled), 10 ** places)
    digits = f"{frac:0{places}d}"
    if scaled.denominator == 1:
        digits = digits.rstrip("0")
    text = str(whole) + (f".{digits}" if digits else "")
    return text if scaled.denominator == 1 else text + "..."


def check_steps(solution):
    """Recompute every multiplication, sum, subtraction and division written in a solution."""
    assert " r " not in solution, solution                          # remainders are worked out now
    assert all(step_checks(solution)), solution                     # the scorer's own step check


@pytest.mark.parametrize("a,b", [(46, 52), (9, 7), (813, 9), (120, 305), (7, 100), (5, 0)])
def test_mul_steps_end_in_the_product(a, b):
    assert _mul_steps(a, b).endswith(f"= {a * b}")
    check_steps(_mul_steps(a, b))


@pytest.mark.parametrize("a,b", [(813, 245), (52722, 2453), (46, 52), (9, 1001)])
def test_partial_products_are_added_two_at_a_time(a, b):
    sums = [part for part in _mul_steps(a, b).split(" = ")[2:] if " x " not in part]
    terms = [len(part.split(" + ")) for part in sums]
    assert terms == list(range(terms[0], 0, -1)), sums       # one addition per step, down to the product
    assert sums[-1] == str(a * b)


def test_long_division_matches_exact_fractions():
    rng = np.random.default_rng(0)
    for _ in range(3000):
        num, den = int(rng.integers(1, 200000)), int(rng.integers(1, 2000))
        steps, value = _long_division(num, den)
        assert value == truncated(Fraction(num, den)), (num, den, value)
        check_steps(steps)


def test_final_answers_are_correct():
    rng = np.random.default_rng(1)
    makers = {**{f"train {k}": v for k, v in SYNTH_KINDS.items()},
              **{f"out {k}": v for k, v in OUT_OF_RANGE.items()}}
    for name, make in makers.items():
        for _ in range(400):
            problem, solution = make(rng)
            check_steps(solution)
            assert problem.startswith("Problem: ") and solution.startswith(" ") and solution.endswith(".\n")
            if "coulomb" in name:
                q1, q2, r = map(int, re.findall(r"(\d+) uC and (\d+) uC are (\d+) m", problem)[0])
                assert f"F = {truncated(Fraction(9 * q1 * q2, r * r), places=2)} mN" in solution
            elif "emf" in name:
                n, dphi, dt = re.findall(r"of (\d+) turns .* by ([\d.]+) Wb in (\d+) s", problem)[0]
                assert f"e = -{truncated(int(n) * Fraction(dphi) / int(dt))} V" in solution
            elif "angular" in name:
                i1, w1, i2 = map(int, re.findall(r"I = (\d+) kg m\^2 and spins at (\d+) rad/s.* to (\d+) kg", problem)[0])
                assert f"w2 = {truncated(Fraction(i1 * w1, i2))} rad/s" in solution
            elif "bond" in name:
                nb, na = map(int, re.findall(r"(\d+) bonding and (\d+) antibonding", problem)[0])
                order = f"{(nb - na) / 2:g}"
                verdict = "is stable" if nb > na else ("does not exist" if nb == na else "is unstable")
                assert f"= {order}, so the species {verdict}." in solution
            elif "king" in name:
                a, b = map(int, re.findall(r"from (\d+) to (\d+) of", problem)[0])
                assert solution.endswith(f"so I = {b - a} / 2 = {(b - a) / 2:g}.\n")
            elif "combinations" in name:
                r, n = map(int, re.findall(r"can (\d+) student\(s\) be chosen from (\d+)", problem)[0])
                assert solution.endswith(f"C({n},{r}) = {math.comb(n, r)}.\n")


def test_edge_cases_of_combinations_need_no_steps():
    rng = np.random.default_rng(2)
    seen = set()
    for _ in range(4000):
        problem, solution = synth_problem(rng, "combinations")
        r, n = map(int, re.findall(r"can (\d+) student\(s\) be chosen from (\d+)", problem)[0])
        seen.add(r == n)
        assert solution.endswith(f"C({n},{r}) = {math.comb(n, r)}.\n")
    assert seen == {True, False}


def test_held_out_problems_never_reach_training():
    rng = np.random.default_rng(3)
    for _ in range(3000):
        assert not is_held_out(synth_problem(rng)[0])
    for _ in range(300):
        assert is_held_out(synth_problem(rng, split="val")[0])
    for kind in set(SYNTH_KINDS) - {"goc"}:                          # every numeric type can be held out
        assert is_held_out(synth_problem(rng, kind, "val")[0])
    with pytest.raises(AssertionError):
        synth_problem(rng, "goc", "val")


def test_built_dataset_keeps_validation_problems_out_of_training(tmp_path):
    n_tr, n_va = build_bins(synthetic_documents(400, 0), tmp_path,
                            val_docs=synthetic_documents(40, 1, split="val"))
    ds = BinDataset(str(tmp_path))
    assert len(ds.data["train"]) == n_tr and len(ds.data["val"]) == n_va

    def problems(tokens):
        text = bytes(int(t) for t in tokens if t < 256).decode()
        return {line for line in text.split("\n") if line.startswith("Problem:")}

    train, val = problems(ds.data["train"]), problems(ds.data["val"])
    assert val and not train & val
    assert synth_doc(np.random.default_rng(0)).split("\nSolution: ", 1)[1]    # format chat_data.py splits on


def test_aligned_windows_start_at_a_document_and_hold_its_whole_problem(tmp_path):
    import torch

    from madscien.data import BOS
    build_bins(synthetic_documents(300, 0), tmp_path, val_docs=synthetic_documents(30, 1, split="val"))
    ds = BinDataset(str(tmp_path), align_docs=True)
    g = torch.Generator().manual_seed(0)
    for split in ("train", "val"):
        x, y = ds.get_batch(split, 64, 128, "cpu", generator=g)
        assert (x[:, 0] == BOS).all()
        assert torch.equal(x[:, 1:], y[:, :-1])                    # targets are still the next tokens
        text = bytes(int(t) for t in x[0, 1:] if t < 256).decode()
        assert text.startswith("Problem: ")
    starts = ds.doc_starts("train", 128)
    assert len(starts) and starts.max() < len(ds.data["train"]) - 129
    x, _ = BinDataset(str(tmp_path)).get_batch("train", 64, 128, "cpu", generator=g)
    assert not (x[:, 0] == BOS).all()                               # the default still samples anywhere


@pytest.mark.parametrize("a,b", [(78, 29), (5, 14), (35, 36), (101, 44), (411, 210), (7, 7), (60, 0), (290, 270)])
def test_sub_steps_end_in_the_difference(a, b):
    text = _sub_steps(a, b)
    assert text.startswith(f"{a} - {b} = ") and text.endswith(f" = {a - b}")
    check_steps(text)
    if a < b:
        assert f"{b} - {a} = " in text                             # the borrow is worked the other way round


def test_remainders_take_off_the_leading_part_first():
    from madscien.data import _division_step
    assert _division_step(5062, 3364) == "5062 / 3364 = 1, 5062 - 3364 = 5062 - 3000 - 364 = 2062 - 364 = 1698"
    assert _division_step(1650, 3364) == "1650 / 3364 = 0, 1650 - 0 = 1650"
    assert _division_step(70, 7) == "70 / 7 = 10, 70 - 70 = 0"                  # nothing left to split
    rng = np.random.default_rng(5)
    for _ in range(2000):
        cur, den = int(rng.integers(0, 10 ** 6)), int(rng.integers(1, 10 ** 4))
        step = _division_step(cur, den)
        check_steps(step)
        assert step.endswith(f"= {cur % den}")


def test_combination_steps_count_up_to_k():
    rng = np.random.default_rng(6)
    for _ in range(500):
        problem, solution = synth_problem(rng, "combinations")
        r, n = map(int, re.findall(r"can (\d+) student\(s\) be chosen from (\d+)", problem)[0])
        k = min(r, n - r)
        assert [int(i) for i in re.findall(r"\((\d+) of (?:\d+)\)", solution)] == list(range(2, k + 1))
        assert set(re.findall(r"\(\d+ of (\d+)\)", solution)) <= {str(k)}
        assert (f"because {n} - {r} = {k}" in solution) == (k != r)
