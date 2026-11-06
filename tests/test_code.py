"""Coding-capability tests. These need no torch:  pytest -q tests/test_code.py"""
import math

import numpy as np
import pytest

from madscien.codegen_data import generate_tasks
from madscien.constraints import DelimiterConstraint
from madscien.verify import extract_solution, is_valid_python, pass_at_k, run_tests


def test_every_task_is_verified_end_to_end_in_the_sandbox():
    tasks = list(generate_tasks(400, seed=7))          # oracle gate runs inside generate_tasks
    assert len({t.entry for t in tasks}) >= 30
    for t in tasks[:60]:
        r = run_tests(t.solution, t.entry, t.cases)
        assert r.passed, (t.entry, r)
        assert is_valid_python(t.solution)
        assert t.document().endswith("### End\n")


def test_documents_are_self_contained_and_prompt_is_a_prefix():
    for t in list(generate_tasks(50, seed=3)):
        assert t.document().startswith(t.prompt())
        assert t.prompt().endswith("### Solution\n")
        assert extract_solution(t.document()[len(t.prompt()):]).strip() == t.solution.strip()


def test_sandbox_catches_wrong_syntax_timeout_and_missing_entry():
    cases = [[[2, 3], 5], [[10, -4], 6]]
    assert run_tests("def add(a, b):\n    return a + b\n", "add", cases).passed
    wrong = run_tests("def add(a, b):\n    return a - b\n", "add", cases)
    assert not wrong.passed and wrong.ok < wrong.total
    assert not is_valid_python("def add(a, b:\n    return a + b\n")
    assert not run_tests("def add(a, b):\n    return a + b\n", "plus", cases).passed
    slow = run_tests("def add(a, b):\n    while True:\n        pass\n", "add", cases, timeout=1.0)
    assert not slow.passed and slow.error == "timeout"
    boom = run_tests("def add(a, b):\n    raise RuntimeError('x')\n", "add", cases)
    assert not boom.passed


def test_pass_at_k_known_values():
    assert pass_at_k(10, 0, 1) == 0.0
    assert pass_at_k(10, 10, 5) == 1.0
    assert math.isclose(pass_at_k(10, 5, 1), 0.5)
    assert math.isclose(pass_at_k(10, 5, 2), 1 - math.comb(5, 2) / math.comb(10, 2))
    assert pass_at_k(4, 3, 2) == 1.0
    with pytest.raises(ValueError):                    # fewer samples than k: undefined, not 1.0
        pass_at_k(2, 0, 4)


def test_eval_skips_pass_at_k_larger_than_samples():
    from code_eval import evaluate, make_mock_generate
    tasks = list(generate_tasks(5, seed=4))
    res = evaluate(make_mock_generate(tasks, 0.0), tasks, n_samples=2, ks=(1, 4), verbose=False)
    assert res["pass@1"] == 0.0 and "pass@4" not in res


def test_heldout_eval_tasks_never_repeat_training_prompts():
    from madscien.codegen_data import heldout_tasks
    train = {t.prompt() for t in generate_tasks(3000, 0)}
    ev = heldout_tasks(100, 12345, train_n_docs=3000, train_seed=0)
    prompts = [t.prompt() for t in ev]
    assert len(ev) == 100 and len(set(prompts)) == 100
    assert not train.intersection(prompts)


def _independent_scan(s):
    """Separate implementation used to check the constraint."""
    stack, mode, q, i = [], "code", None, 0
    while i < len(s):
        ch = s[i]
        if mode == "code":
            if ch == "#":
                mode = "comment"
            elif ch in "'\"":
                mode, q = "str", ch
            elif ch in "([{":
                stack.append(ch)
            elif ch in ")]}":
                if not stack or "([{".index(stack[-1]) != ")]}".index(ch):
                    return False, stack, mode
                stack.pop()
        elif mode == "comment":
            if ch == "\n":
                mode = "code"
        else:
            if ch == "\\":
                i += 1
            elif ch == q:
                mode = "code"
        i += 1
    return True, stack, mode


def test_delimiter_constraint_never_mismatches_and_repair_balances():
    rng = np.random.default_rng(0)
    alphabet = list("()[]{}a \"'\\#\n")
    for _ in range(400):
        c, out = DelimiterConstraint(), []
        for _ in range(int(rng.integers(5, 80))):
            ok = c.allowed_bytes()
            choices = [ch for ch in alphabet if ok[ord(ch)]]
            ch = choices[int(rng.integers(0, len(choices)))]
            c.feed(ord(ch))
            out.append(ch)
        s = "".join(out)
        good, _, _ = _independent_scan(s)
        assert good and c.violations == 0
        good2, stack, mode = _independent_scan(s + c.repair())
        assert good2 and not stack and mode == "code"
        assert c.can_end() == (not stack_after(s))


def stack_after(s):
    _, stack, mode = _independent_scan(s)
    return stack or mode == "str"


def test_constraint_masks_closers_and_eos():
    c = DelimiterConstraint()
    assert not c.allowed_bytes()[ord(")")] and c.can_end()
    c.feed(ord("("))
    c.feed(ord("["))
    ok = c.allowed_bytes()
    assert ok[ord("]")] and not ok[ord(")")] and not ok[ord("}")] and not c.can_end()
    c.feed(ord("]"))
    c.feed(ord(")"))
    assert c.can_end() and c.violations == 0
    c2 = DelimiterConstraint()
    c2.feed_many(list(b"x = '("))                      # bracket inside a string is not structure
    assert c2.quote == "'" and not c2.stack


def test_eval_harness_metrics_behave_like_theory():
    from code_eval import evaluate, make_mock_generate
    tasks = list(generate_tasks(30, seed=99))
    perfect = evaluate(make_mock_generate(tasks, 1.0), tasks, n_samples=4, ks=(1,), verbose=False)
    assert perfect["pass@1"] == 1.0 and perfect["verified_pick"] == 1.0
    half = evaluate(make_mock_generate(tasks, 0.5, seed=1), tasks, n_samples=8, ks=(1,), verbose=False)
    assert 0.3 < half["pass@1"] < 0.7
    assert half["verified_pick"] > half["pass@1"] + 0.2      # sample-and-verify lifts accuracy
