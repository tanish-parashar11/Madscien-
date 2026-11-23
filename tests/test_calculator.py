"""The calculator must fix slips and never touch a correct solution."""
import re
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from eval_synth import OUT_OF_RANGE
from madscien.calculator import DIVISION, STEP, fix, generate
from madscien.data import EOS, SYNTH_KINDS


def solutions(n=150, seed=0):
    rng = np.random.default_rng(seed)
    for make in [*SYNTH_KINDS.values(), *OUT_OF_RANGE.values()]:
        for _ in range(n):
            yield make(rng)[1]


def test_correct_solutions_are_never_changed():
    for solution in solutions():
        for i, c in enumerate(solution):
            if c in " (\n":
                assert fix(solution[:i + 1]) == solution[:i + 1], solution[:i + 1][-200:]


@pytest.mark.parametrize("wrong,right", [
    ("9829 x 7 = 68703. ", "9829 x 7 = 68803. "),                                       # from out_rem
    ("134 x 700 + 134 x 40 + 134 x 6 = 93800 + 5360 + 704 = ", "134 x 700 + 134 x 40 + 134 x 6 = 93800 + 5360 + 804 = "),
    ("1999400 + 799760 + 89973 = 2789160 + 89973 = ", "1999400 + 799760 + 89973 = 2799160 + 89973 = "),
    ("2799160 + 89973 = 3889133. ", "2799160 + 89973 = 2889133. "),
    ("23612 / 3364 = 6, ", "23612 / 3364 = 7, "),
    ("23612 / 3364 = 7, 23612 - 23448 = ", "23612 / 3364 = 7, 23612 - 23548 = "),
    ("5062 - 3364 = 5062 - 3000 - 264 = ", "5062 - 3364 = 5062 - 3000 - 364 = "),
    ("5062 - 3000 - 364 = 2162 - 364 = ", "5062 - 3000 - 364 = 2062 - 364 = "),
    ("1160 - 156 = 1904, ", "1160 - 156 = 1004, "),
    ("9134 - 1109 = 9134 - 1000 - 10 - 9 = ", "9134 - 1109 = 9134 - 1000 - 100 - 9 = "),
    ("because 26 - 15 = 10. ", "because 26 - 15 = 11. "),
    ("650 / 2 = 324 (", "650 / 2 = 325 ("),
    ("27 x 26 = 348, ", "27 x 26 = 702, "),
])
def test_real_slips_are_fixed(wrong, right):
    assert fix("Solution: " + wrong) == "Solution: " + right


def test_copying_and_multiplication_splits_are_left_to_the_model():
    for text in ("9e9 x 585e-6 x 245e-6 / 590^2 N = 9 x 585 x 245 / 590^2 mN. ",   # a cut-short copy
                 "813 x 245 = 813 x 200 + 813 x 40 + 813 x 4 = ",                  # a wrong split
                 "bond order = 2393 / 2 = 1196.4, ",                               # decimals
                 "6 - 906 = -(906 - 6), "):
        assert fix(text) == text


class SlippingModel(torch.nn.Module):
    """Writes the right solution byte by byte, except that it gets the last digit of every result the
    calculator handles wrong. It reads what was generated so far, so it follows the calculator's fixes."""

    def __init__(self, prompt, solution):
        super().__init__()
        self.cfg = SimpleNamespace(block_size=4096)
        self.w = torch.nn.Parameter(torch.zeros(1))
        self.prompt, self.text = len(prompt), solution.encode()
        self.slips = set()
        for i in range(len(solution)):
            m = STEP.search(solution[:i + 1]) or DIVISION.search(solution[:i + 1])
            if m and m.re is DIVISION:
                self.slips.add(m.end(5) - 1)                       # the product in "a - p"
            elif m:
                first = m.group(2).split(" ")[0]
                if first == m.group(1).split(" ")[0]:              # a subtraction split: its last part
                    self.slips.add(m.end(2) - 1)
                else:                                              # otherwise the first number
                    self.slips.add(m.start(2) + len(first) - 1)

    def new_cache(self):
        return {"seq": []}

    def forward(self, inp, last_only=True, caches=None):
        caches["seq"] += inp[0].tolist()
        done = len(caches["seq"]) - self.prompt
        logits = torch.zeros(1, 1, 300)
        if done >= len(self.text):
            nxt = EOS
        else:
            nxt = self.text[done]
            if done in self.slips:
                nxt = ord(str((int(chr(nxt)) + 1) % 10))
        logits[0, 0, nxt] = 1
        return SimpleNamespace(logits=logits)


def test_generation_with_the_calculator_recovers_the_right_solution():
    rng = np.random.default_rng(1)
    for name, make in [*SYNTH_KINDS.items(), *OUT_OF_RANGE.items()]:
        problem, solution = make(rng)
        prompt = list(problem.encode())
        model = SlippingModel(prompt, solution)
        out, fixes = generate(model, prompt, 3000, EOS, calculator=False)
        slipped = bytes(out).decode()
        assert fixes == 0 and (slipped != solution) == bool(model.slips), name
        out, fixes = generate(model, prompt, 3000, EOS)
        assert bytes(out).decode() == solution and fixes == len(model.slips), name
