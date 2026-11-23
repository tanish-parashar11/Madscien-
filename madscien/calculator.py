"""
A calculator for greedy generation of the synthetic worked solutions (madscien/data.py).

The model still decides what to write: which numbers to copy, how to split them, which step comes next.
Each time it finishes a result, the calculator checks it against the expression it was computed from,
and replaces it with the exact value if the model slipped. Generation then carries on from the fixed text.

What it fixes, after "E = " once the right side R is complete (R ends in ", ", " = ", ". ", " (" or ".\\n"):
  a single number      9829 x 7 = 68703    -> 68803      (also a / b, as whole-number division)
  partial products     134 x 700 + 134 x 6 = 93800 + 704   -> 93800 + 804
  one addition         1999400 + 799760 + 89973 = 2789160 + 89973   -> 2799160 + 89973
  a subtraction split  5062 - 3364 = 5062 - 3000 - 264   -> 5062 - 3000 - 364   (and digit by digit)
and in long division, the product in "a / b = q, a - p = " is set to q x b.

It never changes what the model copied from the problem, a multiplication split (813 x 200 + ...), or
decimals: copying stays the model's job, so scores with the calculator show how well it sets problems up.
"""
import re

TERM = r"\d+(?: x \d+)*"
EXPR = rf"\d+ / \d+|{TERM}(?: [+-] {TERM})+|\d+(?: x \d+)+"          # at least one operator
RESULT = r"(?:\d+ [+-] )*\d+"
STEP = re.compile(rf"(?<![\d.])({EXPR}) = ({RESULT})(, | = |\. | \(|\.\n)$")
DIVISION = re.compile(r"(?<![\d.])(\d+) / (\d+) = (\d+), (\d+) - (\d+) = $")
TAIL = 800                       # longer than any expression and result in the training format


def _split(expr):
    """'a x b + c - d' -> terms ['a x b', 'c', 'd'] and operators ['+', '-']."""
    parts = re.split(r" ([+-]) ", expr)
    return parts[::2], parts[1::2]


def _term(term):
    value = 1
    for factor in term.split(" x "):
        value *= int(factor)
    return value


def value(expr):
    if " / " in expr:
        a, b = map(int, expr.split(" / "))
        return a // b
    terms, ops = _split(expr)
    total = _term(terms[0])
    for op, term in zip(ops, terms[1:]):
        total += _term(term) if op == "+" else -_term(term)
    return total


def _lead(n):
    return int(str(n)[0]) * 10 ** (len(str(n)) - 1)


def _digit_parts(n):
    s = str(n)
    return [int(d) * 10 ** (len(s) - 1 - i) for i, d in enumerate(s) if d != "0"]


def correct(expr, result):
    """The result the training format writes after 'expr = ', given the shape the model chose for it."""
    if " / " in expr:
        return str(value(expr)) if result.isdigit() and int(expr.split(" / ")[1]) else result
    if result.isdigit():
        return str(value(expr)) if value(expr) >= 0 else result
    e_terms, e_ops = _split(expr)
    r_terms, r_ops = _split(result)
    if len(r_terms) == len(e_terms) and r_ops == e_ops and any(" x " in t for t in e_terms):
        fixed = [str(_term(t)) for t in e_terms]                                   # partial products
    elif (len(r_terms) == len(e_terms) - 1 and r_ops == e_ops[1:] and r_terms[1:] == e_terms[2:]
          and " x " not in e_terms[0] + e_terms[1]):
        fixed = [str(value(f"{e_terms[0]} {e_ops[0]} {e_terms[1]}"))] + r_terms[1:]  # one addition
    elif (len(e_terms) == 2 and e_ops == ["-"] and r_ops == ["-"] * len(r_ops) and r_terms[0] == e_terms[0]
          and " x " not in expr and len(r_terms) >= 3):                             # subtraction split
        b = int(e_terms[1])
        parts = [_lead(b), b - _lead(b)] if len(r_terms) == 3 else _digit_parts(b)
        if len(parts) != len(r_terms) - 1 or 0 in parts:
            return result
        fixed = [r_terms[0]] + [str(p) for p in parts]
    else:
        return result
    ops = r_ops + [""]
    return " ".join(f"{t} {o}" if o else t for t, o in zip(fixed, ops))


def fix(text):
    """text as generated so far. Returns it with its last finished result corrected (or unchanged)."""
    head, tail = text[:-TAIL], text[-TAIL:]
    m = STEP.search(tail)
    if m:
        new = correct(m.group(1), m.group(2))
        if new != m.group(2):
            tail = tail[:m.start(2)] + new + m.group(3)
    m = DIVISION.search(tail)
    if m and m.group(1) == m.group(4):
        p = str(int(m.group(2)) * int(m.group(3)))
        if p != m.group(5):
            tail = tail[:m.start(5)] + p + " = "
    return head + tail


def generate(model, ids, max_new_tokens, eos_id, calculator=True):
    """Greedy decoding of one sequence, with the calculator. Returns (new token ids, corrections made).
    After a correction the cache is rebuilt from the fixed text, so the model sees what it would have
    seen had it written the right value itself."""
    import torch
    device = next(model.parameters()).device
    bs = model.cfg.block_size
    reserve = min(64, max(1, bs // 4))
    ids, start = list(ids), len(ids)
    caches, cache_len, fixes = None, 0, 0
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            if caches is None or cache_len >= bs:                  # (re)prefill from the latest tokens
                caches = model.new_cache()
                inp = ids[-(bs - reserve):]
                cache_len = len(inp)
            else:
                inp = ids[-1:]
                cache_len += 1
            logits = model(torch.tensor([inp], device=device), last_only=True, caches=caches).logits
            nxt = int(logits[0, -1].argmax())
            if nxt == eos_id:
                break
            ids.append(nxt)
            if calculator and nxt in (32, 40, 10):                 # every result ends in " ", "(" or "\n"
                text = bytes(t for t in ids[start:] if t < 256).decode("utf-8", "replace")
                fixed = fix(text)
                if fixed != text:
                    ids = ids[:start] + list(fixed.encode("utf-8"))
                    caches, fixes = None, fixes + 1
    model.train(was_training)
    return ids[start:], fixes
