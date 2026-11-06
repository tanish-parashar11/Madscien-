"""
Verified synthetic CODE data.

Every task = description + visible tests + a reference solution TEXT. Before a task is emitted:
  * the solution text is exec'd and compared with an independent oracle on 8 random cases;
    a mismatch raises (so every emitted training document is correct w.r.t. its tests);
  * cases = 2 visible tests (shown in the prompt) + 6 hidden tests (used by code_eval.py).

Document layout (what the LM trains on):
    ### Task
    <description>
    ### Tests
    assert f(...) == ...
    ### Solution
    <code>
    ### End

    python -m madscien.codegen_data --n_docs 20000 --out data/code
"""
import argparse
import math
from dataclasses import dataclass

import numpy as np

N_CASES, N_SHOWN = 8, 2


@dataclass
class CodeTask:
    entry: str
    description: str
    solution: str
    cases: list                      # [[args], expected]

    @property
    def shown(self):
        return self.cases[:N_SHOWN]

    def prompt(self):
        tests = "\n".join(f"assert {self.entry}({', '.join(map(repr, a))}) == {e!r}"
                          for a, e in self.shown)
        return f"### Task\n{self.description}\n### Tests\n{tests}\n### Solution\n"

    def document(self):
        return self.prompt() + self.solution.rstrip() + "\n### End\n"


def _close(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b))
    return a == b


def _build(entry, description, solution, oracle, gen_args, rng):
    cases = []
    for _ in range(N_CASES):
        args = gen_args(rng)
        cases.append([args, oracle(*args)])
    ns = {}                                              # no free imports: solution must be self-contained
    exec(solution, ns)                                   # our own templates (trusted)
    fn = ns[entry]
    for args, expected in cases:                         # the verification gate
        got = fn(*args)
        if not _close(got, expected):
            raise AssertionError(f"template bug in {entry}: {args} -> {got!r}, oracle {expected!r}")
    return CodeTask(entry, description, solution, cases)


def _pick(rng, xs):
    return xs[int(rng.integers(0, len(xs)))]


def _fl(rng, lo, hi, nd=2):
    return round(float(rng.uniform(lo, hi)), nd)


# ---------------------------------------------------------------- task families
_PRIMES = [n for n in range(2, 120) if all(n % d for d in range(2, int(n ** 0.5) + 1))]


def t_coulomb(rng):
    entry = _pick(rng, ["coulomb_force", "electric_force", "force_between"])
    k = _pick(rng, [9e9, 1.0, 8.99e9])
    body = _pick(rng, [f"abs(q1 * q2) * {k!r} / (r * r)", f"{k!r} * abs(q1 * q2) / r ** 2"])
    return _build(entry,
                  f"Write `{entry}(q1, q2, r)` returning the magnitude of the electrostatic force between "
                  f"charges q1 and q2 at distance r, using k = {k!r}.",
                  f"def {entry}(q1, q2, r):\n    return {body}\n",
                  lambda q1, q2, r: k * abs(q1 * q2) / r ** 2,
                  lambda g: [_fl(g, -9, 9, 1), _fl(g, 1, 9, 1), _fl(g, 0.5, 5, 1)], rng)


def t_gauss(rng):
    entry = _pick(rng, ["gauss_flux", "enclosed_flux", "flux_through_surface"])
    eps0 = _pick(rng, [8.854e-12, 1.0, 8.85e-12])
    return _build(entry,
                  f"Write `{entry}(charges)` returning the electric flux through a closed surface that "
                  f"encloses the given list of charges (Gauss law), with epsilon naught = {eps0!r}.",
                  f"def {entry}(charges):\n    return sum(charges) / {eps0!r}\n",
                  lambda ch: sum(ch) / eps0,
                  lambda g: [[_fl(g, -5, 5, 1) for _ in range(int(g.integers(1, 6)))]], rng)


def t_emf(rng):
    entry = _pick(rng, ["induced_emf", "faraday_emf", "coil_emf"])
    body = _pick(rng, ["-n * dphi / dt", "-(n * dphi) / dt"])
    return _build(entry,
                  f"Write `{entry}(n, dphi, dt)` returning the induced emf of a coil with n turns whose "
                  f"flux changes by dphi in time dt (Faraday + Lenz).",
                  f"def {entry}(n, dphi, dt):\n    return {body}\n",
                  lambda n, dphi, dt: -n * dphi / dt,
                  lambda g: [int(g.integers(1, 100)), _fl(g, -2, 2), _fl(g, 0.5, 5, 1)], rng)


def t_angular(rng):
    entry = _pick(rng, ["new_spin_rate", "angular_speed_after", "spin_after_change"])
    return _build(entry,
                  f"Write `{entry}(i1, w1, i2)` returning the new angular speed when the moment of inertia "
                  f"changes from i1 to i2 with no external torque (angular momentum is conserved).",
                  f"def {entry}(i1, w1, i2):\n    return i1 * w1 / i2\n",
                  lambda i1, w1, i2: i1 * w1 / i2,
                  lambda g: [_fl(g, 1, 9, 1), _fl(g, 1, 20, 1), _fl(g, 1, 9, 1)], rng)


def t_bond_order(rng):
    entry = _pick(rng, ["bond_order", "mot_bond_order", "molecular_bond_order"])
    return _build(entry,
                  f"Write `{entry}(bonding, antibonding)` returning the bond order from molecular orbital "
                  f"theory, given the numbers of bonding and antibonding electrons.",
                  f"def {entry}(bonding, antibonding):\n    return (bonding - antibonding) / 2\n",
                  lambda b, a: (b - a) / 2,
                  lambda g: [int(g.integers(0, 15)), int(g.integers(0, 12))], rng)


def t_goc(rng):
    entry = _pick(rng, ["most_stable_carbocation", "goc_most_stable", "stablest_cation"])
    names = ["primary", "secondary", "tertiary"]
    rank = {"primary": 1, "secondary": 2, "tertiary": 3}
    sol = (f"def {entry}(types):\n    rank = {{'primary': 1, 'secondary': 2, 'tertiary': 3}}\n"
           f"    return max(types, key=lambda t: rank[t])\n")

    def gen(g):
        k = int(g.integers(2, 4))
        return [[names[i] for i in g.permutation(3)[:k]]]
    return _build(entry,
                  f"Write `{entry}(types)` returning the most stable carbocation from a list of "
                  f"'primary', 'secondary', 'tertiary' (stability 3 > 2 > 1).",
                  sol, lambda ts: sorted(ts, key=lambda t: rank[t])[-1], gen, rng)


def t_king(rng):
    entry = _pick(rng, ["king_integral", "symmetric_integral", "king_property_value"])
    body = _pick(rng, ["(b - a) / 2", "0.5 * (b - a)"])
    return _build(entry,
                  f"Write `{entry}(a, b)` returning the integral from a to b of f(x) / (f(x) + f(a + b - x)) dx "
                  f"for any positive f (use the King property).",
                  f"def {entry}(a, b):\n    return {body}\n",
                  lambda a, b: (b - a) / 2,
                  lambda g: (lambda a: [a, round(a + float(g.uniform(0.5, 6)), 2)])(_fl(g, -3, 3, 1)), rng)


def t_ncr(rng):
    entry = _pick(rng, ["ncr", "choose", "combinations_count"])
    sol = _pick(rng, [
        f"import math\n\n\ndef {entry}(n, r):\n    return math.comb(n, r)\n",
        f"def {entry}(n, r):\n    r = min(r, n - r)\n    out = 1\n    for i in range(1, r + 1):\n"
        f"        out = out * (n - r + i) // i\n    return out\n",
        f"def {entry}(n, r):\n    row = [1]\n    for _ in range(n):\n"
        f"        row = [1] + [row[i] + row[i + 1] for i in range(len(row) - 1)] + [1]\n    return row[r]\n",
    ])
    return _build(entry, f"Write `{entry}(n, r)` returning the number of ways to choose r items from n "
                         f"(combinations).", sol, lambda n, r: math.comb(n, r),
                  lambda g: (lambda n: [n, int(g.integers(0, n + 1))])(int(g.integers(1, 21))), rng)


def t_gcd(rng):
    entry = _pick(rng, ["gcd_of", "greatest_common_divisor", "euclid_gcd"])
    sol = _pick(rng, [
        f"def {entry}(a, b):\n    while b:\n        a, b = b, a % b\n    return a\n",
        f"def {entry}(a, b):\n    return a if b == 0 else {entry}(b, a % b)\n",
    ])
    return _build(entry, f"Write `{entry}(a, b)` returning the greatest common divisor of two positive "
                         f"integers.", sol, math.gcd,
                  lambda g: [int(g.integers(1, 500)), int(g.integers(1, 500))], rng)


def t_powmod(rng):
    entry = _pick(rng, ["power_mod", "modexp", "mod_pow"])
    sol = _pick(rng, [
        f"def {entry}(b, e, m):\n    return pow(b, e, m)\n",
        f"def {entry}(b, e, m):\n    out, b = 1, b % m\n    while e:\n        if e & 1:\n"
        f"            out = out * b % m\n        b = b * b % m\n        e >>= 1\n    return out % m\n",
    ])
    return _build(entry, f"Write `{entry}(b, e, m)` returning b to the power e modulo m.", sol,
                  lambda b, e, m: pow(b, e, m),
                  lambda g: [int(g.integers(0, 50)), int(g.integers(0, 60)), int(g.integers(2, 100))], rng)


def t_prime(rng):
    entry = _pick(rng, ["is_prime", "check_prime", "prime_test"])
    sol = (f"def {entry}(n):\n    if n < 2:\n        return False\n    i = 2\n    while i * i <= n:\n"
           f"        if n % i == 0:\n            return False\n        i += 1\n    return True\n")
    return _build(entry, f"Write `{entry}(n)` returning True if the integer n is prime, else False.", sol,
                  lambda n: n > 1 and sum(1 for d in range(1, n + 1) if n % d == 0) == 2,
                  lambda g: [_pick(g, _PRIMES) if g.random() < 0.5 else int(g.integers(0, 120))], rng)


def t_trapezoid(rng):
    entry = _pick(rng, ["trapezoid", "trapz_area", "integrate_samples"])
    sol = _pick(rng, [
        f"def {entry}(y, h):\n    return h * (sum(y) - 0.5 * (y[0] + y[-1]))\n",
        f"def {entry}(y, h):\n    s = 0.0\n    for i in range(len(y) - 1):\n"
        f"        s += 0.5 * (y[i] + y[i + 1]) * h\n    return s\n",
    ])

    def oracle(y, h):
        return sum(0.5 * (y[i] + y[i + 1]) * h for i in range(len(y) - 1))
    return _build(entry, f"Write `{entry}(y, h)` returning the composite trapezoid-rule integral of equally "
                         f"spaced samples y with spacing h.", sol, oracle,
                  lambda g: [[_fl(g, -5, 5) for _ in range(int(g.integers(3, 9)))],
                             _pick(g, [0.5, 1.0, 0.25, 0.1])], rng)


FAMILIES = [t_coulomb, t_gauss, t_emf, t_angular, t_bond_order, t_goc, t_king, t_ncr, t_gcd,
            t_powmod, t_prime, t_trapezoid]


def generate_tasks(n, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        yield FAMILIES[int(rng.integers(0, len(FAMILIES)))](rng)


def heldout_tasks(n, seed, train_n_docs=0, train_seed=0, max_draws=None):
    """n eval tasks whose prompts are unique and do NOT occur in the training set that
    generate_tasks(train_n_docs, train_seed) produced (some families have only a few hundred
    distinct prompts, so plain fresh draws would repeat training documents verbatim)."""
    seen = {t.prompt() for t in generate_tasks(train_n_docs, train_seed)}
    out, draws = [], 0
    max_draws = max_draws or 50 * n
    rng = np.random.default_rng(seed)
    while len(out) < n and draws < max_draws:
        t = FAMILIES[int(rng.integers(0, len(FAMILIES)))](rng)
        draws += 1
        if t.prompt() not in seen:
            seen.add(t.prompt())
            out.append(t)
    if len(out) < n:
        raise RuntimeError(f"only found {len(out)} held-out tasks in {max_draws} draws")
    return out


def code_documents(n, seed=0):
    for t in generate_tasks(n, seed):
        yield t.document()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_docs", type=int, default=20000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sandbox_check", type=int, default=30,
                    help="re-verify this many tasks end-to-end through the subprocess runner")
    a = ap.parse_args()

    from madscien.data import build_bins
    from madscien.verify import run_tests

    tasks = list(generate_tasks(a.n_docs, a.seed))
    bad = sum(not run_tests(t.solution, t.entry, t.cases).passed for t in tasks[:a.sandbox_check])
    print(f"sandbox re-check: {a.sandbox_check - bad}/{a.sandbox_check} passed")
    n_tr, n_va = build_bins((t.document() for t in tasks), a.out)
    print(f"wrote {a.out}: train {n_tr} tokens, val {n_va} tokens ({a.n_docs} verified docs)")
