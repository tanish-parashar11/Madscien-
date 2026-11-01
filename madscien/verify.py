"""
Verification tools for generated code (stdlib only).

 * is_valid_python   : syntax check with ast
 * run_tests         : run a candidate against test cases in a separate process (timeout + rlimits)
 * pass_at_k         : unbiased pass@k estimator (Chen et al. 2021)
 * extract_solution  : cut the raw model continuation at the "### End" marker

NOTE: the subprocess has a timeout and CPU/memory limits, but this is NOT a security sandbox.
For untrusted code run it inside a container / VM.
"""
import ast
import json
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Optional

END_MARK = "### End"

_HARNESS = r'''
import json, math, sys
payload = json.load(open(sys.argv[1]))

def close(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(close(x, y) for x, y in zip(a, b))
    return a == b

def emit(**kw):
    sys.stdout.write("\n@@RESULT@@" + json.dumps(kw) + "\n")

total = len(payload["cases"])
ns = {"__name__": "candidate"}
try:
    exec(compile(payload["code"], "<candidate>", "exec"), ns)
    fn = ns[payload["entry"]]
except BaseException as e:
    emit(ok=0, total=total, error="load: %s: %s" % (type(e).__name__, str(e)[:200]))
    sys.exit(0)
ok = 0
for args, expected in payload["cases"]:
    try:
        if close(fn(*args), expected):
            ok += 1
    except BaseException:
        pass
emit(ok=ok, total=total, error=None)
'''


@dataclass
class TestResult:
    ok: int
    total: int
    error: Optional[str] = None

    @property
    def passed(self):
        return self.total > 0 and self.ok == self.total and self.error is None


def is_valid_python(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return False


def extract_solution(generated: str) -> str:
    return generated.split(END_MARK)[0].rstrip() + "\n"


def _limits(cpu_s, mem_mb):
    def apply():
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
        resource.setrlimit(resource.RLIMIT_AS, (mem_mb * 1024 * 1024, mem_mb * 1024 * 1024))
    return apply


def run_tests(code, entry, cases, timeout=5.0, mem_mb=1024) -> TestResult:
    """cases: [[args_list, expected], ...] (JSON-serialisable)."""
    total = len(cases)
    with tempfile.TemporaryDirectory() as td:
        pj, hp = os.path.join(td, "payload.json"), os.path.join(td, "harness.py")
        with open(pj, "w") as f:
            json.dump({"code": code, "entry": entry, "cases": cases}, f)
        with open(hp, "w") as f:
            f.write(_HARNESS)
        kw = {}
        if os.name == "posix":
            kw["preexec_fn"] = _limits(int(math.ceil(timeout)) + 1, mem_mb)
        try:
            p = subprocess.run([sys.executable, "-I", hp, pj], cwd=td, capture_output=True,
                               text=True, timeout=timeout,
                               env={"PATH": os.environ.get("PATH", "")}, **kw)
        except subprocess.TimeoutExpired:
            return TestResult(0, total, "timeout")
        lines = [l for l in p.stdout.splitlines() if l.startswith("@@RESULT@@")]
        if not lines:
            return TestResult(0, total, f"crash (exit {p.returncode})")
        r = json.loads(lines[-1][len("@@RESULT@@"):])
        return TestResult(r["ok"], r["total"], r["error"])


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator of P(at least one of k samples passes) given c of n passed."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)
