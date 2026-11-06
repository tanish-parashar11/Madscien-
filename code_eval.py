"""
Evaluate coding ability with EXECUTION, not by eye.

  python code_eval.py --selftest                              # harness check, no model needed
  python code_eval.py --ckpt out/ckpt.pt --n_tasks 100 --n_samples 8

Metrics (per task: n sampled solutions, 2 visible tests, 6 hidden tests):
  syntax_valid   fraction of samples that parse
  pass@k         unbiased estimator, a sample passes if it solves ALL visible + hidden tests
  verified_pick  sample-and-verify: take the first sample that passes the VISIBLE tests, score it on
                 the hidden tests. This is how accuracy is raised at inference time: generate several
                 candidates, keep one that survives the tests.
"""
import argparse

import numpy as np

from madscien.codegen_data import generate_tasks, heldout_tasks
from madscien.verify import END_MARK, extract_solution, is_valid_python, pass_at_k, run_tests


def evaluate(generate_fn, tasks, n_samples=8, ks=(1, 4), timeout=5.0, verbose=True):
    """generate_fn(prompt:str, n:int) -> list of raw continuations (text after '### Solution\\n')."""
    n_syntax = n_total = 0
    counts, picked_ok, picked_any = [], 0, 0
    for t in tasks:
        outs = generate_fn(t.prompt(), n_samples)
        c, first_visible = 0, None
        for raw in outs:
            code = extract_solution(raw)
            n_total += 1
            if not is_valid_python(code):
                continue
            n_syntax += 1
            full = run_tests(code, t.entry, t.cases, timeout)
            if full.passed:
                c += 1
            if first_visible is None and run_tests(code, t.entry, t.shown, timeout).passed:
                first_visible = full.passed
        counts.append(c)
        if first_visible is not None:
            picked_any += 1
            picked_ok += int(first_visible)
    nt = len(tasks)
    res = {"syntax_valid": n_syntax / max(1, n_total),
           "verified_pick": picked_ok / nt,
           "tasks_with_visible_pass": picked_any / nt}
    for k in ks:
        if k > n_samples:                                # pass@k is undefined with fewer than k samples
            print(f"(skipping pass@{k}: needs at least {k} samples, got {n_samples})")
            continue
        res[f"pass@{k}"] = float(np.mean([pass_at_k(n_samples, c, k) for c in counts]))
    if verbose:
        print(" | ".join(f"{k} {v:.3f}" for k, v in res.items()))
    return res


# ------------------------------------------------------------------ mock generator (self-test)
def corrupt(code, rng):
    muts = [("+", "-"), ("-", "+"), ("*", "/"), ("/", "*"), ("<=", "<"), ("== 0", "== 1")]
    for i in rng.permutation(len(muts)):
        a, b = muts[i]
        if a in code.split("\n", 1)[-1]:                     # do not touch the def line
            head, body = code.split("\n", 1)
            return head + "\n" + body.replace(a, b, 1)
    if rng.random() < 0.5:
        return code.rstrip().rstrip(")") + "\n"              # unbalanced paren -> syntax error
    return code.replace("return ", "return 1 + ", 1)


def make_mock_generate(tasks, p_correct, seed=0):
    rng = np.random.default_rng(seed)
    by_prompt = {t.prompt(): t for t in tasks}

    def gen(prompt, n):
        t = by_prompt[prompt]
        outs = []
        for _ in range(n):
            code = t.solution if rng.random() < p_correct else corrupt(t.solution, rng)
            outs.append(code + END_MARK + "\n")
        return outs
    return gen


# ------------------------------------------------------------------ real model
def make_model_generate(ckpt, device=None, temperature=0.8, top_k=40, max_new_tokens=400,
                        use_constraint=True):
    import torch
    from madscien.config import X5Config
    from madscien.constraints import DelimiterConstraint
    from madscien.data import ByteTokenizer
    from madscien.model import MadScienLM

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt, map_location=device)
    model = MadScienLM(X5Config(**ck["cfg"])).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    tok = ByteTokenizer()

    def gen(prompt, n):
        outs = []
        for _ in range(n):
            ids = [tok.bos_id] + tok.encode(prompt)
            idx = torch.tensor([ids], dtype=torch.long, device=device)
            c = DelimiterConstraint() if use_constraint else None      # tracks the solution only
            out = model.generate(idx, max_new_tokens, temperature=temperature, top_k=top_k,
                                 eos_id=tok.eos_id, constraint=c)
            text = tok.decode(out[0, len(ids):].tolist())
            if c is not None and END_MARK not in text:
                text += c.repair()                                     # close what is still open
            outs.append(text)
        return outs
    return gen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n_tasks", type=int, default=100)
    ap.add_argument("--n_samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--no_constraint", action="store_true")
    ap.add_argument("--seed", type=int, default=12345, help="eval tasks use fresh random draws")
    ap.add_argument("--train_n_docs", type=int, default=20000,
                    help="--n_docs used to build the training set; its prompts are excluded from eval")
    ap.add_argument("--train_seed", type=int, default=0, help="--seed used to build the training set")
    a = ap.parse_args()

    if a.selftest:                                      # harness check only: no model, no contamination
        tasks = list(generate_tasks(40, a.seed))
        for p in (1.0, 0.5, 0.0):
            print(f"mock model, P(correct sample) = {p}:")
            evaluate(make_mock_generate(tasks, p), tasks, n_samples=8)
        return
    assert a.ckpt, "--ckpt is required (or use --selftest)"
    tasks = heldout_tasks(a.n_tasks, a.seed, a.train_n_docs, a.train_seed)
    gen = make_model_generate(a.ckpt, temperature=a.temperature, use_constraint=not a.no_constraint)
    evaluate(gen, tasks, n_samples=a.n_samples)


if __name__ == "__main__":
    main()
