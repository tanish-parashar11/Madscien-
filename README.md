# MADSCIEN-X5 LM (PyTorch)

Decoder-only language model whose token mixer is your 7-step algorithm.
Each layer runs the full pipeline on every token's causal window ("problem") and adds the
result back into the residual stream.

## Quickstart

```bash
pip install -r requirements.txt
pytest -q                                   # RUN THIS FIRST (see "Verification")

python -m madscien.data --source synth --n_docs 50000 --out data/synth      # or your own text
python train.py --data_dir data/synth --preset tiny --max_iters 2000
python sample.py --ckpt out/ckpt.pt --prompt "Problem: Two charges 3 uC and 4 uC"
```

Your own synthetic data: put it in a text file, documents separated by blank lines, then
`python -m madscien.data --source text --path my.txt --out data/mine`.

Checkpoints: `out/ckpt.pt` = best validation loss (used by `sample.py`), `out/last.pt` = most recent
eval (used by `--resume`). The final iteration is always evaluated and saved.

Score a checkpoint on the synthetic problems, by exact full solution (uses the GPU if there is one):
```bash
python eval_synth.py --ckpt out/ckpt.pt --data_dir data/synth
```
It reports problems seen in training, problems not seen, and numbers outside the training ranges.
Only the last two test more than recall: a low loss can still mean 0% out of range.

The synthetic solutions work the arithmetic out step by step, and about 10% of the numeric problems are
held out of training to form the validation set, so `best val` measures problems the model never saw.
Datasets and checkpoints made before this format must be rebuilt and retrained. Answers that do not end
exactly are cut off at 3 decimals (2 for Coulomb, to fit 512 tokens) and marked `...`, not rounded.
Subtractions are written out too, including the one inside every long-division remainder.
Many solutions are longer than the `small` preset's 256-token context (all fit in 512), so train them with
whole problems in view:
```bash
python train.py --data_dir data/synth --preset small --block_size 512 --batch_size 6 --align_docs --max_iters 20000
```
`--align_docs` starts every training window at the beginning of a problem. Without it, most windows that
reach the later steps of a long solution have lost the problem's numbers, so the model learns to guess
them instead of copying them. On an 8 GB GPU, `--block_size 512 --batch_size 6` uses about 6.5 GB.

## Your algorithm -> code

| Step | Your pseudo-code | Where | What it does |
|---|---|---|---|
| 1 | `encode_to_charges`, `apply_gauss_law`, `F = k q1 q2 / r^2` | `x5_core.py` Step 1 | token -> signed charge (`q_proj`). Coulomb flux `|q_i| + k|q_i| sum_j |q_j|/r_ij^2`. Gauss gate keeps the `keep_k` strongest, the rest is earthed. |
| 2 | `calculate_angular_momentum`, `if dL_dt != 0: correct_with_angular_impulse` | Step 2 | Cayley rotation (exactly orthogonal) + torque. `L = I*omega`, `I = sum m r^2`. Angular impulse rescales so L is restored. |
| 3 | `faraday_induction`, `e = -dPhi/dt` | Step 3 | flux through `flux_loops` loops, EMF = minus the difference along the sequence. |
| 4 | `generate_all_cases`, `bayes_probability`, `king_property_integral`, complex/locus | Step 4 | every subset of kept tokens (size 1..`max_subset`) is a case. Phasor `z = emf + i*flux`, modulus and phase. Bayes posterior over cases. King property `f(x) -> f(a+b-x)` with a cumulative trapezoid integral. |
| 5 | `bond_order = (bonding - antibonding)/2`, `goc_stability_check` 3 > 2 > 1 | Step 5 | Hueckel orbitals `alpha -/+ beta`, Aufbau fill (lowest energy first, 2 per orbital), bond order. Stability = number of similar "alkyl" neighbours (3 deg 1.0, 2 deg 0.66, 1 deg 0.33). Keep if bond order > 0 and stable. |
| 6 | `singularity_compress` | Step 6 | weighted average of survivors -> one vector (then `out_proj` back to the residual stream). |
| 7 | `self_evolve(loss = physics_loss(...))` | `evolve.py`, `model.physics_loss` | `physics_loss = CE + lam_cons*(dL/L) + lam_fallback*(no-survivor rate)`. Every update is a self-evolve step; LR and `lam_cons` self-adapt within hard bounds. |

Differentiability: forward values are the exact hard algorithm (top-k, floor, Aufbau, degree
thresholds). Backward uses straight-through estimators (sigmoid) for each discrete decision.

## What is NOT in your algorithm but needed for an LLM
Token embedding, RMSNorm, residual stream, `out_proj`, LM head, causal windows, optional SwiGLU
(`ffn_mult`, default 0 = pure X5). Heads = parallel X5 cores with shared core weights.

## Presets
`tiny` (fast CPU test), `small` (`ffn_mult=4`, 8 layers), `base` (12 layers, ~90M params with the
byte vocab, gradient checkpointing on). Memory scales with `batch * seq * n_heads * cases^2`;
lower `keep_k` / `max_subset` / batch size if you run out (cases = 41 at k=6, subsets<=3).


## Hybrid global attention (like a standard transformer)
Every `attn_every`-th layer (default 2) is a standard pre-norm transformer layer: causal
grouped-query attention (GQA) with RoPE and a SwiGLU FFN, using PyTorch SDPA (flash kernels on GPU).
X5 layers keep doing the local physics/chemistry/maths mixing, attention layers add global recall,
so the stack is X5 -> attention -> X5 -> attention ...

```bash
python train.py --data_dir data/code --preset small                 # hybrid (default)
python train.py --data_dir data/code --preset small --attn_every 3  # fewer attention layers
python train.py --data_dir data/code --preset small --attn_every 0  # pure X5, local only
```
Attention is a standard component, so how much it helps vs. pure X5 is something to measure with an
ablation on your data (`--attn_every 0` vs `2`), not assume. Capability like a frontier model comes
from scale (parameters, tokens, compute) and post-training, which no architecture change provides.

## Chat behaviour (ChatGPT-style conversation)
Chat format: `<BOS><|system|>..<|end|><|user|>..<|end|><|assistant|>..<|end|>` (`madscien/chat.py`).
SFT trains **only on assistant text and its `<|end|>`** (loss mask), so the model learns what to say
and when to stop. The default persona (a one-line system prompt, short so it fits the small contexts):
friendly, honest, concise, mirror the user's language (English / Hinglish). The habits themselves
(answer first, one clarifying question when ambiguous, admit uncertainty) are taught by the SFT data.
If a prompt does not fit, `chat_cli.py` drops old turns first, then shortens the system prompt, and only
then trims the user's message.

Recipe (pretrain first, then SFT):
```bash
python -m madscien.data --source text --path general_text.txt --out data/pre      # your large text
python train.py --data_dir data/pre --preset small --max_iters 20000
python -m madscien.chat_data --source synth --n_convs 20000 --out data/chat       # + your own JSONL
python train.py --data_dir data/chat --sft --init_from out/ckpt.pt --lr 3e-4 --out_dir out_chat
python chat_cli.py --ckpt out_chat/ckpt.pt        # multi-turn chat, /reset /system /temp
```
Your own conversations: `--source jsonl --path chats.jsonl` with lines
`{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}`.

Honest scope: the built-in synthetic set teaches conversation *habits* (friendly tone, clarifying
questions, honest uncertainty, Hinglish mirroring, follow-ups, verified code) using templates.
ChatGPT-level range of human-like behaviour comes from large, diverse, human-quality conversation data,
a large pretrained model, and preference tuning (RLHF/DPO). This repo gives you the full pipeline for
the first two; preference tuning is not included yet.

Attention upgrades: QK-norm (stable logits), GQA + RoPE, and incremental decoding: attention layers keep
a KV cache and X5 layers keep a window cache, so generation costs O(1) per token instead of
re-running the whole context.

## Coding capability (verified, not guaranteed)

No model can be "100% accurate" at coding (deciding correctness of arbitrary programs is
undecidable - Rice's theorem). What this repo gives you instead is a pipeline where code is
**checked by execution**, so accuracy is measured and pushed up honestly.

| Piece | File | What it does |
|---|---|---|
| Verified training data | `madscien/codegen_data.py` | 12 task families (Coulomb, Gauss, EMI, angular momentum, MOT bond order, GOC, King property, nCr, gcd, modpow, primes, trapezoid). Every solution text is checked against an independent oracle before it is written; 2 visible + 6 hidden tests per task. |
| Sandboxed runner | `madscien/verify.py` | syntax check (`ast`), run tests in a subprocess with timeout + CPU/memory limits, unbiased pass@k. Not a security sandbox: use a container for untrusted code. |
| Constrained decoding | `madscien/constraints.py`, `generate(..., constraint=)` | the model can never emit a mismatched bracket or end inside an open string/bracket. |
| Evaluation | `code_eval.py` | `syntax_valid`, `pass@k`, `verified_pick` (sample N, keep the first that passes the visible tests, score on hidden tests). |

```bash
python -m madscien.codegen_data --n_docs 20000 --out data/code
python train.py --data_dir data/code --preset small --max_iters 5000
python code_eval.py --ckpt out/ckpt.pt --n_tasks 100 --n_samples 8
python code_eval.py --selftest          # harness check, no model needed
```
Eval tasks are held out: any prompt that occurs in the training set is skipped. If you built the data
with a different `--n_docs`/`--seed`, pass the same values as `--train_n_docs`/`--train_seed`.

Honest limits: the model only sees these task families, so it learns to solve *these kinds* of
problems (and variants), not arbitrary programming. General coding needs a large model trained on
a large real-code corpus; you can add your own verified tasks as new functions in `FAMILIES`.
Passing tests is not proof of correctness: weak tests let wrong code through (the generator makes
sure e.g. prime tests contain primes). Tests: `pytest -q tests/test_code.py tests/test_chat.py` (no torch needed).

## Verification
`pytest -q` checks:
- torch core == the numpy reference in `reference/` (the exact-copy test)
- rotation orthogonal and L-conserving, King property
- causality (future tokens never change past logits), also with dilation and attention
- cached incremental decoding == full forward; masked-loss correctness
- every parameter receives a finite gradient; checkpointing matches plain
- overfit sanity with SelfEvolver, generation modes, constrained decoding

## Known limitations (be aware)
- **Context.** X5 layers are local (a `window` of tokens, widened by dilation). Global context comes
  from the hybrid attention layers (below); with `--attn_every 0` the model is local-only.
- Decoding is cached; when the context passes `block_size` the cache is rebuilt from the latest tokens.
- **Self-evolution** adapts weights, LR and a loss weight inside bounds. It does not change code
  or architecture.
- Scaling behaviour of this architecture is unproven; expect to tune `keep_k`, `window`,
  `flux_loops` and the straight-through temperatures.
