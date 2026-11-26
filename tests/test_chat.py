"""Chat-format and SFT-data tests (no torch needed):  pytest -q tests/test_chat.py"""
import json
import os
import re
import tempfile

import numpy as np

from madscien.chat import DEFAULT_SYSTEM, encode_chat, fit_to_context
from madscien.chat_data import build_chat_bins, synthetic_conversations
from madscien.data import ASSISTANT, BOS, END, SYSTEM, USER, VOCAB_SIZE, ByteTokenizer, slice_xy
from madscien.verify import is_valid_python

TOK = ByteTokenizer()


def test_encode_chat_masks_only_assistant_text_and_its_end():
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    ids, mask = encode_chat(msgs)
    assert ids[0] == BOS and ids[1] == SYSTEM                      # default system prompt added
    assert len(ids) == len(mask) and max(ids) < VOCAB_SIZE
    trained = [i for i, m in zip(ids, mask) if m]
    assert trained == list(b"hello") + [END]
    assert ids.count(USER) == 1 and ids.count(ASSISTANT) == 1
    assert mask[ids.index(ASSISTANT)] == 0                         # role token is context, not target
    p_ids, p_mask = encode_chat(msgs[:1], add_generation_prompt=True)
    assert p_ids[-1] == ASSISTANT and sum(p_mask) == 0
    assert TOK.decode(ids).startswith(DEFAULT_SYSTEM[:20])


def test_fit_to_context_keeps_system_and_newest_turns():
    msgs = [{"role": "system", "content": "be nice"}]
    for i in range(30):
        msgs += [{"role": "user", "content": f"question number {i} " * 5},
                 {"role": "assistant", "content": f"answer number {i} " * 5}]
    msgs.append({"role": "user", "content": "final question"})
    out = fit_to_context(msgs, 400)
    assert out[0]["role"] == "system" and out[-1]["content"] == "final question"
    assert len(encode_chat(out, add_generation_prompt=True)[0]) <= 400
    assert out[1]["role"] == "user"                                # never starts mid-exchange
    huge = [{"role": "user", "content": "x" * 5000}]
    assert len(encode_chat(fit_to_context(huge, 300), add_generation_prompt=True)[0]) <= 300


def test_fit_to_context_shortens_persona_before_the_user_message():
    question = "What is the force between two 3 uC charges 2 m apart?"
    for max_len in (64, 96, 192):                       # chat_cli budgets for tiny / small presets
        long_system = [{"role": "system", "content": "persona " * 100}]
        for msgs in (long_system, []):                 # explicit persona, and the default one
            out = fit_to_context(msgs + [{"role": "user", "content": question}], max_len)
            assert out[-1]["content"] == question
            ids = encode_chat(out, add_generation_prompt=True)[0]
            assert len(ids) <= max_len and ids[:2] == [BOS, SYSTEM]


def test_default_system_prompt_fits_small_contexts():
    ids, _ = encode_chat([{"role": "user", "content": "hi"}], add_generation_prompt=True)
    assert len(ids) <= 128 - 128 // 4                   # fits chat_cli's prompt budget in the tiny preset


def test_synthetic_coulomb_answers_never_round_to_zero():
    from madscien.data import synth_doc
    rng = np.random.default_rng(0)
    for _ in range(5000):
        d = synth_doc(rng)
        if d.startswith("Problem: Two charges"):
            q1, q2, r = map(int, re.findall(r"(\d+) uC and (\d+) uC are (\d+) m", d)[0])
            got = float(re.findall(r"F = ([0-9.]+?)(?:\.\.\.)? mN", d)[0])
            want = 9 * q1 * q2 / r ** 2                 # 9e9 x 1e-6 x 1e-6 N = 9 mN
            assert got > 0 and 0 <= want - got < 0.01, d    # cut off at 2 decimals, never rounded


def test_synthetic_conversations_are_well_formed_and_varied():
    convs = list(synthetic_conversations(400, seed=1))
    text = json.dumps(convs)
    for msgs in convs:
        roles = [m["role"] for m in msgs]
        assert roles[0] == "user" and roles[-1] == "assistant" and len(roles) % 2 == 0
        assert roles == ["user", "assistant"] * (len(roles) // 2)
        assert all(m["content"].strip() for m in msgs)
    assert "bhai" in text.lower() and "```python" in text and "which language" in text
    blocks = re.findall(r"```python\n(.*?)```", text.replace("\\n", "\n"), flags=re.S)
    assert blocks and all(is_valid_python(b) for b in blocks)


def test_sft_bins_and_loss_masks_line_up():
    with tempfile.TemporaryDirectory() as d:
        sizes = build_chat_bins(synthetic_conversations(300, seed=2), d)
        assert sizes["train"] > 10000 and sizes["val"] > 100
        tokens = np.fromfile(os.path.join(d, "train.bin"), dtype=np.uint16)
        mask = np.fromfile(os.path.join(d, "train_mask.bin"), dtype=np.uint8)
        assert len(tokens) == len(mask) and tokens.max() < VOCAB_SIZE
        meta = json.load(open(os.path.join(d, "meta.json")))
        assert meta["vocab_size"] == VOCAB_SIZE
        rng = np.random.default_rng(0)
        seen_valid = 0
        for _ in range(200):
            i = int(rng.integers(0, len(tokens) - 130))
            x, y = slice_xy(tokens, mask, i, 128)
            assert x.shape == y.shape == (128,)
            valid = y != -100
            nxt = tokens[i + 1:i + 129].astype(np.int64)
            assert (y[valid] == nxt[valid]).all()
            assert not np.isin(y[valid], [SYSTEM, USER, ASSISTANT, BOS]).any()   # never train on role tokens
            seen_valid += valid.sum()
        assert seen_valid > 0
