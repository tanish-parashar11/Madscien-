"""
Chat format for MADSCIEN (stdlib only).

    <BOS><|system|>...<|end|><|user|>...<|end|><|assistant|>...<|end|><|user|>...

Training signal: only assistant text and the <|end|> that closes it (so the model learns WHAT to
say and WHEN to stop). System/user tokens are context only.
"""
from .data import ASSISTANT, BOS, END, SYSTEM, USER, ByteTokenizer

# Kept short on purpose: every byte is a token, and the small presets only have 128-256 tokens of context.
DEFAULT_SYSTEM = "You are MADSCIEN by Youtri: friendly, honest, concise; reply in the user's language."
ROLE_ID = {"system": SYSTEM, "user": USER, "assistant": ASSISTANT}


def encode_chat(messages, tok=None, add_generation_prompt=False, default_system=DEFAULT_SYSTEM):
    """messages: [{"role": "system"|"user"|"assistant", "content": str}, ...]
    Returns (ids, mask); mask[j] = 1 when token j is a training target."""
    tok = tok or ByteTokenizer()
    msgs = list(messages)
    if not msgs or msgs[0]["role"] != "system":
        msgs = [{"role": "system", "content": default_system}] + msgs
    ids, mask = [BOS], [0]
    for m in msgs:
        body = tok.encode(m["content"])
        train = 1 if m["role"] == "assistant" else 0
        ids += [ROLE_ID[m["role"]]] + body + [END]
        mask += [0] + [train] * len(body) + [train]
    if add_generation_prompt:
        ids.append(ASSISTANT)
        mask.append(0)
    return ids, mask


def fit_to_context(messages, max_len, tok=None, default_system=DEFAULT_SYSTEM):
    """Drop the oldest turns until the prompt fits in max_len tokens. If it still does not fit, shorten
    the system prompt (keeping its start), and as a last resort trim the start of the newest message."""
    tok = tok or ByteTokenizer()
    msgs = list(messages)
    system = [msgs.pop(0)] if msgs and msgs[0]["role"] == "system" else []
    size = lambda turns: len(encode_chat(system + turns, tok, True, default_system)[0])
    while len(msgs) > 1 and size(msgs) > max_len:
        msgs.pop(0)
        while len(msgs) > 1 and msgs[0]["role"] != "user":
            msgs.pop(0)
    if msgs and size(msgs) > max_len:                  # the user's message matters more than the persona
        system = system or [{"role": "system", "content": default_system}]
        raw = system[0]["content"].encode("utf-8")
        keep = max(0, len(raw) - (size(msgs) - max_len))
        system = [{"role": "system", "content": raw[:keep].decode("utf-8", errors="ignore")}]
    if msgs and size(msgs) > max_len:
        last = dict(msgs[-1])
        raw = last["content"].encode("utf-8")
        over = size(msgs) - max_len
        last["content"] = raw[min(over, max(0, len(raw) - 1)):].decode("utf-8", errors="ignore")
        msgs[-1] = last
    return system + msgs
