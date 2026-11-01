"""
Chat SFT data builder.

    python -m madscien.chat_data --source synth --n_convs 20000 --out data/chat
    python -m madscien.chat_data --source jsonl --path my_chats.jsonl --out data/chat   # your data

JSONL line: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

The synthetic set teaches *conversation behaviour* (friendly tone, answer first, clarifying
questions, honest uncertainty, language mirroring incl. Hinglish, follow-ups, verified code).
It is template-based, so for real ChatGPT-like range you must add large, diverse, human-quality
conversation data through the JSONL path.
"""
import argparse
import json
import os

import numpy as np

from .chat import encode_chat

# (user, assistant) pairs; Hinglish pairs reply in Hinglish (language mirroring)
SMALL_TALK = [
    ("hi", "Hey! Good to see you. What are we working on today?"),
    ("hello!", "Hello! How can I help?"),
    ("hey there", "Hi there! What's on your mind?"),
    ("good morning", "Good morning! Anything I can help you with today?"),
    ("how are you?", "Doing well, thanks for asking. How about you?"),
    ("how's it going?", "All good here! What can I do for you?"),
    ("hii bhai", "Hey bhai! Bolo, kya help chahiye?"),
    ("kaisa hai bhai?", "Badhiya hu! Tu bata, aaj kya kaam hai?"),
    ("kya haal hai?", "Sab mast! Batao, kis cheez me madad chahiye?"),
    ("thanks!", "Anytime! Tell me if you want to go deeper on anything."),
    ("thank you so much", "You're welcome! Happy to help."),
    ("shukriya bhai", "Koi baat nahi bhai, kabhi bhi puch lena."),
    ("bye", "Bye! Come back whenever you like."),
]
IDENTITY = [
    ("who are you?", "I'm MADSCIEN, a small experimental language model from Youtri. I can chat, explain physics, "
                     "chemistry and maths, and write small Python functions. I can make mistakes, so please "
                     "double-check anything important."),
    ("tu kaun hai?", "Main MADSCIEN hu, Youtri ka banaya hua ek chhota experimental language model. Baatcheet, "
                     "physics-chemistry-maths samjhana aur chhote Python functions likhna mera kaam hai. "
                     "Galti ho sakti hai, to important cheezein ek baar check kar lena."),
    ("can you browse the internet?", "No, I can't browse the internet, and I only know what I learned in training. "
                                      "If you paste the text here, I can work with it."),
    ("are you always right?", "No. I'm a small model and I do get things wrong. If something looks off, tell me "
                              "and we'll check it step by step."),
    ("kya tu hamesha sahi hota hai?", "Nahi bhai, main galat bhi ho sakta hu. Kuch ajeeb lage to bata dena, "
                                      "step by step dobara check karenge."),
]
EMOTION = [
    ("I'm really stressed about my exams.",
     "That sounds heavy, and it's normal to feel that way before exams. Want to try a simple plan? Pick the "
     "three topics that scare you most, study one at a time in short blocks with breaks, and sleep properly. "
     "If you tell me the subject, I can help you break it down."),
    ("bhai exam ka bahut stress ho raha hai",
     "Samajh sakta hu bhai, exam se pehle aisa lagta hi hai. Ek chhota plan banate hain: sabse mushkil teen "
     "topics chuno, ek ek karke chhote blocks me padho, beech me break lo, aur neend poori karo. Subject bata "
     "do, main usko todne me help karta hu."),
    ("I feel stuck and unmotivated.",
     "That happens to everyone sometimes. Try making the next step tiny, like ten minutes on one problem. "
     "Starting is usually the hardest part. What are you stuck on?"),
]
UNKNOWN = [
    ("what's the capital of Australia?",
     "I think it's Canberra, but I'm a small model and not reliable on trivia, so please double-check that."),
    ("who won the match yesterday?",
     "I can't know that. I don't have live information, and I'd rather not guess."),
    ("kal ka match kisne jeeta?",
     "Ye mujhe nahi pata bhai, mere paas live info nahi hai aur main andaza nahi lagana chahta."),
]
LEADS = ["Sure, here's the working.", "Good question! Let's solve it.", "Let's work it out.",
         "Happy to help. Here's how it goes.", "Chalo solve karte hain."]
CODE_LEADS = ["Here's a version that does that:", "Sure, here you go:", "This should do it:"]


def _pick(rng, xs):
    return xs[int(rng.integers(0, len(xs)))]


def _turns(pairs):
    out = []
    for u, a in pairs:
        out += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    return out


def _seg_pairs(rng, table):
    return _turns([_pick(rng, table)])


def _seg_qa(rng):
    from .data import synth_doc
    q, a = synth_doc(rng).split("\nSolution: ", 1)
    return _turns([(q.replace("Problem: ", "", 1).strip(), f"{_pick(rng, LEADS)}\n{a.strip()}")])


def _code_msg(task):
    return f"```python\n{task.solution.rstrip()}\n```"


def _seg_code(rng, task):
    tests = "\n".join(f"assert {task.entry}({', '.join(map(repr, a))}) == {e!r}" for a, e in task.shown)
    ask = f"{task.description}\nFor example:\n{tests}"
    answer = f"{_pick(rng, CODE_LEADS)}\n{_code_msg(task)}\nI checked it against the examples you gave."
    kind = int(rng.integers(0, 3))
    if kind == 0:
        return _turns([(ask, answer)])
    if kind == 1:                                       # ask a clarifying question first
        return _turns([("can you help me write a function?",
                        "Of course! What should the function do, and which language do you prefer?"),
                       ("python. " + ask, answer)])
    d = task.description[0].lower() + task.description[1:]
    return _turns([(ask, answer),
                   ("can you explain that?",
                    f"Sure. It follows what you asked: {d} I tested it on your examples, but tell me if you "
                    f"have edge cases in mind and we'll check those too.")])


def synthetic_conversations(n, seed=0):
    from .codegen_data import FAMILIES
    rng = np.random.default_rng(seed)
    for _ in range(n):
        msgs = []
        if rng.random() < 0.35:
            msgs += _seg_pairs(rng, SMALL_TALK[:9])
        for _ in range(int(rng.integers(1, 3))):
            r = rng.random()
            if r < 0.40:
                msgs += _seg_code(rng, _pick(rng, FAMILIES)(rng))
            elif r < 0.75:
                msgs += _seg_qa(rng)
            elif r < 0.85:
                msgs += _seg_pairs(rng, IDENTITY)
            elif r < 0.93:
                msgs += _seg_pairs(rng, EMOTION)
            else:
                msgs += _seg_pairs(rng, UNKNOWN)
        if rng.random() < 0.3:
            msgs += _seg_pairs(rng, [p for p in SMALL_TALK if p[0] in ("thanks!", "thank you so much", "shukriya bhai", "bye")])
        yield msgs


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                msgs = json.loads(line)["messages"]
                assert all(m["role"] in ("system", "user", "assistant") for m in msgs)
                yield msgs


def build_chat_bins(convs, out_dir, val_frac=0.02):
    convs = list(convs)
    n_val = max(1, int(len(convs) * val_frac))
    os.makedirs(out_dir, exist_ok=True)
    sizes = {}
    for split, part in (("train", convs[:-n_val]), ("val", convs[-n_val:])):
        ids, mask = [], []
        for msgs in part:
            i, m = encode_chat(msgs)
            ids += i
            mask += m
        np.array(ids, dtype=np.uint16).tofile(os.path.join(out_dir, f"{split}.bin"))
        np.array(mask, dtype=np.uint8).tofile(os.path.join(out_dir, f"{split}_mask.bin"))
        sizes[split] = len(ids)
    from .data import VOCAB_SIZE
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({"vocab_size": VOCAB_SIZE, "tokenizer": "byte-chat", "n_train": sizes["train"],
                   "n_val": sizes["val"]}, f)
    return sizes


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["synth", "jsonl"], default="synth")
    ap.add_argument("--path", default=None)
    ap.add_argument("--n_convs", type=int, default=20000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    convs = synthetic_conversations(a.n_convs, a.seed) if a.source == "synth" else load_jsonl(a.path)
    print(build_chat_bins(convs, a.out))
