"""
Talk to a trained checkpoint (multi-turn, remembers the conversation).

    python chat_cli.py --ckpt out/ckpt.pt
Commands:  /reset  clear history   /system <text>  set persona   /temp <x>  set temperature   /quit
"""
import argparse

import torch

from madscien.chat import DEFAULT_SYSTEM, encode_chat, fit_to_context
from madscien.config import X5Config
from madscien.data import ByteTokenizer
from madscien.model import MadScienLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--system", default=DEFAULT_SYSTEM)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--repetition_penalty", type=float, default=1.1)
    ap.add_argument("--max_new_tokens", type=int, default=300)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location=device)
    cfg = X5Config(**ck["cfg"])
    model = MadScienLM(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    tok = ByteTokenizer()
    system, temp = a.system, a.temperature
    history = []
    print("MADSCIEN chat. /reset, /system <text>, /temp <x>, /quit")
    while True:
        try:
            user = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user:
            continue
        if user == "/quit":
            break
        if user == "/reset":
            history = []
            print("(history cleared)")
            continue
        if user.startswith("/system "):
            system, history = user[8:], []
            print("(persona set, history cleared)")
            continue
        if user.startswith("/temp "):
            temp = float(user[6:])
            continue
        history.append({"role": "user", "content": user})
        msgs = fit_to_context([{"role": "system", "content": system}] + history,
                              max(64, cfg.block_size - a.max_new_tokens), tok)
        ids, _ = encode_chat(msgs, tok, add_generation_prompt=True)
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        out = model.generate(idx, a.max_new_tokens, temperature=temp, top_k=a.top_k, top_p=a.top_p,
                             repetition_penalty=a.repetition_penalty, eos_id=tok.end_id)
        reply = tok.decode(out[0, len(ids):].tolist()).strip()
        history.append({"role": "assistant", "content": reply})
        print(f"bot > {reply}")


if __name__ == "__main__":
    main()
