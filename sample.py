"""
Generate text from a trained checkpoint.

    python sample.py --ckpt out/ckpt.pt --prompt "Problem: Two charges 3 uC and 4 uC" --max_new_tokens 200
"""
import argparse

import torch

from madscien.config import X5Config
from madscien.data import ByteTokenizer
from madscien.model import MadScienLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompt", default="Problem:")
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location=device)
    cfg = X5Config(**ck["cfg"])
    model = MadScienLM(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    tok = ByteTokenizer()
    ids = [tok.bos_id] + tok.encode(a.prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, a.max_new_tokens, temperature=a.temperature, top_k=a.top_k,
                         top_p=a.top_p, eos_id=tok.eos_id)
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
