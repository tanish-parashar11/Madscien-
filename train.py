"""
Train MADSCIEN-X5 LM.

    python -m madscien.data --source synth --n_docs 50000 --out data/synth
    python train.py --data_dir data/synth --preset tiny --max_iters 2000

Every update is Step 7:  self_evolve(loss = physics_loss(final_answer)).
"""
import argparse
import json
import os
import time
from contextlib import nullcontext

import torch

from madscien.config import PRESETS, X5Config, make_config
from madscien.data import BinDataset, ChatBinDataset
from madscien.evolve import SelfEvolver
from madscien.model import MadScienLM, physics_loss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", default="out")
    p.add_argument("--preset", choices=list(PRESETS), default="tiny")
    # model overrides
    for name in ("n_layer", "d_model", "n_heads", "window", "core_dim", "keep_k", "flux_loops",
                 "block_size", "ffn_mult", "attn_every", "attn_heads", "attn_kv_heads"):
        p.add_argument(f"--{name}", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--grad_checkpoint", action="store_true")
    # optimisation
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--max_iters", type=int, default=2000)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--min_lr", type=float, default=2e-4)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--no_self_adapt", action="store_true", help="freeze LR/lambda adaptation")
    # logging / io
    p.add_argument("--eval_interval", type=int, default=200)
    p.add_argument("--eval_iters", type=int, default=20)
    p.add_argument("--log_interval", type=int, default=20)
    p.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--threads", type=int, default=None, help="cap CPU threads (torch.set_num_threads)")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--sft", action="store_true", help="chat SFT data: train only on assistant tokens")
    p.add_argument("--init_from", default=None, help="start from a pretrained checkpoint (its config is used)")
    return p.parse_args()


@torch.no_grad()
def estimate_loss(model, ds, a, ctx, device, block_size):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(a.eval_iters)
        for i in range(a.eval_iters):
            x, y = ds.get_batch(split, a.batch_size, block_size, device)
            with ctx:
                losses[i] = model(x, y).loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    a = parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dev_type = "cuda" if device.startswith("cuda") else "cpu"
    dtype = a.dtype or ("bfloat16" if dev_type == "cuda" and torch.cuda.is_bf16_supported() else "float32")
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    ctx = nullcontext() if dtype == "float32" else torch.autocast(device_type=dev_type, dtype=ptdtype)
    if dev_type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    ds = (ChatBinDataset if a.sft else BinDataset)(a.data_dir)
    os.makedirs(a.out_dir, exist_ok=True)
    ckpt_path = os.path.join(a.out_dir, "ckpt.pt")          # best validation loss
    last_path = os.path.join(a.out_dir, "last.pt")          # most recent eval (used by --resume)

    overrides = {k: getattr(a, k) for k in ("n_layer", "d_model", "n_heads", "window", "core_dim",
                                              "keep_k", "flux_loops", "block_size", "ffn_mult", "dropout",
                                              "attn_every", "attn_heads", "attn_kv_heads")}
    overrides["vocab_size"] = ds.meta["vocab_size"]
    if a.grad_checkpoint:
        overrides["grad_checkpoint"] = True
    init_ck = torch.load(a.init_from, map_location=device) if a.init_from else None
    cfg = X5Config(**init_ck["cfg"]) if init_ck else make_config(a.preset, **overrides)
    assert cfg.vocab_size == ds.meta["vocab_size"], "dataset and model vocab sizes differ"

    model = MadScienLM(cfg).to(device)
    if init_ck:
        model.load_state_dict(init_ck["model"])
        print(f"initialised from {a.init_from}")
    optimizer = model.configure_optimizers(a.weight_decay, a.lr, (a.beta1, a.beta2), dev_type)
    scaler = None
    if dtype == "float16":
        try:
            scaler = torch.amp.GradScaler("cuda")
        except AttributeError:
            scaler = torch.cuda.amp.GradScaler()
    evolver = SelfEvolver(model, optimizer, a.lr, a.min_lr, a.warmup, a.max_iters,
                          grad_clip=a.grad_clip, scaler=scaler, adapt=not a.no_self_adapt)

    it0, best_val, resumed = 0, float("inf"), False
    resume_path = last_path if os.path.exists(last_path) else ckpt_path
    if a.resume and os.path.exists(resume_path):
        ck = torch.load(resume_path, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        evolver.load_state_dict(ck["evolver"])
        # checkpoints are written at eval time, BEFORE the update of that iteration runs
        it0, best_val, resumed = ck["iter"], ck["best_val"], True
        print(f"resumed from {resume_path} at iter {it0}")

    print(f"params {model.num_params() / 1e6:.2f}M | cases/problem {cfg.n_cases} | "
          f"receptive field {cfg.receptive_field()} tokens | device {device} | dtype {dtype}")
    run_model = torch.compile(model) if a.compile else model

    t0 = time.time()
    for it in range(it0, a.max_iters + 1):
        skip_eval = resumed and it == it0                    # already evaluated before the checkpoint
        if (it % a.eval_interval == 0 or it == a.max_iters) and not skip_eval:
            ev = estimate_loss(run_model, ds, a, ctx, device, cfg.block_size)
            print(f"[eval] iter {it} | train {ev['train']:.4f} | val {ev['val']:.4f} "
                  f"| val ppl {torch.exp(torch.tensor(ev['val'])).item():.2f}")
            is_best = ev["val"] < best_val and (it > 0 or a.max_iters == 0)
            if is_best:
                best_val = ev["val"]
            state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "evolver": evolver.state_dict(), "cfg": cfg.to_dict(),
                     "iter": it, "best_val": best_val}
            torch.save(state, last_path)
            if is_best:
                torch.save(state, ckpt_path)
            with open(os.path.join(a.out_dir, "cfg.json"), "w") as f:
                json.dump(cfg.to_dict(), f, indent=2)
        if it == a.max_iters:
            break

        ce_acc, dL_acc, aux_acc = 0.0, 0.0, None
        for _ in range(a.grad_accum):
            x, y = ds.get_batch("train", a.batch_size, cfg.block_size, device)
            with ctx:
                out = run_model(x, y)
                loss = physics_loss(out.loss, out.aux, evolver.lam_cons, evolver.lam_fallback)
            evolver.backward(loss / a.grad_accum)               # Step 7: self-evolve
            ce_acc += out.loss.item() / a.grad_accum
            dL_acc += out.aux[:, 0].mean().item() / a.grad_accum
            aux_acc = out.aux.detach() if aux_acc is None else aux_acc + out.aux.detach()
        gnorm, lr = evolver.step(it, ce_acc, dL_acc)

        if it % a.log_interval == 0:
            aux = (aux_acc / a.grad_accum).mean(0).tolist()
            print(f"iter {it:6d} | ce {ce_acc:.4f} | dL/L {aux[0]:.3f} | fallback {aux[1]:.2%} "
                  f"| survivors {aux[2]:.1f} | BO {aux[3]:.2f} | gnorm {gnorm:.2f} | lr {lr:.2e} "
                  f"| lam_cons {evolver.lam_cons:.4f} | skipped {evolver.skipped} "
                  f"| {(time.time() - t0) / (it - it0 + 1):.2f}s/it")


if __name__ == "__main__":
    main()
