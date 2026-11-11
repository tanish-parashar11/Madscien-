"""MADSCIEN-X5 language model: hybrid decoder-only LM (X5 pipeline layers + global GQA attention)."""
import inspect
import math
from typing import NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import X5Config
from .x5_core import X5Core


DIGIT_0, DIGIT_9 = ord("0"), ord("9")          # byte tokenizer: a token id is its byte


def digit_places(idx, run_before=None):
    """Place of every token inside its run of digits: 1 for the first digit of a number, 2 for the
    second, ...; 0 for every other token. run_before (B,) = digits already in a run just before idx,
    so cached decoding can continue a number across calls."""
    B, T = idx.shape
    is_digit = (idx >= DIGIT_0) & (idx <= DIGIT_9)
    t = torch.arange(T, device=idx.device).expand(B, T)
    start = -1 if run_before is None else (-1 - run_before).view(B, 1)
    last_other = torch.where(is_digit, torch.as_tensor(start, device=idx.device).expand(B, T), t)
    last_other = torch.cummax(last_other, dim=1).values
    return torch.where(is_digit, t - last_other, torch.zeros_like(t))


class LMOutput(NamedTuple):
    logits: torch.Tensor
    loss: Optional[torch.Tensor]     # plain cross-entropy (ignore_index = -100)
    aux: torch.Tensor                # (n_x5_layers, 5): dL_rel, fallback, survivors, mean_BO, enclosed


def _fp32_if_half(x):
    return x.float() if x.dtype in (torch.float16, torch.bfloat16) else x


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = _fp32_if_half(x)
        out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (out * self.weight.to(out.dtype)).to(x.dtype)


class X5Block(nn.Module):
    """x -> norm -> [charges + contents per head] -> causal windows -> X5Core (steps 1-6)
       -> norm -> out_proj -> residual.   Optional SwiGLU FFN (cfg.ffn_mult > 0).
       cache (dict) = last (window-1)*dilation projected charges/contents, for incremental decoding."""

    def __init__(self, cfg: X5Config, x5_idx: int):
        super().__init__()
        self.cfg = cfg
        H, d = cfg.n_heads, cfg.core_dim
        self.dilation = cfg.dilation(x5_idx)               # index among X5 layers, not all layers
        self.norm1 = RMSNorm(cfg.d_model)
        self.q_proj = nn.Linear(cfg.d_model, H, bias=False)        # word -> charge
        self.x_proj = nn.Linear(cfg.d_model, H * d, bias=False)    # word -> content
        self.core = X5Core(cfg)
        self.feat_norm = RMSNorm(H * cfg.feat_dim)
        self.out_proj = nn.Linear(H * cfg.feat_dim, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)
        self.norm2 = None
        if cfg.ffn_mult > 0:
            hid = cfg.ffn_mult * cfg.d_model
            self.norm2 = RMSNorm(cfg.d_model)
            self.w_gate = nn.Linear(cfg.d_model, hid, bias=False)
            self.w_up = nn.Linear(cfg.d_model, hid, bias=False)
            self.w_down = nn.Linear(hid, cfg.d_model, bias=False)

    def _windows(self, q, X, n_new):
        """q:(B,L,H) X:(B,L,H,d) hold L consecutive positions. Causal (dilated) windows for the
        LAST n_new positions -> (B*n_new*H, W), (B*n_new*H, W, d)."""
        B, L, H = q.shape
        W, dil = self.cfg.window, self.dilation
        t = torch.arange(L - n_new, L, device=q.device)[:, None]
        j = torch.arange(W, device=q.device)[None, :]
        idx = t - (W - 1 - j) * dil                    # (n_new, W) oldest -> newest (= t)
        valid = (idx >= 0).to(q.dtype)                 # before the start of text = padding
        idx = idx.clamp_min(0)
        qw = q[:, idx] * valid[None, :, :, None]                       # (B,n_new,W,H)
        Xw = X[:, idx] * valid[None, :, :, None, None]                 # (B,n_new,W,H,d)
        qw = qw.permute(0, 1, 3, 2).reshape(B * n_new * H, W)
        Xw = Xw.permute(0, 1, 3, 2, 4).reshape(B * n_new * H, W, X.shape[-1])
        return qw, Xw

    def forward(self, x, cache=None):
        B, Tn, _ = x.shape
        H, d = self.cfg.n_heads, self.cfg.core_dim
        # Detached: X5 output and weights still train, but gradients through the core must not reach
        # the embeddings or earlier layers, otherwise the model stops learning from its input.
        h = self.norm1(x.detach())
        cdt = _fp32_if_half(x).dtype
        q = self.q_proj(h).to(cdt)
        X = self.x_proj(h).to(cdt).view(B, Tn, H, d)
        if cache is not None:                                # incremental decoding
            if "q" in cache:
                q = torch.cat([cache["q"], q], dim=1)
                X = torch.cat([cache["X"], X], dim=1)
            need = (self.cfg.window - 1) * self.dilation
            cache["q"], cache["X"] = q[:, -need:], X[:, -need:]
        qw, Xw = self._windows(q, X, Tn)
        with torch.autocast(device_type=x.device.type, enabled=False):
            feats, aux = self.core(qw, Xw)
        feats = feats.view(B, Tn, H * self.cfg.feat_dim).to(x.dtype)
        x = x + self.drop(self.out_proj(self.feat_norm(feats)))
        if self.norm2 is not None:
            h2 = self.norm2(x)
            x = x + self.drop(self.w_down(F.silu(self.w_gate(h2)) * self.w_up(h2)))
        return x, aux


class CausalSelfAttention(nn.Module):
    """Causal self-attention: grouped-query (GQA), QK-norm, rotary embeddings (RoPE), KV cache."""

    def __init__(self, cfg: X5Config):
        super().__init__()
        D, H, G = cfg.d_model, cfg.attn_heads, cfg.attn_kv_heads
        self.H, self.G, self.hd = H, G, D // H
        self.dropout = cfg.dropout
        self.q = nn.Linear(D, H * self.hd, bias=False)
        self.kv = nn.Linear(D, 2 * G * self.hd, bias=False)
        self.o = nn.Linear(H * self.hd, D, bias=False)
        self.qn, self.kn = RMSNorm(self.hd), RMSNorm(self.hd)     # QK-norm: stable attention logits
        inv_freq = 1.0 / (cfg.rope_base ** (torch.arange(0, self.hd, 2).float() / self.hd))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _rope(self, x, T, pos0=0):                      # x: (B, heads, T, hd), positions pos0..pos0+T-1
        dt = _fp32_if_half(x).dtype
        t = torch.arange(pos0, pos0 + T, device=x.device, dtype=dt)
        freqs = torch.outer(t, self.inv_freq.to(dt))     # (T, hd/2)
        cos, sin = freqs.cos()[None, None], freqs.sin()[None, None]
        x1, x2 = x[..., ::2].to(dt), x[..., 1::2].to(dt)
        out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)
        return out.to(x.dtype)

    def forward(self, x, cache=None):
        B, T, _ = x.shape
        pos0 = cache["k"].shape[2] if (cache is not None and "k" in cache) else 0
        assert T == 1 or pos0 == 0, "cached decoding feeds one token at a time after the prefill"
        q = self.q(x).view(B, T, self.H, self.hd).transpose(1, 2)           # (B,H,T,hd)
        k, v = self.kv(x).view(B, T, 2, self.G, self.hd).unbind(2)          # (B,T,G,hd) each
        k, v = k.transpose(1, 2), v.transpose(1, 2)                         # (B,G,T,hd)
        q, k = self._rope(self.qn(q), T, pos0), self._rope(self.kn(k), T, pos0)
        if cache is not None:
            if "k" in cache:
                k, v = torch.cat([cache["k"], k], dim=2), torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v                                   # stored before GQA repeat
        if self.G != self.H:
            rep = self.H // self.G
            k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=(T > 1),
                                           dropout_p=self.dropout if self.training else 0.0)
        return self.o(y.transpose(1, 2).reshape(B, T, self.H * self.hd))


class AttnBlock(nn.Module):
    """Pre-norm transformer layer: causal GQA attention + SwiGLU. Gives global context."""

    def __init__(self, cfg: X5Config):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(cfg.d_model), RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        hid = cfg.attn_ffn_mult * cfg.d_model
        self.w_gate = nn.Linear(cfg.d_model, hid, bias=False)
        self.w_up = nn.Linear(cfg.d_model, hid, bias=False)
        self.w_down = nn.Linear(hid, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cache=None):
        x = x + self.drop(self.attn(self.norm1(x), cache))
        h = self.norm2(x)
        return x + self.drop(self.w_down(F.silu(self.w_gate(h)) * self.w_up(h)))


class MadScienLM(nn.Module):
    def __init__(self, cfg: X5Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.digit_emb = nn.Embedding(cfg.digit_positions, cfg.d_model) if cfg.digit_positions else None
        self.blocks = nn.ModuleList([AttnBlock(cfg) if cfg.is_attn_layer(i) else X5Block(cfg, cfg.x5_index(i))
                                     for i in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)
        std = 0.02 / math.sqrt(2 * cfg.n_layer)          # GPT-2 style residual scaling
        for blk in self.blocks:
            if isinstance(blk, AttnBlock):
                nn.init.normal_(blk.attn.o.weight, mean=0.0, std=std)
                nn.init.normal_(blk.w_down.weight, mean=0.0, std=std)
            else:
                nn.init.normal_(blk.out_proj.weight, mean=0.0, std=std)
                if blk.norm2 is not None:
                    nn.init.normal_(blk.w_down.weight, mean=0.0, std=std)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def new_cache(self):
        """One dict per layer, plus one for the model's own state (the digit run in progress).
        Pass to forward(): first call = prefill (any length), then 1 token at a time."""
        return [dict() for _ in self.blocks] + [dict()]

    def _embed(self, idx, caches):
        x = self.tok_emb(idx)
        if self.digit_emb is None:
            return x
        state = None if caches is None else caches[-1]
        places = digit_places(idx, None if state is None else state.get("run"))
        if state is not None:
            state["run"] = places[:, -1]
        if self.training and self.cfg.digit_offset:
            shift = torch.randint(0, self.cfg.digit_offset + 1, (idx.size(0), 1), device=idx.device)
            places = torch.where(places > 0, places + shift, places)
        is_digit = (places > 0).unsqueeze(-1).to(x.dtype)
        return x + self.digit_emb(places.clamp(max=self.cfg.digit_positions - 1)) * is_digit

    def forward(self, idx, targets=None, last_only=False, caches=None):
        B, T = idx.shape
        if caches is None:
            assert T <= self.cfg.block_size, f"sequence {T} > block_size {self.cfg.block_size}"
        x = self._embed(idx, caches)
        auxs = []
        for i, blk in enumerate(self.blocks):
            if caches is None and self.cfg.grad_checkpoint and self.training and torch.is_grad_enabled():
                out = checkpoint(blk, x, use_reentrant=False)
            else:
                out = blk(x, cache=None if caches is None else caches[i])
            if isinstance(blk, AttnBlock):
                x = out
            else:
                x, aux = out
                auxs.append(aux)
        x = self.norm_f(x)
        if last_only:
            x = x[:, -1:, :]
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                                   targets.reshape(-1), ignore_index=-100)
        return LMOutput(logits, loss, torch.stack(auxs))

    def configure_optimizers(self, weight_decay, lr, betas, device_type):
        params = [p for p in self.parameters() if p.requires_grad]
        groups = [
            {"params": [p for p in params if p.dim() >= 2], "weight_decay": weight_decay},
            {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0},
        ]
        use_fused = device_type == "cuda" and "fused" in inspect.signature(torch.optim.AdamW).parameters
        extra = {"fused": True} if use_fused else {}
        return torch.optim.AdamW(groups, lr=lr, betas=betas, **extra)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None, eos_id=None,
                 constraint=None, repetition_penalty=1.0, use_cache=True):
        """Incremental decoding (KV cache + X5 window cache). When the context would exceed block_size
        the cache is rebuilt from the most recent tokens.
        constraint: optional object with allowed_mask(vocab, eos_id, device) and feed(token),
        e.g. madscien.constraints.DelimiterConstraint (batch size 1; caller feeds the prompt)."""
        if constraint is not None:
            assert idx.size(0) == 1, "constrained decoding supports batch size 1"
        was_training = self.training
        self.eval()
        bs = self.cfg.block_size
        reserve = min(64, max(1, bs // 4))
        caches, cache_len = None, 0
        done = torch.zeros(idx.size(0), 1, dtype=torch.bool, device=idx.device)
        for _ in range(max_new_tokens):
            if use_cache:
                if caches is None or cache_len >= bs:        # (re)prefill from the latest tokens
                    caches = self.new_cache()
                    inp = idx[:, -(bs - reserve):]
                    cache_len = inp.size(1)
                else:
                    inp = idx[:, -1:]
                    cache_len += 1
                logits = self(inp, last_only=True, caches=caches).logits[:, -1, :].float()
            else:
                logits = self(idx[:, -bs:], last_only=True).logits[:, -1, :].float()
            if repetition_penalty != 1.0:
                hist = idx[:, -256:]
                score = logits.gather(1, hist)
                score = torch.where(score > 0, score / repetition_penalty, score * repetition_penalty)
                logits = logits.scatter(1, hist, score)
            if constraint is not None:
                allowed = constraint.allowed_mask(logits.size(-1), eos_id, logits.device)
                logits = logits.masked_fill(~allowed, float("-inf"))
            if temperature <= 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
                if top_p:
                    s_logits, s_idx = torch.sort(logits, descending=True)
                    s_probs = torch.softmax(s_logits, dim=-1)
                    drop = (s_probs.cumsum(-1) - s_probs) > top_p
                    s_logits = s_logits.masked_fill(drop, float("-inf"))
                    logits = torch.full_like(logits, float("-inf")).scatter(1, s_idx, s_logits)
                nxt = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
            if eos_id is not None:                       # finished rows keep emitting EOS (padding)
                nxt = torch.where(done, torch.full_like(nxt, eos_id), nxt)
                done = done | (nxt == eos_id)
            idx = torch.cat([idx, nxt], dim=1)
            if constraint is not None:
                constraint.feed(int(nxt.item()))
            if eos_id is not None and bool(done.all()):
                break
        self.train(was_training)
        return idx


def physics_loss(ce, aux, lam_cons, lam_fallback):
    """Step 7 objective:  CE  +  lam_cons * (dL/L before the angular impulse)
                              +  lam_fallback * (rate of problems where no stable candidate survived)"""
    return ce + lam_cons * aux[:, 0].mean() + lam_fallback * aux[:, 1].mean()
