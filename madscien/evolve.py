"""
Step 7: Self-Evolution  ->  self_evolve(loss = physics_loss(final_answer))

SelfEvolver owns the update rule of the model:
  * backward + clip + AdamW step on physics_loss   (the model updates itself)
  * warmup + cosine learning-rate schedule
  * bounded self-adaptation, every `adapt_every` updates:
        - plateau in the loss  -> shrink the LR multiplier (never below `min_mult`)
        - dL/L above target    -> raise lam_cons, below target -> lower it (bounded)
It does NOT rewrite code or architecture; everything it can change is bounded and logged.
"""
import math


class SelfEvolver:
    def __init__(self, model, optimizer, base_lr, min_lr, warmup_iters, max_iters,
                 grad_clip=1.0, scaler=None, adapt=True, adapt_every=200,
                 plateau_tol=0.002, min_mult=0.25, cons_target=0.25,
                 lam_cons_bounds=(1e-4, 1.0)):
        self.model, self.opt, self.scaler = model, optimizer, scaler
        self.base_lr, self.min_lr = base_lr, min_lr
        self.warmup, self.max_iters = warmup_iters, max_iters
        self.grad_clip = grad_clip
        self.adapt, self.adapt_every = adapt, adapt_every
        self.plateau_tol, self.min_mult = plateau_tol, min_mult
        self.cons_target, self.lam_bounds = cons_target, lam_cons_bounds
        self.lr_mult = 1.0
        self.lam_cons = model.cfg.lam_cons
        self.lam_fallback = model.cfg.lam_fallback
        self.skipped = 0
        self._ema = None
        self._ema_prev = None
        self._dL_sum, self._n = 0.0, 0
        self._updates = 0

    # ---- schedule
    def lr_at(self, it):
        if it < self.warmup:
            lr = self.base_lr * (it + 1) / max(1, self.warmup)
        else:
            ratio = (it - self.warmup) / max(1, self.max_iters - self.warmup)
            coeff = 0.5 * (1.0 + math.cos(math.pi * min(1.0, ratio)))
            lr = self.min_lr + coeff * (self.base_lr - self.min_lr)
        return lr * self.lr_mult

    # ---- update
    def backward(self, loss):
        (self.scaler.scale(loss) if self.scaler is not None else loss).backward()

    def step(self, it, ce_value, dL_value):
        import torch
        lr = self.lr_at(it)
        for g in self.opt.param_groups:
            g["lr"] = lr
        if self.scaler is not None:
            self.scaler.unscale_(self.opt)
        gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        if self.scaler is not None:
            self.scaler.step(self.opt)
            self.scaler.update()
        elif torch.isfinite(gnorm):
            self.opt.step()
        else:
            self.skipped += 1                       # never corrupt weights with NaN/inf grads
        self.opt.zero_grad(set_to_none=True)
        self._observe(ce_value, dL_value)
        return float(gnorm), lr

    # ---- bounded self-adaptation
    def _observe(self, ce, dL):
        self._ema = ce if self._ema is None else 0.98 * self._ema + 0.02 * ce
        self._dL_sum += dL
        self._n += 1
        self._updates += 1
        if not self.adapt or self._updates % self.adapt_every != 0:
            return
        if self._ema_prev is not None:
            improvement = (self._ema_prev - self._ema) / max(abs(self._ema_prev), 1e-9)
            if improvement < self.plateau_tol:
                self.lr_mult = max(self.min_mult, self.lr_mult * 0.9)
            else:
                self.lr_mult = min(1.0, self.lr_mult * 1.05)
        self._ema_prev = self._ema
        mean_dL = self._dL_sum / max(1, self._n)
        lo, hi = self.lam_bounds
        if mean_dL > self.cons_target:
            self.lam_cons = min(hi, self.lam_cons * 1.25)
        else:
            self.lam_cons = max(lo, self.lam_cons / 1.25)
        self._dL_sum, self._n = 0.0, 0

    def state_dict(self):
        return dict(lr_mult=self.lr_mult, lam_cons=self.lam_cons, lam_fallback=self.lam_fallback,
                    ema=self._ema, ema_prev=self._ema_prev, updates=self._updates,
                    skipped=self.skipped)

    def load_state_dict(self, s):
        self.lr_mult, self.lam_cons = s["lr_mult"], s["lam_cons"]
        self.lam_fallback = s["lam_fallback"]
        self._ema, self._ema_prev = s["ema"], s["ema_prev"]
        self._updates, self.skipped = s["updates"], s["skipped"]
