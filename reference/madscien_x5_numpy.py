"""
MADSCIEN X5  -  Electrostatics x Rotation x EMI x Calculus x P&C x Chemistry
=============================================================================
Numpy-only, runnable model that implements the 7-step algorithm:

  1. Field Filter       Coulomb flux + Gauss gate + earthing   (drop low-charge tokens)
  2. Conservation Check Rotation (Cayley) + torque + angular impulse (keep L constant)
  3. Logic Induce       Faraday: e = -dPhi/dt along the sequence
  4. Solve              P&C (all subsets) + Bayes + King-property integral + complex phasors
  5. Chemistry Filter   MOT bond order via Aufbau fill + GOC (3 > 2 > 1) stability
  6. Singularity Core   weighted compression of survivors into ONE vector
  7. Self-Evolution     gradient-free evolution (antithetic ES + Adam) on physics_loss

Training = ES evolves the pipeline genome, ridge regression (closed form) fits the
read-out head for every genome.  No autograd needed -> numpy only.

Run:   python madscien_x5.py            (train + report)
       python madscien_x5.py --test     (invariant checks only)
"""
import argparse
import itertools
import time
from dataclasses import dataclass

import numpy as np


# ------------------------------------------------------------------ config
@dataclass
class Config:
    vocab: int = 20          # words
    seq_len: int = 12        # words per problem
    d: int = 6               # charge-space dimension
    keep_k: int = 4          # Gaussian surface encloses k strongest charges
    m_f: int = 6             # flux loops (grid points of the integral)
    max_subset: int = 3      # P&C: all subsets of size 1..max_subset
    n_pairs: int = 4         # MOT: orbital pairs (bonding + antibonding each)
    coulomb_k: float = 0.1   # F = k q1 q2 / r^2
    eps: float = 0.05        # softening so r -> 0 does not blow up
    omega: float = 1.0       # spin rate, L = I * omega
    torque: float = 0.5      # strength of the non-conservative step in layer 2
    # physics_loss weights
    lam_cons: float = 0.05   # penalise big angular-momentum violations
    lam_fallback: float = 0.05  # penalise "no stable candidate survived"


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


# ------------------------------------------------------------------ genome
def param_shapes(cfg):
    s_dim = 3 * cfg.m_f + 1
    return {
        "E": (cfg.vocab, cfg.d),          # word -> content vector
        "charge": (cfg.vocab,),           # word -> charge  (step 1)
        "S": (cfg.d, cfg.d),              # skew generator of the rotation (step 2)
        "Wt": (cfg.d, cfg.d),             # torque map (step 2)
        "B": (cfg.d, cfg.m_f),            # flux loops (step 3)
        "u": (2 * cfg.m_f,),              # Bayes log-likelihood weights (step 4)
        "Aalpha": (s_dim, cfg.n_pairs),   # orbital centre energies (step 5)
        "Ah": (s_dim, cfg.n_pairs),       # resonance integrals (step 5)
        "we": (s_dim,),                   # electron count (step 5)
        "tau": (1,),                      # similarity threshold for GOC (step 5)
    }


def init_params(cfg, rng):
    sh = param_shapes(cfg)
    scale = {"E": 1 / np.sqrt(cfg.d), "charge": 0.5, "S": 0.3, "Wt": 0.3,
             "B": 1 / np.sqrt(cfg.d), "u": 0.5, "Aalpha": 0.3, "Ah": 0.5,
             "we": 0.3, "tau": 0.0}
    return {k: rng.standard_normal(s) * scale[k] for k, s in sh.items()}


def flatten(p, cfg):
    return np.concatenate([p[k].ravel() for k in param_shapes(cfg)])


def unflatten(theta, cfg):
    out, i = {}, 0
    for k, s in param_shapes(cfg).items():
        n = int(np.prod(s))
        out[k] = theta[i:i + n].reshape(s)
        i += n
    return out


# ------------------------------------------------------------------ ridge head
class RidgeHead:
    """Closed-form linear read-out on standardised features."""

    def __init__(self, lam=1e-2):
        self.lam = lam

    def fit(self, Z, y):
        self.mu, self.sd = Z.mean(0), Z.std(0) + 1e-8
        A = np.hstack([(Z - self.mu) / self.sd, np.ones((len(Z), 1))])
        reg = self.lam * np.eye(A.shape[1])
        reg[-1, -1] = 0.0
        self.w = np.linalg.solve(A.T @ A + reg * len(Z), A.T @ y)
        return self

    def predict(self, Z):
        A = np.hstack([(Z - self.mu) / self.sd, np.ones((len(Z), 1))])
        return A @ self.w


# ------------------------------------------------------------------ the model
class MadScienX5:
    def __init__(self, cfg=None, seed=0):
        self.cfg = cfg or Config()
        self.rng = np.random.default_rng(seed)
        self.p = init_params(self.cfg, self.rng)
        self.head = None
        self.sigma = 0.15                 # ES step size (itself self-adapted)
        self._adam_m = self._adam_v = None
        self._t = 0
        k = self.cfg.keep_k
        self.subsets = [c for r in range(1, self.cfg.max_subset + 1)
                        for c in itertools.combinations(range(k), r)]
        self.M = np.zeros((len(self.subsets), k))     # subset -> averaging weights
        for i, c in enumerate(self.subsets):
            self.M[i, list(c)] = 1.0 / len(c)

    # -------- helpers
    def _ang_mom(self, X, m):
        """L = I * omega,  I = sum m_i |x_i|^2   (moment of inertia of kept charges)."""
        return self.cfg.omega * (m * (X ** 2).sum(-1)).sum(-1)

    @staticmethod
    def _cumtrapz(y, h):
        mid = 0.5 * (y[..., 1:] + y[..., :-1]) * h
        return np.concatenate([np.zeros_like(y[..., :1]), np.cumsum(mid, -1)], -1)

    # -------- forward pass: Steps 1-6
    def forward(self, tokens, p=None, diag=False):
        p, cfg = (p or self.p), self.cfg
        Bn, n = tokens.shape
        rows = np.arange(Bn)[:, None]
        D = {}

        # ===== Step 1: Field Filter [Electrostatics + Earthing] =====
        q = p["charge"][tokens]                         # har word ko charge
        aq = np.abs(q)
        pos = np.arange(n) / n
        inv_r2 = 1.0 / ((pos[:, None] - pos[None, :]) ** 2 + cfg.eps)
        np.fill_diagonal(inv_r2, 0.0)
        # flux_i = |q_i| + k * sum_j |q_i||q_j| / r_ij^2      (Coulomb: F = k q1 q2 / r^2)
        flux = aq + cfg.coulomb_k * aq * (aq @ inv_r2)
        top = np.argsort(-flux, axis=1, kind="stable")[:, :cfg.keep_k]   # Gauss gate
        keep_idx = np.sort(top, axis=1)                  # bekar info ground, order preserved
        tok_k = tokens[rows, keep_idx]
        flux_k = flux[rows, keep_idx]
        mass = aq[rows, keep_idx] + 0.1
        enclosed = flux_k.sum(1) / (flux.sum(1) + 1e-9)  # enclosed flux / total flux
        X = p["E"][tok_k] * (flux_k / (flux_k.mean(1, keepdims=True) + 1e-9))[..., None]

        # ===== Step 2: Conservation Check [Rotational] =====
        A = 0.5 * (p["S"] - p["S"].T)                    # skew-symmetric
        I = np.eye(cfg.d)
        R = np.linalg.solve(I - A, I + A)                # Cayley transform -> orthogonal
        X1 = X @ R                                       # pure rotation: L unchanged
        X2 = X1 + cfg.torque * np.tanh(X1 @ p["Wt"])     # torque: L changes
        L_in = self._ang_mom(X, mass)
        L2 = self._ang_mom(X2, mass)
        dL_rel = np.abs(L2 - L_in) / (L_in + 1e-9)       # dL/dt != 0 ?
        X3 = X2 * np.sqrt(L_in / (L2 + 1e-9))[:, None, None]  # angular impulse restores L

        # ===== Step 3: Logic Induce [EMI] =====
        Phi = X3 @ p["B"]                                # flux through each loop
        Phi_prev = np.concatenate([np.zeros_like(Phi[:, :1]), Phi[:, :-1]], 1)
        emf = -(Phi - Phi_prev)                          # e = -dPhi/dt  (dt = 1 word)

        # ===== Step 4: Solve [Calculus + P&C + Complex] =====
        re = np.einsum("ck,bkm->bcm", self.M, emf)       # P&C: every subset = one case
        im = np.einsum("ck,bkm->bcm", self.M, Phi)
        z = re + 1j * im                                 # phasor (EMF + i*flux)
        f, theta = np.abs(z), np.angle(z)                # locus: modulus & phase
        h = 1.0 / (cfg.m_f - 1)
        f_sym = 0.5 * (f + f[..., ::-1])                 # King: f(x) -> f(a+b-x)
        area = self._cumtrapz(f, h)[..., -1]
        area_k = self._cumtrapz(f_sym, h)[..., -1]
        D["king_err"] = float(np.abs(area - area_k).max())
        sol = np.concatenate([self._cumtrapz(f, h), area_k[..., None],
                              np.cos(theta), np.sin(theta)], -1)          # (B, C, 3m+1)
        prior = np.einsum("ck,bk->bc", self.M, flux_k)
        prior = prior / prior.sum(1, keepdims=True)
        logit = np.log(prior + 1e-9) + np.concatenate([re, im], -1) @ p["u"]
        logit -= logit.max(1, keepdims=True)
        prob = np.exp(logit)
        prob /= prob.sum(1, keepdims=True)               # Bayes posterior over cases

        # ===== Step 5: Chemistry Filter [Bonding + GOC] =====
        P = cfg.n_pairs
        alpha = sol @ p["Aalpha"]
        beta = np.abs(sol @ p["Ah"])
        E_orb = np.concatenate([alpha - beta, alpha + beta], -1)   # Huckel: alpha -/+ beta
        is_bond = np.arange(2 * P) < P
        n_e = np.clip(np.floor(sigmoid(sol @ p["we"]) * (4 * P + 1)), 0, 4 * P)
        rank = np.argsort(np.argsort(E_orb, -1), -1)               # Aufbau: low energy first
        occ = np.clip(n_e[..., None] - 2 * rank, 0, 2)             # Pauli: max 2 per orbital
        bond_order = ((occ * is_bond).sum(-1) - (occ * ~is_bond).sum(-1)) / 2.0   # (b - a)/2
        sc = sol - sol.mean(1, keepdims=True)
        sc = sc / (np.linalg.norm(sc, axis=-1, keepdims=True) + 1e-9)
        sim = sc @ sc.transpose(0, 2, 1)
        thr = 0.2 + 0.6 * sigmoid(p["tau"][0])
        degree = (sim > thr).sum(-1) - 1                           # "alkyl" neighbours
        stab = np.select([degree >= 3, degree == 2, degree == 1], [1.0, 0.66, 0.33], 0.1)
        stable = degree >= 2                                       # 3 deg / 2 deg only
        keep = (bond_order > 0) & stable
        none = ~keep.any(1)                                        # nobody survived
        keep[none, prob[none].argmax(1)] = True                    # fallback: best posterior

        # ===== Step 6: Singularity Core =====
        w = keep * prob * stab * np.maximum(bond_order, 0.5)
        w = w / w.sum(1, keepdims=True)
        point = np.einsum("bc,bcs->bs", w, sol)                    # everything -> one point
        feats = np.concatenate([point, keep.mean(1, keepdims=True),
                                enclosed[:, None]], 1)

        if diag:
            D.update(keep_idx=keep_idx, dL_rel=float(dL_rel.mean()),
                     fallback=float(none.mean()), mean_bo=float(bond_order[keep].mean()),
                     survivors=float(keep.sum(1).mean()),
                     rot_err=float(np.abs(self._ang_mom(X1, mass) - L_in).max()),
                     impulse_err=float(np.abs(self._ang_mom(X3, mass) - L_in).max()),
                     R=R)
            return feats, D
        return feats, {"dL_rel": float(dL_rel.mean()), "fallback": float(none.mean())}

    # -------- physics loss (Step 7 input)
    def physics_loss(self, mse, d):
        c = self.cfg
        return mse + c.lam_cons * d["dL_rel"] + c.lam_fallback * d["fallback"]

    def _fitness_loss(self, theta, tokens, y, n_fit):
        feats, d = self.forward(tokens, unflatten(theta, self.cfg))
        if not np.all(np.isfinite(feats)):
            return 1e3
        head = RidgeHead().fit(feats[:n_fit], y[:n_fit])
        mse = float(np.mean((head.predict(feats[n_fit:]) - y[n_fit:]) ** 2))
        return self.physics_loss(mse, d)

    # ===== Step 7: Self-Evolution =====
    def self_evolve(self, tokens, y, pop=48, lr=0.05):
        """One generation: mutate genome, score every child by physics_loss,
        move toward the better ones (ES + Adam) and self-adapt the step size."""
        theta = flatten(self.p, self.cfg)
        n_fit = int(0.65 * len(tokens))
        base = self._fitness_loss(theta, tokens, y, n_fit)
        eps = self.rng.standard_normal((pop // 2, theta.size))
        eps = np.concatenate([eps, -eps])                 # antithetic sampling
        losses = np.array([self._fitness_loss(theta + self.sigma * e, tokens, y, n_fit)
                           for e in eps])
        ranks = np.argsort(np.argsort(-losses))           # low loss -> high rank
        shaped = ranks / (len(ranks) - 1) - 0.5
        grad = (shaped[:, None] * eps).sum(0) / (len(eps) * self.sigma)   # ascent dir
        if self._adam_m is None:
            self._adam_m, self._adam_v = np.zeros_like(theta), np.zeros_like(theta)
        self._t += 1
        self._adam_m = 0.9 * self._adam_m + 0.1 * grad
        self._adam_v = 0.999 * self._adam_v + 0.001 * grad ** 2
        mh = self._adam_m / (1 - 0.9 ** self._t)
        vh = self._adam_v / (1 - 0.999 ** self._t)
        theta = theta + lr * mh / (np.sqrt(vh) + 1e-8) - 1e-4 * theta
        self.p = unflatten(theta, self.cfg)
        success = float((losses < base).mean())           # 1/5-th style rule
        self.sigma = float(np.clip(self.sigma * (1.03 if success > 0.2 else 0.985), 0.03, 0.3))
        return base, float(losses.min())

    # -------- train / predict
    def fit(self, tokens, y, generations=200, batch=400, pop=48, verbose=True, log_every=20):
        t0 = time.time()
        for g in range(1, generations + 1):
            idx = self.rng.choice(len(tokens), min(batch, len(tokens)), replace=False)
            base, best = self.self_evolve(tokens[idx], y[idx], pop=pop)
            if verbose and (g % log_every == 0 or g == 1):
                print(f"  gen {g:4d} | physics_loss {base:.4f} (best child {best:.4f}) "
                      f"| sigma {self.sigma:.3f} | {time.time() - t0:5.1f}s")
        feats, _ = self.forward(tokens)
        self.head = RidgeHead().fit(feats, y)             # final head on all data
        return self

    def predict(self, tokens):
        feats, _ = self.forward(tokens)
        return np.sign(self.head.predict(feats))


# ------------------------------------------------------------------ toy problem
def make_needle_data(n, cfg, rng):
    """Words 0-3 are signal (0,1 = '+', 2,3 = '-'), the rest are noise.
    3 signal words hide among noise; label = majority polarity of the signal words."""
    L = cfg.seq_len
    tokens = rng.integers(4, cfg.vocab, size=(n, L))
    where = np.argsort(rng.random((n, L)), axis=1)[:, :3]
    sig = rng.integers(0, 4, size=(n, 3))
    r = np.arange(n)[:, None]
    tokens[r, where] = sig
    y = np.where(np.where(sig < 2, 1, -1).sum(1) > 0, 1.0, -1.0)
    is_sig = np.zeros((n, L), bool)
    is_sig[r, where] = True
    return tokens, y, is_sig


def signal_recall(model, tokens, is_sig):
    _, d = model.forward(tokens, diag=True)
    r = np.arange(len(tokens))[:, None]
    return float(is_sig[r, d["keep_idx"]].sum(1).mean() / 3.0)


# ------------------------------------------------------------------ invariants
def selftest():
    cfg = Config()
    m = MadScienX5(cfg, seed=1)
    tok, y, _ = make_needle_data(64, cfg, np.random.default_rng(1))
    feats, d = m.forward(tok, diag=True)
    R = d["R"]
    assert np.allclose(R.T @ R, np.eye(cfg.d), atol=1e-9), "rotation not orthogonal"
    assert d["rot_err"] < 1e-8, f"rotation changed L: {d['rot_err']}"
    assert d["impulse_err"] < 1e-8, f"angular impulse failed: {d['impulse_err']}"
    assert d["king_err"] < 1e-9, f"King property violated: {d['king_err']}"
    assert np.all(np.isfinite(feats)), "non-finite features"
    assert feats.shape == (64, 3 * cfg.m_f + 1 + 2)
    print("selftest OK:")
    print(f"  rotation keeps L        (max err {d['rot_err']:.1e})")
    print(f"  angular impulse fixes L (max err {d['impulse_err']:.1e})")
    print(f"  King property holds     (max err {d['king_err']:.1e})")
    print(f"  cases per problem (P&C) {len(m.subsets)}, "
          f"avg survivors after chemistry filter {d['survivors']:.1f}")


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--gens", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    selftest()
    if a.test:
        raise SystemExit

    cfg = Config()
    rng = np.random.default_rng(a.seed)
    Xtr, ytr, _ = make_needle_data(2000, cfg, rng)
    Xte, yte, sig_te = make_needle_data(1000, cfg, rng)

    model = MadScienX5(cfg, seed=a.seed)

    # before evolution (random genome, ridge head only)
    f0, _ = model.forward(Xtr)
    h0 = RidgeHead().fit(f0, ytr)
    acc0 = float(np.mean(np.sign(h0.predict(model.forward(Xte)[0])) == yte))
    rec0 = signal_recall(model, Xte, sig_te)
    print(f"\nBefore self-evolution: test acc {acc0:.3f} | field-filter signal recall {rec0:.3f}"
          f" (random keep = {cfg.keep_k / cfg.seq_len:.3f})")

    print("\nSelf-evolving ...")
    model.fit(Xtr, ytr, generations=a.gens)

    acc = float(np.mean(model.predict(Xte) == yte))
    rec = signal_recall(model, Xte, sig_te)
    _, d = model.forward(Xte, diag=True)
    print(f"\nAfter  self-evolution: test acc {acc:.3f} | field-filter signal recall {rec:.3f}")
    print(f"  uncorrected dL/L before angular impulse: {d['dL_rel']:.3f}")
    print(f"  chemistry filter: {d['survivors']:.1f} survivors/problem, "
          f"mean bond order {d['mean_bo']:.2f}, fallback rate {d['fallback']:.2%}")

    # reference: plain bag-of-words ridge (this toy task is linearly easy; it is only a sanity bar)
    bow = lambda T: np.stack([np.bincount(t, minlength=cfg.vocab) for t in T]).astype(float)
    hb = RidgeHead().fit(bow(Xtr), ytr)
    print(f"  reference bag-of-words ridge acc: {np.mean(np.sign(hb.predict(bow(Xte))) == yte):.3f}")
