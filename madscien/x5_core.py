"""
X5Core - Steps 1-6 of the MADSCIEN algorithm as a differentiable PyTorch module.

Input  : one "problem" per row = W tokens, each with a signed charge q and a content vector x.
Output : the singularity point (+2 diagnostics) per problem, and an aux vector of physics stats.

Forward values are EXACTLY the hard algorithm (top-k, floor, Aufbau fill, 3>2>1 degree, thresholds).
Gradients use straight-through estimators (hard forward, sigmoid backward) for every discrete
decision, so the whole pipeline trains with ordinary backprop.

Line-for-line equivalent to reference/madscien_x5_numpy.py (tests/test_x5.py checks that).
"""
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

AUX_NAMES = ("dL_rel", "fallback", "survivors", "mean_bond_order", "enclosed_flux")


def st_step(x, temp):
    """Heaviside in the forward pass, sigmoid(x/temp) gradient in the backward pass."""
    hard = (x > 0).to(x.dtype)
    if not torch.is_grad_enabled():
        return hard
    soft = torch.sigmoid(x / temp)
    return hard + soft - soft.detach()


def case_matrix(k, max_subset):
    """P&C: every subset of the k kept tokens is one case; row = averaging weights."""
    subsets = [c for r in range(1, max_subset + 1) for c in itertools.combinations(range(k), r)]
    M = torch.zeros(len(subsets), k)
    for i, c in enumerate(subsets):
        M[i, list(c)] = 1.0 / len(c)
    return M


def cumtrapz(y, h):
    mid = 0.5 * (y[..., 1:] + y[..., :-1]) * h
    return torch.cat([torch.zeros_like(y[..., :1]), torch.cumsum(mid, dim=-1)], dim=-1)


def normalize_weights(w):
    """w / (sum + 1e-9) in the forward pass, with a larger eps in the backward pass: when every survivor
    has near-zero probability (e.g. padding-only cases in a partial window) the plain gradient explodes."""
    total = w.sum(-1, keepdim=True)
    safe = w / (total + 1e-4)
    return safe + (w / (total + 1e-9) - safe).detach()


class X5Core(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        W, d, m, P, S = cfg.window, cfg.core_dim, cfg.flux_loops, cfg.n_orbital_pairs, cfg.sol_dim

        # --- learnable genome (same names / init scales as the numpy reference)
        self.S = nn.Parameter(torch.randn(d, d) * 0.3)         # skew generator of rotation (Step 2)
        self.Wt = nn.Parameter(torch.randn(d, d) * 0.3)        # torque map            (Step 2)
        self.B = nn.Parameter(torch.randn(d, m) / d ** 0.5)    # flux loops            (Step 3)
        self.u = nn.Parameter(torch.randn(2 * m) * 0.5)        # Bayes log-likelihood  (Step 4)
        self.Aalpha = nn.Parameter(torch.randn(S, P) * 0.3)    # orbital centres       (Step 5)
        self.Ah = nn.Parameter(torch.randn(S, P) * 0.5)        # resonance integrals   (Step 5)
        self.we = nn.Parameter(torch.randn(S) * 0.3)           # electron count        (Step 5)
        self.tau = nn.Parameter(torch.zeros(1))                # GOC similarity thr    (Step 5)

        # --- fixed physics buffers
        pos = torch.arange(W, dtype=torch.float32) / W
        inv_r2 = 1.0 / ((pos[:, None] - pos[None, :]) ** 2 + cfg.eps)
        inv_r2.fill_diagonal_(0.0)
        self.register_buffer("inv_r2", inv_r2, persistent=False)
        self.register_buffer("M", case_matrix(cfg.keep_k, cfg.max_subset), persistent=False)

    def forward(self, q, X):
        """q: (N, W) signed charges.  X: (N, W, d) contents.  Returns feats (N, S+2), aux (5,)."""
        cfg = self.cfg
        k, m, P, d = cfg.keep_k, cfg.flux_loops, cfg.n_orbital_pairs, cfg.core_dim
        N = q.shape[0]
        C = self.M.shape[0]

        # ===== Step 1: Field Filter [Electrostatics + Earthing] =====
        aq = q.abs()
        # flux_i = |q_i| + k * sum_j |q_i||q_j| / r_ij^2        (F = k q1 q2 / r^2)
        flux = aq + cfg.coulomb_k * aq * (aq @ self.inv_r2)
        idx = flux.topk(k, dim=-1).indices.sort(dim=-1).values   # Gauss gate: keep high charge
        flux_k = flux.gather(1, idx)                             # rest is earthed (dropped)
        mass = aq.gather(1, idx) + 0.1
        enclosed = flux_k.sum(-1) / (flux.sum(-1) + 1e-9)        # enclosed flux / total flux
        Xk = X.gather(1, idx.unsqueeze(-1).expand(-1, -1, d))
        Xk = Xk * (flux_k / (flux_k.mean(-1, keepdim=True) + 1e-9)).unsqueeze(-1)

        # ===== Step 2: Conservation Check [Rotational] =====
        I = torch.eye(d, dtype=Xk.dtype, device=Xk.device)
        A = 0.5 * (self.S - self.S.t())
        R = torch.linalg.solve(I - A, I + A)                     # Cayley: orthogonal rotation
        X1 = Xk @ R                                              # rotation alone keeps L
        X2 = X1 + cfg.torque * torch.tanh(X1 @ self.Wt)          # torque: L changes
        L_in = cfg.omega * (mass * Xk.pow(2).sum(-1)).sum(-1)    # L = I*omega, I = sum m r^2
        L2 = cfg.omega * (mass * X2.pow(2).sum(-1)).sum(-1)
        dL_rel = (L2 - L_in).abs() / (L_in + 1e-9)               # dL/dt != 0 ?
        scale = torch.sqrt(L_in.clamp_min(1e-12) / (L2 + 1e-9))
        X3 = X2 * scale[:, None, None]                           # angular impulse restores L

        # ===== Step 3: Logic Induce [EMI] =====
        Phi = X3 @ self.B                                        # flux through each loop
        Phi_prev = F.pad(Phi[:, :-1], (0, 0, 1, 0))
        emf = -(Phi - Phi_prev)                                  # e = -dPhi/dt  (dt = 1 token)

        # ===== Step 4: Solve [Calculus + P&C + Complex] =====
        re = torch.einsum("ck,nkm->ncm", self.M, emf)            # P&C: every subset = a case
        im = torch.einsum("ck,nkm->ncm", self.M, Phi)
        mod = torch.sqrt(re * re + im * im + 1e-12)              # phasor z = re + i*im
        cos_t, sin_t = re / mod, im / mod                        # locus: modulus and phase
        h = 1.0 / (m - 1)
        f_sym = 0.5 * (mod + mod.flip(-1))                       # King: f(x) -> f(a+b-x)
        area_k = cumtrapz(f_sym, h)[..., -1:]
        sol = torch.cat([cumtrapz(mod, h), area_k, cos_t, sin_t], dim=-1)     # (N, C, 3m+1)
        prior = torch.einsum("ck,nk->nc", self.M, flux_k)
        prior = prior / prior.sum(-1, keepdim=True)
        logit = torch.log(prior + 1e-9) + torch.cat([re, im], dim=-1) @ self.u
        prob = torch.softmax(logit, dim=-1)                      # Bayes posterior over cases

        # ===== Step 5: Chemistry Filter [Bonding + GOC] =====
        alpha = sol @ self.Aalpha
        beta = (sol @ self.Ah).abs()
        E_orb = torch.cat([alpha - beta, alpha + beta], dim=-1)  # Huckel: alpha -/+ beta
        z_e = sol @ self.we
        n_hard = torch.floor(torch.sigmoid(z_e) * (4 * P + 1)).clamp(0, 4 * P)
        rank = E_orb.argsort(-1).argsort(-1).to(E_orb.dtype)     # Aufbau: lowest energy first
        if torch.is_grad_enabled():
            n_soft = torch.sigmoid(z_e) * (4 * P)
            n_e = n_hard + (n_soft - n_soft.detach())
            diff = E_orb.unsqueeze(-1) - E_orb.unsqueeze(-2)
            rank_soft = torch.sigmoid(diff / cfg.rank_temp).sum(-1) - 0.5
            rank = rank + (rank_soft - rank_soft.detach())
        else:
            n_e = n_hard
        occ = (n_e.unsqueeze(-1) - 2.0 * rank).clamp(0.0, 2.0)   # Pauli: max 2 per orbital
        bond_order = (occ[..., :P].sum(-1) - occ[..., P:].sum(-1)) / 2.0      # (bonding-anti)/2

        sc = sol - sol.mean(dim=1, keepdim=True)
        sc = sc / torch.sqrt((sc * sc).sum(-1, keepdim=True) + 1e-12)
        sim = sc @ sc.transpose(1, 2)
        thr = 0.2 + 0.6 * torch.sigmoid(self.tau)
        off = 1.0 - torch.eye(C, dtype=sim.dtype, device=sim.device)
        deg = ((sim > thr).to(sim.dtype) * off).sum(-1)          # number of "alkyl" neighbours
        if torch.is_grad_enabled():
            soft_deg = (torch.sigmoid((sim - thr) / cfg.sim_temp) * off).sum(-1)
            deg = deg + (soft_deg - soft_deg.detach())
        stab = (0.1 + 0.23 * st_step(deg - 0.5, cfg.st_temp)     # 0 deg 0.10 | 1 deg 0.33
                + 0.33 * st_step(deg - 1.5, cfg.st_temp)         # 2 deg 0.66
                + 0.34 * st_step(deg - 2.5, cfg.st_temp))        # 3 deg 1.00
        stable = st_step(deg - 1.5, cfg.st_temp)                 # stable = 3 deg or 2 deg
        keep_pre = st_step(bond_order, cfg.st_temp) * stable     # bond_order > 0 and stable
        deficit = (1.0 - keep_pre.sum(-1)).clamp_min(0.0)        # 1 if nobody survived
        none = keep_pre.detach().sum(-1) < 0.5
        onehot = F.one_hot(prob.argmax(-1), C).to(prob.dtype)    # fallback: best posterior
        keep = torch.where(none.unsqueeze(-1), onehot, keep_pre)

        # ===== Step 6: Singularity Core =====
        w = keep * prob * stab * bond_order.clamp_min(0.5)
        w = normalize_weights(w)
        point = torch.einsum("nc,ncs->ns", w, sol)               # everything -> ONE point
        feats = torch.cat([point, keep.mean(-1, keepdim=True), enclosed.unsqueeze(-1)], dim=-1)

        kd = keep.detach()
        aux = torch.stack([
            dL_rel.mean(),                                        # conservation violation
            deficit.mean(),                                       # == fallback rate (has grad)
            kd.sum(-1).mean(),                                    # survivors per problem
            (bond_order.detach() * kd).sum() / (kd.sum() + 1e-9),  # mean bond order of survivors
            enclosed.detach().mean(),
        ])
        return feats, aux
