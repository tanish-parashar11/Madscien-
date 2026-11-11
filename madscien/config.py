"""Model configuration for MADSCIEN-X5 LM."""
import math
from dataclasses import asdict, dataclass


@dataclass
class X5Config:
    # ---- vocabulary / sequence
    vocab_size: int = 263            # 256 bytes + BOS/EOS/PAD + 4 chat tokens (see data.py)
    block_size: int = 256            # max training context

    # ---- skeleton
    n_layer: int = 6
    d_model: int = 256
    n_heads: int = 4                 # parallel X5 cores per layer (core weights are shared)
    ffn_mult: int = 0                # 0 = pure X5 blocks (exact algorithm); >0 adds SwiGLU
    dropout: float = 0.0
    tie_embeddings: bool = True
    grad_checkpoint: bool = False    # trade compute for memory (recommended for big runs)

    # ---- hybrid global attention (standard transformer layers mixed in between X5 layers)
    attn_every: int = 2              # every k-th layer = causal GQA attention + SwiGLU (0 = none)
    attn_heads: int = 8
    attn_kv_heads: int = 2           # grouped-query attention: attn_heads % attn_kv_heads == 0
    attn_ffn_mult: int = 4
    rope_base: float = 10000.0

    # ---- digit positions ("Abacus"-style): each digit token also gets an embedding of its place
    # inside its number (1st, 2nd, ... digit). In training every place is shifted by a random
    # 0..digit_offset per sequence, so places past the longest training number are trained too.
    digit_positions: int = 0         # size of the place table (0 = off)
    digit_offset: int = 0            # largest random shift used in training

    # ---- the algorithm (Steps 1-6) - same names/defaults as the numpy reference
    window: int = 32                 # one "problem" = the last `window` tokens (causal)
    dilation_cycle: bool = True      # l-th X5 layer looks at every 2^(l mod c)-th token
    core_dim: int = 16               # d    : dimension of the charge space
    keep_k: int = 6                  # k    : Gauss gate keeps the k strongest charges
    flux_loops: int = 8              # m_f  : flux loops == grid points of the integral
    max_subset: int = 3              # P&C  : all subsets of size 1..max_subset are cases
    n_orbital_pairs: int = 4         # MOT  : bonding/antibonding orbital pairs
    coulomb_k: float = 0.1           # F = k q1 q2 / r^2
    eps: float = 0.05                # softening of 1/r^2
    omega: float = 1.0               # spin rate, L = I * omega
    torque: float = 0.5              # strength of the non-conservative step in Step 2

    # ---- straight-through temperatures (backward pass only; forward is hard/exact)
    rank_temp: float = 0.25
    st_temp: float = 0.5
    sim_temp: float = 0.1

    # ---- physics_loss weights (Step 7); lam_* are adapted online by SelfEvolver
    lam_cons: float = 0.01
    lam_fallback: float = 0.01

    def __post_init__(self):
        assert self.keep_k >= 2, "keep_k must be >= 2"
        assert self.window >= self.keep_k, "window must be >= keep_k"
        assert self.flux_loops >= 2, "flux_loops must be >= 2"
        assert 1 <= self.max_subset <= self.keep_k, "need 1 <= max_subset <= keep_k"
        if self.attn_every > 0:
            assert self.d_model % self.attn_heads == 0, "d_model must divide by attn_heads"
            assert self.attn_heads % self.attn_kv_heads == 0, "attn_heads must divide by attn_kv_heads"
            assert (self.d_model // self.attn_heads) % 2 == 0, "RoPE needs an even head dim"
            assert self.attn_ffn_mult >= 1
        assert self.digit_positions == 0 or self.digit_positions > self.digit_offset + 1, \
            "digit_positions must leave room for places beyond the largest offset"
        assert self.n_x5_layers >= 1, "need at least one X5 layer (attn_every=1 would remove all)"

    # ---- derived
    @property
    def sol_dim(self):               # [cumtrapz(f) | area | cos(theta) | sin(theta)]
        return 3 * self.flux_loops + 1

    @property
    def feat_dim(self):              # + keep fraction + enclosed flux
        return self.sol_dim + 2

    def is_attn_layer(self, i):
        return self.attn_every > 0 and (i + 1) % self.attn_every == 0

    @property
    def n_x5_layers(self):
        return sum(not self.is_attn_layer(i) for i in range(self.n_layer))

    @property
    def n_cases(self):
        return sum(math.comb(self.keep_k, r) for r in range(1, self.max_subset + 1))

    def x5_index(self, i):
        """Position of layer i among the X5 layers only (attention layers are skipped)."""
        return sum(not self.is_attn_layer(j) for j in range(i))

    def dilation(self, layer):
        """Dilation of the `layer`-th X5 layer (counted among X5 layers, see x5_index)."""
        if not self.dilation_cycle:
            return 1
        max_dil = max(1, self.block_size // self.window)
        n_cycle = int(math.floor(math.log2(max_dil))) + 1
        return 2 ** (layer % n_cycle)

    def receptive_field(self):
        if self.attn_every > 0:
            return self.block_size            # attention layers see the whole context
        return 1 + sum((self.window - 1) * self.dilation(l) for l in range(self.n_x5_layers))

    def to_dict(self):
        return asdict(self)


PRESETS = {
    "tiny": dict(n_layer=4, d_model=128, n_heads=4, window=16, keep_k=5, flux_loops=6,
                 max_subset=3, block_size=128),
    "small": dict(n_layer=8, d_model=384, n_heads=8, window=32, ffn_mult=4, block_size=256),
    "base": dict(n_layer=12, d_model=768, n_heads=12, window=32, ffn_mult=4, block_size=512,
                 attn_heads=12, attn_kv_heads=4, grad_checkpoint=True),
}


def make_config(preset=None, **overrides):
    kw = dict(PRESETS[preset]) if preset else {}
    kw.update({k: v for k, v in overrides.items() if v is not None})
    return X5Config(**kw)
