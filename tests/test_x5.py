"""
Run first:   pytest -q

1. test_core_matches_numpy_reference  -> torch X5Core == your numpy algorithm (the 'hubhu copy' check)
2. physics invariants                 -> rotation keeps L, angular impulse restores L, King property
3. causality                          -> future tokens never change past logits
4. gradients                          -> every parameter gets a finite gradient
5. overfit + generation sanity
"""
import numpy as np
import pytest
import torch

from madscien.config import X5Config
from madscien.evolve import SelfEvolver
from madscien.model import MadScienLM, physics_loss
from madscien.x5_core import X5Core, cumtrapz, normalize_weights


def tiny_cfg(**kw):
    base = dict(vocab_size=64, block_size=32, n_layer=2, d_model=64, n_heads=2, window=8,
                core_dim=8, keep_k=4, flux_loops=6, max_subset=2, n_orbital_pairs=2)
    base.update(kw)
    return X5Config(**base)


# ------------------------------------------------------------------ 1. exact algorithm copy
def test_core_matches_numpy_reference():
    import madscien_x5_numpy as ref

    W, d, k, m, P = 12, 6, 4, 6, 4
    ncfg = ref.Config(vocab=20, seq_len=W, d=d, keep_k=k, m_f=m, max_subset=3, n_pairs=P)
    npm = ref.MadScienX5(ncfg, seed=3)
    tokens = np.random.default_rng(0).integers(0, 20, size=(48, W))
    feats_np, _ = npm.forward(tokens)

    cfg = X5Config(vocab_size=20, window=W, core_dim=d, keep_k=k, flux_loops=m, max_subset=3,
                   n_orbital_pairs=P, n_layer=1, d_model=16, n_heads=1)
    core = X5Core(cfg).double()
    with torch.no_grad():
        for name in ["S", "Wt", "B", "u", "Aalpha", "Ah", "we", "tau"]:
            getattr(core, name).copy_(torch.from_numpy(npm.p[name]))
        q = torch.from_numpy(npm.p["charge"][tokens])
        X = torch.from_numpy(npm.p["E"][tokens])
        feats_t, _ = core(q, X)
    np.testing.assert_allclose(feats_t.numpy(), feats_np, atol=1e-5, rtol=1e-5)


# ------------------------------------------------------------------ 2. physics invariants
def test_rotation_is_orthogonal_and_conserves_L():
    cfg = tiny_cfg()
    core = X5Core(cfg).double()
    d = cfg.core_dim
    A = 0.5 * (core.S - core.S.t())
    I = torch.eye(d, dtype=torch.double)
    R = torch.linalg.solve(I - A, I + A)
    assert torch.allclose(R.t() @ R, I, atol=1e-9)
    X = torch.randn(5, 4, d, dtype=torch.double)
    m = torch.rand(5, 4, dtype=torch.double) + 0.1
    L = lambda Z: (m * Z.pow(2).sum(-1)).sum(-1)
    assert torch.allclose(L(X), L(X @ R), atol=1e-9)


def test_core_runs_and_king_property():
    cfg = tiny_cfg()
    core = X5Core(cfg).double()
    q = torch.randn(16, cfg.window, dtype=torch.double)
    X = torch.randn(16, cfg.window, cfg.core_dim, dtype=torch.double)
    feats, aux = core(q, X)
    assert torch.isfinite(feats).all() and feats.shape == (16, cfg.feat_dim)
    assert aux.shape == (5,)
    # King property: integral of f equals integral of the symmetrised f
    f = torch.rand(7, 3, cfg.flux_loops, dtype=torch.double)
    h = 1.0 / (cfg.flux_loops - 1)
    f_sym = 0.5 * (f + f.flip(-1))
    assert torch.allclose(cumtrapz(f, h)[..., -1], cumtrapz(f_sym, h)[..., -1], atol=1e-12)


# ------------------------------------------------------------------ 3. causality
@pytest.mark.parametrize("cycle,attn", [(False, 0), (True, 0), (True, 2), (False, 3)])
def test_causality(cycle, attn):
    torch.manual_seed(0)
    cfg = tiny_cfg(dilation_cycle=cycle, n_layer=3, attn_every=attn)
    model = MadScienLM(cfg).eval()
    a = torch.randint(0, cfg.vocab_size, (2, 24))
    b = a.clone()
    b[:, 12:] = torch.randint(0, cfg.vocab_size, (2, 12))
    with torch.no_grad():
        la, lb = model(a).logits, model(b).logits
    assert torch.allclose(la[:, :12], lb[:, :12], atol=1e-5)
    assert not torch.allclose(la[:, 12:], lb[:, 12:], atol=1e-5)


# ------------------------------------------------------------------ 4. gradients
def test_every_parameter_gets_finite_gradient():
    torch.manual_seed(0)
    cfg = tiny_cfg(ffn_mult=2)
    model = MadScienLM(cfg).train()
    x = torch.randint(0, cfg.vocab_size, (2, 24))
    y = torch.randint(0, cfg.vocab_size, (2, 24))
    out = model(x, y)
    loss = physics_loss(out.loss, out.aux, cfg.lam_cons, cfg.lam_fallback)
    loss.backward()
    assert out.aux.shape == (cfg.n_x5_layers, 5)
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"


def test_normalize_weights_exact_forward_bounded_gradient():
    """Regression: dividing by a near-zero survivor weight sent training gradient norms past 1e10."""
    w = torch.full((1, 4), 1e-10, dtype=torch.float64, requires_grad=True)
    out = normalize_weights(w)
    assert torch.allclose(out, w / (w.sum(-1, keepdim=True) + 1e-9), rtol=1e-12)
    out[0, 0].backward()
    assert w.grad.abs().max() < 1e5, f"gradient exploded: {w.grad.abs().max():.2e}"

    normal = torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float64)
    assert torch.allclose(normalize_weights(normal), normal, rtol=1e-6)


def test_x5_gradients_do_not_reach_earlier_layers():
    """Regression: gradients through the core stopped the model learning from its input."""
    torch.manual_seed(0)
    cfg = tiny_cfg(attn_every=0, ffn_mult=0)
    block = MadScienLM(cfg).blocks[0].train()
    x = torch.randn(2, 16, cfg.d_model, requires_grad=True)
    out, aux = block(x)
    ((out - x).pow(2).sum() + aux[0]).backward()
    assert x.grad is None or (x.grad == 0).all()
    for name in ("q_proj", "x_proj", "out_proj"):
        assert getattr(block, name).weight.grad.abs().sum() > 0, f"{name} stopped training"


def test_grad_checkpoint_matches_plain():
    torch.manual_seed(0)
    cfg = tiny_cfg()
    m1 = MadScienLM(cfg).train()
    cfg2 = tiny_cfg(grad_checkpoint=True)
    m2 = MadScienLM(cfg2).train()
    m2.load_state_dict(m1.state_dict())
    x = torch.randint(0, cfg.vocab_size, (2, 24))
    y = torch.randint(0, cfg.vocab_size, (2, 24))
    l1, l2 = m1(x, y).loss, m2(x, y).loss
    l1.backward()
    l2.backward()
    assert torch.allclose(l1, l2, atol=1e-6)
    g1 = m1.blocks[0].out_proj.weight.grad
    g2 = m2.blocks[0].out_proj.weight.grad
    assert torch.allclose(g1, g2, atol=1e-5)


# ------------------------------------------------------------------ 5. sanity: learns + generates
def test_overfits_a_tiny_batch_with_self_evolver():
    torch.manual_seed(0)
    cfg = tiny_cfg(tie_embeddings=False)
    model = MadScienLM(cfg).train()
    opt = model.configure_optimizers(0.0, 3e-3, (0.9, 0.95), "cpu")
    evolver = SelfEvolver(model, opt, 3e-3, 3e-4, 5, 200, adapt=True, adapt_every=50)
    perm = torch.randperm(cfg.vocab_size)[:33]
    x, y = perm[:32].unsqueeze(0), perm[1:33].unsqueeze(0)
    first = last = None
    for it in range(200):
        out = model(x, y)
        loss = physics_loss(out.loss, out.aux, evolver.lam_cons, evolver.lam_fallback)
        evolver.backward(loss)
        evolver.step(it, out.loss.item(), out.aux[:, 0].mean().item())
        first = out.loss.item() if first is None else first
        last = out.loss.item()
    assert last < 0.5 * first, f"loss did not fall enough: {first:.3f} -> {last:.3f}"


def test_evolver_ignores_a_non_finite_loss_when_adapting():
    model = MadScienLM(tiny_cfg())
    opt = model.configure_optimizers(0.0, 1e-3, (0.9, 0.95), "cpu")
    evolver = SelfEvolver(model, opt, 1e-3, 1e-4, 1, 10)
    evolver._observe(2.0, 0.1)
    evolver._observe(float("nan"), 0.1)
    evolver._observe(2.0, float("inf"))
    assert evolver._ema == 2.0 and evolver._updates == 1


def test_generate_shapes_and_sampling_modes():
    torch.manual_seed(0)
    cfg = tiny_cfg()
    model = MadScienLM(cfg)
    idx = torch.randint(0, cfg.vocab_size, (1, 5))
    out = model.generate(idx, 6, temperature=0.8, top_k=10, top_p=0.9)
    assert out.shape == (1, 11)
    out = model.generate(idx, 4, temperature=0.0)
    assert out.shape == (1, 9)
    assert out.min() >= 0 and out.max() < cfg.vocab_size


def test_config_dilations_and_receptive_field():
    cfg = X5Config(block_size=256, window=32, n_layer=6)
    assert [cfg.dilation(i) for i in range(6)] == [1, 2, 4, 8, 1, 2]
    assert cfg.receptive_field() > 31 * 6


@pytest.mark.parametrize("preset", ["tiny", "small", "base"])
def test_hybrid_x5_layers_walk_the_whole_dilation_cycle(preset):
    """Attention layers must not eat dilation slots: X5 layers get 1, 2, 4, ... in order."""
    from madscien.config import make_config
    from madscien.model import X5Block
    cfg = make_config(preset)
    model = MadScienLM(cfg)
    dils = [b.dilation for b in model.blocks if isinstance(b, X5Block)]
    assert dils == [cfg.dilation(j) for j in range(cfg.n_x5_layers)]
    assert dils[:3] == [1, 2, 4][:len(dils)]


def test_generate_batch_pads_finished_rows_with_eos():
    torch.manual_seed(0)
    cfg = tiny_cfg()
    model = MadScienLM(cfg)
    idx = torch.randint(0, cfg.vocab_size, (4, 3))
    eos = 7
    out = model.generate(idx, 40, temperature=1.5, eos_id=eos)
    for row in out[:, 3:]:
        hits = (row == eos).nonzero()
        if len(hits):
            assert (row[hits[0, 0]:] == eos).all()        # nothing after a row's first EOS


def test_constrained_generate_never_breaks_brackets():
    from madscien.constraints import DelimiterConstraint
    torch.manual_seed(0)
    cfg = tiny_cfg(vocab_size=259, block_size=64)
    model = MadScienLM(cfg)
    c = DelimiterConstraint()
    idx = torch.tensor([[256]])                       # BOS
    out = model.generate(idx, 40, temperature=1.0, constraint=c, eos_id=257)
    assert c.violations == 0
    assert (out[0, 1:] < 256).all() or c.can_end()    # EOS only when nothing is left open


def test_attention_layers_give_global_context():
    """X5-only model with a short window cannot see token 0 from position 31; with attention it can."""
    torch.manual_seed(0)
    a = torch.randint(0, 64, (1, 32))
    b = a.clone()
    b[0, 0] = (a[0, 0] + 1) % 64
    local = MadScienLM(tiny_cfg(n_layer=2, attn_every=0, dilation_cycle=False)).eval()
    hybrid = MadScienLM(tiny_cfg(n_layer=2, attn_every=2)).eval()
    with torch.no_grad():
        assert torch.allclose(local(a).logits[:, 31], local(b).logits[:, 31], atol=1e-6)
        assert not torch.allclose(hybrid(a).logits[:, 31], hybrid(b).logits[:, 31], atol=1e-6)


def test_hybrid_layout_and_rope_is_relative():
    from madscien.model import AttnBlock, CausalSelfAttention, X5Block
    cfg = tiny_cfg(n_layer=4, attn_every=2)
    model = MadScienLM(cfg)
    kinds = [type(b) for b in model.blocks]
    assert kinds == [X5Block, AttnBlock, X5Block, AttnBlock] and cfg.n_x5_layers == 2
    att = CausalSelfAttention(cfg)
    hd = cfg.d_model // cfg.attn_heads
    q = torch.randn(1, 1, 1, hd).expand(1, 1, 10, hd).contiguous()
    k = torch.randn(1, 1, 1, hd).expand(1, 1, 10, hd).contiguous()
    rq, rk = att._rope(q, 10), att._rope(k, 10)
    dot = lambda i, j: (rq[0, 0, i] * rk[0, 0, j]).sum()
    assert torch.allclose(dot(5, 3), dot(8, 6), atol=1e-5)


@pytest.mark.parametrize("cycle", [False, True])
def test_cached_decoding_matches_full_forward(cycle):
    """Prefill + one-token-at-a-time decoding (KV cache + X5 window cache) == one full forward."""
    torch.manual_seed(0)
    cfg = tiny_cfg(n_layer=4, attn_every=2, dilation_cycle=cycle, block_size=32)
    model = MadScienLM(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (2, 24))
    with torch.no_grad():
        full = model(idx).logits
        caches = model.new_cache()
        parts = [model(idx[:, :10], caches=caches).logits]
        for t in range(10, 24):
            parts.append(model(idx[:, t:t + 1], caches=caches).logits)
        inc = torch.cat(parts, dim=1)
    assert torch.allclose(full, inc, atol=1e-4)


def test_generate_with_and_without_cache_agree_when_greedy():
    torch.manual_seed(0)
    model = MadScienLM(tiny_cfg(n_layer=4, attn_every=2))
    idx = torch.randint(0, 64, (1, 6))
    a = model.generate(idx, 12, temperature=0.0, use_cache=True)
    b = model.generate(idx, 12, temperature=0.0, use_cache=False)
    assert torch.equal(a, b)


def test_generate_survives_context_overflow_and_repetition_penalty():
    torch.manual_seed(0)
    cfg = tiny_cfg(block_size=32)
    model = MadScienLM(cfg)
    idx = torch.randint(0, 64, (1, 20))
    out = model.generate(idx, 60, temperature=0.8, top_p=0.9, repetition_penalty=1.2)   # crosses block_size
    assert out.shape == (1, 80) and out.max() < 64


def test_loss_ignores_masked_targets():
    import torch.nn.functional as F
    torch.manual_seed(0)
    cfg = tiny_cfg()
    model = MadScienLM(cfg).eval()
    x = torch.randint(0, 64, (2, 16))
    y = torch.randint(0, 64, (2, 16))
    y[:, :8] = -100
    out = model(x, y)
    manual = F.cross_entropy(out.logits[:, 8:].reshape(-1, 64), y[:, 8:].reshape(-1))
    assert torch.allclose(out.loss, manual, atol=1e-6)


# ------------------------------------------------------------------ digit positions
def test_digit_places_count_each_digit_inside_its_number():
    from madscien.model import digit_places
    text = "x 123, 45 6"
    idx = torch.tensor([list(text.encode())])
    want = [0, 0, 1, 2, 3, 0, 0, 1, 2, 0, 1]
    assert digit_places(idx).tolist() == [want]
    for cut in range(1, len(text)):                     # split anywhere, e.g. inside "123"
        first = digit_places(idx[:, :cut])
        rest = digit_places(idx[:, cut:], first[:, -1])
        assert torch.cat([first, rest], 1).tolist() == [want], cut


def test_cached_decoding_matches_full_forward_with_digit_places():
    torch.manual_seed(0)
    cfg = tiny_cfg(n_layer=4, attn_every=2, digit_positions=12, digit_offset=4)
    model = MadScienLM(cfg).eval()
    idx = torch.randint(44, 60, (2, 24))                # mostly digits, so runs cross every split
    with torch.no_grad():
        full = model(idx).logits
        caches = model.new_cache()
        parts = [model(idx[:, :7], caches=caches).logits]
        for t in range(7, 24):
            parts.append(model(idx[:, t:t + 1], caches=caches).logits)
    assert torch.allclose(full, torch.cat(parts, dim=1), atol=1e-4)


def test_digit_offset_is_random_in_training_only_and_trains_the_place_table():
    torch.manual_seed(0)
    cfg = tiny_cfg(digit_positions=12, digit_offset=6)
    model = MadScienLM(cfg)
    idx = torch.tensor([list(b"9 * 4521 = 40689, 7 + 12 = 19")])          # every byte < tiny vocab 64
    with torch.no_grad():
        model.eval()
        assert torch.equal(model(idx).logits, model(idx).logits)
        model.train()
        outs = [model(idx).logits for _ in range(8)]
    assert any(not torch.allclose(outs[0], o) for o in outs[1:])
    model.train()
    model(idx, targets=idx).loss.backward()
    grad = model.digit_emb.weight.grad
    assert grad[1:].abs().sum() > 0 and grad[0].abs().sum() == 0   # place 0 (not a digit) is never used
    plain = MadScienLM(tiny_cfg())
    assert plain.digit_emb is None and sum(p.numel() for p in plain.parameters()) < model.num_params()
