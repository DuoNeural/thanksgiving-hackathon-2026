"""
Mathematical, Causality, Manifold-Norm, and Hardware Smoke Verification for CartanPredLM.
"""
import math
import time
import torch
import aura_net
from aura_net import CartanPredLM, CartanRouter4, chunked_reflexive_delta_scan


def get_device():
    if torch.cuda.is_available():
        idx = 1 if torch.cuda.device_count() > 1 else 0
        return torch.device(f"cuda:{idx}")
    return torch.device("cpu")


def test_cartan_manifold_bounds(dev):
    for label, kw in [
        ("v1", dict(sinkhorn_steps=10, gamma_floor=0.0, w_out_max=2.0)),
        ("v2", dict(sinkhorn_steps=5, gamma_floor=0.08, w_out_max=1.5)),
    ]:
        router = CartanRouter4(512, n_heads=8, **kw).to(dev)
        X = torch.randn(2, 64, 4, 512, device=dev)
        err = torch.rand(2, 64, 8, device=dev)
        h_in, C_t, w_out, gamma = router.mix_and_read(X, err)
        svals = torch.linalg.svdvals(C_t.reshape(-1, 4, 4))
        max_sval = svals.max().item()
        min_g = gamma.min().item()
        print(f"[Cartan Norm {label}] max ||C_t||_2 = {max_sval:.6f} (<= 1.0), min_gamma={min_g:.4f}, max_wout={w_out.max().item():.4f}")
        assert max_sval <= 1.001, f"Cartan transition spectral norm exceeded 1: {max_sval}"
        assert min_g >= kw["gamma_floor"] - 1e-6

    # Adversarial Householder-Delta stability check: 100% aligned keys with beta=1.99
    q = torch.ones(2, 128, 8, 64, device=dev) / 8.0
    k = torch.ones(2, 128, 8, 64, device=dev) / 8.0
    v = torch.randn(2, 128, 8, 64, device=dev)
    a = torch.full((2, 128, 8), 0.99, device=dev)
    b = torch.full((2, 128, 8), 1.99, device=dev)
    for cs in (64, 32):
        out_d, surp_d = chunked_reflexive_delta_scan(q, k, v, a, b, chunk_size=cs)
        assert torch.isfinite(out_d).all() and torch.isfinite(surp_d).all(), f"Delta scan overflowed (cs={cs})!"
        print(f"[Delta Stability cs={cs}] beta=1.99 aligned keys finite! max|out|={out_d.abs().max().item():.2f}")


def test_causality_and_contract(dev):
    import os
    v1_ckpt = "/workspace/hackathon/aura/runs/cartan_v1_30k/ckpt_best.pt"
    if os.path.exists(v1_ckpt):
        ckpt = torch.load(v1_ckpt, map_location="cpu", weights_only=False)
        m1 = CartanPredLM(**ckpt["config"])
        m1.load_state_dict(ckpt["model"], strict=True)
        print(f"[v1 Compat] strict=True loaded {v1_ckpt} cleanly ({sum(p.numel() for p in m1.parameters()):,} params)!")
        del m1, ckpt

    torch.manual_seed(42)
    v2_cfg = dict(
        dim=512, depth=16, ffn_dim=2048, d_emb=512, attn_every=2, chunk_size=32,
        dropout=0.10, sinkhorn_steps=5, gamma_floor=0.08, w_out_max=1.5, qk_norm=True, v_resid=True,
    )
    m = CartanPredLM(**v2_cfg).to(dev)
    n_p = sum(p.numel() for p in m.parameters())
    print(f"[Params v2] CartanPredLM(d_emb=512, dim=512, depth=16, ffn=2048, cs=32, qk_norm, v_resid) = {n_p:,} ({n_p/1e6:.2f}M)")
    assert n_p <= 100_000_000, f"Exceeded 100M cap: {n_p}"

    m.eval()
    x1 = torch.randint(0, 49152, (2, 128), device=dev)
    x2 = x1.clone()
    x2[:, 65:] = torch.randint(0, 49152, (2, 63), device=dev)
    with torch.no_grad():
        l1 = m(x1)
        l2 = m(x2)
    diff = (l1[:, :65] - l2[:, :65]).abs().max().item()
    print(f"[Causality v2] prefix max diff across cut=65: {diff:.2e}")
    assert diff < 1e-4, f"Causality leak! diff={diff}"

    # Verify chunked CE loss == explicit cross_entropy(forward(x), y) in eval mode (no dropout noise)
    y = torch.randint(0, 49152, (2, 128), device=dev)
    with torch.no_grad():
        ce_chunk, _ = m.forward_loss(x1, y)
        logits = m(x1)
        ce_ref = torch.nn.functional.cross_entropy(logits.reshape(-1, 49152), y.reshape(-1))
    loss_diff = abs(ce_chunk.item() - ce_ref.item())
    print(f"[Head Parity v2] chunk_ce={ce_chunk.item():.4f} ref_ce={ce_ref.item():.4f} diff={loss_diff:.2e} lnV={math.log(49152):.4f}")
    assert loss_diff < 1e-3, f"Chunked vs explicit CE mismatch: {loss_diff}"

    # Verify backward + dropout in train mode
    m.train()
    ce_train, _ = m.forward_loss(x1, y)
    ce_train.backward()
    grad_norms = [p.grad.abs().max().item() for p in m.parameters() if p.grad is not None]
    assert all(math.isfinite(g) for g in grad_norms) and len(grad_norms) == sum(1 for _ in m.parameters())
    print(f"[Grad Check v2] all {len(grad_norms)} params received finite grads (max={max(grad_norms):.4f})")
    del m, x1, x2, l1, l2, y, ce_chunk, ce_train, logits, ce_ref
    if dev.type == "cuda":
        torch.cuda.empty_cache()


def bench_vram_and_speed(dev):
    if dev.type != "cuda":
        return
    for ckpt_mode, bsz in [(True, 16)]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)
        m = CartanPredLM(
            dim=512, depth=16, ffn_dim=2048, d_emb=512, attn_every=2, chunk_size=32,
            dropout=0.10, sinkhorn_steps=5, gamma_floor=0.08, w_out_max=1.5, qk_norm=True, v_resid=True,
            grad_ckpt=ckpt_mode,
        ).to(dev)
        x = torch.randint(0, 49152, (bsz, 512), device=dev)
        y = torch.randint(0, 49152, (bsz, 512), device=dev)
        for _ in range(2):
            m.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = m.forward_loss(x, y)
            loss.backward()
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        n_iters = 5
        for _ in range(n_iters):
            m.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = m.forward_loss(x, y)
            loss.backward()
        torch.cuda.synchronize(dev)
        dt = time.perf_counter() - t0
        peak_mb = torch.cuda.max_memory_allocated(dev) / 1e6
        tok_s = (n_iters * bsz * 512) / dt
        print(f"[Bench v2] grad_ckpt={str(ckpt_mode):<5} batch={bsz:<2} | {tok_s:,.0f} tok/s | peak_vram={peak_mb:,.1f} MB")

        # Inference speed probe matching eval_hackathon.py (T=32, 64 gen steps)
        m.eval()
        prompt = torch.randint(0, 49152, (1, 32), device=dev)
        with torch.no_grad():
            for _ in range(3):
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    _ = m(prompt)
            torch.cuda.synchronize(dev)
            ti0 = time.perf_counter()
            for _ in range(64):
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = m(prompt)
                prompt = torch.cat([prompt[:, 1:], logits[:, -1:, :].argmax(-1)], dim=1)
            torch.cuda.synchronize(dev)
            inf_tps = 64 / (time.perf_counter() - ti0)
        print(f"[Infer v2] T=32 window | {inf_tps:.2f} tok/s")
        del m, x, y


if __name__ == "__main__":
    dev = get_device()
    print(f"Testing on device: {dev}")
    test_cartan_manifold_bounds(dev)
    test_causality_and_contract(dev)
    bench_vram_and_speed(dev)
    print("\nALL CARTAN-PRED MATH, CAUSALITY & VRAM BENCHMARKS PASSED!")
