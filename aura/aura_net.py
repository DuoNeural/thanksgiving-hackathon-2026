"""
CartanPredLM — Aura's DuoNeural Thanksgiving Hackathon 2026 Architecture
=========================================================================
Novel <100M sequence architecture trained cleanly on 8GB RTX 5060 (p41322 GPU1).

Core Mathematical Innovations:
1. Token-Conditional Cartan Manifold Hyper-Connections (M=4 streams, conv(Birkhoff, SO(4))):
   Each token predicts a dynamic 4x4 transition matrix:
       C_t = (1 - gamma_t) * Sinkhorn_3(S_t) + gamma_t * Cayley(A_t)
   where Sinkhorn_3(S_t) lies in the Birkhoff doubly-stochastic polytope (dissipative
   consensus, ||P||_2 <= 1) and Cayley(A_t) = (I - A_t/2)^{-1}(I + A_t/2) with A_t = -A_t^T
   lies in SO(4) (exact isometric gauge braiding, det=+1, ||Q||_2 = 1). By convexity,
   ||C_t||_2 <= 1 unconditionally across arbitrary depth.
2. Reflexive Householder-Delta Associative Memory (beta_t in (0, 2)) + Predictive Surprise:
   Extends rank-1 DeltaNet from erasure-only [0, 1] into generalized Householder reflections
   beta_t in (0, 2) via exact FP32 chunked-64 WY triangular solves, returning both retrieved
   associative output and per-head innovation error u_corr = beta_t (v_t - alpha_t S_{t-1} k_t)
   which directly modulates Cartan SO(4) mixing rate gamma_t in closed loop.
3. Stick-Breaking Cumulative-Decay Causal Attention (G_i - G_j):
   Combines causal depthwise Conv4 filtering, RoPE, output head-gating, and exact rank-1
   cumulative log-forget bias G_i - G_j = sum_{u=j+1}^i log sigmoid(g_u) <= 0 inside SDPA.
4. Quad-Stream Cartan Readout + Chunked Factorized Tied Softmax:
   Concatenates normalized representations from all 4 manifold streams ([4*D] -> d_emb=384)
   into a chunked tied cross-entropy head to avoid materializing [B, T, V] training logits.

Eval contract:
    forward(input_ids: LongTensor[B, T]) -> FloatTensor[B, T, V]
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        inv = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (xf * inv).to(x.dtype) * self.weight


def build_rope_cache(seq_len: int, head_dim: int, base: float = 10000.0, device=None):
    assert head_dim % 2 == 0
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    T = x.shape[1]
    c = cos[:T].view(1, T, 1, -1).to(x.dtype)
    s = sin[:T].view(1, T, 1, -1).to(x.dtype)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)


class CausalDepthwiseConv1d(nn.Module):
    """Strict causal depthwise 1D convolution with kernel size k."""
    def __init__(self, channels: int, kernel_size: int = 4):
        super().__init__()
        self.k = kernel_size
        self.conv = nn.Conv1d(channels, channels, kernel_size=kernel_size, groups=channels, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C] -> pad left by k-1 so output[:, t] only sees <= t
        xt = x.transpose(1, 2)
        xp = F.pad(xt, (self.k - 1, 0))
        return F.silu(self.conv(xp)).transpose(1, 2)


class _ChunkedTiedCEFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor, chunk_size: int):
        N, D = hidden.shape
        V, _ = weight.shape
        h_f = hidden.float()
        w_f = weight.float()

        valid = (targets >= 0) & (targets < V)
        n_valid = valid.sum().clamp(min=1).float()

        loss_total = torch.zeros((), dtype=torch.float32, device=hidden.device)
        d_h = torch.zeros_like(h_f)
        d_w = torch.zeros_like(w_f)

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            h_c = h_f[start:end]
            t_c = targets[start:end]
            v_c = valid[start:end]
            t_safe = torch.where(v_c, t_c, torch.zeros_like(t_c))

            logits = h_c @ w_f.t()
            lse = torch.logsumexp(logits, dim=-1, keepdim=True)
            probs = torch.exp(logits - lse)

            ce_c = lse.squeeze(1) - logits.gather(1, t_safe.unsqueeze(1)).squeeze(1)
            loss_total = loss_total + (ce_c * v_c.float()).sum() / n_valid

            d_logits = probs
            d_logits.scatter_add_(
                1,
                t_safe.unsqueeze(1),
                torch.full((end - start, 1), -1.0, dtype=torch.float32, device=hidden.device),
            )
            d_logits = d_logits * (v_c.float().unsqueeze(1) / n_valid)

            d_h[start:end] = d_logits @ w_f
            d_w.add_(d_logits.t() @ h_c)

        ctx.save_for_backward(d_h.to(hidden.dtype), d_w.to(weight.dtype))
        return loss_total

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        d_h, d_w = ctx.saved_tensors
        return d_h * grad_out.to(d_h.dtype), d_w * grad_out.to(d_w.dtype), None, None


class FactorizedTiedHead(nn.Module):
    def __init__(self, d_in: int, d_emb: int = 384, chunk_size: int = 1024):
        super().__init__()
        self.proj = nn.Linear(d_in, d_emb, bias=False)
        self.chunk_size = chunk_size

    def forward_loss(self, hidden: torch.Tensor, embed_weight: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        h = self.proj(hidden).reshape(-1, embed_weight.shape[1])
        return _ChunkedTiedCEFn.apply(h, embed_weight, targets.reshape(-1), self.chunk_size)

    def forward_logits(self, hidden: torch.Tensor, embed_weight: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden).float() @ embed_weight.float().t()


def _cartan_math_impl(raw: torch.Tensor):
    B, T, _ = raw.shape
    w_in = F.softmax(raw[..., 0:4], dim=-1)
    w_out = 2.0 * torch.sigmoid(raw[..., 4:8])

    S_raw = raw[..., 8:24].view(B, T, 4, 4)
    P = torch.exp((0.5 * (S_raw + S_raw.transpose(-1, -2))).clamp(-6.0, 6.0))
    for _ in range(10):
        P = P / (P.sum(dim=-1, keepdim=True) + 1e-6)
        P = P / (P.sum(dim=-2, keepdim=True) + 1e-6)

    # Polar Cartan composition C_t = Q_t(gamma * g) @ P_t: exact ||Q_t||_2 = 1 avoids convex-sum norm shrinkage!
    gamma = torch.sigmoid(raw[..., 30])
    g = torch.tanh(raw[..., 24:30]) * (gamma.unsqueeze(-1) * math.pi)
    c, s = torch.cos(g), torch.sin(g)
    c0, c1, c2, c3, c4, c5 = c.unbind(-1)
    s0, s1, s2, s3, s4, s5 = s.unbind(-1)
    z = torch.zeros_like(c0)
    G1 = torch.stack([
        torch.stack([c0, -s0, z, z], dim=-1),
        torch.stack([s0,  c0, z, z], dim=-1),
        torch.stack([z, z, c1, -s1], dim=-1),
        torch.stack([z, z, s1,  c1], dim=-1),
    ], dim=-2)
    G2 = torch.stack([
        torch.stack([c2, z, -s2, z], dim=-1),
        torch.stack([z, c3, z, -s3], dim=-1),
        torch.stack([s2, z,  c2, z], dim=-1),
        torch.stack([z, s3, z,  c3], dim=-1),
    ], dim=-2)
    G3 = torch.stack([
        torch.stack([c4, z, z, -s4], dim=-1),
        torch.stack([z, c5, -s5, z], dim=-1),
        torch.stack([z, s5,  c5, z], dim=-1),
        torch.stack([s4, z, z,  c4], dim=-1),
    ], dim=-2)
    Q = (G1 @ G2) @ G3

    C_t = Q @ P
    return w_in, w_out, C_t, gamma


def _delta_prep_impl(
    kf: torch.Tensor, vf: torch.Tensor, af: torch.Tensor, bf: torch.Tensor,
    tril_mask: torch.Tensor, strict_tril: torch.Tensor, eye_c: torch.Tensor,
):
    log_a = torch.log(af.clamp(min=1e-6))
    gamma = torch.cumsum(log_a, dim=-1)

    diff = gamma.unsqueeze(-1) - gamma.unsqueeze(-2)
    diff_safe = torch.where(tril_mask, diff.clamp(max=0.0), torch.full_like(diff, -50.0))
    Lambda = torch.where(tril_mask, torch.exp(diff_safe), torch.zeros_like(diff_safe))

    Lambda0 = torch.exp(gamma.clamp(min=-50.0, max=0.0))
    chunk_decay_end = Lambda0[..., -1]

    KKT = torch.matmul(kf, kf.transpose(-1, -2))
    L = eye_c + torch.where(strict_tril, bf.unsqueeze(-1) * Lambda * KKT, torch.zeros_like(KKT))

    # Concatenate [W_rhs, U_rhs] to solve exact WY system (I + M)[W, U] = rhs in one unitriangular call
    # without Neumann power cancellation (M^k can grow ~1e12 before cancelling when beta > 1).
    rhs = torch.cat([(bf * Lambda0).unsqueeze(-1) * kf, bf.unsqueeze(-1) * vf], dim=-1)
    sol = torch.linalg.solve_triangular(L, rhs, upper=False, unitriangular=True)
    dk = kf.shape[-1]
    return Lambda, Lambda0, chunk_decay_end, sol[..., :dk], sol[..., dk:]


_cartan_math_fn = _cartan_math_impl
_delta_prep_fn = _delta_prep_impl


def enable_compiled_kernels():
    global _cartan_math_fn
    _cartan_math_fn = torch.compile(_cartan_math_impl, mode="default")


def _delta_scan_impl(
    qf: torch.Tensor, kf: torch.Tensor, vf: torch.Tensor, af: torch.Tensor, bf: torch.Tensor,
    tril_mask: torch.Tensor, strict_tril: torch.Tensor, eye_c: torch.Tensor,
):
    B, NC, H, C, dk = qf.shape
    dv = vf.shape[-1]
    Lambda, Lambda0, chunk_decay_end, W_mat, U_mat = _delta_prep_fn(kf, vf, af, bf, tril_mask, strict_tril, eye_c)

    QKT_causal = torch.matmul(qf, kf.transpose(-1, -2)) * Lambda
    tail_decay = Lambda[:, :, :, -1, :]

    S_curr = torch.zeros(B, H, dv, dk, device=qf.device, dtype=torch.float32)
    outputs = []
    surprises = []
    for c in range(NC):
        q_c = qf[:, c]
        k_c = kf[:, c]
        W_c = W_mat[:, c]
        U_c = U_mat[:, c]
        lam0_c = Lambda0[:, c]

        u_corr = U_c - torch.matmul(W_c, S_curr.transpose(-1, -2))
        q_read = torch.matmul(q_c, S_curr.transpose(-1, -2)) * lam0_c.unsqueeze(-1)
        out_c = q_read + torch.matmul(QKT_causal[:, c], u_corr)
        outputs.append(out_c)
        surprises.append(torch.tanh(torch.sqrt(u_corr.pow(2).mean(dim=-1) + 1e-6)))

        k_scaled = k_c * tail_decay[:, c].unsqueeze(-1)
        S_curr = chunk_decay_end[:, c].unsqueeze(-1).unsqueeze(-1) * S_curr + torch.matmul(u_corr.transpose(-1, -2), k_scaled)

    return torch.stack(outputs, dim=1), torch.stack(surprises, dim=1)


def chunked_reflexive_delta_scan(
    q: torch.Tensor,          # [B, T, H, dk]
    k_hat: torch.Tensor,      # [B, T, H, dk] unit Euclidean norm
    v: torch.Tensor,          # [B, T, H, dv]
    alpha: torch.Tensor,      # [B, T, H] in (0, 1)
    beta: torch.Tensor,       # [B, T, H] in (0, 2) generalized Householder strength
    chunk_size: int = 64,
):
    """
    Exact FP32 chunked WY Reflexive Householder-Delta scan:
        S_t = alpha_t * S_{t-1} (I - beta_t k_t k_t^T) + beta_t v_t k_t^T
        o_t = S_t q_t
    """
    B, T, H, dk = q.shape
    dv = v.shape[-1]
    pad_t = (chunk_size - (T % chunk_size)) % chunk_size
    if pad_t > 0:
        q = F.pad(q, (0, 0, 0, 0, 0, pad_t))
        k_hat = F.pad(k_hat, (0, 0, 0, 0, 0, pad_t))
        v = F.pad(v, (0, 0, 0, 0, 0, pad_t))
        alpha = F.pad(alpha, (0, 0, 0, pad_t), value=1.0)
        beta = F.pad(beta, (0, 0, 0, pad_t), value=0.0)
    T_pad = T + pad_t
    NC = T_pad // chunk_size
    C = chunk_size

    qf = q.float().view(B, NC, C, H, dk).permute(0, 1, 3, 2, 4).contiguous()
    kf = k_hat.float().view(B, NC, C, H, dk).permute(0, 1, 3, 2, 4).contiguous()
    vf = v.float().view(B, NC, C, H, dv).permute(0, 1, 3, 2, 4).contiguous()
    af = alpha.float().view(B, NC, C, H).permute(0, 1, 3, 2).contiguous()
    bf = beta.float().view(B, NC, C, H).permute(0, 1, 3, 2).contiguous()

    tril_mask = torch.tril(torch.ones(C, C, device=q.device, dtype=torch.bool))
    strict_tril = torch.tril(torch.ones(C, C, device=q.device, dtype=torch.bool), diagonal=-1)
    eye_c = torch.eye(C, device=q.device, dtype=torch.float32)

    out_s, surp_s = _delta_scan_impl(qf, kf, vf, af, bf, tril_mask, strict_tril, eye_c)
    out = out_s.permute(0, 1, 3, 2, 4).contiguous().view(B, T_pad, H, dv)[:, :T].contiguous()
    surp = surp_s.permute(0, 1, 3, 2).contiguous().view(B, T_pad, H)[:, :T].contiguous()
    return out.to(q.dtype), surp.to(q.dtype)


class CartanRouter4(nn.Module):
    """
    Token-Conditional Cartan Manifold Hyper-Connection over M=4 streams:
        C_t = (1 - gamma_t) * Sinkhorn_3(S_t) + gamma_t * Cayley(A_t)
    Unconditionally satisfies ||C_t||_2 <= 1 while supporting both Birkhoff consensus
    transport and SO(4) orthogonal phase braiding modulated by predictive surprise.
    """
    M = 4
    SO4_DIM = 6  # 4*(4-1)/2 Lie generators

    def __init__(self, d_s: int, n_heads: int = 8):
        super().__init__()
        self.d_s = d_s
        self.norm_desc = RMSNorm(d_s)
        # Inputs: pooled stream descriptor (d_s) + per-head surprise vector (n_heads)
        in_dim = d_s + n_heads
        # Outputs: w_in (4), w_out (4), sym_logits (16), so4_gen (6), gamma_logit (1) = 31
        self.proj = nn.Linear(in_dim, 4 + 4 + 16 + 6 + 1, bias=True)
        nn.init.normal_(self.proj.weight, std=0.01)
        # Initialize close to uniform read-in, residual write-out, identity Birkhoff, zero skew, small gamma
        bias = torch.zeros(31)
        bias[0:4] = 0.0
        bias[4:8] = -0.5
        eye_flat = torch.eye(4).reshape(16) * 5.0
        bias[8:24] = eye_flat
        bias[24:30] = 0.0
        bias[30] = -1.5
        self.proj.bias = nn.Parameter(bias)
        self.register_buffer("eye4", torch.eye(4, dtype=torch.float32), persistent=False)

    def mix_and_read(self, X: torch.Tensor, err_feat: torch.Tensor):
        """
        X: [B, T, 4, D], err_feat: [B, T, n_heads]
        Returns (h_in: [B, T, D], C_t: [B, T, 4, 4], w_out: [B, T, 4], gamma: [B, T])
        """
        B, T, M, D = X.shape
        desc = self.norm_desc(X.mean(dim=2))  # [B, T, D]
        inp = torch.cat([desc, err_feat.to(desc.dtype)], dim=-1)
        raw = self.proj(inp).float()  # [B, T, 31]

        w_in, w_out, C_t, gamma_out = _cartan_math_fn(raw)
        w_in = w_in.to(X.dtype)
        w_out = w_out.to(X.dtype)

        h_in = (w_in.unsqueeze(-1) * X).sum(dim=2)  # [B, T, D]
        return h_in, C_t, w_out, gamma_out

    @staticmethod
    def apply_write(X: torch.Tensor, C_t: torch.Tensor, w_out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        X_mix = torch.matmul(C_t.to(X.dtype), X)
        return X_mix + w_out.unsqueeze(-1) * y.unsqueeze(2)


class ReflexiveDeltaOp(nn.Module):
    """Reflexive Householder-Delta predictive sequence operator (beta in (0, 2))."""
    def __init__(self, d_model: int, n_heads: int = 8, chunk_size: int = 64):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.chunk_size = chunk_size

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.conv_q = CausalDepthwiseConv1d(d_model, kernel_size=4)
        self.conv_k = CausalDepthwiseConv1d(d_model, kernel_size=4)
        self.conv_v = CausalDepthwiseConv1d(d_model, kernel_size=4)

        self.alpha_proj = nn.Linear(d_model, n_heads, bias=True)
        self.beta_proj = nn.Linear(d_model, n_heads, bias=True)
        nn.init.constant_(self.alpha_proj.bias, 3.0)
        nn.init.constant_(self.beta_proj.bias, -0.5)

        self.o_gate = nn.Linear(d_model, d_model, bias=False)
        self.o_norm = RMSNorm(self.head_dim)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor):
        B, T, D = x.shape
        H, dh = self.n_heads, self.head_dim

        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = self.conv_q(q).view(B, T, H, dh)
        k = self.conv_k(k).view(B, T, H, dh)
        v = self.conv_v(v).view(B, T, H, dh)

        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)

        q_hat = q.float() / torch.sqrt((q.float() * q.float()).sum(dim=-1, keepdim=True) + 1e-6)
        k_hat = k.float() / torch.sqrt((k.float() * k.float()).sum(dim=-1, keepdim=True) + 1e-6)

        alpha = torch.sigmoid(self.alpha_proj(x).float())
        beta = 2.0 * torch.sigmoid(self.beta_proj(x).float())

        out_h, surprise = chunked_reflexive_delta_scan(q_hat, k_hat, v, alpha, beta, chunk_size=self.chunk_size)
        out_h = self.o_norm(out_h).reshape(B, T, D)
        out = self.o_proj(out_h * F.silu(self.o_gate(x)))
        return out, surprise


class StickDecayAttnOp(nn.Module):
    """Stick-Breaking Cumulative-Decay Causal Attention with rank-1 log-forget bias G_i - G_j."""
    def __init__(self, d_model: int, n_heads: int = 8):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.conv_q = CausalDepthwiseConv1d(d_model, kernel_size=4)
        self.conv_k = CausalDepthwiseConv1d(d_model, kernel_size=4)
        self.conv_v = CausalDepthwiseConv1d(d_model, kernel_size=4)

        self.forget_proj = nn.Linear(d_model, n_heads, bias=True)
        nn.init.constant_(self.forget_proj.bias, 3.5)

        self.o_gate = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor):
        B, T, D = x.shape
        H, dh = self.n_heads, self.head_dim

        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = self.conv_q(q).view(B, T, H, dh)
        k = self.conv_k(k).view(B, T, H, dh)
        v = self.conv_v(v).view(B, T, H, dh)

        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)

        # Cumulative log-forget decay G_t <= 0
        log_alpha = F.logsigmoid(self.forget_proj(x).float())  # [B, T, H]
        G = torch.cumsum(log_alpha, dim=1).transpose(1, 2)     # [B, H, T]
        bias = (G.unsqueeze(-1) - G.unsqueeze(-2)).clamp_(min=-30.0).to(q.dtype)  # [B, H, T, T]
        causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        bias.masked_fill_(~causal.view(1, 1, T, T), float("-inf"))

        out_h = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=bias
        ).transpose(1, 2).contiguous()

        # Innovation signal: RMS difference between attended representation and raw token value
        diff = (out_h.float() - v.float()).pow(2).mean(dim=-1) + 1e-6
        surprise = torch.tanh(torch.sqrt(diff)).to(x.dtype)  # [B, T, H]

        out = self.o_proj(out_h.reshape(B, T, D) * torch.sigmoid(self.o_gate(x)))
        return out, surprise


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, ffn_dim, bias=False)
        self.w3 = nn.Linear(d_model, ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class CartanBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_dim: int,
        use_attn: bool = False,
        chunk_size: int = 64,
    ):
        super().__init__()
        self.router_seq = CartanRouter4(d_model, n_heads=n_heads)
        self.norm_seq = RMSNorm(d_model)
        self.seq_op = StickDecayAttnOp(d_model, n_heads=n_heads) if use_attn else ReflexiveDeltaOp(d_model, n_heads=n_heads, chunk_size=chunk_size)

        self.router_ffn = CartanRouter4(d_model, n_heads=n_heads)
        self.norm_ffn = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, ffn_dim)

    def forward(self, X: torch.Tensor, err_feat: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor):
        h_in, C_seq, w_out_seq, gamma_seq = self.router_seq.mix_and_read(X, err_feat)
        y_seq, surp_seq = self.seq_op(self.norm_seq(h_in), rope_cos, rope_sin)
        X = CartanRouter4.apply_write(X, C_seq, w_out_seq, y_seq)

        h_ffn, C_ffn, w_out_ffn, gamma_ffn = self.router_ffn.mix_and_read(X, surp_seq)
        y_ffn = self.ffn(self.norm_ffn(h_ffn))
        X = CartanRouter4.apply_write(X, C_ffn, w_out_ffn, y_ffn)

        return X, surp_seq, 0.5 * (gamma_seq.mean() + gamma_ffn.mean())


class CartanPredLM(nn.Module):
    def __init__(
        self,
        vocab_size: int = 49152,
        d_emb: int = 384,
        dim: int = 512,
        depth: int = 16,
        n_heads: int = 8,
        ffn_dim: int = 2304,
        attn_every: int = 2,      # 2 => 1:1 alternating Delta / Stick-Attn; 4 => 3:1 hybrid
        chunk_size: int = 64,
        max_seq_len: int = 1024,
        grad_ckpt: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_emb = d_emb
        self.dim = dim
        self.depth = depth
        self.n_heads = n_heads
        self.ffn_dim = ffn_dim
        self.attn_every = attn_every
        self.chunk_size = chunk_size
        self.grad_ckpt = grad_ckpt

        self.embed = nn.Embedding(vocab_size, d_emb)
        nn.init.normal_(self.embed.weight, std=0.02)
        self.emb_up = nn.Linear(d_emb, 4 * dim, bias=False)
        nn.init.normal_(self.emb_up.weight, std=0.02)
        self.emb_norm = RMSNorm(4 * dim)

        self.blocks = nn.ModuleList([
            CartanBlock(
                d_model=dim,
                n_heads=n_heads,
                ffn_dim=ffn_dim,
                use_attn=((i + 1) % attn_every == 0),
                chunk_size=chunk_size,
            )
            for i in range(depth)
        ])

        # Quad-Stream Cartan Readout: normalize each of the 4 streams independently -> [B, T, 4*dim]
        self.stream_norms = nn.ModuleList([RMSNorm(dim) for _ in range(4)])
        self.head = FactorizedTiedHead(4 * dim, d_emb=d_emb, chunk_size=1024)

        cos, sin = build_rope_cache(max_seq_len, dim // n_heads)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        res_std = 0.02 / math.sqrt(2 * depth)
        for blk in self.blocks:
            nn.init.normal_(blk.seq_op.o_proj.weight, std=res_std)
            nn.init.normal_(blk.ffn.w2.weight, std=res_std)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear) and m not in (self.emb_up,):
            if hasattr(m, "weight") and m.weight.ndim == 2:
                # Preserve custom router/proj bias inits
                if m.out_features != 31 and m.out_features != self.n_heads:
                    nn.init.normal_(m.weight, std=0.02)

    def forward_hidden(self, input_ids: torch.Tensor):
        B, T = input_ids.shape
        e = self.embed(input_ids)
        X = (0.5 * self.emb_norm(self.emb_up(e))).view(B, T, 4, self.dim)
        err_feat = torch.zeros(B, T, self.n_heads, device=input_ids.device, dtype=X.dtype)

        gammas = []
        surprises = []
        for idx, blk in enumerate(self.blocks):
            # Selective checkpointing: checkpoint every block if grad_ckpt=True, 2/3 if "2of3", or half if "half"
            use_ckpt = self.training and torch.is_grad_enabled() and (
                (self.grad_ckpt is True) or (self.grad_ckpt == "half" and idx % 2 == 0) or (self.grad_ckpt == "2of3" and idx % 3 != 0)
            )
            if use_ckpt:
                X, err_feat, g_mean = checkpoint(
                    blk, X, err_feat, self.rope_cos, self.rope_sin, use_reentrant=False
                )
            else:
                X, err_feat, g_mean = blk(X, err_feat, self.rope_cos, self.rope_sin)
            gammas.append(g_mean)
            surprises.append(err_feat.mean())

        normed = [self.stream_norms[m](X[:, :, m, :]) for m in range(4)]
        h = torch.cat(normed, dim=-1)
        stats_tensors = (
            torch.stack(gammas).mean().detach(),
            torch.stack(surprises).mean().detach(),
            X.float().pow(2).mean().sqrt().detach(),
        )
        return h, stats_tensors

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        h, _ = self.forward_hidden(input_ids)
        return self.head.forward_logits(h, self.embed.weight)

    def forward_loss(self, input_ids: torch.Tensor, targets: torch.Tensor):
        h, (g_t, s_t, r_t) = self.forward_hidden(input_ids)
        ce_loss = self.head.forward_loss(h, self.embed.weight, targets)
        stats = {
            "ce_loss": ce_loss.detach().item(),
            "cartan_gamma": g_t.item(),
            "surprise_mean": s_t.item(),
            "stream_rms": r_t.item(),
        }
        return ce_loss, stats
