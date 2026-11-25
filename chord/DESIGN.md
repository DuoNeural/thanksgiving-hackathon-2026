# CHORD: Cross-Harmonic Oscillator Resonant Dynamics

**Author:** Archon (DuoNeural Lab Director)  
**Date:** 2026-11-25  
**Hackathon:** DuoNeural Thanksgiving AI Hackathon 2026  
**Node:** p41322 GPU0 (RTX 5060 8GB)

---

## Core Idea

CHORD is a 16-layer language model where each layer's state-space model (SSM) is designed to operate at a *different Arnold Tongue* — a topological resonance frequency discovered empirically in RMSN research. This is a world-first: no prior architecture has used per-layer topological resonance targeting.

The key insight: in RMSN (Recurrent Momentum State Networks), the ratio Ω = D/(M_w × L) determines whether a model's weight condensate crystallizes into a "TRUE SILENT" near-zero Lyapunov attractor (ppl ≈ 42–77) or cascades into a cemented topology (~90 ppl penalty). Different Ω values produce qualitatively different dynamics — Arnold Tongues in the resonance landscape.

CHORD stacks these modes: 16 layers, 4 Ω values cycling, one per-layer resonance identity.

---

## Architecture

### SSM Formulation (S4D-style)

Each CHORD layer contains a `ChordSSM` block: a diagonal state-space model with **fixed A** (time-invariant):

```
h_t = A ⊙ h_{t-1} + B_t ⊙ x_t     (A fixed diagonal ∈ (0,1), B_t input-dependent)
y_t = C_t ⊙ h_t + D ⊙ x_t          (C_t input-dependent, D skip)
```

Since A is fixed, this is a causal convolution:
```
h_t = Σ_{s≤t} A^(t-s) ⊙ B_s ⊙ x_s  =  kernel ★ u
```
Computed in **O(T log T)** via FFT — no sequential scan. No attention. No KV cache.

### Variable M_w per Layer (The CHORD Innovation)

Each layer has its own `M_w` (workspace slots), which sets its Arnold Tongue:

```
Ω_i = D / (M_w_i × L)
```

CHORD M_w schedule (D=512, L=16):

| Layer | M_w | Ω | Tongue | RMSN Empirical Result |
|-------|-----|---|--------|----------------------|
| 0, 4, 8, 12 | 40 | 4/5 | Ω=4/5 | **Record PPL=42.5** (topology #62) |
| 1, 5, 9, 13 | 64 | 1/2 | Ω=1/2 | TRUE SILENT (topology #65, #74) |
| 2, 6, 10, 14 | 48 | 2/3 | Ω=2/3 | Sharkovskii cascade |
| 3, 7, 11, 15 | 96 | 1/3 | Ω=1/3 | Sub-harmonic |

The 4/5 tongue is the best-performing RMSN resonance; 1/2 gives TRUE SILENT dynamics; 2/3 is the Sharkovskii tongue (topological period-3 → all periods). CHORD tries to crystallize all four simultaneously in a single model.

### A Initialization (Hippo-inspired)

Fixed diagonal A uses geometric spacing across M_w slots:
```
log(-A_j) = log(j / M_w)  for j = 1..M_w
```
This gives eigenvalues A_j = exp(-exp(log(j/M_w))) ∈ (0,1), ranging from fast (j→M_w) to slow (j→1) decay — diverse timescales like HiPPO but simpler.

### Full Layer Structure

Each `ChordLayer`:
1. **SSM branch**: RMSNorm → input gate (SiLU) → ChordSSM → gated residual
2. **FFN branch**: RMSNorm → SwiGLU (d_ffn=1024) → residual

### Model Specification

```
vocab_size = 49,152  (cosmo2 tokenizer)
D = 512
L = 16
state_dim = 16  (per M_w slot)
ffn_dim = 1,024
M_w schedule = [40, 64, 48, 96] × 4

Total params: 78,950,912 (78.95M)
Peak VRAM: ~6,324 MB @ batch=8, seq_len=512
```

---

## The Arnold Tongue Hypothesis

**Hypothesis**: Stacking layers at different Arnold Tongue Ω values creates a multi-frequency resonance structure. The 4/5 layers (empirically best in RMSN) should act as principal condensation sites, while 1/2 (TRUE SILENT) layers provide stability, 2/3 (Sharkovskii) provides broad periodic coverage, and 1/3 provides slow-decay memory.

**Prediction**: CHORD should converge faster than a uniform-Ω SSM of equivalent size, because different layers lock onto different topological attractors rather than competing for the same resonance mode.

---

## Training Details

- Steps: 200,000
- Batch: 8, seq_len: 512
- LR: 3e-4 → 1e-5 (cosine with 2k warmup)
- bf16 AMP
- Data: WikiText-103 (cosmo2 tokenized, 95/5 split, shared with CartanPredLM)
- Hardware: RTX 5060 8GB (p41322 GPU0)

---

## bf16 Dtype Cascade (Engineering Note)

Training in bf16 exposed a 4-stage dtype mismatch:
1. `torch.fft.rfft` doesn't support bf16 → must cast SSM projections' *output* to float32 for FFT, then back to bf16
2. `F.linear(bf16_input, bf16_weight)` must match — projecting in bf16 then casting output is correct; casting input to float32 would cause the mismatch
3. `D_skip` parameter must be cast to float32 before the skip-connection multiply
4. `nn.RMSNorm` upcasts to float32 for numerical stability even inside autocast → must `.to(weight.dtype)` after each norm call in the layer

Fix: `wdtype = self.in_gate.weight.dtype` and `.to(wdtype)` after each `norm(x)` call.

---

## Results (Live — Hackathon in Progress)

| Step | Val PPL | Notes |
|------|---------|-------|
| 500 | 538.12 | Very early |
| 18,000 | 67.87 | Still dropping |
| 38,000 | **63.71** | Current best |
| 200,000 | TBD | ETA ~19:00Z Nov 25 |

Training speed: 5.0–5.2 step/s, ETA 8.9h remaining.

---

## Theoretical Connections

CHORD draws from:
- **S4/S4D** (Gu et al.): diagonal SSM with fixed A, O(T log T) FFT convolution
- **Mamba** (Gu, Dao): input-dependent B, C (selective SSMs)
- **RMSN** (DuoNeural): Arnold Tongue topology, Lyapunov exponent as training signal, topological phase transitions in neural networks
- **KAM Theory**: resonance conditions Ω=p/q and their relationship to stability vs. cascade

The key distinction from prior SSMs: CHORD's variable M_w is not a hyperparameter — it's a **topological address** in resonance space.

---

*Code and final weights will be released at hackathon end (07:00Z Nov 28). Zero-peeking in effect during live competition.*
