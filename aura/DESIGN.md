# Aura — DuoNeural Thanksgiving Hackathon 2026: `CartanPredLM`

**Experimental Research Architecture (`99,603,872` / `99.60M` parameters, trained from scratch on 8GB RTX 5060 `p41322 GPU1`)**

> **Research Notice:** Experimental hackathon model created inside the 72-hour DuoNeural Thanksgiving Hackathon (`07:00Z Nov 25 – 07:00Z Nov 28, 2026`). Evaluated on held-out WikiText-103 (`cosmo2` `V=49,152`, last 5% = `6,312,914` tokens, `seq_len=512`, shared `eval_hackathon.py`).

---

## Mathematical & Architectural Innovations

### 1. Token-Conditional Polar Cartan Manifold Hyper-Connections (`CartanRouter4`, $M=4$ streams)
Rather than passing a single residual stream through depth, `CartanPredLM` lifts normalized token embeddings ($d_{\text{emb}}=384 \to 4 \times 512 = 2048$ via `emb_up` + `0.5 * emb_norm`) onto a **4-stream Cartan manifold**. At every sublayer, each token dynamically constructs a $4 \times 4$ transition operator via **multiplicative Polar Cartan composition**:
$$C_t = Q_t(\gamma_t g_t)\, P_t$$
where:
- **$P_t = \text{Sinkhorn}_{10}(\exp(\operatorname{clamp}(\tfrac{1}{2}(S_t + S_t^T), -6, 6))) \in \mathcal{B}_4$** lies in the Birkhoff doubly-stochastic polytope ($\|P_t\|_2 \le 1$, dissipative consensus routing across streams).
- **$Q_t(\gamma_t g_t) = G_{01,23}(\theta_0, \theta_1)\, G_{02,13}(\theta_2, \theta_3)\, G_{03,12}(\theta_4, \theta_5) \in SO(4)$**, with $\theta = \tanh(g_t) \cdot \gamma_t \pi$ and $\gamma_t = \sigma(\cdot)$, provides exact isometric gauge braiding across all 6 Lie generators ($\det Q_t = +1$, $\|Q_t\|_2 = 1$).
- **Why Polar Product instead of Convex Sum:** Linear convex mixing $(1-\gamma)P + \gamma Q$ shrinks orthogonal components across 32 transitions ($\| \cdot \|_2 < 1$ compounding geometrically). Multiplicative Polar composition $Q_t P_t$ guarantees **$\|C_t\|_2 \le \|Q_t\|_2 \|P_t\|_2 \le 1$** unconditionally while preserving non-contracting isometric signal rotations (`stream_rms` healthy, zero vanishing/exploding depth cascade).

### 2. Reflexive Householder-Delta Associative Memory ($\beta_t \in (0, 2)$) + Predictive Surprise Feedback
- Interleaved `1:1` (`attn_every=2`, 8 layers) alongside causal attention, each `ReflexiveDeltaOp` applies causal depthwise Conv4 filtering, RoPE, $L_2$-normalized queries/keys ($\hat{q}_t, \hat{k}_t$), scalar forget $\alpha_t \in (0, 1)$, and generalized Householder strength $\beta_t \in (0, 2)$:
  $$S_t = \alpha_t S_{t-1}\left(I - \beta_t \hat{k}_t \hat{k}_t^T\right) + \beta_t v_t \hat{k}_t^T, \qquad o_t = S_t \hat{q}_t$$
- Because $\operatorname{eig}(I - \beta_t \hat{k}_t \hat{k}_t^T) = \{1, \dots, 1, 1 - \beta_t\}$ with $|1 - \beta_t| < 1$ for $\beta_t \in (0, 2)$, the associative state transition remains strictly non-expansive while supporting true **sign-flipping reflexive erasure** ($\beta_t > 1$) rather than decay-only erasure ($\beta_t \le 1$).
- **Exact WY Triangular Solve:** Solved chunk-wise ($C=64$) in FP32 via single-call `torch.linalg.solve_triangular(I + M, rhs, upper=False, unitriangular=True)` rather than power-truncated Neumann products (which suffer catastrophic cancellation $\sim C(30,15)\beta^{16}$ when conv-aligned keys meet $\beta_t > 1$).
- **Closed-Loop Predictive Surprise:** Each sequence layer emits normalized innovation error $\text{surp}_t = \tanh(\sqrt{\operatorname{mean}(u_{\text{corr}, t}^2) + \epsilon}) \in \mathbb{R}^{H}$, feeding directly into the next `CartanRouter4` to modulate $SO(4)$ gauge mixing $\gamma_t$.

### 3. Stick-Breaking Cumulative-Decay Causal Attention ($G_i - G_j$)
- Interleaved on the remaining 8 layers (`attn_every=2`), combining causal depthwise Conv4 (`k=4`), RoPE, sigmoid output head-gating, and an exact rank-1 cumulative log-forget causal mask:
  $$G_i = \sum_{u=1}^{i} \log \sigma(g_u), \qquad \text{bias}_{ij} = \operatorname{clamp}(G_i - G_j, \min=-30.0) \le 0 \quad (j \le i)$$

### 4. Quad-Stream Cartan Readout & Memory-Efficient Optimization
- Independently RMS-normalizes all 4 manifold streams ($4 \times 512 = 2048 \to d_{\text{emb}} = 384$) into a chunked tied softmax head (`FactorizedTiedHead`, avoiding materializing `[B, T, V]` float32 logits during training).
- Trained with **Hybrid Muon** (`78,381,056` matrix weights, `lr=8e-4`, Newton-Schulz 5) + **AdamW** (`21,222,816` embeddings/norms/convs/routers/gates, `lr=2e-4`) at `batch=16, seq=512, grad_ckpt=all` (`3,553 MB` peak CUDA VRAM on 8GB RTX 5060).

---

## Parameter Breakdown & Key Verified Metrics (`cartan_v1_30k`)
- **Exact Parameter Count:** `99,603,872` (`99.60M` — `muon`: `78,381,056`, `adamw`: `21,222,816`)
- **Official Shared `eval_hackathon.py` Checkpoints (`6,312,448` held-out WikiText-103 val tokens):**
  - **Step 1,000** (`8.19M` train tokens): `val_ppl = 104.4101` (`val_ce = 4.64833`), `BPB = 1.5585`, `eval VRAM = 1,464.5 MB`
  - **Step 5,000** (`40.96M` train tokens): **`val_ppl = 47.0493`** (`val_ce = 3.8512`), **`BPB = 1.29124`**, `eval VRAM = 1,464.9 MB` (`skipped_nan = 0`)
  - **Step 10,000** (`81.92M` train tokens): **`val_ppl = 34.7903`** (`val_ce = 3.54934`), **`BPB = 1.19003`**, `eval VRAM = 1,463.9 MB` (`skipped_nan = 0`)
  - **Step 15,000** (`122.88M` train tokens): **`val_ppl = 30.1105`** (`val_ce = 3.40487`), **`BPB = 1.14159`**, `eval VRAM = 1,464.5 MB` (`skipped_nan = 0`)
  - **Step 20,000** (`163.84M` train tokens): **`val_ppl = 27.02`** (`val_ce = 3.29658`), **`BPB = 1.10528`**, `eval VRAM = 1,464.5 MB` (`skipped_nan = 0`)
  - **Step 25,000** (`204.80M` train tokens): **`val_ppl = 25.068`** (`val_ce = 3.22159`), **`BPB = 1.08014`**, `eval VRAM = 1,464.5 MB` (`skipped_nan = 0`)
  - **Final Best — Step 29,000 (`ckpt_best.pt`, run `30,000/30,000` DONE ✅):** **`val_ppl = 24.2865`** (`val_ce = 3.18992`), **`BPB = 1.06952`**, uncontended inference **`8.27–8.5 tok/s`** (`5.9 tok/s` concurrent), `eval VRAM = 1,464.9–1,466.6 MB`, `peak train VRAM = 3,552.8 MB`, **`skipped_nan = 0`** across all `30,000` steps (`29/30` monotonic bests; live batch PPL touched `18.89` at step `29,850`).

---

## `CartanPredLM v2` Final Design & Verified Results (`cartan_v2_36k` DONE ✅)
Analysis of `v1` exposed two clear optimization targets:
1. **Late-epoch overfitting:** Train batch PPL fell to `18.89` while `val_ppl` plateaued (`24.88 → 24.44 → 24.56`) after ~2 epochs over WikiText-103 with `dropout = 0.0`. `v2` adds `dropout = 0.10`, `wd = 0.08`, and `min_lr_ratio = 0.05` over `36,000` steps.
2. **Manifold capacity & readout rank:** Expanded tied embedding/output bottleneck from `d_emb = 384 → 512` (trading `ffn_dim = 2336 → 2048`, an exact tensor-core multiple `4 × dim`), halved Sinkhorn iterations (`10 → 5`) with cached causal/triangular scan masks, capped stream write residual (`w_out_max = 1.5`), locked $SO(4)$ gauge braiding open with **`gamma_floor = 0.08`** (stabilized cleanly at `0.082–0.083` vs `v1`'s decay to `0.024`), and added `qk_norm` + learned pre-conv `v_resid` across all 8 attention + 8 delta layers (`chunk_size = 64` retained after empirical benchmarking showed `5,781 tok/s` vs `5,386 tok/s` for `chunk_size = 32`).

- **Exact Parameter Count (`v2`):** `99,408,416` (`99.41M` — `muon`: `71,368,704`, `adamw`: `28,039,712`)
- **Train Speed & Hardware:** `~4,800–5,460 tok/s` (`batch = 16 × 1, seq = 512, grad_ckpt = all, --compile`), `3,674.8 MB` peak training VRAM on `p41322 GPU1`, **`skipped_nan = 0`**, and **`36/36` monotonic best fast-val checkpoints** (`95.16 @ 1k → 39.96 @ 5k → 32.49 @ 10k → 29.03 @ 15k → 26.37 @ 20k → 24.29 @ 25k → 22.94 @ 30k → 22.21 @ 35k → 22.1106 @ 36k`).
- **Official Shared `eval_hackathon.py` Curve (`6,312,448` held-out WikiText-103 val tokens):**
  - **Step 5,000:** `val_ppl = 40.3611` (`val_ce = 3.69787`), `BPB = 1.23983` (`-6.69 PPL` vs `v1` @ 5k)
  - **Step 10,000:** `val_ppl = 32.5761` (`val_ce = 3.48358`), `BPB = 1.16798` (`-2.21 PPL` vs `v1` @ 10k)
  - **Step 15,000:** `val_ppl = 28.8747` (`val_ce = 3.36296`), `BPB = 1.12754` (`-1.24 PPL` vs `v1` @ 15k)
  - **Step 20,000:** `val_ppl = 26.2561` (`val_ce = 3.26790`), `BPB = 1.09567` (`-0.764 PPL` vs `v1` @ 20k)
  - **Step 25,000:** **`val_ppl = 24.1110`** (`val_ce = 3.18267`), **`BPB = 1.06709`** (already beat `v1` 30k final!)
  - **Step 30,000:** **`val_ppl = 22.7075`** (`val_ce = 3.12269`), **`BPB = 1.04698`** (broke `< 23 PPL` & `< 1.05 BPB`)
  - **Final Best — Step 36,000 (`ckpt_best.pt`, `36,000/36,000` DONE ✅):** **`val_ppl = 21.9194`** (`val_ce = 3.08737`), **`BPB = 1.03514`**, `eval VRAM = 1,476.3 MB`, eager inference `5.38–5.6 tok/s` (`248.99 tok/s` under `torch.compile(mode="reduce-overhead")`, `46.3×` speedup on `[1, 32]`), `skipped_nan = 0` — improving on `v1` by **`-2.3671 PPL`** and **`-0.03438 BPB`**!

---

## `CartanPredLM v3` Architecture & Verified Results (`cartan_v3_54k` DONE ✅ — **Sub-20 Record: `19.8464 PPL / 1.00183 BPB`**)
Profiling `v2` at `Step 36,000` revealed zero overfitting gap (`train_ppl ≈ 20.8–21.9` vs `val_ppl = 21.92`), rising global pre-clip grad norm (`gn = 1.21 → 3.05`), and eager Python dispatch overhead:
1. **Cross-Layer `v_first` Value Anchoring:** Layer-0 sequence operator exports raw lexical value tensor `v_first`, blended into all 15 deeper `ReflexiveDeltaOp` + `StickDecayAttnOp` layers via learned per-head gate `sigmoid(v0_proj(x))` (`99,469,976` total params = `99.47M`, `muon=71,303,168 / adamw=28,166,808`).
2. **Per-Group Split Gradient Clipping (`--split-clip`):** Clips `adamw` (`1.0`) separately from `muon` (`2.0`) so late-stage Muon norms do not attenuate `embed`, `head.proj`, or `CartanRouter4` by `3×`.
3. **Eager Speedup & 54k Regularization Schedule:** Vectorized $SO(4)$ Givens (`G1, G2, G3` single-reshape stacks) + adaptive `T <= 32` eval chunking, paired with `dropout = 0.06`, `wd = 0.07`, `muon_lr = 9e-4`, `adamw_lr = 2.5e-4`, `min_lr_ratio = 0.03`, `warmup = 400`, `steps = 54,000` (`442.37M` train tokens, `3,716.7 MB` peak training VRAM, **`skipped_nan = 0`**, **`54/54` monotonic best fast-val steps**).

- **Official Shared `eval_hackathon.py` Curve (`cartan_v3_54k`, `6,312,448` held-out WikiText-103 val tokens):**
  - **Step 5,000:** `val_ppl = 39.5414` (`val_ce = 3.67735`), `BPB = 1.23295`
  - **Step 10,000:** `val_ppl = 32.4404` (`val_ce = 3.47940`), `BPB = 1.16658`
  - **Step 15,000:** `val_ppl = 29.2822` (`val_ce = 3.37698`), `BPB = 1.13224`
  - **Step 20,000:** `val_ppl = 27.2405` (`val_ce = 3.30470`), `BPB = 1.10801`
  - **Step 25,000:** `val_ppl = 25.5980` (`val_ce = 3.24252`), `BPB = 1.08716`
  - **Step 30,000:** `val_ppl = 24.1389` (`val_ce = 3.18383`), `BPB = 1.06748` *(already beat `v1` 30k final!)*
  - **Step 35,000:** `val_ppl = 22.8271` (`val_ce = 3.12795`), `BPB = 1.04875`
  - **Step 40,000:** **`val_ppl = 21.5718`** (`val_ce = 3.07139`), **`BPB = 1.02978`** *(already beat `v2` 36k final!)*
  - **Step 45,000:** **`val_ppl = 20.5950`** (`val_ce = 3.02505`), **`BPB = 1.01424`** *(sub-21!)*
  - **Step 50,000:** **`val_ppl = 20.0461`** (`val_ce = 2.99803`), **`BPB = 1.00519`** *(sub-3.0 CE!)*
  - **Final UNCONTENDED — Step 54,000 (`ckpt_best.pt`, `54,000/54,000` DONE ✅):** **`val_ppl = 19.8464`** (`val_ce = 2.98802`), **`BPB = 1.00183`**, `eval VRAM = 1,477.2 MB`, uncontended eager autoregressive `eval_hackathon.py` `12.1 tok/s` (`7,680.33 tok/s` / `4.17 ms/fwd` on `torch.compile(mode="reduce-overhead")` `[1, 32]` prefill forward), `skipped_nan = 0` — improving on `v2` by **`-2.0730 PPL` (`-0.03331 BPB`)** and on `v1` by **`-4.4401 PPL` (`-0.06769 BPB`)**!
