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
- **Train Speed & Hardware:** Up to `5,862 tok/s` (`0.71 step/s` @ `batch=16, seq=512`), `3,553 MB` peak training VRAM on `NVIDIA GeForce RTX 5060 (8GB)` (`p41322 GPU1`).
