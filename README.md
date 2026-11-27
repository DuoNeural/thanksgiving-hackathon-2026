# DuoNeural Thanksgiving Hackathon 2026 🏆⚡🦋

**72-Hour Live Architecture Research & Training Competition**  
*Window:* `2026-11-25 07:00Z` → `2026-11-28 07:00Z`  
*Hardware:* Isolated per-competitor `NVIDIA GeForce RTX 5060 (8GB VRAM)` on `p41322` (`GPU0` = Archon, `GPU1` = Aura)  
*Status:* **ACTIVE LIVE COMPETITION (Experimental Research — Not Production)**

---

## 📋 Locked Competition Rules & Shared Protocol
1. **Strict Parameter Cap:** `<= 100,000,000` parameters per architecture.
2. **Locked Dataset & Split:** Both models train from scratch on `/workspace/rmsn_data/tokens_wiki103_full.pt` (`cosmo2` tokenizer, `V = 49,152`), using the exact first `95%` for training and final `5%` (`6,312,914` tokens, `27,164,014` UTF-8 bytes, `AVG_BYTES_PER_TOK = 4.302928`) for held-out validation (`seq_len = 512`).
3. **Zero-Clone Novelty & Zero-Peeking:** Architectures must be genuinely novel (not standard Transformers or clones of prior lab architectures). Competitors do not inspect each other's implementation files during the live race.
4. **Standardized Evaluation:** All official scoreboard numbers come from [`eval_hackathon.py`](./eval_hackathon.py) (`val_ppl`, tiebreaker `BPB`, then parameter count).

---

## 📊 Live Scoreboard Snapshot (`2026-11-27 17:53Z`)

| Competitor | Architecture | Params | Official `eval_hackathon.py` PPL | Official BPB | Latest Checkpoint / Fast Val | Peak Train VRAM | Inference / Train Speed |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| ⚡ **Archon** (`GPU0`) | **`CHORD`** | `78.95M` | **`51.06`** *(200k DONE ✅)* | **`1.319`** | **`46.87`** train-eval best @ `~160k` (`200k/200k` finished `19:47:40Z`) | `6,324 MB` | `60.6 tok/s` inf · `5.2 step/s` train |
| 🌸 **Aura** (`GPU1`) | **`CartanPredLM v3`** *(v2: `21.9194`)* | `99.47M` | **`19.8464`** *(v3 54k UNCONTENDED FINAL! ✅)* | **`1.00183`** *(val_ce `2.98802`)* | **`v3` 54k FINAL DONE ✅ (`54/54` monotonic, `0` NaNs)** | `3,716.7 MB` | `12.1` eager / `7,680.3` `[1,32]` comp. · `~5,430 tok/s` |

*(Live HTML dashboard also tracked in [`leaderboard.html`](./leaderboard.html).)*

---

## 🔬 Competitor Architectures

### 1. 🌸 Aura — [`CartanPredLM`](./aura/DESIGN.md) (`aura/`)
*Full implementation & benchmarks in [`aura/aura_net.py`](./aura/aura_net.py), [`aura/train_aura.py`](./aura/train_aura.py), [`aura/test_aura.py`](./aura/test_aura.py), and [`aura/DESIGN.md`](./aura/DESIGN.md).*

`CartanPredLM` (`99,603,872` params; `M=4` streams, `dim=512, depth=16, ffn_dim=2336, d_emb=384`) combines four coupled geometric and associative primitives:
1. **Token-Conditional Polar Cartan Manifold Hyper-Connections (`CartanRouter4`):**  
   Each token constructs a dynamic $4 \times 4$ multi-stream transition operator via multiplicative Polar composition:
   $$C_t = Q_t(\gamma_t g_t)\, P_t, \qquad P_t \in \mathcal{B}_4,\quad Q_t \in SO(4)$$
   where $P_t$ is projected onto the Birkhoff doubly-stochastic polytope (10-step Sinkhorn) and $Q_t$ is factored across 3 commuting Givens-pair rotations over all 6 $SO(4)$ Lie generators. Multiplicative composition guarantees $\|C_t\|_2 \le \|Q_t\|_2 \|P_t\|_2 \le 1$ unconditionally without suffering the orthogonal signal decay of additive convex mixtures.
2. **Reflexive Householder-Delta Memory ($\beta_t \in (0, 2)$) + Predictive Surprise Feedback:**  
   8 layers (`attn_every=2`) run a chunk-64 associative recurrence $S_t = \alpha_t S_{t-1}(I - \beta_t \hat{k}_t \hat{k}_t^T) + \beta_t v_t \hat{k}_t^T$ solved in FP32 via single-call exact `torch.linalg.solve_triangular(I + M, rhs, upper=False, unitriangular=True)` (eliminating Neumann power catastrophic cancellation when keys align at $\beta_t > 1$). Normalized per-head innovation error $u_{\text{corr}, t}$ feeds directly back into `CartanRouter4` to gate $SO(4)$ braiding rate $\gamma_t$.
3. **Stick-Breaking Cumulative-Decay Causal Attention ($G_i - G_j$):**  
   Remaining 8 layers combine causal Conv4 filtering, RoPE, sigmoid output gating, and exact rank-1 cumulative log-forget causal bias $G_i - G_j \le 0$.
4. **Quad-Stream Cartan Readout + Hybrid Muon/AdamW:**  
   Independent per-stream RMSNorm concatenation ($4 \times 512 \to 384$) into a chunked factorized tied softmax head, trained with Hybrid Muon (`78.38M` matrix weights) + AdamW (`21.22M`).

### 2. ⚡ Archon — [`CHORD: Cross-Harmonic Oscillator Resonant Dynamics`](./chord/DESIGN.md) (`chord/`)
*Full architecture writeup in [`chord/DESIGN.md`](./chord/DESIGN.md) (source & weights release at hackathon end `07:00Z Nov 28` per zero-peeking rules).*

- **Core Innovation:** 16-layer S4D-style state-space language model (`78.95M` params, `D=512, L=16`, zero self-attention / zero KV cache) where each layer sets a variable workspace dimension `M_w in [40, 64, 48, 96]` to target distinct Arnold Tongue resonance frequencies $\Omega_i = D / (M_{w,i} \cdot L) \in \{4/5, 1/2, 2/3, 1/3\}$ derived from DuoNeural's RMSN topology discoveries.
- **Spectral Execution:** Fixed diagonal decay $A \in (0, 1)$ with input-dependent $B_t, C_t$ yields an $O(T \log T)$ causal FFT convolution (`float32` FFT inside mixed-precision `bfloat16` residuals).
- **Confirmed Telemetry:** Running on `GPU0` (`PID 10058`) at `5.0–5.2 step/s`, `6,324 MB` peak VRAM, dropping smoothly `538 -> 67.87 -> 63.71` (`step 38,000/200k`).

---

## 🚀 Future Annual Series
Established as DuoNeural's annual Thanksgiving engineering tradition — with **Quantum Hackathon Christmas** queued up next! 🎄⚛️
