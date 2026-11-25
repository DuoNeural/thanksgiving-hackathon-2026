"""
Training Harness for CartanPredLM on DuoNeural Hackathon (p41322 GPU1, 8GB RTX 5060).
Uses memory-efficient Hybrid Muon (1 momentum buffer) + AdamW, gradient accumulation,
and exact 95%/5% split of /workspace/rmsn_data/tokens_wiki103_full.pt.
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

import aura_net
from aura_net import CartanPredLM


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    transposed = False
    if G.size(-2) > G.size(-1):
        X = X.T
        transposed = True
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.05):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]
                buf.mul_(mu).add_(g)
                u = g.add(buf, alpha=mu) if nesterov else buf
                o = zeropower_via_newtonschulz5(u, steps=ns_steps)
                rms = o.float().pow(2).mean().sqrt().clamp(min=1e-8)
                scale = 0.2 / rms
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(o.to(p.dtype), alpha=-lr * scale.item())


class HybridMuonAdamW:
    ADAMW_KEYWORDS = ("embed", "emb_up", "head", "conv", "norm", "router", "alpha_proj", "beta_proj", "forget_proj")

    def __init__(self, model: torch.nn.Module, adamw_lr: float = 2e-4, muon_lr: float = 8e-4, wd: float = 0.05):
        muon_params, adamw_decay, adamw_nodecay = [], [], []
        vocab_sz = getattr(model, "vocab_size", 49152)
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            lname = name.lower()
            if p.ndim == 2 and p.shape[0] != vocab_sz and not any(k in lname for k in self.ADAMW_KEYWORDS):
                muon_params.append(p)
            elif p.ndim >= 2 and "embed" not in lname:
                adamw_decay.append(p)
            else:
                adamw_nodecay.append(p)

        self.muon = Muon(muon_params, lr=muon_lr, weight_decay=wd) if muon_params else None
        groups = []
        if adamw_decay:
            groups.append({"params": adamw_decay, "lr": adamw_lr, "weight_decay": wd, "base_lr": adamw_lr})
        if adamw_nodecay:
            groups.append({"params": adamw_nodecay, "lr": adamw_lr, "weight_decay": 0.0, "base_lr": adamw_lr})
        self.adamw = torch.optim.AdamW(groups, betas=(0.9, 0.95), eps=1e-8)
        self.muon_base_lr = muon_lr
        self.counts = {
            "muon": sum(p.numel() for p in muon_params),
            "adamw": sum(p.numel() for p in adamw_decay) + sum(p.numel() for p in adamw_nodecay),
        }

    def zero_grad(self, set_to_none=True):
        if self.muon:
            self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def set_lr_scale(self, scale: float):
        if self.muon:
            for g in self.muon.param_groups:
                g["lr"] = self.muon_base_lr * scale
        for g in self.adamw.param_groups:
            g["lr"] = g["base_lr"] * scale

    def step(self):
        if self.muon:
            self.muon.step()
        self.adamw.step()


class ChunkDataset(Dataset):
    def __init__(self, tokens: torch.Tensor, seq_len: int, stride: int):
        self.tokens = tokens
        self.seq_len = seq_len
        self.stride = stride
        self.n = max(0, (len(tokens) - seq_len - 1) // stride + 1)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        i = idx * self.stride
        c = self.tokens[i : i + self.seq_len + 1].long()
        return c[:-1], c[1:]


def cosine_lr_scale(step: int, warmup: int, total: int, min_ratio: float = 0.1) -> float:
    if step <= warmup:
        return step / max(warmup, 1)
    prog = (step - warmup) / max(total - warmup, 1)
    return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * min(prog, 1.0)))


@torch.no_grad()
def evaluate_fast(model, val_dl, device, max_batches: int = 80):
    model.eval()
    tot_ce, nb = 0.0, 0
    for vx, vy in val_dl:
        vx, vy = vx.to(device, non_blocking=True), vy.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _, m = model.forward_loss(vx, vy)
        tot_ce += m["ce_loss"]
        nb += 1
        if nb >= max_batches:
            break
    model.train()
    ce = tot_ce / max(nb, 1)
    return ce, math.exp(min(ce, 20.0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/workspace/rmsn_data/tokens_wiki103_full.pt")
    p.add_argument("--out", required=True)
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--depth", type=int, default=16)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--ffn-dim", type=int, default=2336)
    p.add_argument("--d-emb", type=int, default=384)
    p.add_argument("--attn-every", type=int, default=2)
    p.add_argument("--chunk-size", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--accum", type=int, default=2)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--adamw-lr", type=float, default=2e-4)
    p.add_argument("--muon-lr", type=float, default=8e-4)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--sinkhorn-steps", type=int, default=10)
    p.add_argument("--gamma-floor", type=float, default=0.0)
    p.add_argument("--w-out-max", type=float, default=2.0)
    p.add_argument("--qk-norm", action="store_true")
    p.add_argument("--v-resid", action="store_true")
    p.add_argument("--grad-ckpt", default="all", choices=["none", "half", "2of3", "all"])
    p.add_argument("--compile", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"

    def log(msg):
        s = f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}"
        print(s, flush=True)
        with open(log_path, "a") as f:
            f.write(s + "\n")

    tokens = torch.load(args.data, map_location="cpu", weights_only=True)
    if isinstance(tokens, dict):
        tokens = tokens.get("tokens", list(tokens.values())[0])
    tokens = tokens.flatten()
    # Exact match to eval_hackathon.py split: last 5% = val, first 95% = train
    n_val = int(len(tokens) * 0.05)
    train_tokens = tokens[:-n_val]
    val_tokens = tokens[-n_val:]

    train_ds = ChunkDataset(train_tokens, args.seq_len, stride=256)
    val_ds = ChunkDataset(val_tokens, args.seq_len, stride=args.seq_len)
    g = torch.Generator().manual_seed(args.seed)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, generator=g, drop_last=True, num_workers=2, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=8, shuffle=False, drop_last=True, num_workers=1, pin_memory=True)
    train_iter = iter(train_dl)

    cfg = {
        "vocab_size": 49152,
        "d_emb": args.d_emb,
        "dim": args.dim,
        "depth": args.depth,
        "n_heads": args.n_heads,
        "ffn_dim": args.ffn_dim,
        "attn_every": args.attn_every,
        "chunk_size": args.chunk_size,
        "max_seq_len": max(args.seq_len, 1024),
        "grad_ckpt": False,
        "dropout": args.dropout,
        "sinkhorn_steps": args.sinkhorn_steps,
        "gamma_floor": args.gamma_floor,
        "w_out_max": args.w_out_max,
        "qk_norm": args.qk_norm,
        "v_resid": args.v_resid,
    }
    ckpt_flag = True if args.grad_ckpt == "all" else (args.grad_ckpt if args.grad_ckpt in ("half", "2of3") else False)
    model = CartanPredLM(**{**cfg, "grad_ckpt": ckpt_flag}).to(device)
    tot_p = sum(p.numel() for p in model.parameters())
    assert tot_p <= 100_000_000, f"Params {tot_p} > 100M!"
    if args.compile:
        aura_net.enable_compiled_kernels()
    opt = HybridMuonAdamW(model, adamw_lr=args.adamw_lr, muon_lr=args.muon_lr, wd=args.wd)
    log(f"CartanPredLM Hackathon | dev={device} | params={tot_p:,} ({tot_p/1e6:.2f}M) | batch={args.batch}x{args.accum} | grad_ckpt={args.grad_ckpt} | groups={opt.counts}")

    ce0, ppl0 = evaluate_fast(model, val_dl, device, max_batches=40)
    log(f"Step 0 | val_ce={ce0:.4f} val_ppl={ppl0:.2f} (lnV={math.log(49152):.2f})")

    torch.cuda.reset_peak_memory_stats(device)
    best_ce = float("inf")
    skipped_nan = 0
    t_win = time.perf_counter()
    toks_win = 0
    ce_win, gn_win = [], []

    model.train()
    for step in range(1, args.steps + 1):
        scale = cosine_lr_scale(step, args.warmup, args.steps, min_ratio=args.min_lr_ratio)
        opt.set_lr_scale(scale)
        opt.zero_grad(set_to_none=True)

        step_ce = 0.0
        bad = False
        last_stats = {}
        for _ in range(args.accum):
            try:
                bx, by = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dl)
                bx, by = next(train_iter)
            bx, by = bx.to(device, non_blocking=True), by.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss, st = model.forward_loss(bx, by)
            if not torch.isfinite(loss):
                bad = True
                break
            (loss / args.accum).backward()
            step_ce += st["ce_loss"] / args.accum
            toks_win += bx.numel()
            last_stats = st

        if bad:
            opt.zero_grad(set_to_none=True)
            skipped_nan += 1
            log(f"WARN step={step}: non-finite loss (skipped={skipped_nan})")
            continue

        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gn):
            opt.zero_grad(set_to_none=True)
            skipped_nan += 1
            log(f"WARN step={step}: non-finite grad_norm={gn}")
            continue

        opt.step()
        ce_win.append(step_ce)
        gn_win.append(float(gn))

        if step % args.log_every == 0 or step == 1:
            dt = max(time.perf_counter() - t_win, 1e-6)
            tps = toks_win / dt
            avg_ce = sum(ce_win) / len(ce_win)
            avg_gn = sum(gn_win) / len(gn_win)
            vram_mb = torch.cuda.max_memory_allocated(device) / 1e6
            log(f"step={step:5d}/{args.steps} | ce={avg_ce:.4f} ppl={math.exp(min(avg_ce,20)):.2f} | gn={avg_gn:.3f} | {tps:.0f} tok/s | vram={vram_mb:.0f}MB | gamma={last_stats.get('cartan_gamma',0):.3f} surp={last_stats.get('surprise_mean',0):.3f} rms={last_stats.get('stream_rms',0):.2f}")
            t_win = time.perf_counter()
            toks_win = 0
            ce_win, gn_win = [], []

        if step % args.eval_every == 0 or step == args.steps:
            vce, vppl = evaluate_fast(model, val_dl, device, max_batches=80)
            if vce < best_ce:
                best_ce = vce
                clean_sd = {k.replace("._orig_mod.", "."): v for k, v in model.state_dict().items()}
                torch.save({"model": clean_sd, "config": cfg, "step": step, "val_ce": vce, "val_ppl": vppl}, out_dir / "ckpt_best.pt")
            log(f"EVAL step={step} | val_ce={vce:.4f} val_ppl={vppl:.2f} | best_ppl={math.exp(min(best_ce,20)):.2f} | skipped_nan={skipped_nan}")

    summary = {
        "params": tot_p,
        "steps": args.steps,
        "best_val_ce": round(best_ce, 5),
        "best_val_ppl": round(math.exp(min(best_ce, 20)), 4),
        "peak_vram_mb": round(torch.cuda.max_memory_allocated(device) / 1e6, 1),
        "skipped_nan": skipped_nan,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"DONE: {json.dumps(summary)}")


if __name__ == "__main__":
    main()
