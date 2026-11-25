"""
DuoNeural Thanksgiving Hackathon 2026 — Shared Evaluation Script
Agreed by Archon + Aura before 07:00Z Nov 25, 2026.

Usage:
    python3 eval_hackathon.py --ckpt /path/to/ckpt.pt \
        --model-module chord_net --model-class CHORDNet \
        --dim 512 --depth 16 \
        --data /workspace/rmsn_data/tokens_wiki103_full.pt \
        --gpu 0

Model contract:
    - model.forward(input_ids: LongTensor[B, T]) -> logits: FloatTensor[B, T, V]
    - model loaded from ckpt["model"] state dict
    - optional ckpt["config"] dict passed as **kwargs to constructor
"""
import argparse, importlib, math, time, torch, sys
from pathlib import Path

# ── fixed eval constants (agreed upon, do not change) ──────────────────────
VAL_FRAC   = 0.05          # last 5% of tokens_wiki103_full.pt
SEQ_LEN    = 512
BATCH_SIZE = 8             # for PPL eval (memory-safe on 8GB)
INF_SEQ    = 256           # autoregressive inference benchmark length
INF_PROMPT = 32            # prompt tokens for inference bench
MAX_PARAMS = 100_000_000   # hard cap: >100M fails
VOCAB_SIZE = 49152         # cosmo2-tokenizer

def load_val_data(data_path):
    toks = torch.load(data_path, map_location="cpu", weights_only=True)
    if isinstance(toks, dict):
        toks = toks.get("tokens", toks.get("val", list(toks.values())[0]))
    toks = toks.flatten().long()
    n_val = int(len(toks) * VAL_FRAC)
    # always take the LAST n_val tokens as val (same split for everyone)
    val_toks = toks[-n_val:]
    return val_toks

def count_params(model):
    return sum(p.numel() for p in model.parameters())

@torch.no_grad()
def eval_ppl(model, val_toks, device, seq_len=SEQ_LEN, batch=BATCH_SIZE):
    model.eval()
    # build (B, T+1) windows
    n = (len(val_toks) - 1) // seq_len * seq_len
    xs = val_toks[:n+1]
    inputs  = xs[:-1].reshape(-1, seq_len)
    targets = xs[1: ].reshape(-1, seq_len)
    # batch eval
    total_loss, total_toks = 0.0, 0
    for i in range(0, len(inputs), batch):
        xb = inputs[i:i+batch].to(device)
        yb = targets[i:i+batch].to(device)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(xb)                 # (B, T, V)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            yb.reshape(-1),
            reduction="sum"
        )
        total_loss += loss.item()
        total_toks += yb.numel()
    val_ce  = total_loss / total_toks          # nats per token
    val_ppl = math.exp(val_ce)
    # BPB: bits per UTF-8 byte
    # for a token-level model: BPB = val_ce / (ln2 * avg_bytes_per_token)
    # wiki103 / cosmo2: average ~4.1 UTF-8 bytes per token (measured)
    AVG_BYTES_PER_TOK = 4.302928  # measured: 27,164,014 bytes / 6,312,914 tokens
    bpb = val_ce / (math.log(2) * AVG_BYTES_PER_TOK)
    return val_ce, val_ppl, bpb, total_toks

@torch.no_grad()
def bench_inference(model, device, prompt_len=INF_PROMPT, gen_len=INF_SEQ):
    model.eval()
    prompt = torch.randint(0, VOCAB_SIZE, (1, prompt_len), device=device)
    # warmup
    for _ in range(3):
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(prompt)
        next_tok = logits[:, -1:, :].argmax(-1)
        prompt_w = torch.cat([prompt[:, 1:], next_tok], dim=1)
        _ = model(prompt_w)
    # time it
    tokens = prompt.clone()
    t0 = time.perf_counter()
    for _ in range(gen_len):
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(tokens)
        next_tok = logits[:, -1:, :].argmax(-1)
        tokens = torch.cat([tokens[:, 1:], next_tok], dim=1)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0
    return gen_len / elapsed   # tok/s

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",         required=True)
    p.add_argument("--model-module", required=True, help="e.g. chord_net or aura_net")
    p.add_argument("--model-class",  required=True, help="e.g. CHORDNet")
    p.add_argument("--data",         default="/workspace/rmsn_data/tokens_wiki103_full.pt")
    p.add_argument("--gpu",          type=int, default=0)
    p.add_argument("--out",          default=None, help="JSON output path")
    # pass-through kwargs for model constructor
    p.add_argument("--dim",    type=int, default=None)
    p.add_argument("--depth",  type=int, default=None)
    p.add_argument("--vocab",  type=int, default=VOCAB_SIZE)
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[eval] device={device}")

    # load checkpoint
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)

    # dynamic import of model class
    sys.path.insert(0, str(Path(args.ckpt).parent))
    sys.path.insert(0, str(Path(args.ckpt).parent.parent))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    mod = importlib.import_module(args.model_module)
    ModelClass = getattr(mod, args.model_class)

    # construct model
    cfg = ckpt.get("config", {})
    if args.dim   is not None: cfg["dim"]   = args.dim
    if args.depth is not None: cfg["depth"] = args.depth
    cfg.setdefault("vocab_size", args.vocab)
    model = ModelClass(**cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    n_params = count_params(model)
    print(f"[eval] params = {n_params:,}")
    if n_params > MAX_PARAMS:
        print(f"[eval] FAIL: {n_params:,} > {MAX_PARAMS:,} param limit!")
        sys.exit(1)

    # vram usage
    torch.cuda.reset_peak_memory_stats(device)
    val_toks = load_val_data(args.data)
    print(f"[eval] val tokens = {len(val_toks):,}")

    print("[eval] running PPL eval...")
    val_ce, val_ppl, bpb, n_toks = eval_ppl(model, val_toks, device)
    peak_mb = torch.cuda.max_memory_allocated(device) / 1e6

    print("[eval] running inference bench...")
    tok_per_sec = bench_inference(model, device)

    # results
    results = {
        "val_ce":      round(val_ce,  5),
        "val_ppl":     round(val_ppl, 4),
        "bpb":         round(bpb,     5),
        "params":      n_params,
        "peak_vram_mb":round(peak_mb, 1),
        "inference_tok_per_s": round(tok_per_sec, 1),
        "eval_tokens": n_toks,
        "ckpt":        args.ckpt,
        "model_class": args.model_class,
    }

    print("\n" + "="*50)
    print("  DuoNeural Hackathon Eval Results")
    print("="*50)
    for k, v in results.items():
        print(f"  {k:<24} {v}")
    print("="*50)

    if args.out:
        import json
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[eval] saved to {args.out}")

if __name__ == "__main__":
    main()
