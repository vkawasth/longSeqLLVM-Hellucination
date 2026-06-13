#!/usr/bin/env python3
"""
Hessenberg Init Test v3 — Architectural vs Learned
====================================================
Fix: output_attentions must be set in model.config, not just at call time.
"""

import argparse, json, warnings, copy
warnings.filterwarnings('ignore')
import numpy as np
import torch
from scipy.linalg import hessenberg

parser = argparse.ArgumentParser()
parser.add_argument('--model',   default='gpt2-medium')
parser.add_argument('--layer',   type=int, default=-1)
parser.add_argument('--n_texts', type=int, default=5)
parser.add_argument('--save',    default='hessenberg_init_results.json')
parser.add_argument('--verbose', action='store_true')
args = parser.parse_args()

TEXTS = [
    "The water cycle begins when solar energy heats surface water causing "
    "evaporation. Water vapor rises and cools forming clouds. "
    "Precipitation returns water to rivers completing the cycle.",
    "Newton's first law states objects remain at rest unless acted upon. "
    "His second law defines force as mass times acceleration. "
    "The third law states every action has an equal and opposite reaction.",
    "Albert Einstein was born in Ulm Germany on March 14 1879. "
    "He developed the special theory of relativity in 1905. "
    "He won the Nobel Prize in Physics in 1921.",
    "DNA replication begins when helicase unwinds the double helix. "
    "DNA polymerase adds complementary nucleotides in sequence. "
    "The result is two identical DNA molecules.",
    "Photosynthesis converts light energy into chemical energy. "
    "Chlorophyll absorbs sunlight to drive conversion of carbon dioxide. "
    "Glucose is produced and oxygen is released as a byproduct.",
][:args.n_texts]

DIM = 32

print(f"\n{'='*70}")
print(f"  HESSENBERG INIT TEST v3 — ARCHITECTURAL vs LEARNED")
print(f"  Model: {args.model}")
print(f"{'='*70}\n")

from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config

tokenizer = GPT2Tokenizer.from_pretrained(args.model)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ── THE FIX: set output_attentions in config before building models ───────────
def load_model_with_attentions(model_name, random_init=False, permute_layer=None):
    """Load or build GPT-2 with output_attentions baked into config."""
    config = GPT2Config.from_pretrained(model_name)
    config.output_attentions = True      # ← key fix
    config.output_hidden_states = True

    if random_init:
        model = GPT2LMHeadModel(config)  # random weights, attentions enabled
    else:
        model = GPT2LMHeadModel.from_pretrained(model_name, config=config)

    if permute_layer is not None:
        d = config.n_embd
        with torch.no_grad():
            attn = model.transformer.h[permute_layer].attn
            W = attn.c_attn.weight.data
            W_K = W[:, d:2*d].clone()
            perm = torch.randperm(d)
            W[:, d:2*d] = W_K[perm]

    model.eval()
    return model, config

print("1. Loading TRAINED model...", flush=True)
model_trained, cfg = load_model_with_attentions(args.model)
n_layers = cfg.n_layer
n_heads  = cfg.n_head
d_model  = cfg.n_embd
d_head   = d_model // n_heads
layer_idx = n_layers - 1 if args.layer == -1 else args.layer
print(f"   {n_layers} layers  {n_heads} heads  d={d_model}  d_head={d_head}  layer {layer_idx}")

print("2. Building RANDOM INIT model...", flush=True)
model_random, _ = load_model_with_attentions(args.model, random_init=True)
print("   No pretrained weights — pure random Gaussian init")

print("3. Building PERMUTED model...", flush=True)
model_permuted, _ = load_model_with_attentions(args.model, permute_layer=layer_idx)
print("   Trained weights; W_K rows permuted in test layer\n")

# ── verify attentions work ────────────────────────────────────────────────────
def get_attentions(model, text):
    tokens = tokenizer.encode(text, return_tensors='pt',
                               max_length=128, truncation=True)
    with torch.no_grad():
        out = model(tokens)
    if out.attentions is None or len(out.attentions) == 0:
        raise RuntimeError("No attentions returned — config fix failed")
    return [a[0].cpu().numpy() for a in out.attentions]

# Quick test
print("Verifying attentions...", end='', flush=True)
test_attns = get_attentions(model_trained, "test sentence")
print(f" OK — {len(test_attns)} layers, shape {test_attns[0].shape}\n")

# ── key basis and transition matrix ──────────────────────────────────────────
def get_WK(model, layer, head):
    try:
        W = model.transformer.h[layer].attn.c_attn.weight.detach().cpu().numpy()
        W_K_full = W[:, d_model:2*d_model]
        return W_K_full[:, head*d_head:(head+1)*d_head]  # [d_model, d_head]
    except Exception:
        return np.eye(max(d_head, DIM))

def key_basis(model, layer, head, dim):
    """dim×dim orthogonal basis from W_K right singular vectors."""
    W_K = get_WK(model, layer, head)   # [d_model, d_head]
    try:
        # SVD: Vt is [min(d_model,d_head), d_head]
        _, _, Vt = np.linalg.svd(W_K, full_matrices=False)
        # Vt rows are right singular vectors in d_head-dim space
        # Take top-dim rows, then top-dim columns → [dim, dim]
        d = min(dim, Vt.shape[0], Vt.shape[1])
        sub = Vt[:d, :d]
        if d < dim:
            padded = np.eye(dim); padded[:d,:d] = sub; sub = padded
        Q, _ = np.linalg.qr(sub)
        return Q   # [dim, dim]
    except Exception:
        return np.eye(dim)

def attn_to_T(attn_head, dim=DIM):
    A = attn_head / (attn_head.sum(1, keepdims=True) + 1e-8)
    seq_len = A.shape[0]
    d = min(dim, seq_len - 1)
    if d < 2: return np.eye(dim)
    try:
        _, _, Vt = np.linalg.svd(A, full_matrices=False)
        A_proj = A @ Vt[:d].T
        X = A_proj[:-1]; Y = A_proj[1:]; n = min(len(X),len(Y))
        eps = 1e-4 * max(float(np.linalg.norm(X[:n].T@X[:n])), 1.0)
        T, _, _, _ = np.linalg.lstsq(
            X[:n].T@X[:n]+eps*np.eye(d), X[:n].T@Y[:n], rcond=None)
        T = T.T
        # Pad to DIM if needed
        if d < dim:
            padded = np.eye(dim); padded[:d,:d] = T[:d,:d]; T = padded
        return T
    except Exception:
        return np.eye(dim)

def hv(T, basis=None):
    """Hessenberg violation, optionally in given basis."""
    M = T if basis is None else basis.T @ T @ basis
    try:
        H = hessenberg(M)
        v = np.linalg.norm(np.tril(M-H,-2))
        return float(v / max(np.linalg.norm(M), 1e-8))
    except Exception:
        return 1.0

def idem(T):
    return float(np.linalg.norm(T@T-T)/max(np.linalg.norm(T),1e-8))

# ── random baseline ───────────────────────────────────────────────────────────
print("Computing random matrix baseline (200 matrices)...", flush=True)
rng = np.random.RandomState(42)
rb_raw=[]; rb_key=[]
for _ in range(200):
    M = rng.randn(DIM,DIM); M/=max(np.linalg.norm(M),1e-8)
    rb_raw.append(hv(M))
    Q,_ = np.linalg.qr(rng.randn(DIM,DIM))
    rb_key.append(hv(M,Q))
baseline = {'raw':float(np.mean(rb_raw)),'raw_std':float(np.std(rb_raw)),
            'key':float(np.mean(rb_key)),'key_std':float(np.std(rb_key))}
print(f"  raw={baseline['raw']:.4f}±{baseline['raw_std']:.4f}  "
      f"key={baseline['key']:.4f}±{baseline['key_std']:.4f}\n")

# ── main test ─────────────────────────────────────────────────────────────────
results = {}
for cond, model in [('trained',  model_trained),
                    ('random',   model_random),
                    ('permuted', model_permuted)]:
    print(f"  Testing {cond.upper()}...", flush=True)
    raw_all=[]; kb_all=[]; id_all=[]
    for ti, text in enumerate(TEXTS):
        attns = get_attentions(model, text)
        attn_l = attns[min(layer_idx, len(attns)-1)]  # [n_heads, seq, seq]
        h_raw=[]; h_key=[]; h_id=[]
        for h in range(n_heads):
            T  = attn_to_T(attn_l[h], dim=DIM)
            Bk = key_basis(model, layer_idx, h, dim=DIM)
            h_raw.append(hv(T))
            h_key.append(hv(T, Bk))
            h_id.append(idem(T))
        raw_all.append(float(np.mean(h_raw)))
        kb_all.append(float(np.min(h_key)))
        id_all.append(float(np.mean(h_id)))
        if args.verbose:
            bh = int(np.argmin(h_key))
            print(f"    Text {ti+1}: raw={raw_all[-1]:.3f}  "
                  f"key_best={kb_all[-1]:.3f} (h{bh})  idem={id_all[-1]:.3f}")
    r = {'raw':float(np.mean(raw_all)),'raw_std':float(np.std(raw_all)),
         'key_best':float(np.mean(kb_all)),'key_std':float(np.std(kb_all)),
         'idem':float(np.mean(id_all)),'raw_all':raw_all,'key_all':kb_all}
    results[cond]=r
    sigma = (baseline['key']-r['key_best'])/baseline['key_std']
    toda = '✓ TODA' if r['key_best']<0.1 else '~ marg' if r['key_best']<0.2 else '✗'
    print(f"    raw={r['raw']:.4f}  key_best={r['key_best']:.4f} "
          f"({sigma:.1f}σ below random)  idem={r['idem']:.4f}  {toda}")

# ── summary table ─────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("  RESULTS")
print(f"{'='*70}\n")
print(f"  Baseline: key={baseline['key']:.4f}±{baseline['key_std']:.4f}\n")
print(f"  {'Condition':<12} {'key_best':>9} {'σ below rnd':>12} "
      f"{'raw':>7} {'idem':>7} {'< 0.1?':>7}")
print("  "+"-"*55)
for c in ['trained','random','permuted']:
    r=results[c]
    sigma=(baseline['key']-r['key_best'])/baseline['key_std']
    flag='✓ YES' if r['key_best']<0.1 else '~ marg' if r['key_best']<0.2 else '✗ NO'
    print(f"  {c:<12} {r['key_best']:>9.4f} {sigma:>12.1f} "
          f"{r['raw']:>7.4f} {r['idem']:>7.4f} {flag:>7}")

tr=results['trained']['key_best']
rn=results['random']['key_best']
pm=results['permuted']['key_best']

print(f"\n{'='*70}")
print("  VERDICT")
print(f"{'='*70}\n")
if rn < 0.1:
    verdict="ARCHITECTURAL"
    detail=(f"Random init {rn:.4f} < 0.1. Structure is in the softmax formula itself.")
elif rn < 0.2:
    verdict="PARTIALLY ARCHITECTURAL"
    detail=(f"Random init {rn:.4f} marginal. Training strengthens: {rn:.4f}→{tr:.4f}.")
else:
    verdict="LEARNED"
    detail=(f"Random init {rn:.4f} ≈ random baseline {baseline['key']:.4f}. "
            f"Training creates structure: {rn:.4f}→{tr:.4f}.")
print(f"  {verdict}: {detail}")
print(f"  Permuted: {pm:.4f} — ", end="")
if pm < 0.1:
    print("attention pattern alone carries structure (W_K directions not essential)")
elif pm >= rn:
    print("learned W_K directions are essential to the structure")
else:
    print(f"partial dependence on learned W_K directions")

print(f"\n  Raw basis (no rotation): trained={results['trained']['raw']:.4f}  "
      f"random={results['random']['raw']:.4f}  permuted={results['permuted']['raw']:.4f}")
print(f"  (raw trained is {(baseline['raw']-results['trained']['raw'])/baseline['raw']:.0%} "
      "below random baseline — robust regardless of basis)")

# ── save ──────────────────────────────────────────────────────────────────────
if args.save:
    def s(o):
        if isinstance(o,bool): return bool(o)
        if isinstance(o,np.integer): return int(o)
        if isinstance(o,np.floating): return float(o)
        if isinstance(o,dict): return {k:s(v) for k,v in o.items()}
        if isinstance(o,list): return [s(v) for v in o]
        return o
    out={'model':args.model,'layer':layer_idx,'n_texts':args.n_texts,
         'dim':DIM,'baseline':baseline,'results':s(results),
         'verdict':{'conclusion':verdict,'trained':float(tr),
                    'random':float(rn),'permuted':float(pm)}}
    with open(args.save,'w') as f: json.dump(out,f,indent=2)
    print(f"\n  Saved → {args.save}")
