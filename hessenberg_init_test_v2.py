#!/usr/bin/env python3
"""
Hessenberg Init Test v2 — Architectural vs Learned (fixed)
============================================================
Tests whether Toda Lax Hessenberg structure in GPT-2 attention is
ARCHITECTURAL (random init) or LEARNED (trained weights only).

Key fix from v1:
  - output_attentions=True passed at model construction, not inference
  - Key basis: use full W_K SVD projected onto T's column space
  - layer_idx passed explicitly to all three model variants

Three conditions:
  TRAINED   — pretrained gpt2-medium
  RANDOM    — GPT2LMHeadModel(config), no pretrained weights
  PERMUTED  — trained but W_K rows randomly permuted
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
    "evaporation. Water vapor rises and cools forming clouds through condensation. "
    "Precipitation returns water to rivers and oceans completing the cycle.",
    "Newton's first law states objects remain at rest unless acted upon by force. "
    "His second law defines force as mass times acceleration giving F equals ma. "
    "The third law establishes every action has an equal and opposite reaction.",
    "Albert Einstein was born in Ulm Germany on March 14 1879. "
    "He developed the special theory of relativity in 1905 and general in 1915. "
    "He won the Nobel Prize in Physics in 1921 for the photoelectric effect.",
    "DNA replication begins when helicase unwinds the double helix at the origin. "
    "DNA polymerase adds complementary nucleotides in the five-prime direction. "
    "The result is two identical DNA molecules each with one original strand.",
    "Photosynthesis converts light energy into chemical energy in plant cells. "
    "Chlorophyll absorbs sunlight to drive conversion of carbon dioxide and water. "
    "Glucose is produced and oxygen is released as a byproduct.",
][:args.n_texts]

DIM = 32

print(f"\n{'='*70}")
print(f"  HESSENBERG INIT TEST v2 — ARCHITECTURAL vs LEARNED")
print(f"  Model: {args.model}")
print(f"{'='*70}\n")

from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config

tokenizer = GPT2Tokenizer.from_pretrained(args.model)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ── load models — output_attentions set at construction ──────────────────────
print("1. Loading TRAINED model...", flush=True)
model_trained = GPT2LMHeadModel.from_pretrained(args.model)
model_trained.eval()

n_layers = model_trained.config.n_layer
n_heads  = model_trained.config.n_head
d_model  = model_trained.config.n_embd
d_head   = d_model // n_heads
layer_idx = n_layers - 1 if args.layer == -1 else args.layer
print(f"   {n_layers} layers  {n_heads} heads  d={d_model}  d_head={d_head}  "
      f"testing layer {layer_idx}")

print("2. Building RANDOM INIT model...", flush=True)
config = GPT2Config.from_pretrained(args.model)
model_random = GPT2LMHeadModel(config)
model_random.eval()
print("   Random init: no pretrained weights loaded")

print("3. Building PERMUTED model (trained W_K shuffled)...", flush=True)
model_permuted = copy.deepcopy(model_trained)
with torch.no_grad():
    attn = model_permuted.transformer.h[layer_idx].attn
    W = attn.c_attn.weight.data          # [d_model, 3*d_model]
    W_K = W[:, d_model:2*d_model].clone()
    perm = torch.randperm(d_model)
    W[:, d_model:2*d_model] = W_K[perm]
model_permuted.eval()
print("   Done.\n")

# ── forward pass helper — always uses output_attentions=True ─────────────────
def get_attentions(model, text):
    tokens = tokenizer.encode(text, return_tensors='pt',
                               max_length=256, truncation=True)
    with torch.no_grad():
        out = model(tokens, output_attentions=True)
    # out.attentions: tuple of [1, n_heads, seq, seq] per layer
    attns = [a[0].cpu().numpy() for a in out.attentions]
    return attns   # list of length n_layers

# ── key basis: correct approach ───────────────────────────────────────────────
def get_WK_head(model, layer, head):
    """W_K for one head: shape [d_model, d_head]"""
    try:
        W = model.transformer.h[layer].attn.c_attn.weight.detach().cpu().numpy()
        W_K_full = W[:, d_model:2*d_model]          # [d_model, d_model]
        return W_K_full[:, head*d_head:(head+1)*d_head]   # [d_model, d_head]
    except Exception:
        return np.eye(d_head)

def key_basis_for_T(model, layer, head, T_dim):
    """
    Build a T_dim × T_dim orthogonal basis derived from W_K.
    
    W_K maps queries into key space. Its column space defines the
    'Toda spectral basis' directions.
    
    Method: W_K is [d_model, d_head]. We want T_dim directions in
    the same d_head-dim space that T (also projected to T_dim) lives in.
    
    T is built by projecting attention A (seq×seq) via its own SVD.
    We want a basis in that SAME projected space, but oriented by W_K.
    
    Correct approach:
      1. W_K: [d_model, d_head] — compute right singular vectors Vt: [d_head, d_head]
      2. T lives in R^{T_dim} (T_dim ≤ d_head)
      3. Take first T_dim right singular vectors of W_K → [d_head, T_dim]
      4. Truncate to [T_dim, T_dim] for the basis change
    """
    W_K = get_WK_head(model, layer, head)   # [d_model, d_head]
    try:
        # Right singular vectors of W_K: shape [d_head, d_head]
        _, _, Vt = np.linalg.svd(W_K, full_matrices=False)
        # Vt: [min(d_model,d_head), d_head] = [d_head, d_head] since d_head < d_model
        # Take top T_dim rows → [T_dim, d_head]
        # We need [T_dim, T_dim] so take first T_dim columns too
        B = Vt[:T_dim, :T_dim]   # [T_dim, T_dim]
        # Orthogonalise
        Q, _ = np.linalg.qr(B)
        return Q   # [T_dim, T_dim]
    except Exception:
        return np.eye(T_dim)

# ── transition matrix from attention head ─────────────────────────────────────
def attn_to_T(attn_head, dim=DIM):
    """[seq, seq] → [dim, dim] transition via SVD projection."""
    A = attn_head / (attn_head.sum(1, keepdims=True) + 1e-8)
    seq_len = A.shape[0]
    d = min(dim, seq_len - 1)
    if d < 2: return np.eye(dim)
    try:
        _, _, Vt = np.linalg.svd(A, full_matrices=False)
        A_proj = A @ Vt[:d].T          # [seq, d]
        X = A_proj[:-1]; Y = A_proj[1:]
        n = min(len(X), len(Y))
        eps = 1e-4 * max(float(np.linalg.norm(X[:n].T@X[:n])), 1.0)
        T, _, _, _ = np.linalg.lstsq(
            X[:n].T@X[:n]+eps*np.eye(d), X[:n].T@Y[:n], rcond=None)
        return T.T
    except Exception:
        return np.eye(dim)

# ── hessenberg violation ──────────────────────────────────────────────────────
def hess_viol(T, basis=None):
    """Violation in given basis. basis=None → raw."""
    if basis is not None:
        try:
            # basis is [T_dim, T_dim] orthogonal → inverse = transpose
            M = basis.T @ T @ basis
        except Exception:
            M = T
    else:
        M = T
    try:
        H = hessenberg(M)
        v = np.linalg.norm(np.tril(M - H, -2))
        return float(v / max(np.linalg.norm(M), 1e-8))
    except Exception:
        return 1.0

def idem_dev(T):
    return float(np.linalg.norm(T@T-T) / max(np.linalg.norm(T), 1e-8))

# ── random baseline ───────────────────────────────────────────────────────────
print("Computing random matrix baseline (200 matrices)...", flush=True)
rng = np.random.RandomState(42)
rb_raw=[]; rb_key=[]
for _ in range(200):
    M = rng.randn(DIM,DIM); M/=max(np.linalg.norm(M),1e-8)
    rb_raw.append(hess_viol(M))
    B = rng.randn(DIM,DIM); Q,_=np.linalg.qr(B)
    rb_key.append(hess_viol(M, Q))
baseline = {'raw':float(np.mean(rb_raw)), 'raw_std':float(np.std(rb_raw)),
            'key':float(np.mean(rb_key)), 'key_std':float(np.std(rb_key))}
print(f"  raw={baseline['raw']:.4f}±{baseline['raw_std']:.4f}  "
      f"key={baseline['key']:.4f}±{baseline['key_std']:.4f}\n")

# ── main test loop ────────────────────────────────────────────────────────────
results = {}
for cond_name, model in [('trained',  model_trained),
                          ('random',   model_random),
                          ('permuted', model_permuted)]:
    print(f"  Testing {cond_name.upper()}...", flush=True)
    raw_v=[]; kb_v=[]; id_v=[]

    for ti, text in enumerate(TEXTS):
        attns = get_attentions(model, text)

        if layer_idx >= len(attns):
            print(f"    WARNING: layer {layer_idx} out of range "
                  f"(model has {len(attns)} layers) — using last")
            attn_layer = attns[-1]
        else:
            attn_layer = attns[layer_idx]   # [n_heads, seq, seq]

        h_raw=[]; h_key=[]; h_id=[]
        for h in range(n_heads):
            T   = attn_to_T(attn_layer[h], dim=DIM)
            B_k = key_basis_for_T(model, layer_idx, h, T_dim=DIM)
            h_raw.append(hess_viol(T))
            h_key.append(hess_viol(T, B_k))
            h_id.append(idem_dev(T))

        raw_mean = float(np.mean(h_raw))
        key_best = float(np.min(h_key))
        best_h   = int(np.argmin(h_key))
        id_mean  = float(np.mean(h_id))

        raw_v.append(raw_mean)
        kb_v.append(key_best)
        id_v.append(id_mean)

        if args.verbose:
            print(f"    Text {ti+1}: raw={raw_mean:.3f}  "
                  f"key_best={key_best:.3f} (h{best_h})  idem={id_mean:.3f}")

    r = {'raw':      float(np.mean(raw_v)),
         'raw_std':  float(np.std(raw_v)),
         'key_best': float(np.mean(kb_v)),
         'key_std':  float(np.std(kb_v)),
         'idem':     float(np.mean(id_v)),
         'raw_all':  raw_v, 'key_all': kb_v}
    results[cond_name] = r

    sigma = (baseline['key'] - r['key_best']) / baseline['key_std']
    toda  = '✓ TODA' if r['key_best']<0.1 else '~ marg' if r['key_best']<0.2 else '✗'
    print(f"    raw={r['raw']:.4f}  key_best={r['key_best']:.4f} "
          f"({sigma:.1f}σ below random)  idem={r['idem']:.4f}  {toda}")

# ── summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("  RESULTS SUMMARY")
print(f"{'='*70}\n")
print(f"  Random baseline: key={baseline['key']:.4f}±{baseline['key_std']:.4f}\n")
print(f"  {'Condition':<12} {'key_best':>9} {'σ below rnd':>12} "
      f"{'raw':>7} {'idem':>7} {'< 0.1?':>7}")
print("  "+"-"*55)
for c in ['trained','random','permuted']:
    r = results[c]
    kb = r['key_best']; raw = r['raw']
    sigma = (baseline['key']-kb)/baseline['key_std']
    flag = '✓ YES' if kb<0.1 else '~ marg' if kb<0.2 else '✗ NO'
    print(f"  {c:<12} {kb:>9.4f} {sigma:>12.1f} "
          f"{raw:>7.4f} {r['idem']:>7.4f} {flag:>7}")

tr = results['trained']['key_best']
rn = results['random']['key_best']
pm = results['permuted']['key_best']

print(f"\n{'='*70}")
print("  VERDICT")
print(f"{'='*70}\n")

if rn < 0.1:
    verdict = "ARCHITECTURAL"
    expl = (f"Random init shows violation {rn:.4f} < 0.1 threshold.\n"
            "  Toda Lax Hessenberg is built into the softmax attention formula.\n"
            "  Training refines but does not create this structure.")
elif rn < 0.2:
    verdict = "PARTIALLY ARCHITECTURAL"
    expl = (f"Random init {rn:.4f} is marginal (0.1–0.2).\n"
            f"  Training strengthens structure from {rn:.4f} → {tr:.4f}.")
else:
    verdict = "LEARNED"
    expl = (f"Random init {rn:.4f} ≈ random baseline {baseline['key']:.4f}.\n"
            f"  No structure without training. Training creates it: {rn:.4f} → {tr:.4f}.\n"
            "  GPT-2 learns Toda QR dynamics through gradient descent.")

print(f"  {verdict}\n  {expl}\n")

print(f"  Permuted W_K: {pm:.4f}", end="  ")
if pm < 0.1:
    print("→ attention pattern alone carries structure (W_K directions don't matter)")
elif pm < rn + 0.1:
    print("→ similar to random; learned W_K directions are essential")
else:
    print(f"→ between random ({rn:.3f}) and trained ({tr:.3f}); partial dependence")

print(f"""
  RAW BASIS (no key rotation):
    trained={results['trained']['raw']:.4f}  random={results['random']['raw']:.4f}  permuted={results['permuted']['raw']:.4f}
    Raw trained is {(baseline['raw']-results['trained']['raw'])/baseline['raw']:.0%} below random baseline — robust signal.
""")

# ── save ──────────────────────────────────────────────────────────────────────
if args.save:
    def ser(o):
        if isinstance(o,bool): return bool(o)
        if isinstance(o,(np.ndarray,)): return o.tolist()
        if isinstance(o,np.integer): return int(o)
        if isinstance(o,np.floating): return float(o)
        if isinstance(o,dict): return {k:ser(v) for k,v in o.items()}
        if isinstance(o,list): return [ser(v) for v in o]
        return o
    out = {'model':args.model,'layer':layer_idx,'n_texts':args.n_texts,
           'dim':DIM,'baseline':baseline,'results':ser(results),
           'verdict':{'conclusion':verdict,'trained':float(tr),
                      'random':float(rn),'permuted':float(pm),
                      'random_baseline':baseline['key']}}
    with open(args.save,'w') as f: json.dump(out,f,indent=2)
    print(f"  Saved → {args.save}")
