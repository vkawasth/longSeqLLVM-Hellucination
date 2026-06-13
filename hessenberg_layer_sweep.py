#!/usr/bin/env python3
"""
Hessenberg Layer Sweep — Cross-layer spectral stability test
============================================================
The decisive experiment to move from "analogy" to "mechanism".

Tests ALL layers of gpt2-medium. For each layer measures:
  1. Raw Hessenberg violation (does it decrease with depth?)
  2. Approximate spectral preservation λ(T_ℓ) ≈ λ(T_{ℓ+1})
  3. Idempotency deviation (does T² approach T with depth?)
  4. Attention entropy (is softmax concentration the driver?)

Predictions if Toda/QR interpretation is correct:
  A. Hessenberg violation decreases monotonically with layer depth
  B. Spectral distance between consecutive layers decreases with depth
  C. Idempotency deviation decreases with depth (T²→T at deep layers)
  D. Attention entropy decreases with depth (softmax concentrates)

If any of A-D hold strongly, the "depth = QR iterations" claim
is supported. If none hold, we have Hessenberg structure only in
the final layer — a weaker but still interesting result.

Usage:
    python hessenberg_layer_sweep.py
    python hessenberg_layer_sweep.py --model gpt2-medium --save sweep.json
"""

import argparse, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
from scipy.linalg import hessenberg
from scipy.stats import spearmanr

parser = argparse.ArgumentParser()
parser.add_argument('--model',   default='gpt2-medium')
parser.add_argument('--n_texts', type=int, default=5)
parser.add_argument('--save',    default='hessenberg_sweep.json')
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
    "He developed special relativity in 1905 and general relativity in 1915. "
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
print(f"  HESSENBERG LAYER SWEEP — ALL LAYERS")
print(f"  Model: {args.model}  |  Texts: {args.n_texts}")
print(f"{'='*70}\n")

from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config

tokenizer = GPT2Tokenizer.from_pretrained(args.model)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

config = GPT2Config.from_pretrained(args.model)
config.output_attentions = True
model = GPT2LMHeadModel.from_pretrained(args.model, config=config)
model.eval()

n_layers = model.config.n_layer
n_heads  = model.config.n_head
d_model  = model.config.n_embd
print(f"  {n_layers} layers  {n_heads} heads  d={d_model}\n")

# ── helpers ───────────────────────────────────────────────────────────────────
def get_all_attentions(text):
    tokens = tokenizer.encode(text, return_tensors='pt',
                               max_length=256, truncation=True)
    with torch.no_grad():
        out = model(tokens)
    return [a[0].cpu().numpy() for a in out.attentions]  # list[n_layers] of [n_heads,seq,seq]

def attn_to_T(A_head, dim=DIM):
    A = A_head / (A_head.sum(1, keepdims=True) + 1e-8)
    seq_len = A.shape[0]
    d = min(dim, seq_len - 1)
    if d < 2: return np.eye(dim), np.ones(dim)*float('nan')
    try:
        _, _, Vt = np.linalg.svd(A, full_matrices=False)
        A_proj = A @ Vt[:d].T
        X = A_proj[:-1]; Y = A_proj[1:]; n = min(len(X),len(Y))
        eps = 1e-4*max(float(np.linalg.norm(X[:n].T@X[:n])),1.0)
        T, _,_,_ = np.linalg.lstsq(X[:n].T@X[:n]+eps*np.eye(d),
                                     X[:n].T@Y[:n], rcond=None)
        T = T.T
        if d < dim:
            padded=np.eye(dim); padded[:d,:d]=T[:d,:d]; T=padded
        # eigenvalues of T (for spectral stability)
        eigs = np.sort(np.abs(np.linalg.eigvals(T)))[::-1]
        return T, eigs
    except Exception:
        return np.eye(dim), np.ones(dim)*float('nan')

def hess_viol_raw(T):
    try:
        H = hessenberg(T)
        v = np.linalg.norm(np.tril(T-H,-2))
        return float(v/max(np.linalg.norm(T),1e-8))
    except: return 1.0

def idem_dev(T):
    return float(np.linalg.norm(T@T-T)/max(np.linalg.norm(T),1e-8))

def attn_entropy(A_head):
    """Mean row entropy of attention matrix (lower = more concentrated)."""
    A = A_head / (A_head.sum(1,keepdims=True)+1e-8)
    # Only rows with non-trivial attention (avoid pure causal mask zeros)
    eps = 1e-9
    ent = -np.sum(A * np.log(A + eps), axis=1)
    return float(np.mean(ent))

def spectral_distance(eigs1, eigs2):
    """L2 distance between sorted eigenvalue magnitudes."""
    valid = ~(np.isnan(eigs1) | np.isnan(eigs2))
    if valid.sum() < 4: return float('nan')
    return float(np.linalg.norm(eigs1[valid] - eigs2[valid]))

# ── layer sweep ────────────────────────────────────────────────────────────────
print("  Running sweep across all layers...\n")
print(f"  {'Layer':>6} {'hess_viol':>10} {'idem':>7} {'entropy':>8} "
      f"{'spec_dist':>10}  Trend")
print("  "+"-"*55)

layer_data = []   # one entry per layer, averaged over texts and heads

for layer_idx in range(n_layers):
    hv_vals=[]; id_vals=[]; ent_vals=[]; spec_dists=[]
    prev_eigs_by_head = {}

    for text in TEXTS:
        attns = get_all_attentions(text)
        attn_l = attns[layer_idx]   # [n_heads, seq, seq]

        for h in range(n_heads):
            T, eigs = attn_to_T(attn_l[h], dim=DIM)
            hv_vals.append(hess_viol_raw(T))
            id_vals.append(idem_dev(T))
            ent_vals.append(attn_entropy(attn_l[h]))

            # Spectral distance to PREVIOUS layer
            if layer_idx > 0 and (text, h) in prev_eigs_by_head:
                sd = spectral_distance(prev_eigs_by_head[(text,h)], eigs)
                if not np.isnan(sd):
                    spec_dists.append(sd)
            prev_eigs_by_head[(text,h)] = eigs

    mean_hv  = float(np.mean(hv_vals))
    mean_id  = float(np.mean(id_vals))
    mean_ent = float(np.mean(ent_vals))
    mean_sd  = float(np.mean(spec_dists)) if spec_dists else float('nan')

    layer_data.append({
        'layer': layer_idx,
        'hess_viol': mean_hv,
        'idem': mean_id,
        'entropy': mean_ent,
        'spec_dist': mean_sd,
    })

    trend = ('↓' if mean_hv < 0.1 else
             '↓' if len(layer_data) > 1 and mean_hv < layer_data[-2]['hess_viol'] else
             '→' if len(layer_data) > 1 and abs(mean_hv - layer_data[-2]['hess_viol']) < 0.01 else '↑')
    sd_str = f"{mean_sd:.4f}" if not np.isnan(mean_sd) else "  ---"
    print(f"  L{layer_idx:>4}:  {mean_hv:>10.4f} {mean_id:>7.4f} "
          f"{mean_ent:>8.4f} {sd_str:>10}  {trend}")

# ── analysis ───────────────────────────────────────────────────────────────────
layers = [d['layer'] for d in layer_data]
hvs    = [d['hess_viol'] for d in layer_data]
ids    = [d['idem'] for d in layer_data]
ents   = [d['entropy'] for d in layer_data]
sds    = [d['spec_dist'] for d in layer_data if not np.isnan(d['spec_dist'])]

print(f"\n{'='*70}")
print("  TREND ANALYSIS")
print(f"{'='*70}\n")

# Spearman correlation with depth
r_hv,  p_hv  = spearmanr(layers, hvs)
r_id,  p_id  = spearmanr(layers, ids)
r_ent, p_ent = spearmanr(layers, ents)

print(f"  Hessenberg violation vs depth: r={r_hv:+.3f}  p={p_hv:.4f}  "
      f"{'↓ decreases with depth ✓' if r_hv<-0.3 and p_hv<0.05 else '→ no clear trend'}")
print(f"  Idempotency deviation vs depth: r={r_id:+.3f}  p={p_id:.4f}  "
      f"{'↓ T→T² with depth ✓' if r_id<-0.3 and p_id<0.05 else '→ no clear trend'}")
print(f"  Attention entropy vs depth:     r={r_ent:+.3f}  p={p_ent:.4f}  "
      f"{'↓ concentrates with depth ✓' if r_ent<-0.3 and p_ent<0.05 else '→ no clear trend'}")

# Layers below 0.1
below_01 = [d['layer'] for d in layer_data if d['hess_viol'] < 0.1]
print(f"\n  Layers with hess_viol < 0.1: {below_01}")
print(f"  ({len(below_01)}/{n_layers} = {len(below_01)/n_layers:.0%} of all layers)")

# Best and worst
best_layer  = min(layer_data, key=lambda x: x['hess_viol'])
worst_layer = max(layer_data, key=lambda x: x['hess_viol'])
print(f"  Best layer:  L{best_layer['layer']} = {best_layer['hess_viol']:.4f}")
print(f"  Worst layer: L{worst_layer['layer']} = {worst_layer['hess_viol']:.4f}")

# Monotone fraction
mono = sum(1 for i in range(1, len(hvs)) if hvs[i] <= hvs[i-1])
print(f"  Monotone decreasing fraction: {mono}/{n_layers-1} = {mono/(n_layers-1):.0%}")

print(f"\n{'='*70}")
print("  VERDICT ON QR-ITERATION CLAIM")
print(f"{'='*70}\n")

supports_qr = (r_hv < -0.4 and p_hv < 0.05) or (len(below_01) > n_layers//2)
if r_hv < -0.4 and p_hv < 0.05:
    print("  ✓ SUPPORTS QR ITERATION: Hessenberg violation decreases monotonically")
    print("    with depth. Consistent with each layer performing one QR step.")
    print(f"    Spearman r={r_hv:.3f}, p={p_hv:.4f}")
elif len(below_01) > n_layers // 4:
    print("  ~ PARTIAL SUPPORT: Multiple layers satisfy Hessenberg < 0.1")
    print(f"    but no clear monotone decrease (r={r_hv:.3f}, p={p_hv:.4f})")
    print("    Consistent with 'each layer is Hessenberg' but not 'depth = QR steps'")
else:
    print("  ✗ DOES NOT SUPPORT QR ITERATION ACROSS LAYERS")
    print(f"    Only {len(below_01)} layer(s) below 0.1 threshold.")
    print("    The Hessenberg result is specific to the last layer, not progressive.")
    print("    Downgrade: 'final layer exhibits Hessenberg structure consistent")
    print("    with integrable dynamics' not 'each layer performs one QR step'")

if r_ent < -0.3 and p_ent < 0.05:
    print(f"\n  ✓ SOFTMAX CONCENTRATION INCREASES WITH DEPTH (r={r_ent:.3f})")
    print("    Supports: attention entropy decreasing = Hessenberg bias growing")
    print("    This is the mechanistic link: entropy → concentration → Hessenberg")

print(f"\n  RECOMMENDED PAPER LANGUAGE:")
if supports_qr:
    print("  'GPT-2-medium attention exhibits layer-progressive Hessenberg structure")
    print("   consistent with a discrete Toda/QR flow: violation decreases from")
    print(f"  {max(hvs[:3]):.3f} (early layers) to {min(hvs[-3:]):.3f} (deep layers),'")
    print("  'supporting the depth-as-QR-iterations interpretation.'")
else:
    print("  'The final layer (L23) of GPT-2-medium exhibits Hessenberg structure")
    print("   (violation 0.081) consistent with a Toda/QR-like dynamical regime.")
    print("   Whether this structure progressively accumulates across layers")
    print("   requires further cross-layer spectral stability measurements.'")

# ── save ──────────────────────────────────────────────────────────────────────
if args.save:
    def s(o):
        if isinstance(o,np.integer): return int(o)
        if isinstance(o,np.floating): return float(o)
        if isinstance(o,dict): return {k:s(v) for k,v in o.items()}
        if isinstance(o,list): return [s(v) for v in o]
        return o
    out = {'model':args.model,'n_layers':n_layers,'n_texts':args.n_texts,
           'dim':DIM,'layer_data':s(layer_data),
           'trends':{'hess_r':float(r_hv),'hess_p':float(p_hv),
                     'idem_r':float(r_id),'idem_p':float(p_id),
                     'ent_r':float(r_ent),'ent_p':float(p_ent)},
           'below_01_layers':below_01,'supports_qr':bool(supports_qr)}
    with open(args.save,'w') as f: json.dump(out,f,indent=2)
    print(f"\n  Saved → {args.save}")
