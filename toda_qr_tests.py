#!/usr/bin/env python3
"""
Toda/QR Confirmation Tests
===========================
Two tests to move from "consistent with" to "implements":

TEST 1 — ISOSPECTRALITY
  For each consecutive layer pair (ℓ, ℓ+1):
  Compute eigenvalues of T_ℓ and T_{ℓ+1}.
  Measure spectral distance ||sort|λ(T_ℓ)| − sort|λ(T_{ℓ+1})||.
  Prediction: small and DECREASING with depth.
  If eigenvalues are conserved across layers → Toda isospectral flow confirmed.

TEST 2 — QR PREDICTION
  For each layer ℓ:
  QR-factorise T_ℓ = Q_ℓ R_ℓ.
  Predict T_{ℓ+1} ≈ R_ℓ Q_ℓ  (the QR update rule).
  Measure ||T_{ℓ+1} − R_ℓ Q_ℓ||_F / ||T_{ℓ+1}||_F.
  Compare to random baseline: ||T_{ℓ+1} − T_random||_F / ||T_{ℓ+1}||_F.
  Prediction: QR error << random baseline.
  If QR error < random → A_{ℓ+1} ≈ R_ℓ Q_ℓ directly confirmed.

Usage:
    python toda_qr_tests.py
    python toda_qr_tests.py --model gpt2-medium --save toda_results.json
"""

import argparse, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
from scipy.stats import spearmanr

parser = argparse.ArgumentParser()
parser.add_argument('--model',   default='gpt2-medium')
parser.add_argument('--n_texts', type=int, default=5)
parser.add_argument('--save',    default='toda_qr_results.json')
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
print(f"  TODA/QR CONFIRMATION TESTS")
print(f"  Test 1: Isospectrality  |  Test 2: QR Prediction")
print(f"  Model: {args.model}")
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
    return [a[0].cpu().numpy() for a in out.attentions]

def attn_to_T(A_head, dim=DIM):
    """[seq,seq] → [dim,dim] transition matrix via SVD projection."""
    A = A_head / (A_head.sum(1, keepdims=True) + 1e-8)
    seq_len = A.shape[0]
    d = min(dim, seq_len - 1)
    if d < 2: return np.eye(dim)
    try:
        _, _, Vt = np.linalg.svd(A, full_matrices=False)
        A_proj = A @ Vt[:d].T
        X = A_proj[:-1]; Y = A_proj[1:]; n = min(len(X),len(Y))
        eps = 1e-4*max(float(np.linalg.norm(X[:n].T@X[:n])),1.0)
        T,_,_,_ = np.linalg.lstsq(X[:n].T@X[:n]+eps*np.eye(d),
                                    X[:n].T@Y[:n],rcond=None)
        T = T.T
        if d < dim:
            padded=np.eye(dim); padded[:d,:d]=T[:d,:d]; T=padded
        return T
    except Exception:
        return np.eye(dim)

def spectral_distance(T1, T2):
    """||sort|λ(T1)| − sort|λ(T2)|||  — signed eigenvalue mismatch."""
    try:
        e1 = np.sort(np.abs(np.linalg.eigvals(T1)))[::-1]
        e2 = np.sort(np.abs(np.linalg.eigvals(T2)))[::-1]
        return float(np.linalg.norm(e1 - e2))
    except Exception:
        return float('nan')

def qr_prediction_error(T_ell, T_next):
    """
    QR update prediction error:
      T_ell = Q R  (QR decomp)
      T_predicted = R Q
      error = ||T_next - T_predicted||_F / ||T_next||_F
    """
    try:
        Q, R = np.linalg.qr(T_ell)
        T_pred = R @ Q                 # the QR Toda update
        err = np.linalg.norm(T_next - T_pred)
        return float(err / max(np.linalg.norm(T_next), 1e-8)), T_pred
    except Exception:
        return float('nan'), T_ell

def random_prediction_error(T_next, rng):
    """Baseline: how well does a random matrix predict T_next?"""
    T_rand = rng.randn(*T_next.shape)
    T_rand /= max(np.linalg.norm(T_rand), 1e-8)
    T_rand *= np.linalg.norm(T_next)   # match scale
    err = np.linalg.norm(T_next - T_rand)
    return float(err / max(np.linalg.norm(T_next), 1e-8))

# ── random baseline for spectral distance ─────────────────────────────────────
print("Computing random baselines...", flush=True)
rng = np.random.RandomState(42)
rand_spec_dists = []
rand_qr_errors  = []
for _ in range(200):
    M1 = rng.randn(DIM,DIM); M1/=max(np.linalg.norm(M1),1e-8)
    M2 = rng.randn(DIM,DIM); M2/=max(np.linalg.norm(M2),1e-8)*np.linalg.norm(M1)
    rand_spec_dists.append(spectral_distance(M1, M2))
    err, _ = qr_prediction_error(M1, M2)
    rand_qr_errors.append(err)

base_spec = float(np.mean(rand_spec_dists))
base_spec_std = float(np.std(rand_spec_dists))
base_qr   = float(np.mean(rand_qr_errors))
base_qr_std = float(np.std(rand_qr_errors))
print(f"  Spectral distance baseline: {base_spec:.4f} ± {base_spec_std:.4f}")
print(f"  QR prediction error baseline: {base_qr:.4f} ± {base_qr_std:.4f}\n")

# ── main loop — collect T per layer ───────────────────────────────────────────
# Structure: layer_Ts[layer][text][head] = T matrix
all_spec_dists = []   # [n_layer_pairs, n_texts, n_heads]
all_qr_errors  = []   # [n_layer_pairs, n_texts, n_heads]

# Per-layer, per-text averages
layer_spec = [[] for _ in range(n_layers-1)]
layer_qr   = [[] for _ in range(n_layers-1)]

print("Running tests across all layers...\n")

for ti, text in enumerate(TEXTS):
    print(f"  Text {ti+1}/{len(TEXTS)}: {text[:50]}...", flush=True)
    attns = get_all_attentions(text)

    # Build T for every layer and head
    T_grid = []  # [n_layers][n_heads]
    for ell in range(n_layers):
        T_row = [attn_to_T(attns[ell][h], dim=DIM) for h in range(n_heads)]
        T_grid.append(T_row)

    # Test consecutive layer pairs
    for ell in range(n_layers - 1):
        spec_dists_pair = []
        qr_errs_pair    = []
        for h in range(n_heads):
            T_ell  = T_grid[ell][h]
            T_next = T_grid[ell+1][h]

            # Test 1: isospectrality
            sd = spectral_distance(T_ell, T_next)
            spec_dists_pair.append(sd)

            # Test 2: QR prediction
            qr_err, _ = qr_prediction_error(T_ell, T_next)
            qr_errs_pair.append(qr_err)

        layer_spec[ell].append(float(np.nanmean(spec_dists_pair)))
        layer_qr[ell].append(float(np.nanmean(qr_errs_pair)))

        if args.verbose:
            print(f"    L{ell}→L{ell+1}: spec_dist={np.nanmean(spec_dists_pair):.4f}  "
                  f"qr_err={np.nanmean(qr_errs_pair):.4f}")

# ── aggregate ─────────────────────────────────────────────────────────────────
mean_spec = [float(np.nanmean(layer_spec[ell])) for ell in range(n_layers-1)]
mean_qr   = [float(np.nanmean(layer_qr[ell]))   for ell in range(n_layers-1)]
pair_idxs = list(range(n_layers-1))

# ── results table ─────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("  RESULTS: Layer-to-Layer Tests")
print(f"{'='*70}")
print(f"  Baselines: spec_dist={base_spec:.4f}  qr_error={base_qr:.4f}\n")
print(f"  {'Pair':>8} {'spec_dist':>10} {'vs baseline':>12} "
      f"{'qr_error':>10} {'vs baseline':>12} {'QR predicts?':>13}")
print("  "+"-"*68)

qr_confirmed_pairs = []
iso_confirmed_pairs = []

for ell in range(n_layers-1):
    sd = mean_spec[ell]
    qr = mean_qr[ell]
    iso_ok = sd < base_spec * 0.5   # spec dist < half random baseline
    qr_ok  = qr < base_qr  * 0.5   # QR error < half random baseline
    if iso_ok: iso_confirmed_pairs.append(ell)
    if qr_ok:  qr_confirmed_pairs.append(ell)
    print(f"  L{ell:>2}→L{ell+1:<2}   {sd:>10.4f} {sd/base_spec:>11.2f}×  "
          f"{qr:>10.4f} {qr/base_qr:>11.2f}×  "
          f"{'✓ QR' if qr_ok else '':>13}")

# ── trend analysis ────────────────────────────────────────────────────────────
valid_spec = [(i,v) for i,v in enumerate(mean_spec) if not np.isnan(v)]
valid_qr   = [(i,v) for i,v in enumerate(mean_qr)   if not np.isnan(v)]

if valid_spec:
    r_spec, p_spec = spearmanr([x[0] for x in valid_spec],
                                [x[1] for x in valid_spec])
else: r_spec, p_spec = 0, 1

if valid_qr:
    r_qr, p_qr = spearmanr([x[0] for x in valid_qr],
                             [x[1] for x in valid_qr])
else: r_qr, p_qr = 0, 1

print(f"\n{'='*70}")
print("  TREND ANALYSIS")
print(f"{'='*70}\n")
print(f"  Spectral distance vs depth:  r={r_spec:+.3f}  p={p_spec:.4f}  "
      f"{'↓ decreasing ✓' if r_spec < -0.3 and p_spec < 0.05 else '→ no trend'}")
print(f"  QR error vs depth:           r={r_qr:+.3f}  p={p_qr:.4f}  "
      f"{'↓ QR improves ✓' if r_qr < -0.3 and p_qr < 0.05 else '→ no trend'}")

print(f"\n  Layer pairs where QR predicts well (error < 50% baseline): "
      f"{qr_confirmed_pairs} ({len(qr_confirmed_pairs)}/{n_layers-1})")
print(f"  Layer pairs where spectra are preserved (dist < 50% baseline): "
      f"{iso_confirmed_pairs} ({len(iso_confirmed_pairs)}/{n_layers-1})")

# ── verdict ───────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("  VERDICT")
print(f"{'='*70}\n")

qr_frac   = len(qr_confirmed_pairs)  / (n_layers-1)
iso_frac  = len(iso_confirmed_pairs) / (n_layers-1)
mean_qr_overall  = float(np.nanmean(mean_qr))
mean_iso_overall = float(np.nanmean(mean_spec))

print(f"  TEST 1 (Isospectrality):")
print(f"    Mean spec dist  = {mean_iso_overall:.4f}  (baseline: {base_spec:.4f})")
print(f"    Ratio           = {mean_iso_overall/base_spec:.3f}×")
if mean_iso_overall < base_spec * 0.5:
    print(f"    ✓ CONFIRMED: consecutive layers are approximately isospectral")
    print(f"      ({mean_iso_overall/base_spec:.0%} of random baseline)")
elif mean_iso_overall < base_spec * 0.75:
    print(f"    ~ PARTIAL: spec dist below baseline but not strongly isospectral")
else:
    print(f"    ✗ NOT CONFIRMED: spec dist comparable to random baseline")

print(f"\n  TEST 2 (QR Prediction):")
print(f"    Mean QR error   = {mean_qr_overall:.4f}  (baseline: {base_qr:.4f})")
print(f"    Ratio           = {mean_qr_overall/base_qr:.3f}×")
if mean_qr_overall < base_qr * 0.5:
    print(f"    ✓ CONFIRMED: A_{{ℓ+1}} ≈ R_ℓ Q_ℓ  — QR update rule holds")
    print(f"      QR prediction is {mean_qr_overall/base_qr:.0%} of random baseline")
elif mean_qr_overall < base_qr * 0.75:
    print(f"    ~ PARTIAL: QR prediction better than random but not strongly")
else:
    print(f"    ✗ NOT CONFIRMED: QR prediction ≈ random — update rule not exact")

# combined
if mean_iso_overall < base_spec * 0.5 and mean_qr_overall < base_qr * 0.5:
    verdict = "STRONG: Both isospectrality and QR update confirmed"
    upgrade = ("'Implements' language fully justified. "
               "Each attention layer performs one QR step of a Toda flow.")
elif mean_iso_overall < base_spec * 0.75 or mean_qr_overall < base_qr * 0.75:
    verdict = "PARTIAL: At least one test shows signal below random baseline"
    upgrade = ("'Consistent with' language appropriate. "
               "Stronger than Hessenberg alone but not yet 'implements'.")
else:
    verdict = "INCONCLUSIVE: Neither test shows strong signal"
    upgrade = ("Maintain calibrated language. "
               "Hessenberg result stands; QR update not directly confirmed.")

print(f"\n  COMBINED VERDICT: {verdict}")
print(f"  PAPER LANGUAGE: {upgrade}")

print(f"""
  PAPER IMPLICATION TABLE:
  ┌────────────────────────────────────────────────────────────────┐
  │ Evidence                     Status    Paper claim             │
  ├────────────────────────────────────────────────────────────────┤
  │ Hessenberg < 0.1 (trained)   ✓ Done    Approximate structure   │
  │ Layer-depth r=-0.914         ✓ Done    Progressive convergence │
  │ 3× concordant correlations   ✓ Done    Consistent with Toda    │
  │ Isospectrality               {('✓ Done' if mean_iso_overall<base_spec*0.5 else '? Pending'):8s}  λ(T_ℓ)≈λ(T_ℓ₊₁)          │
  │ QR update A_ℓ₊₁≈R_ℓQ_ℓ      {('✓ Done' if mean_qr_overall<base_qr*0.5 else '? Pending'):8s}  Implements Toda QR        │
  └────────────────────────────────────────────────────────────────┘
""")

# ── save ──────────────────────────────────────────────────────────────────────
if args.save:
    def s(o):
        if isinstance(o,np.integer): return int(o)
        if isinstance(o,np.floating): return float(o) if not np.isnan(o) else None
        if isinstance(o,float) and np.isnan(o): return None
        if isinstance(o,dict): return {k:s(v) for k,v in o.items()}
        if isinstance(o,list): return [s(v) for v in o]
        return o
    out = {
        'model': args.model, 'n_layers': n_layers,
        'n_texts': args.n_texts, 'dim': DIM,
        'baselines': {'spec_dist': base_spec, 'spec_std': base_spec_std,
                      'qr_error': base_qr, 'qr_std': base_qr_std},
        'layer_mean_spec': s(mean_spec),
        'layer_mean_qr':   s(mean_qr),
        'trends': {'spec_r': float(r_spec), 'spec_p': float(p_spec),
                   'qr_r':   float(r_qr),   'qr_p':   float(p_qr)},
        'summary': {
            'mean_spec_overall': float(mean_iso_overall),
            'mean_qr_overall':   float(mean_qr_overall),
            'spec_ratio': float(mean_iso_overall/base_spec),
            'qr_ratio':   float(mean_qr_overall/base_qr),
            'iso_confirmed_pairs': iso_confirmed_pairs,
            'qr_confirmed_pairs':  qr_confirmed_pairs,
            'verdict': verdict,
        }
    }
    with open(args.save,'w') as f: json.dump(out,f,indent=2)
    print(f"  Saved → {args.save}")
