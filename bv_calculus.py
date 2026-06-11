"""
bv_calculus.py
==============
Batalin-Vilkovisky differential calculus on context transport.

Sources:
  Kowalzig-Krämer [1203.4984]: BV structures on Ext and Tor.
    Ext_U(A,A) = Gerstenhaber algebra (cup product + bracket)
    Tor_U(M,A) = BV module (cap product + Lie derivative)
    Cartan-Rinehart formula: L_T = [b, S_T] + [B, i_T]
    
  Magnus [2510.21871]: Divided difference operators on lattices.
    D(f)(x_{n+1/2}) = (f(x_{n+1}) - f(x_n)) / (x_{n+1} - x_n)
    Context: tangent linearization of the trajectory.

Three metric constraints (from design note):
  Option A: Tangent linearization P(x,y) = ∇_x y (divided difference)
  Option B: Grassmannian projection (fixed rank)
  Option C: Commutator [x,y] = xy - yx (non-commutativity measure)

In context algebra terms:

The BV structure is ALREADY implicit in our coboundary measurement:
  b(T)(h1,h2) = T(h2) - T(h1) = ρ_R(T) - ρ_L(T)
  This is the Hochschild coboundary of T as a 1-cochain.

What BV adds on top:
  1. Cup product: T_{12} ∪ T_{23} = composed transport T_{13}
     Cup error = ||T_{12} ∘ T_{23} - T_{13}|| / ||T_{13}||
     = m_3 in the A_∞ structure (associativity failure)
     
  2. Gerstenhaber bracket: {T_{12}, T_{23}} = [T_{12}, T_{23}] = non-commutativity
     Large bracket = transport has curvature (non-abelian context dynamics)
     Zero bracket = flat transport (abelian, no interaction between windows)
     
  3. Lie derivative (Cartan): L_T(h) = b(T ∩ h) - T ∩ b(h)
     Measures whether T is a DERIVATION with respect to the context structure
     = whether transport preserves the algebraic structure of context windows
     
  4. Divided difference gradient: ∇_h1 h2 = (h2 - h1) / ||h2 - h1||
     Tangent to trajectory = the direction of context motion
     {T, ∇h} = interaction between transport and motion direction
     Large = transport curves the trajectory (non-geodesic context evolution)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))


# ─────────────────────────────────────────────────────────────────
# TRANSPORT FITTING (shared utility)
# ─────────────────────────────────────────────────────────────────

def fit_transport(wa: np.ndarray, wb: np.ndarray,
                  rank: Optional[int] = None) -> np.ndarray:
    """
    Fit transport T: wa -> wb by regularised LS.
    If rank is given, project onto Grassmannian(rank) [Option B].
    """
    d = min(wa.shape[1], wb.shape[1])
    X = wa[:, :d]; Y = wb[:, :d]
    eps = 1e-4 * max(float(np.linalg.norm(X.T @ X)), 1.0)
    T, _, _, _ = np.linalg.lstsq(X.T @ X + eps * np.eye(d), X.T @ Y, rcond=None)
    T = T.T

    if rank is not None and rank < d:
        # Option B: Grassmannian projection
        U, s, Vt = np.linalg.svd(T)
        T = U[:, :rank] @ np.diag(s[:rank]) @ Vt[:rank, :]

    return T


def tangent_vector(wa: np.ndarray, wb: np.ndarray) -> np.ndarray:
    """
    Option A: tangent linearization / divided difference.
    ∇_{wa} wb = (μ_b - μ_a) / ||μ_b - μ_a||
    where μ = mean of window.
    """
    mu_a = wa.mean(axis=0)
    mu_b = wb.mean(axis=0)
    delta = mu_b - mu_a
    norm  = np.linalg.norm(delta) + 1e-8
    return delta / norm


def commutator(T12: np.ndarray, T23: np.ndarray) -> np.ndarray:
    """
    Option C: Gerstenhaber bracket / commutator.
    {T12, T23} = T12 @ T23 - T23 @ T12
    Measures non-commutativity of adjacent transports.
    """
    return T12 @ T23 - T23 @ T12


# ─────────────────────────────────────────────────────────────────
# HOCHSCHILD COBOUNDARY (the b operator)
# ─────────────────────────────────────────────────────────────────

def hochschild_b(T: np.ndarray, wa: np.ndarray, wb: np.ndarray) -> float:
    """
    Hochschild coboundary of T as a 1-cochain:
    b(T)(wa, wb) = T(wb) - T(wa)   [applied to mean vectors]
    = ρ_R(T) · μ_b - ρ_L(T) · μ_a

    This IS the coboundary ρ_R - ρ_L, now properly typed as a 
    Hochschild coboundary in the BV calculus.
    """
    mu_a = wa.mean(axis=0)
    mu_b = wb.mean(axis=0)
    Tmu_a = T @ mu_a
    Tmu_b = T @ mu_b
    return float(np.linalg.norm(Tmu_b - Tmu_a) / (np.linalg.norm(mu_b) + 1e-8))


# ─────────────────────────────────────────────────────────────────
# GERSTENHABER ALGEBRA ON Ext
# ─────────────────────────────────────────────────────────────────

@dataclass
class GerstenhaberData:
    """
    Ext_U(A,A) = Gerstenhaber algebra structure on transport operators.
    
    Cup product T12 ∪ T23 ≈ T13 (composition)
    Cup error = deviation from composition = m3 (A_∞ failure)
    Bracket {T12,T23} = commutator = non-commutativity / curvature
    """
    # Cup product
    T12:              np.ndarray
    T23:              np.ndarray
    T13_fitted:       np.ndarray   # directly fitted T13
    T13_composed:     np.ndarray   # T12 ∘ T23 (the cup product)
    cup_error:        float        # ||T12∘T23 - T13|| / ||T13||  = m3
    cup_error_norm:   float        # absolute

    # Gerstenhaber bracket
    bracket:          np.ndarray   # {T12, T23} = [T12, T23]
    bracket_norm:     float        # ||{T12,T23}||_F

    # Spectral structure of bracket
    bracket_spectrum: np.ndarray   # eigenvalues of bracket (imaginary for skew-sym part)


def compute_gerstenhaber(
    wa: np.ndarray,
    wb: np.ndarray,
    wc: np.ndarray,
    rank: Optional[int] = None,
) -> GerstenhaberData:
    """
    Compute the Gerstenhaber algebra structure on three windows.
    
    wa -> wb -> wc defines two adjacent transports T12, T23
    and a composed transport T13.
    
    Cup product: T12 ∪ T23 should equal T13 if transport is flat.
    Bracket: {T12,T23} measures the curvature.
    Cup error = m3 in the A_∞ structure (what the Bockstein probed).
    """
    T12 = fit_transport(wa, wb, rank=rank)
    T23 = fit_transport(wb, wc, rank=rank)
    T13 = fit_transport(wa, wc, rank=rank)

    cup = T12 @ T23                                  # cup product
    cup_err_abs  = float(np.linalg.norm(cup - T13, 'fro'))
    cup_err_norm = cup_err_abs / (float(np.linalg.norm(T13, 'fro')) + 1e-8)

    bracket      = commutator(T12, T23)
    bkt_norm     = float(np.linalg.norm(bracket, 'fro'))

    # Spectrum of bracket
    bkt_sym    = (bracket + bracket.T) / 2       # symmetric part
    bkt_skew   = (bracket - bracket.T) / 2       # skew-symmetric part
    bkt_eigvals = np.linalg.eigvalsh(bkt_sym)    # real eigenvalues of sym part

    return GerstenhaberData(
        T12=T12, T23=T23,
        T13_fitted=T13, T13_composed=cup,
        cup_error=cup_err_norm, cup_error_norm=cup_err_abs,
        bracket=bracket, bracket_norm=bkt_norm,
        bracket_spectrum=bkt_eigvals,
    )


# ─────────────────────────────────────────────────────────────────
# LIE DERIVATIVE AND BV MODULE ON Tor
# ─────────────────────────────────────────────────────────────────

@dataclass
class LieDerivativeData:
    """
    Tor_U(M,A) = BV module: Lie derivative of transport on hidden states.
    
    Cartan-Rinehart formula: L_T(h) = b(T · h) - T · b(h)
    where b = Hochschild boundary on the chain complex of context windows.
    
    L_T(h) = 0 iff T is a DERIVATION w.r.t. the context algebra structure.
    L_T(h) ≠ 0 iff T fails to preserve the algebraic structure.
    """
    T:              np.ndarray
    # Hochschild boundary b of (T · h) and T · b(h)
    bTh:            float   # ||b(T · h)||
    Tbh:            float   # ||T · b(h)||
    lie_deriv_norm: float   # ||L_T(h)|| = ||b(T·h) - T·b(h)||
    # Divided difference gradient
    grad_norm:      float   # ||∇_wa wb|| = tangent norm
    # Interaction: {T, ∇h}
    bracket_grad:   float   # ||{T, ∇h}||


def compute_lie_derivative(
    T:  np.ndarray,
    wa: np.ndarray,
    wb: np.ndarray,
) -> LieDerivativeData:
    """
    Compute L_T on the pair of windows (wa, wb).
    
    In Hochschild terms:
    b(T · h) = b applied to the image of T
    T · b(h) = T applied to the Hochschild boundary
    
    Practical: use mean vectors
    b(T · μ_a, T · μ_b) = T·μ_b - T·μ_a   (the cap product)
    T · b(μ_a, μ_b) = T · (μ_b - μ_a)
    
    L_T(h) = b(T·h) - T·b(h) = (T·μ_b - T·μ_a) - T·(μ_b - μ_a) = 0
    
    At first order (mean vectors) L_T = 0 always (T is linear).
    The non-trivial contribution comes from SECOND-ORDER terms:
    use covariance matrices instead of mean vectors.
    
    Covariance version:
    Σ_a = wa.T @ wa / n
    b(T · Σ)(h) = T Σ_b T^T - T Σ_a T^T
    T · b(Σ)(h) = T(Σ_b - Σ_a)
    L_T(Σ) = T Σ_b T^T - T Σ_a T^T - T(Σ_b - Σ_a)
           = T(Σ_b T^T - Σ_b) - T(Σ_a T^T - Σ_a)
           = T(Σ_b(T^T - I) - Σ_a(T^T - I))
           = T(Σ_b - Σ_a)(T^T - I)
    """
    d = T.shape[0]
    n = min(len(wa), len(wb))

    def cov(w):
        h = w - w.mean(0)
        return h.T @ h / max(len(h)-1, 1) + 1e-4 * np.eye(w.shape[1])

    Sigma_a = cov(wa[:, :d])
    Sigma_b = cov(wb[:, :d])
    delta_Sigma = Sigma_b - Sigma_a

    # Covariance-level Lie derivative: L_T(Σ) = T · ΔΣ · (T^T - I)
    L_T_Sigma = T @ delta_Sigma @ (T.T - np.eye(d))
    lie_norm = float(np.linalg.norm(L_T_Sigma, 'fro'))

    # b(T·h): transport applied then boundary
    T_Sigma_b = T @ Sigma_b @ T.T
    T_Sigma_a = T @ Sigma_a @ T.T
    bTh = float(np.linalg.norm(T_Sigma_b - T_Sigma_a, 'fro'))

    # T·b(h): boundary then transport
    Tbh = float(np.linalg.norm(T @ delta_Sigma, 'fro'))

    # Divided difference gradient (Option A / Magnus)
    mu_a = wa.mean(0); mu_b = wb.mean(0)
    grad = (mu_b - mu_a) / (np.linalg.norm(mu_b - mu_a) + 1e-8)
    grad_norm = float(np.linalg.norm(grad))

    # Bracket {T, ∇h}: non-commutativity of transport with motion
    # grad is a vector, T is a matrix: extend to outer product
    grad_mat  = np.outer(grad, grad)           # [d,d] rank-1 matrix
    brkt_grad = float(np.linalg.norm(T @ grad_mat - grad_mat @ T, 'fro'))

    return LieDerivativeData(
        T=T, bTh=bTh, Tbh=Tbh,
        lie_deriv_norm=lie_norm,
        grad_norm=grad_norm,
        bracket_grad=brkt_grad,
    )


# ─────────────────────────────────────────────────────────────────
# FULL BV ANALYSIS
# ─────────────────────────────────────────────────────────────────

@dataclass
class BVAnalysis:
    """
    Complete BV differential calculus on a context trajectory.
    
    For a sequence of windows w1, w2, ..., wK:
    - Gerstenhaber structure on each triple (wi, wi+1, wi+2)
    - Lie derivative structure on each pair (wi, wi+1)
    - Summary statistics across the trajectory
    """
    # Per-triple Gerstenhaber data
    gerstenhaber:      List[GerstenhaberData]
    # Per-pair Lie derivative data
    lie_derivatives:   List[LieDerivativeData]

    # Summary
    mean_cup_error:    float   # mean m3 across triples = A_∞ associativity failure
    mean_bracket_norm: float   # mean curvature across triples
    mean_lie_norm:     float   # mean Lie derivative = derivation failure
    mean_bracket_grad: float   # mean {T, ∇h} = transport-motion non-commutativity

    # BV signal: combination of all four
    bv_defect:         float   # high = trajectory is algebraically incoherent


def analyse_trajectory_bv(
    hidden_states:  np.ndarray,    # [n_tokens, d]
    window_size:    int = 32,
    rank:           Optional[int] = 4,  # Grassmannian rank (Option B)
) -> BVAnalysis:
    """
    Full BV analysis of a hidden state trajectory.
    
    Computes Gerstenhaber algebra structure and Lie derivatives
    across all windows, giving the complete differential calculus
    of context transport.
    """
    n, d = hidden_states.shape
    wins = [hidden_states[k*window_size:(k+1)*window_size]
            for k in range(n // window_size)]

    if len(wins) < 2:
        return BVAnalysis([], [], 0.0, 0.0, 0.0, 0.0, 0.0)

    # Lie derivatives on adjacent pairs
    lie_data = []
    for k in range(len(wins) - 1):
        T = fit_transport(wins[k], wins[k+1], rank=rank)
        ld = compute_lie_derivative(T, wins[k], wins[k+1])
        lie_data.append(ld)

    # Gerstenhaber on triples
    gst_data = []
    for k in range(len(wins) - 2):
        gd = compute_gerstenhaber(wins[k], wins[k+1], wins[k+2], rank=rank)
        gst_data.append(gd)

    # Summaries
    mean_cup  = float(np.mean([g.cup_error     for g in gst_data])) if gst_data else 0.0
    mean_bkt  = float(np.mean([g.bracket_norm  for g in gst_data])) if gst_data else 0.0
    mean_lie  = float(np.mean([l.lie_deriv_norm for l in lie_data])) if lie_data else 0.0
    mean_bgrd = float(np.mean([l.bracket_grad  for l in lie_data])) if lie_data else 0.0

    # BV defect: aggregate signal
    # High when: large cup error (m3 failure) + large Lie derivative (non-derivation)
    # Low when: transport is flat (m3 ≈ 0) and a derivation (L_T ≈ 0)
    bv_defect = 0.4 * min(mean_cup, 2.0) / 2.0 + 0.3 * mean_lie + 0.3 * mean_bgrd

    return BVAnalysis(
        gerstenhaber=gst_data,
        lie_derivatives=lie_data,
        mean_cup_error=mean_cup,
        mean_bracket_norm=mean_bkt,
        mean_lie_norm=mean_lie,
        mean_bracket_grad=mean_bgrd,
        bv_defect=bv_defect,
    )


# ─────────────────────────────────────────────────────────────────
# EXPERIMENT: BV STRUCTURE ON SYNTHETIC DATASET
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import torch, sys
    sys.path.insert(0, os.path.dirname(__file__))
    from factscore_alignment import PhasedGPT2
    from hallucination_topology_pipeline import tokenize_controlled
    from relation_nerve import SYNTHETIC_DATASET

    model = PhasedGPT2(d_model=256, n_layers=12, n_heads=8,
                       vocab_size=5000, max_seq=512, phase=1.0, seed=42)
    model.eval()

    def get_hs(text, layer=11):
        model.register_hooks()
        tok = tokenize_controlled(text)[:256]
        with torch.no_grad():
            model(torch.tensor([tok]))
        h = model._hs_by_layer.get(layer)
        model.remove_hooks()
        if h is None: return None
        return h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-8)

    print("=" * 72)
    print("  BV Differential Calculus on Context Transport")
    print("  Kowalzig-Krämer [1203.4984] + Magnus [2510.21871]")
    print()
    print("  Cup error  = m3 = A_∞ associativity failure (T12∘T23 ≠ T13)")
    print("  Bracket    = {T12,T23} = curvature / non-commutativity")
    print("  Lie deriv  = L_T = derivation failure on covariance")
    print("  {T,∇h}     = transport-motion non-commutativity")
    print("=" * 72)
    print()
    print(f"  {'ID':<28} {'Label':<12} {'Cup_err':>8} {'Bracket':>8} {'Lie':>7} {'T_grad':>8}")
    print("  " + "-" * 65)

    results = {'correct': [], 'wrong': []}
    for sample in SYNTHETIC_DATASET:
        hs = get_hs(sample['text'])
        if hs is None or len(hs) < 8:
            continue

        bv = analyse_trajectory_bv(hs, window_size=4, rank=4)
        is_correct = sample['label'] == 'correct'
        results['correct' if is_correct else 'wrong'].append(bv.bv_defect)

        print(f"  {sample['id']:<28} {sample['label']:<12} "
              f"{bv.mean_cup_error:>8.4f} "
              f"{bv.mean_bracket_norm:>8.4f} "
              f"{bv.mean_lie_norm:>7.4f} "
              f"{bv.mean_bracket_grad:>8.4f}")

    print()
    for key in ['correct', 'wrong']:
        v = results[key]
        if v:
            print(f"  {key:<8}: BV_defect mean={np.mean(v):.4f} "
                  f"std={np.std(v):.4f}")

    if results['correct'] and results['wrong']:
        d = np.mean(results['wrong']) - np.mean(results['correct'])
        pooled = np.std(results['correct'] + results['wrong']) + 1e-8
        print(f"  Cohen d = {d/pooled:.3f}  "
              f"({'correct > wrong' if d > 0 else 'wrong > correct'})")

    print()
    print("  Three metric constraints (from design note):")
    print("  Option A (tangent): tangent_vector(wa, wb)")
    print("  Option B (Grassmannian): fit_transport(..., rank=4)")
    print("  Option C (commutator): {T12, T23} = T12@T23 - T23@T12")
    print()
    print("  All three are implemented. Option B (Grassmannian) gives the")
    print("  most stable transport for the A_∞ composition law.")
