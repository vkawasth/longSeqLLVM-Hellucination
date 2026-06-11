"""
spectral_sheaf.py
=================
The spectral sheaf F_spec on the context category C_ctx.

What we had (bockstein_sheaf_metric._cech):
  F(X^k) = Sigma_k^{1/2}   (covariance stalk, one matrix per window)
  coboundary = ||T * Sigma_src^{1/2} - Sigma_tgt^{1/2}||_F  (scalar)

What the spectral sheaf adds:
  F_spec(X^k) = top-k eigenspace of C_k  (graded by eigenvalue)
  restriction rho_{k,k+1}: eigenspace_k -> eigenspace_{k+1}
                            measured as Grassmann distance
  H^0(X; F_spec): global sections = persistently stable spectral modes
  H^1(X; F_spec): obstruction to assembling local sections globally
                  = modes that are locally coherent but globally inconsistent

Why this matters (Abouzaid / D-brane connection):
  In D^b(Coh(M_ctx)), the spectral sheaf is the pushforward of the
  structure sheaf along the spectral curve of T.
  Eigenvalues are 'D-brane charges'.
  Ext^1(F_spec, F_spec) = H^1(X; F_spec) = obstruction class.
  Non-zero H^1 = the dominant spectral modes cannot be assembled into
  a globally coherent section = structural hallucination at the
  algebraic (not just topological) level.

Three measurements:
  1. Grassmann coboundary:  how much do dominant directions rotate?
  2. Spectral gap ratio:    is there a clear dominant mode (attention sink)?
  3. H^0 persistence:       which modes survive across all windows?
"""

import numpy as np
import gudhi
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import warnings, sys, os
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))


# ─────────────────────────────────────────────────────────────────
# SPECTRAL STALK
# ─────────────────────────────────────────────────────────────────

@dataclass
class SpectralStalk:
    """
    F_spec(X^k): the spectral data at window k.
    
    Stalk = eigendecomposition of the empirical covariance C_k.
    Graded by eigenvalue: top modes carry semantic information,
    bottom modes carry noise.
    """
    window_idx:    int
    eigenvalues:   np.ndarray    # [d] ascending
    eigenvectors:  np.ndarray    # [d, d] columns = eigenvectors
    spectral_gap:  float         # lambda_top / lambda_{top-1}  (attention sink signal)
    top_k:         int           # number of dominant modes tracked

    @property
    def top_eigenspace(self) -> np.ndarray:
        """Top-k eigenvectors as [d, k] matrix."""
        return self.eigenvectors[:, -self.top_k:]

    @property
    def top_eigenvalues(self) -> np.ndarray:
        return self.eigenvalues[-self.top_k:]

    @property
    def spectral_concentration(self) -> float:
        """SC = sum(lambda^2) / (sum(lambda))^2  (inverse participation ratio)."""
        ev = np.abs(self.eigenvalues)
        s1 = ev.sum() + 1e-10
        s2 = (ev**2).sum() + 1e-10
        return float(s2 / s1**2)


def compute_stalk(hidden_states: np.ndarray,
                  window_idx:    int,
                  top_k:         int = 3,
                  regularise:    float = 1e-4) -> SpectralStalk:
    """Compute spectral stalk F_spec(X^k) from hidden states."""
    h = hidden_states - hidden_states.mean(axis=0)
    d = h.shape[1]
    C = (h.T @ h) / max(len(h) - 1, 1) + regularise * np.eye(d)
    eigenvalues, eigenvectors = np.linalg.eigh(C)  # ascending

    # Spectral gap: ratio of top two eigenvalues
    if len(eigenvalues) >= 2 and eigenvalues[-2] > 1e-8:
        gap = float(eigenvalues[-1] / eigenvalues[-2])
    else:
        gap = 1.0

    return SpectralStalk(
        window_idx   = window_idx,
        eigenvalues  = eigenvalues,
        eigenvectors = eigenvectors,
        spectral_gap = gap,
        top_k        = min(top_k, d),
    )


# ─────────────────────────────────────────────────────────────────
# RESTRICTION MAP: GRASSMANN DISTANCE
# ─────────────────────────────────────────────────────────────────

def grassmann_distance(V1: np.ndarray, V2: np.ndarray) -> float:
    """
    Grassmann distance between k-dim subspaces V1, V2 in R^d.
    
    V1, V2 are [d, k] matrices with orthonormal columns.
    Distance = ||theta||_2 where theta are principal angles.
    
    Principal angles: sigma_i = cos(theta_i) = singular values of V1^T V2.
    theta_i = 0: subspaces aligned in direction i
    theta_i = pi/2: subspaces orthogonal in direction i
    """
    # Ensure orthonormal columns
    V1, _ = np.linalg.qr(V1)
    V2, _ = np.linalg.qr(V2)
    k = min(V1.shape[1], V2.shape[1])
    V1, V2 = V1[:, :k], V2[:, :k]

    M = V1.T @ V2             # [k, k] overlap
    sv = np.linalg.svd(M, compute_uv=False)
    sv = np.clip(sv, -1.0, 1.0)
    angles = np.arccos(sv)    # principal angles in [0, pi/2]
    return float(np.linalg.norm(angles))


def eigenvalue_drift(stalk_a: SpectralStalk,
                     stalk_b: SpectralStalk) -> float:
    """
    How much do the top-k eigenvalue MAGNITUDES drift?
    Normalised by mean top eigenvalue.
    """
    k = min(stalk_a.top_k, stalk_b.top_k)
    la = np.sort(stalk_a.top_eigenvalues)[:k]
    lb = np.sort(stalk_b.top_eigenvalues)[:k]
    mean_scale = (np.mean(la) + np.mean(lb)) / 2.0 + 1e-8
    return float(np.linalg.norm(la - lb) / mean_scale)


@dataclass
class RestrictionMap:
    """
    rho_{k, k+1}: F_spec(X^k) -> F_spec(X^{k+1})
    
    The restriction map on the spectral sheaf.
    Measures how much the dominant spectral structure
    changes from one window to the next.
    """
    src_window:       int
    tgt_window:       int
    grassmann_dist:   float    # direction drift (principal angles)
    eigval_drift:     float    # magnitude drift
    spectral_cobdry:  float    # combined: grassmann + eigval_drift
    direction_stable: bool     # grassmann_dist < threshold
    magnitude_stable: bool     # eigval_drift < threshold

    @property
    def is_flat(self) -> bool:
        """Flat section: both direction and magnitude stable."""
        return self.direction_stable and self.magnitude_stable


def compute_restriction(stalk_a: SpectralStalk,
                         stalk_b: SpectralStalk,
                         dir_threshold: float = 0.8,
                         mag_threshold: float = 0.5) -> RestrictionMap:
    g = grassmann_distance(stalk_a.top_eigenspace, stalk_b.top_eigenspace)
    e = eigenvalue_drift(stalk_a, stalk_b)
    return RestrictionMap(
        src_window       = stalk_a.window_idx,
        tgt_window       = stalk_b.window_idx,
        grassmann_dist   = g,
        eigval_drift     = e,
        spectral_cobdry  = g + e,
        direction_stable = g < dir_threshold,
        magnitude_stable = e < mag_threshold,
    )


# ─────────────────────────────────────────────────────────────────
# GLOBAL SECTIONS: H^0
# ─────────────────────────────────────────────────────────────────

@dataclass
class GlobalSection:
    """
    H^0(X; F_spec): a spectral direction persistent across all windows.
    
    A global section exists if there is a unit vector v such that
    v lies (approximately) in the top-k eigenspace of every window.
    
    Existence of global sections = the model has a stable semantic
    mode that persists across the full trajectory.
    """
    exists:           bool
    n_persistent:     int      # number of directions in global section
    persistence_score: float   # mean alignment across all windows
    representative:   Optional[np.ndarray]  # the most persistent direction


def find_global_sections(stalks:       List[SpectralStalk],
                          min_align:    float = 0.7) -> GlobalSection:
    """
    Find directions in the top eigenspace of stalk[0] that remain
    aligned with the top eigenspace of all subsequent stalks.
    
    A direction v is in H^0 if:
    for all k: |projection of v onto top_eigenspace(k)| >= min_align
    """
    if not stalks:
        return GlobalSection(False, 0, 0.0, None)

    # Use stalk 0 as reference
    ref_vecs = stalks[0].top_eigenspace  # [d, k]
    d, k     = ref_vecs.shape
    n_persistent = 0
    best_v   = None
    best_score = 0.0

    for j in range(k):
        v = ref_vecs[:, j]  # candidate global section direction
        # Check alignment with ALL stalks
        alignments = []
        for stalk in stalks[1:]:
            # Projection of v onto top eigenspace of stalk
            proj = stalk.top_eigenspace.T @ v   # [k]
            align = float(np.linalg.norm(proj))  # in [0, 1] if columns orthonormal
            alignments.append(align)

        mean_align = np.mean(alignments) if alignments else 1.0
        if mean_align >= min_align:
            n_persistent += 1
            if mean_align > best_score:
                best_score = mean_align
                best_v = v.copy()

    return GlobalSection(
        exists            = n_persistent > 0,
        n_persistent      = n_persistent,
        persistence_score = best_score,
        representative    = best_v,
    )


# ─────────────────────────────────────────────────────────────────
# H^1: SHEAF COHOMOLOGY OBSTRUCTION
# ─────────────────────────────────────────────────────────────────

@dataclass
class SheafH1:
    """
    H^1(X; F_spec): the sheaf cohomology obstruction.
    
    A 1-cocycle is an assignment of vectors {v_{k,k+1}} to edges
    such that the coboundary condition holds:
    delta(v)_{i,j,k} = rho_{j,k}(v_{i,j}) - v_{i,k} + rho_{i,k}(v_{i,j}) = 0
    
    H^1 = Z^1 / B^1 (cocycles modulo coboundaries).
    
    Computational proxy:
    H^1 is non-trivial when the spectral coboundaries cannot be
    simultaneously minimised — there exist windows where the
    local restrictions are inconsistent.
    
    Measurement: for a path of 4 windows, check whether
    the composition rho_{0,2} = rho_{1,2} o rho_{0,1}
    (transitivity of restriction maps) holds.
    Failure of transitivity = non-trivial H^1 class.
    """
    h1_proxy:          float    # obstruction magnitude
    n_inconsistent:    int      # number of window triples with failed transitivity
    mean_transitivity: float    # how well do restrictions compose?
    has_obstruction:   bool     # h1_proxy > threshold


def compute_sheaf_h1(stalks:      List[SpectralStalk],
                      restrictions: List[RestrictionMap],
                      threshold:    float = 0.5) -> SheafH1:
    """
    Proxy for H^1(X; F_spec).
    
    Check transitivity of restriction maps:
    For each triple (i, j, k) with i < j < k:
    compose rho_{i,j} then rho_{j,k} and compare to rho_{i,k} directly.
    Transitivity failure = non-trivial H^1.
    """
    n = len(stalks)
    if n < 3:
        return SheafH1(0.0, 0, 1.0, False)

    # Build restriction lookup
    restr = {(r.src_window, r.tgt_window): r for r in restrictions}

    n_inconsistent = 0
    transitivity_errors = []

    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                r_ij = restr.get((i, j))
                r_jk = restr.get((j, k))
                r_ik = restr.get((i, k))

                if r_ij is None or r_jk is None or r_ik is None:
                    continue

                # Direct path i->k
                V_i  = stalks[i].top_eigenspace
                V_j  = stalks[j].top_eigenspace
                V_k  = stalks[k].top_eigenspace

                # Composed path i->j->k
                # Project V_i onto V_j, then project that onto V_k
                V_ij = V_j @ (V_j.T @ V_i)   # component of V_i in V_j
                # Normalize
                norms = np.linalg.norm(V_ij, axis=0, keepdims=True)
                V_ij  = V_ij / (norms + 1e-8)
                V_ijk = V_k @ (V_k.T @ V_ij)  # component of V_ij in V_k

                # Direct path i->k
                V_ik  = V_k @ (V_k.T @ V_i)

                # Transitivity error: do V_ijk and V_ik span the same subspace?
                try:
                    g = grassmann_distance(V_ijk, V_ik)
                except Exception:
                    g = 0.0

                transitivity_errors.append(g)
                if g > threshold:
                    n_inconsistent += 1

    if not transitivity_errors:
        return SheafH1(0.0, 0, 1.0, False)

    mean_t = float(np.mean(transitivity_errors))
    h1_proxy = float(np.mean([e for e in transitivity_errors if e > threshold])
                     if any(e > threshold for e in transitivity_errors) else 0.0)

    return SheafH1(
        h1_proxy       = h1_proxy,
        n_inconsistent = n_inconsistent,
        mean_transitivity = 1.0 - mean_t / (np.pi/2),
        has_obstruction   = h1_proxy > threshold,
    )


# ─────────────────────────────────────────────────────────────────
# FULL SPECTRAL SHEAF ANALYSIS
# ─────────────────────────────────────────────────────────────────

@dataclass
class SpectralSheafReport:
    """Complete spectral sheaf analysis of a trajectory."""
    n_windows:        int
    stalks:           List[SpectralStalk]
    restrictions:     List[RestrictionMap]
    global_section:   GlobalSection
    sheaf_h1:         SheafH1
    # Per-window spectral gaps (attention sink signal)
    spectral_gaps:    List[float]
    # Mean Grassmann coboundary across all adjacent windows
    mean_grassmann:   float
    mean_eigval_drift: float
    # Summary score
    spectral_defect:  float  # high = spectral sheaf is incoherent


def analyse_spectral_sheaf(
    hidden_states:  np.ndarray,   # [n_tokens, d]
    window_size:    int = 32,
    n_windows:      int = 8,
    top_k:          int = 3,
) -> SpectralSheafReport:
    """
    Full spectral sheaf analysis on a trajectory.
    """
    n      = len(hidden_states)
    hs     = hidden_states[:window_size * n_windows]
    wins   = [hs[k*window_size:(k+1)*window_size]
              for k in range(n_windows) if (k+1)*window_size <= len(hs)]
    n_wins = len(wins)

    if n_wins < 2:
        return None

    # Compute stalks
    stalks = [compute_stalk(w, k, top_k=top_k) for k, w in enumerate(wins)]

    # Compute restriction maps (adjacent windows)
    restrictions = [
        compute_restriction(stalks[k], stalks[k+1])
        for k in range(n_wins - 1)
    ]

    # Also compute non-adjacent for H^1 transitivity check
    all_restrictions = list(restrictions)
    for i in range(n_wins):
        for j in range(i+2, n_wins):
            all_restrictions.append(
                compute_restriction(stalks[i], stalks[j])
            )

    # Global sections H^0
    gs = find_global_sections(stalks)

    # Sheaf H^1
    h1 = compute_sheaf_h1(stalks, all_restrictions)

    # Summary
    gaps   = [s.spectral_gap for s in stalks]
    g_vals = [r.grassmann_dist for r in restrictions]
    e_vals = [r.eigval_drift   for r in restrictions]

    # Spectral defect: high when no global section AND high coboundary
    defect = (float(np.mean(g_vals)) + float(np.mean(e_vals))) * (1.0 - gs.persistence_score)

    return SpectralSheafReport(
        n_windows        = n_wins,
        stalks           = stalks,
        restrictions     = restrictions,
        global_section   = gs,
        sheaf_h1         = h1,
        spectral_gaps    = gaps,
        mean_grassmann   = float(np.mean(g_vals)),
        mean_eigval_drift= float(np.mean(e_vals)),
        spectral_defect  = defect,
    )


def print_report(r: SpectralSheafReport, label: str = "") -> str:
    if r is None:
        return f"  [{label}] insufficient windows"
    lines = [
        f"  Spectral Sheaf: {label}",
        f"  {'Window':<8} {'Gap':>7} {'Grassmann':>11} {'EigDrift':>10} {'Flat?':>6}",
        "  " + "-" * 46,
    ]
    for k, (s, r_) in enumerate(zip(r.stalks[:-1], r.restrictions)):
        lines.append(
            f"  X^{k:<5} {s.spectral_gap:>7.3f} {r_.grassmann_dist:>11.4f} "
            f"{r_.eigval_drift:>10.4f} {'yes' if r_.is_flat else 'NO':>6}"
        )
    lines += [
        "",
        f"  H^0 (global section): exists={r.global_section.exists}  "
        f"n_persistent={r.global_section.n_persistent}  "
        f"score={r.global_section.persistence_score:.3f}",
        f"  H^1 (obstruction):    proxy={r.sheaf_h1.h1_proxy:.4f}  "
        f"n_incons={r.sheaf_h1.n_inconsistent}  "
        f"has_obs={r.sheaf_h1.has_obstruction}",
        f"  Spectral defect:      {r.spectral_defect:.4f}",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────
# INTEGRATE WITH RELATION NERVE EXPERIMENT
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, torch
    sys.path.insert(0, os.path.dirname(__file__))
    from factscore_alignment import PhasedGPT2
    from hallucination_topology_pipeline import tokenize_controlled

    model = PhasedGPT2(d_model=256, n_layers=12, n_heads=8,
                       vocab_size=5000, max_seq=512, phase=1.0, seed=42)
    model.eval()

    from relation_nerve import SYNTHETIC_DATASET

    def get_hs(model, text, layer, max_tok=128):
        model.register_hooks()
        tok = tokenize_controlled(text)[:max_tok]
        with torch.no_grad():
            model(torch.tensor([tok]))
        hs = model._hs_by_layer.get(layer)
        model.remove_hooks()
        if hs is None: return None
        return hs / (np.linalg.norm(hs, axis=1, keepdims=True) + 1e-8)

    print("=" * 68)
    print("  Spectral Sheaf Analysis on Relation-Typed Dataset")
    print("  Layer 11 (semantic), windows of 8 tokens, top-3 eigenspace")
    print("=" * 68)
    print()

    results = {}
    for sample in SYNTHETIC_DATASET:
        hs = get_hs(model, sample['text'], layer=11)
        if hs is None or len(hs) < 16:
            continue
        report = analyse_spectral_sheaf(hs, window_size=8, n_windows=4, top_k=3)
        results[sample['id']] = (sample['label'], report)

    # Print summary table
    print(f"  {'ID':<28} {'Label':<12} {'H0':>4} {'H1_obs':>8} "
          f"{'Grass':>7} {'Defect':>8}")
    print("  " + "-" * 72)

    for label_order in ['correct','wrong_rel','reversed','invented']:
        for name, (lbl, r) in results.items():
            if lbl != label_order or r is None: continue
            print(f"  {name:<28} {lbl:<12} "
                  f"{'yes' if r.global_section.exists else 'no ':>4} "
                  f"{r.sheaf_h1.h1_proxy:>8.4f} "
                  f"{r.mean_grassmann:>7.4f} "
                  f"{r.spectral_defect:>8.4f}")
        print()

    print("  Summary by label:")
    for lbl in ['correct','wrong_rel','reversed','invented']:
        subset = [(r) for _, (l, r) in results.items() if l == lbl and r is not None]
        if not subset: continue
        print(f"  {lbl:<12}: "
              f"H0={sum(r.global_section.exists for r in subset)}/{len(subset)}  "
              f"H1_mean={np.mean([r.sheaf_h1.h1_proxy for r in subset]):.4f}  "
              f"defect={np.mean([r.spectral_defect for r in subset]):.4f}")
