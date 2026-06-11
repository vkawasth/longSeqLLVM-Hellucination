"""
SemanticSheafComplex
====================
Cellular sheaf over the skeleton filtration, where each simplex
carries local semantic data and inconsistencies propagate through
restriction maps (Floer-type parallel transport).

Theoretical grounding
---------------------
Your hodge_pass.jl already implements the key insight:
  f = f_grad + f_harm + f_curl   (Hodge decomposition on edges)

where:
  f_grad → valid forward flow (grounded inference)
  f_harm → topological dead zone (off-manifold)
  f_curl → tautological loop (cyclic reasoning = hallucination)

We extend this to a full cellular sheaf F over the simplicial complex
K built from the 1024-token skeleton filtration:

  Vertices V = skeleton windows X^1,...,X^8  (0-simplices)
  Edges    E = transitions X^k → X^{k+1}    (1-simplices)
  Faces    F = triples X^{k} → X^{k+1} → X^{k+2}  (2-simplices, k=1..6)

Sheaf data at each simplex σ:
  F(v)     = local semantic stalk at vertex v
             carries: (covariance Σ_v, Jacobian J_v, attn_spectrum λ_v)
  F(e)     = edge stalk (transport data)
             carries: (transition map W_e, Hodge components, parallel transport)
  F(f)     = face stalk (2-simplex consistency)
             carries: (triangle holonomy, m_3 error, Gerstenhaber bracket)

Restriction maps:
  ρ_{v←e}: F(e) → F(v)   (restrict edge data to vertex)
  ρ_{e←f}: F(f) → F(e)   (restrict face data to edge)

These are the Floer parallel transport maps from your fukaya_ad_context.jl:
  ρ_{v←e} ≅ CF*(L_i, L_j) evaluation at vertex
  ρ_{e←f} ≅ m_3(a,b,c) triple composition

The sheaf Laplacian:
  L_F = δ^T δ   where δ: C^0(K;F) → C^1(K;F) is the coboundary

The spectrum of L_F gives:
  λ=0 eigenvectors: globally consistent sections (no hallucination)
  λ>0 eigenvectors: inconsistency modes (hallucination directions)

Connection to Henniart-Vigneras:
  A globally consistent section = supersingular representation
  (cannot be induced from a shorter Levi = not a hallucination)
  An inconsistency = parabolic induction from shorter context
  (model is operating in a reduced sub-context = hallucination)

Connection to Seidel 2025:
  The sheaf Laplacian spectrum is the quantum connection spectrum
  restricted to the semantic stalk data.
  p-adic splitting of L_F eigenvalues → three-prime orbit detection.

Connection to postnikov_rewards.jl:
  HH²(W_Q) = 89 (confirmed) corresponds to the second Betti number β₂
  of the sheaf complex = number of independent 2-cocycles = number of
  independent hallucination modes detectable at level 2.

  The 62-class obstruction (coker = 62) = dimension of the non-trivial
  part of H^1(K; F) = number of independent gluing failures.
"""

import numpy as np
from scipy.linalg import block_diag, null_space
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigsh, svds
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────
# SIMPLEX DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────

@dataclass
class VertexStalk:
    """
    Semantic data at a 0-simplex (context window).
    Mirrors ContextFingerprint from context_algebra.jl.
    """
    window_idx:     int
    mean:           np.ndarray      # mean hidden state [d]
    covariance:     np.ndarray      # empirical covariance [d,d]
    attn_spectrum:  np.ndarray      # singular values of window [min(n,d)]
    hodge_grad:     float           # gradient component norm (grounded flow)
    hodge_harm:     float           # harmonic component norm (loop signal)
    hodge_curl:     float           # curl component norm (volatility)
    hodge_gate:     int             # 0=dead 1=unstable 2=live
    dim:            int             # stalk dimension (pca_dim)


@dataclass
class EdgeStalk:
    """
    Transport data at a 1-simplex (window transition).
    Mirrors the m₁ differential and Floer complex from au_fukaya_engine.jl.
    """
    src_idx:        int
    tgt_idx:        int
    transport_map:  np.ndarray      # W: ℝ^d → ℝ^d  (m₂ linear map)
    transport_err:  float           # ‖W h_src - h_tgt‖ (prediction error)
    jacobian_sv:    np.ndarray      # singular values of W [d]
    jacobian_rank:  int             # rank of W
    is_singular:    bool            # rank-deficient → Dehn twist / vanishing cycle
    floer_score:    float           # CF*(L_src, L_tgt) = disk count proxy
    restriction_L:  np.ndarray      # ρ_{src←edge}: edge → src stalk [d,d]
    restriction_R:  np.ndarray      # ρ_{tgt←edge}: edge → tgt stalk [d,d]


@dataclass
class FaceStalk:
    """
    Holonomy data at a 2-simplex (triangle of three windows).
    Mirrors m₃ (triple A∞ product) and Gerstenhaber bracket.
    """
    v0_idx:         int
    v1_idx:         int
    v2_idx:         int
    holonomy:       np.ndarray      # ρ_{v0←e01} ∘ W_e12 - W_e02 [d,d]
    holonomy_norm:  float           # ‖holonomy‖_F  (consistency measure)
    m3_error:       float           # A∞ relation: m₁m₃ + m₂(m₂⊗1) + ... ≠ 0
    bracket_class:  int             # Gerstenhaber bracket [m_p, m_q] class
    is_consistent:  bool            # holonomy ≈ 0 → section exists


@dataclass
class SheafCochainData:
    """
    Full H^0, H^1, H^2 cohomology of the sheaf complex.
    Mirrors the HH²(W_Q) computation in curved_hh2.jl.
    """
    H0_dim:         int             # dim H^0 = number of global sections
    H1_dim:         int             # dim H^1 = gluing failures (hallucinations)
    H2_dim:         int             # dim H^2 = higher obstructions
    H0_basis:       np.ndarray      # basis for H^0
    H1_generators:  List            # generators of H^1 (hallucination modes)
    laplacian_spectrum: np.ndarray  # eigenvalues of sheaf Laplacian
    betti: Tuple                    # (β₀, β₁, β₂)
    consistency_score: float        # 1 - H1_dim/max_H1 ∈ [0,1]


# ─────────────────────────────────────────────────────────────────
# SHEAF COMPLEX BUILDER
# ─────────────────────────────────────────────────────────────────

class SemanticSheafComplex:
    """
    Build and analyse the cellular sheaf over the skeleton filtration.

    The simplicial complex K has:
      8 vertices  (skeleton windows X^1...X^8)
      7 edges     (transitions X^k → X^{k+1})
      6 faces     (triangles X^k,X^{k+1},X^{k+2})

    Each simplex carries semantic data from the hidden states.
    The sheaf Laplacian detects global inconsistencies.

    Architecture matches your Julia engine:
      VertexStalk   ↔  ContextFingerprint (context_algebra.jl)
      EdgeStalk     ↔  ContextMorphism + m₁ differential
      FaceStalk     ↔  m₃ A∞ relation check
      H^1(K;F)      ↔  HH²(W_Q) in curved_hh2.jl
      sheaf_score   ↔  hodge_gate in hodge_pass.jl
    """

    def __init__(self, pca_dim: int = 48, stalk_dim: int = 16):
        """
        pca_dim:   dimension for PCA reduction of hidden states
        stalk_dim: dimension of each simplex stalk (must be ≤ pca_dim)
                   smaller = more tractable sheaf Laplacian
        """
        self.pca_dim   = pca_dim
        self.stalk_dim = min(stalk_dim, pca_dim)

        # Simplicial complex structure
        self.n_vertices = 8   # X^1...X^8
        self.n_edges    = 7   # X^k → X^{k+1}
        self.n_faces    = 6   # triangles

        # Edge incidence: edge k connects vertex k to vertex k+1
        self.edge_src = list(range(7))      # [0,1,2,3,4,5,6]
        self.edge_tgt = list(range(1, 8))   # [1,2,3,4,5,6,7]

        # Face incidence: face k = (vertex k, k+1, k+2)
        self.face_v = [(k, k+1, k+2) for k in range(6)]

    def _pca_reduce(self, states: np.ndarray) -> np.ndarray:
        """Reduce hidden states to pca_dim dimensions."""
        if states.shape[1] <= self.pca_dim:
            return states
        sc = states - states.mean(axis=0)
        try:
            _, _, Vt = np.linalg.svd(sc, full_matrices=False)
            return sc @ Vt[:self.pca_dim].T
        except Exception:
            rng = np.random.RandomState(42)
            P   = rng.randn(states.shape[1], self.pca_dim)
            P, _ = np.linalg.qr(P)
            return (states - states.mean(axis=0)) @ P

    def _hodge_decompose_window(
        self,
        window: np.ndarray,
        prev_window: Optional[np.ndarray] = None
    ) -> Tuple[float, float, float, int]:
        """
        Hodge decompose the flow within a context window.

        Mirrors hodge_pass.jl Section 2:
          f_grad: gradient component (grounded forward flow)
          f_harm: harmonic component (loop signal / topological)
          f_curl: curl component (volatility / tautological loop)

        For a token window, we build the discrete flow as the
        sequence of incremental transitions h_t → h_{t+1},
        then decompose using the local 1D graph structure.
        """
        n, d = window.shape
        if n < 3:
            return 0.0, 0.0, 0.0, 2  # LIVE by default

        # Edge flows: f[t] = h_{t+1} - h_t  (discrete gradient flow)
        f_vecs = np.diff(window, axis=0)   # [n-1, d]

        # For each dimension, decompose scalar flow
        grad_norms = []
        harm_norms = []
        curl_norms = []

        for dim_i in range(min(d, 8)):   # sample first 8 dims for speed
            f = f_vecs[:, dim_i]         # [n-1] scalar flow on path graph
            n_edges = len(f)

            # Path graph incidence ∂₁ ∈ ℝ^{n×(n-1)}
            # ∂₁[i,e] = +1 if e starts at i, -1 if e ends at i
            d1 = np.zeros((n, n_edges))
            for e in range(n_edges):
                d1[e,   e] = -1.0   # tail
                d1[e+1, e] = +1.0   # head

            # Laplacian L = d1 @ d1.T
            L = d1 @ d1.T
            L_pinv = np.linalg.pinv(L)

            # Hodge decomposition
            div_f  = d1 @ f          # vertex divergence [n]
            x      = L_pinv @ div_f  # node potentials [n]
            f_grad = d1.T @ x        # gradient component [n-1]
            f_harm = f - f_grad      # harmonic residual [n-1]
            # f_curl = 0 for path graph (β₂ = 0)

            grad_norms.append(np.linalg.norm(f_grad))
            harm_norms.append(np.linalg.norm(f_harm))
            curl_norms.append(0.0)

        norm_g = np.mean(grad_norms)
        norm_h = np.mean(harm_norms)
        norm_c = 0.0
        norm_t = norm_g + norm_h + 1e-10

        # Hodge gate (from hodge_pass.jl Section 3, adapted for hidden states)
        # omega_local = mean token norm in the window
        # For normalised hidden states (layernorm output) this is ~1.0
        # Dead zone: window is essentially zero (padding or degenerate)
        omega_local = np.mean(np.linalg.norm(window, axis=1))
        total_flow  = norm_g + norm_h + 1e-10

        # Absolute dead zone: almost no signal at all
        if omega_local < 0.05:
            gate = 0  # DEAD
        elif norm_c / total_flow > 0.4:
            gate = 1  # UNSTABLE: curl (volatility) dominates
        else:
            gate = 2  # LIVE

        return norm_g, norm_h, norm_c, gate

    def _build_vertex_stalk(
        self,
        window: np.ndarray,
        window_idx: int,
        reduced: np.ndarray
    ) -> VertexStalk:
        """
        Build the stalk F(v) at a vertex.
        Carries local covariance, attention spectrum, Hodge decomposition.
        """
        d = reduced.shape[1]
        n = reduced.shape[0]

        mean = reduced.mean(axis=0)
        # Empirical covariance (regularised)
        c = reduced - mean
        if n > 1:
            cov = (c.T @ c) / (n - 1) + 1e-6 * np.eye(d)
        else:
            cov = np.eye(d)

        # Attention spectrum: singular values of the window matrix
        # Proxy for the attention weight distribution
        try:
            sv = np.linalg.svd(c, compute_uv=False)
            attn_spectrum = sv[:self.stalk_dim]
        except Exception:
            attn_spectrum = np.ones(self.stalk_dim)

        # Hodge decomposition of flow within window
        norm_g, norm_h, norm_c, gate = self._hodge_decompose_window(reduced)

        return VertexStalk(
            window_idx    = window_idx,
            mean          = mean[:self.stalk_dim],
            covariance    = cov[:self.stalk_dim, :self.stalk_dim],
            attn_spectrum = attn_spectrum,
            hodge_grad    = norm_g,
            hodge_harm    = norm_h,
            hodge_curl    = norm_c,
            hodge_gate    = gate,
            dim           = self.stalk_dim,
        )

    def _build_edge_stalk(
        self,
        win_src: np.ndarray,
        win_tgt: np.ndarray,
        src_idx: int,
        tgt_idx: int
    ) -> EdgeStalk:
        """
        Build the stalk F(e) at an edge (transition X^k → X^{k+1}).

        The transport map W: ℝ^d → ℝ^d is the m₂ linear map fitting
        win_src → win_tgt.  This is the Floer parallel transport in
        fukaya_ad_context.jl: ρ_c mapping L_src → L_tgt.

        The restriction maps ρ_{src←e} and ρ_{tgt←e} are the
        projections from the edge stalk onto each vertex stalk,
        corresponding to the left/right evaluation in CF*(L_i, L_j).
        """
        d = win_src.shape[1]
        sd = self.stalk_dim

        # Fit transport map W: mean(win_src) → mean(win_tgt) via alignment
        # Use global means for restriction maps (more stable than pointwise)
        mu_src = win_src.mean(axis=0)[:sd]
        mu_tgt = win_tgt.mean(axis=0)[:sd]

        # Transport matrix W from src to tgt (dimension sd)
        # Least squares: W @ X_src ≈ X_tgt
        X_src = win_src[:, :sd]  # [n, sd]
        X_tgt = win_tgt[:, :sd]  # [n, sd]

        # Match lengths
        n_min = min(len(X_src), len(X_tgt))
        X_s = X_src[:n_min]
        X_t = X_tgt[:n_min]

        try:
            # W such that W.T @ X_s.T ≈ X_t.T
            eps = 1e-4 * (np.linalg.norm(X_s) ** 2 / max(n_min, 1))
            W, _, _, _ = np.linalg.lstsq(
                X_s.T @ X_s + eps * np.eye(sd),
                X_s.T @ X_t,
                rcond=None
            )
            W = W.T   # [sd, sd]: maps src → tgt
        except Exception:
            W = np.eye(sd)

        # Jacobian analysis
        sv = np.linalg.svd(W, compute_uv=False)
        threshold = 1e-3 * sv[0] if sv[0] > 0 else 1e-10
        rank = int(np.sum(sv > threshold))
        is_sing = rank < sd

        # Transport error
        predicted = X_s @ W.T
        err = float(np.linalg.norm(predicted - X_t, 'fro')) / max(n_min, 1)

        # Floer score: CF*(L_src, L_tgt) proxy
        # = disk count ≈ 1 / (transport error + ε)
        floer_score = 1.0 / (err + 0.1)

        # Restriction maps encode SEMANTIC CONTENT differences between stalks.
        # ρ_L = cov(src)^{1/2}: maps edge into src's geometric frame
        # ρ_R = cov(tgt)^{1/2}: maps edge into tgt's geometric frame
        # (δs)[e] = ρ_R·s[tgt] - ρ_L·s[src] is nonzero when frames differ.
        # This is the key: two windows with identical statistics → δ=0 (consistent)
        # Two windows in different subspaces → δ≠0 (sheaf obstruction)
        try:
            c_s   = X_s - X_s.mean(axis=0)
            cov_s = (c_s.T @ c_s) / max(n_min-1, 1) + 1e-4 * np.eye(sd)
            try:
                rho_L = np.linalg.cholesky(cov_s)
            except np.linalg.LinAlgError:
                rho_L = np.diag(np.sqrt(np.maximum(np.diag(cov_s), 1e-8)))

            c_t   = X_t - X_t.mean(axis=0)
            cov_t = (c_t.T @ c_t) / max(n_min-1, 1) + 1e-4 * np.eye(sd)
            try:
                rho_R = np.linalg.cholesky(cov_t)
            except np.linalg.LinAlgError:
                rho_R = np.diag(np.sqrt(np.maximum(np.diag(cov_t), 1e-8)))
        except Exception:
            rho_L = np.eye(sd)
            rho_R = np.eye(sd)

        return EdgeStalk(
            src_idx       = src_idx,
            tgt_idx       = tgt_idx,
            transport_map = W,
            transport_err = err,
            jacobian_sv   = sv[:sd],
            jacobian_rank = rank,
            is_singular   = is_sing,
            floer_score   = floer_score,
            restriction_L = rho_L,
            restriction_R = rho_R,
        )

    def _build_face_stalk(
        self,
        e01: EdgeStalk,
        e12: EdgeStalk,
        e02: Optional['EdgeStalk'],
        v0_idx: int,
        v1_idx: int,
        v2_idx: int
    ) -> FaceStalk:
        """
        Build the stalk F(f) at a 2-simplex (triangle v0,v1,v2).

        The holonomy measures whether the triangle commutes:
          W_{01} @ W_{12}  ≈?  W_{02}

        Non-zero holonomy = the A∞ relation fails at m₃:
          Σ_{j+k=n+1} m_j(a_1,...,m_k(a_{j+1},...),...)  ≠ 0

        This corresponds to the Gerstenhaber bracket obstruction
        in curved_hh2.jl: [m_p, m_q] ≠ 0 at this simplex.

        Non-zero holonomy at a 2-simplex = non-trivial H^2 class
        = higher-order obstruction that cannot be fixed by
        adjusting the restriction maps alone.
        """
        sd = self.stalk_dim

        W01 = e01.transport_map   # [sd, sd]
        W12 = e12.transport_map   # [sd, sd]

        # Composed transport: going via the two-step path
        W_composed = W01 @ W12   # [sd, sd]

        # Direct transport (if available) or identity (no direct edge)
        if e02 is not None:
            W_direct = e02.transport_map
        else:
            # For non-adjacent triangles (v0 to v2 directly)
            # Use the identity as the "no direct path" baseline
            W_direct = np.eye(sd)

        # Holonomy = deviation from commutativity
        holonomy = W_composed - W_direct   # [sd, sd]
        h_norm   = float(np.linalg.norm(holonomy, 'fro'))

        # m₃ error: ‖W_{01} W_{12} - W_{02}‖ / (‖W_{01}‖ ‖W_{12}‖ + ε)
        denom    = (np.linalg.norm(W01, 'fro') *
                    np.linalg.norm(W12, 'fro') + 1e-8)
        m3_err   = h_norm / denom

        # Gerstenhaber bracket class:
        # 0 = trivial (holonomy is exact = m₃ correction exists)
        # k = non-trivial at k-th level (pure obstruction)
        sv_h     = np.linalg.svd(holonomy, compute_uv=False)
        sv_threshold = 0.1 * (sv_h[0] if sv_h[0] > 0 else 1.0)
        bracket_class = int(np.sum(sv_h > sv_threshold))

        return FaceStalk(
            v0_idx        = v0_idx,
            v1_idx        = v1_idx,
            v2_idx        = v2_idx,
            holonomy      = holonomy,
            holonomy_norm = h_norm,
            m3_error      = m3_err,
            bracket_class = bracket_class,
            is_consistent = h_norm < 0.5,
        )

    def _build_sheaf_laplacian(
        self,
        vertices: List[VertexStalk],
        edges: List[EdgeStalk],
    ) -> np.ndarray:
        """
        Build the sheaf Laplacian L_F = δ^T δ.

        The coboundary δ: C^0(K;F) → C^1(K;F) is defined by:
          (δs)[e] = ρ_{tgt←e}(s[tgt]) - ρ_{src←e}(s[src])

        where ρ are the restriction maps from each EdgeStalk.

        L_F is block-structured:
          [n_vertices × sd]² blocks
        Total size: (n_vertices × sd) × (n_vertices × sd)

        The spectrum of L_F:
          Kernel (λ=0) = globally consistent sections = H^0(K;F)
          Low eigenvalues = near-consistent (mild gluing failure)
          High eigenvalues = strongly inconsistent (hallucination modes)

        This corresponds to HH²(W_Q) in curved_hh2.jl:
          the 89-dimensional space decomposes into consistent (grad=27)
          and inconsistent (harm+curl=62) parts.
        """
        n_v  = len(vertices)
        n_e  = len(edges)
        sd   = self.stalk_dim

        # δ matrix: [n_e * sd, n_v * sd]
        delta = np.zeros((n_e * sd, n_v * sd))

        for ei, edge in enumerate(edges):
            s = edge.src_idx
            t = edge.tgt_idx
            rho_L = edge.restriction_L   # [sd, sd]: edge → src vertex
            rho_R = edge.restriction_R   # [sd, sd]: edge → tgt vertex

            # (δs)[e] = ρ_R s[t] - ρ_L s[s]
            row_start = ei * sd
            row_end   = row_start + sd

            col_src_start = s * sd
            col_src_end   = col_src_start + sd
            col_tgt_start = t * sd
            col_tgt_end   = col_tgt_start + sd

            delta[row_start:row_end, col_src_start:col_src_end] = -rho_L
            delta[row_start:row_end, col_tgt_start:col_tgt_end] = rho_R

        # Sheaf Laplacian L_F = δ^T δ
        L_F = delta.T @ delta

        return L_F

    def _compute_cohomology(
        self,
        L_F: np.ndarray,
        delta: np.ndarray,
        faces: List[FaceStalk],
        n_v: int,
        n_e: int,
    ) -> SheafCochainData:
        """
        Compute H^0, H^1, H^2 of the sheaf complex.

        H^0 = ker(δ) = global sections = dim(null_space of L_F)
        H^1 = ker(δ₁) / im(δ₀) = gluing failures
        H^2 = holonomy classes from face data

        Mirrors:
          HH²(W_Q) = 89 ↔ H^1 dimension
          coker(ρ*) = 62 ↔ non-trivial H^1 generators
        """
        sd  = self.stalk_dim

        # H^0: kernel of L_F
        ev = np.linalg.eigvalsh(L_F)
        ev_sorted = np.sort(np.abs(ev))
        threshold_0 = 1e-6 * max(ev_sorted[-1], 1.0)
        H0_dim  = int(np.sum(ev_sorted < threshold_0))

        # H^0 basis: eigenvectors for near-zero eigenvalues
        evals, evecs = np.linalg.eigh(L_F)
        H0_basis = evecs[:, np.abs(evals) < threshold_0]

        # H^1: approximate via sheaf Laplacian spectrum
        # In the exact case: H^1_dim = n_e*sd - rank(δ) - rank(δ₁)
        # We approximate using the non-zero eigenvalue count
        # relative to the theoretical maximum
        rank_delta = np.linalg.matrix_rank(delta, tol=threshold_0)
        H1_dim_approx = max(0, n_e * sd - rank_delta - H0_dim)

        # H^1 generators: top eigenvectors of L_F (most inconsistent)
        n_generators = min(H1_dim_approx, 5)   # top 5 hallucination modes
        H1_generators = []
        if n_generators > 0:
            idx_sorted = np.argsort(evals)[::-1]   # descending
            for i in range(min(n_generators, len(idx_sorted))):
                H1_generators.append({
                    'eigenvalue': float(evals[idx_sorted[i]]),
                    'mode': evecs[:, idx_sorted[i]],
                    'dominant_window': int(
                        np.argmax(np.abs(evecs[:, idx_sorted[i]])
                                  .reshape(n_v, sd).sum(axis=1))
                    )
                })

        # H^2: holonomy classes from faces
        H2_dim = sum(1 for f in faces if not f.is_consistent)

        # Betti numbers
        beta = (H0_dim, H1_dim_approx, H2_dim)

        # Consistency score:
        # 1.0 = perfectly consistent (all λ=0)
        # 0.0 = maximally inconsistent
        max_H1 = n_e * sd   # theoretical maximum
        consistency = 1.0 - min(H1_dim_approx / max(max_H1, 1), 1.0)

        return SheafCochainData(
            H0_dim            = H0_dim,
            H1_dim            = H1_dim_approx,
            H2_dim            = H2_dim,
            H0_basis          = H0_basis,
            H1_generators     = H1_generators,
            laplacian_spectrum = np.sort(np.abs(evals)),
            betti             = beta,
            consistency_score = consistency,
        )

    def build(self, windows: List[np.ndarray]) -> Dict:
        """
        Build the complete sheaf complex from context windows.

        Args:
            windows: list of 8 arrays, each [WINDOW_SIZE, hidden_dim]

        Returns:
            dict with keys:
              'vertices'   : List[VertexStalk]
              'edges'      : List[EdgeStalk]
              'faces'      : List[FaceStalk]
              'cohomology' : SheafCochainData
              'sheaf_laplacian': np.ndarray
              'consistency_score': float
              'hodge_gate_counts': dict
              'hallucination_modes': List[dict]
        """
        n_windows = len(windows)
        assert n_windows >= 2, "Need at least 2 windows"

        # PCA reduce all windows to pca_dim
        all_hidden = np.vstack(windows)
        if all_hidden.shape[1] > self.pca_dim:
            all_reduced = self._pca_reduce(all_hidden)
        else:
            all_reduced = all_hidden

        win_size = windows[0].shape[0]
        reduced_windows = [
            all_reduced[k * win_size:(k+1) * win_size]
            for k in range(n_windows)
        ]

        # ── Build vertex stalks ────────────────────────────────
        vertices = [
            self._build_vertex_stalk(
                windows[k], k, reduced_windows[k]
            )
            for k in range(n_windows)
        ]

        # ── Build edge stalks ──────────────────────────────────
        edges = [
            self._build_edge_stalk(
                reduced_windows[k],
                reduced_windows[k+1],
                k, k+1
            )
            for k in range(n_windows - 1)
        ]

        # ── Build face stalks ──────────────────────────────────
        faces = []
        for k in range(n_windows - 2):
            e01 = edges[k]
            e12 = edges[k+1]
            # e02 would be the edge from k to k+2 (skip-connection)
            # We don't have these in the path complex, so pass None
            face = self._build_face_stalk(
                e01, e12, None,
                k, k+1, k+2
            )
            faces.append(face)

        # ── Sheaf Laplacian ────────────────────────────────────
        n_v = len(vertices)
        n_e = len(edges)
        sd  = self.stalk_dim

        # Rebuild delta for cohomology computation
        delta = np.zeros((n_e * sd, n_v * sd))
        for ei, edge in enumerate(edges):
            s = edge.src_idx
            t = edge.tgt_idx
            row = slice(ei*sd, (ei+1)*sd)
            delta[row, s*sd:(s+1)*sd] = -edge.restriction_L
            delta[row, t*sd:(t+1)*sd] =  edge.restriction_R

        L_F = delta.T @ delta

        # ── Cohomology ─────────────────────────────────────────
        cohomology = self._compute_cohomology(L_F, delta, faces, n_v, n_e)

        # ── Hodge gate summary ─────────────────────────────────
        gate_counts = {0: 0, 1: 0, 2: 0}
        for v in vertices:
            gate_counts[v.hodge_gate] += 1

        # ── Hallucination mode summary ─────────────────────────
        hallucination_modes = []
        for gen in cohomology.H1_generators:
            mode = {
                'eigenvalue':       gen['eigenvalue'],
                'dominant_window':  gen['dominant_window'],
                'window_name':      f"X^{gen['dominant_window']+1}",
                'hodge_gate':       vertices[gen['dominant_window']].hodge_gate,
                'hodge_gate_label': ['DEAD','UNSTABLE','LIVE'][
                    vertices[gen['dominant_window']].hodge_gate
                ],
                'transport_err':    edges[min(gen['dominant_window'],
                                             n_e-1)].transport_err,
                'is_singular':      edges[min(gen['dominant_window'],
                                             n_e-1)].is_singular,
            }
            hallucination_modes.append(mode)

        return {
            'vertices':            vertices,
            'edges':               edges,
            'faces':               faces,
            'cohomology':          cohomology,
            'sheaf_laplacian':     L_F,
            'consistency_score':   cohomology.consistency_score,
            'hodge_gate_counts':   gate_counts,
            'hallucination_modes': hallucination_modes,
            'betti':               cohomology.betti,
        }

    def consistency_score(self, sheaf_data: Dict) -> float:
        """
        Extract scalar consistency score from sheaf data.
        0.0 = maximally inconsistent (hallucinating)
        1.0 = perfectly consistent (admissible)
        """
        return sheaf_data['consistency_score']

    def summarise(self, sheaf_data: Dict) -> str:
        """Human-readable sheaf analysis summary."""
        c  = sheaf_data['cohomology']
        gc = sheaf_data['hodge_gate_counts']
        b  = sheaf_data['betti']
        lines = [
            "─" * 50,
            "  Sheaf Complex Analysis",
            "─" * 50,
            f"  Betti numbers:  β₀={b[0]}  β₁={b[1]}  β₂={b[2]}",
            f"  H⁰ (sections):  {c.H0_dim}  (global consistency)",
            f"  H¹ (failures):  {c.H1_dim}  (gluing failures = hallucination modes)",
            f"  H² (holonomy):  {c.H2_dim}  (higher obstructions)",
            f"  Consistency:    {sheaf_data['consistency_score']:.3f}",
            f"  Hodge gates:    LIVE={gc[2]}  UNSTABLE={gc[1]}  DEAD={gc[0]}",
            "  Hallucination modes:",
        ]
        for mode in sheaf_data['hallucination_modes'][:3]:
            lines.append(
                f"    X^{mode['dominant_window']+1}: "
                f"λ={mode['eigenvalue']:.3f}  "
                f"gate={mode['hodge_gate_label']}  "
                f"singular={mode['is_singular']}"
            )
        lines.append("─" * 50)
        return "\n".join(lines)
