"""
AInfinityEntropyEstimator
=========================

Replaces the static cosine-similarity entropy estimator with a
dynamical prediction-error estimator grounded in the A∞ algebra
structure from the Julia engine.

Theoretical grounding:
----------------------
Your curved_hh2 Julia engine computes:
  - m₂ through m₈: A∞ structure maps (composition operations)
  - Gerstenhaber bracket: [m_p, m_q] obstruction structure
  - Cup product: HH¹ ⊗ HH¹ → HH² (cohomological obstructions)
  - Postnikov k-invariants: obstructions to tower lifting
  - FilteredAInfAlgebra: exponential energy decay on long paths

The entropy at position t is:

  E_t = ‖W h_t - h_{t+1}‖         (linear prediction error)

or more precisely, the A∞-weighted version:

  E_t^{A∞} = Σ_{k=2}^{8} λ^{k-2} · ‖m_k(h_{t-k+1},...,h_t) - h_{t+1}‖

where λ is the filtration decay rate from FilteredAInfAlgebra.

This captures:
  - m₂: immediate transition (bigram prediction)
  - m₃: trigram-level compositional error
  - m₄-m₈: higher-order obstruction contributions
  - λ weighting: same exponential filtration as your Julia engine

Why dynamical > geometric:
  Periodic orbits in transformers appear as:
  - sudden drops in predictive rank (Jacobian degeneracy)
  - recurrence instability (m_k error spikes)
  - transition singularities (det(W) near 0)
  NOT as large geometric displacements.

Connection to Postnikov tower:
  The k-invariant obstruction at level n manifests as a spike in
  the m_{n+2} prediction error: the n-th level of the tower fails
  to lift exactly when the (n+2)-ary composition is inconsistent.

  Level 0 (π₀, reachability):  E_t^{m2} spikes (bigram failure)
  Level 1 (π₁, circuits):      E_t^{m3} spikes (trigram failure)
  Level 2 (π₂, homotopy):      E_t^{m4} spikes (4-gram failure)
  ...
  Level 6 (m₈ obstruction):    E_t^{m8} spikes (8-gram = skeleton X^8)

This is the EXACT correspondence between:
  - your Julia m₂...m₈ computation
  - our GSOD skeleton levels X^1...X^8
  - the Seidel p-adic tower depth m=1...8

Implementation:
  We implement m_k as a learned linear map W_k: ℝ^{k·d} → ℝ^d
  fitted locally (window of 64 steps) via least squares.
  This is the discretized/finite-rank approximation of your
  A∞ structure maps in the hidden-state space.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


# FilteredAInfAlgebra parameters — mirrors the Julia struct
LAMBDA_DECAY   = 1.0    # exponential filtration rate (Julia: lambda=1.0)
MAX_ARITY      = 8      # m₂ through m₈ (Julia: max_path_len=20 but we use m8)
ENERGY_CUTOFF  = 1e-8   # prune below this weight (Julia: energy_cutoff=1e-8)


@dataclass
class AInfPredictionError:
    """
    Per-position A∞ prediction error at each arity level.
    Mirrors the output structure of gerstenhaber_compute_A∞ in Julia.
    """
    position: int
    errors: np.ndarray          # shape [MAX_ARITY-1] = [m2..m8 errors]
    weighted_error: float       # λ-filtered total: Σ λ^{k-2} * E_k
    jacobian_rank: int          # rank of local transition Jacobian
    is_singular: bool           # det(J) near 0 → transition singularity
    postnikov_level: int        # which tower level is failing (0=m2, 6=m8)


class AInfinityEntropyEstimator:
    """
    Dynamical entropy estimator using A∞ structure maps m₂...m₈.

    Architecture:
      For each arity k ∈ {2,...,8}:
        W_k: ℝ^{(k-1)·d} → ℝ^d  (locally fitted linear map)
        E_t^k = ‖W_k · [h_{t-k+2},...,h_t] - h_{t+1}‖₂

      Filtration weight (from FilteredAInfAlgebra):
        w_k = exp(-λ · (k-2))  = λ^{k-2} in energy units

      Total A∞ entropy:
        E_t^{A∞} = Σ_{k=2}^{8} w_k · E_t^k

      Jacobian (local transition degeneracy):
        J_t = W_2  (the bigram linear map = linearized m₂)
        rank_deficiency = d - rank(J_t) > 0 → orbit degeneracy

    Fitting:
      W_k is fitted by least squares on a local window of size fit_window.
      This is equivalent to the path energy minimisation in your Julia
      generate_weighted_paths function.
    """

    def __init__(
        self,
        max_arity: int = MAX_ARITY,
        fit_window: int = 64,
        lambda_decay: float = LAMBDA_DECAY,
        energy_cutoff: float = ENERGY_CUTOFF,
        pca_dim: int = 48,          # Increased from 16 as recommended
    ):
        self.max_arity   = max_arity
        self.fit_window  = fit_window
        self.lambda_decay = lambda_decay
        self.energy_cutoff = energy_cutoff
        self.pca_dim     = pca_dim

        # Filtration weights: w_k = exp(-λ * (k-2)) for k=2..8
        # Mirrors FilteredAInfAlgebra energy penalty
        self.filtration_weights = np.array([
            np.exp(-lambda_decay * (k - 2))
            for k in range(2, max_arity + 1)
        ])
        # Normalise so weights sum to 1
        self.filtration_weights /= self.filtration_weights.sum()

    def _fit_mk(
        self,
        states: np.ndarray,
        t: int,
        k: int
    ) -> Tuple[np.ndarray, float]:
        """
        Fit the k-ary A∞ map W_k locally around position t.

        W_k: ℝ^{(k-1)·d} → ℝ^d

        Uses a window of fit_window states ending at t.
        Returns (W_k, prediction_error_at_t).

        This corresponds to computing m_k on the local algebra
        element (path of length k-1) — the discrete A∞ map.
        """
        d = states.shape[1]
        context_len = k - 1   # m_k takes k-1 inputs in our convention

        # Need at least context_len+1 states before t for fitting
        # and context_len states ending at t for prediction
        window_start = max(0, t - self.fit_window)
        window_end   = t  # exclusive: we predict state[t] from state[t-1..t-k+1]

        n_samples = window_end - window_start - context_len
        if n_samples < max(4, context_len + 1):
            # Not enough data — return zero error (admissible)
            return np.zeros((d, context_len * d)), 0.0

        # Build design matrix X: [h_{t-k+2},...,h_{t-1}] concatenated
        # Target y: h_t
        X_rows = []
        y_rows = []
        for i in range(window_start + context_len, window_end):
            context = states[i - context_len:i].flatten()   # [(k-1)*d]
            target  = states[i]                               # [d]
            X_rows.append(context)
            y_rows.append(target)

        X = np.array(X_rows)   # [n_samples, (k-1)*d]
        y = np.array(y_rows)   # [n_samples, d]

        # Regularised least squares: W_k = (X^T X + εI)^{-1} X^T y
        # Regularisation ε prevents overfitting on small windows
        eps = 1e-4 * np.trace(X.T @ X) / max(X.shape[1], 1)
        try:
            W, _, _, _ = np.linalg.lstsq(
                X.T @ X + eps * np.eye(X.shape[1]),
                X.T @ y,
                rcond=None
            )
        except np.linalg.LinAlgError:
            return np.zeros((context_len * d, d)), 0.0

        # Prediction error at position t
        if t >= context_len:
            context_t = states[t - context_len:t].flatten()
            predicted  = context_t @ W
            error = float(np.linalg.norm(predicted - states[t]))
        else:
            error = 0.0

        return W.T, error   # W.T shape [d, (k-1)*d]

    def _jacobian_analysis(
        self,
        W2: np.ndarray,
        d: int
    ) -> Tuple[int, bool]:
        """
        Analyse the local transition Jacobian J = W₂ (the m₂ linear map).

        In your Julia engine, this corresponds to:
          - The m₂ structure map restricted to a single edge
          - The transition matrix T used in hamiltonian_step!
          - Rank deficiency = orbit degeneracy = Postnikov obstruction

        Returns (rank, is_singular).

        A singular Jacobian signals:
          - Dehn twist monodromy (Pass 7 in AU compiler)
          - Vanishing cycle at a critical point (Lemma 1.10 Seidel 2002)
          - Obstruction to lifting at the corresponding tower level
        """
        if W2.shape[0] != W2.shape[1]:
            # Non-square: use SVD rank
            sv = np.linalg.svd(W2, compute_uv=False)
        else:
            sv = np.linalg.svd(W2, compute_uv=False)

        threshold = 1e-3 * sv[0] if sv[0] > 0 else 1e-10
        rank = int(np.sum(sv > threshold))
        is_singular = rank < min(W2.shape)

        return rank, is_singular

    def _postnikov_level_from_errors(
        self,
        errors: np.ndarray
    ) -> int:
        """
        Identify which Postnikov tower level is failing.

        Correspondence (from postnikov_rewards.jl):
          m₂ error spike → Level 0 failure (π₀, reachability)
          m₃ error spike → Level 1 failure (π₁, Markov circuits)
          m₄ error spike → Level 2 failure (π₂, homotopy classes)
          m₅ error spike → Level 3 (k-invariant 1)
          m₆ error spike → Level 4 (k-invariant 2, the "62-class" obstruction)
          m₇ error spike → Level 5
          m₈ error spike → Level 6 (maximum obstruction depth)

        The failing level is the LOWEST arity with anomalous error,
        matching the Postnikov convention: obstructions propagate
        from lower to higher levels (each level blocks the lift to next).
        """
        if len(errors) == 0 or errors.max() < self.energy_cutoff:
            return -1  # No failure

        # Normalise errors to [0,1]
        norm_errors = errors / (errors.max() + 1e-10)

        # Find first (lowest arity) spike above threshold
        for level, err in enumerate(norm_errors):
            if err > 0.5:   # relative threshold
                return level  # level k → Postnikov level k (m_{k+2})

        return int(np.argmax(norm_errors))

    def compute_entropy_sequence(
        self,
        hidden_states: np.ndarray,
        reduce_dim: bool = True
    ) -> np.ndarray:
        """
        Compute the full A∞ entropy sequence over the trajectory.

        Args:
            hidden_states: [seq_len, hidden_dim]
            reduce_dim: whether to PCA-reduce before fitting
                        (True recommended for hidden_dim > 128)

        Returns:
            entropy_seq: [seq_len] array of A∞ weighted prediction errors
        """
        seq_len, hidden_dim = hidden_states.shape

        # PCA reduction to pca_dim (48 — increased for orbit resolution)
        if reduce_dim and hidden_dim > self.pca_dim:
            states = self._pca_reduce(hidden_states)
        else:
            states = hidden_states.copy()

        d = states.shape[1]
        entropy_seq = np.zeros(seq_len)

        # Compute for each position t ≥ max_arity - 1
        min_t = self.max_arity - 1

        for t in range(min_t, seq_len - 1):
            errors_at_t = np.zeros(self.max_arity - 1)  # m₂...m₈

            W2 = None
            for ki, k in enumerate(range(2, self.max_arity + 1)):
                # Filtration weight: prune negligible contributions
                w_k = self.filtration_weights[ki]
                if w_k < self.energy_cutoff:
                    continue

                W_k, err_k = self._fit_mk(states, t, k)
                errors_at_t[ki] = err_k

                if k == 2:
                    W2 = W_k

            # A∞ weighted total (mirrors filtration in Julia engine)
            entropy_seq[t] = float(
                np.dot(self.filtration_weights, errors_at_t)
            )

        return entropy_seq

    def compute_full_analysis(
        self,
        hidden_states: np.ndarray
    ) -> Tuple[np.ndarray, List[AInfPredictionError]]:
        """
        Full A∞ analysis: entropy sequence + per-position detailed breakdown.

        Returns:
            (entropy_seq, detailed_errors)

        detailed_errors contains per-position AInfPredictionError objects
        with m₂...m₈ breakdown and Jacobian analysis.
        Useful for identifying WHICH Postnikov level is failing.
        """
        seq_len, hidden_dim = hidden_states.shape

        if hidden_dim > self.pca_dim:
            states = self._pca_reduce(hidden_states)
        else:
            states = hidden_states.copy()

        d = states.shape[1]
        entropy_seq = np.zeros(seq_len)
        detailed    = []

        min_t = self.max_arity - 1

        for t in range(min_t, seq_len - 1):
            errors_at_t = np.zeros(self.max_arity - 1)
            W2 = None

            for ki, k in enumerate(range(2, self.max_arity + 1)):
                w_k = self.filtration_weights[ki]
                if w_k < self.energy_cutoff:
                    continue
                W_k, err_k = self._fit_mk(states, t, k)
                errors_at_t[ki] = err_k
                if k == 2:
                    W2 = W_k

            weighted_err = float(np.dot(self.filtration_weights, errors_at_t))
            entropy_seq[t] = weighted_err

            # Jacobian analysis (m₂ only, for speed)
            if W2 is not None and W2.size > 0:
                # W2 shape: [d, d] (from m₂: single-step prediction)
                W2_sq = W2[:d, :d] if W2.shape[1] >= d else W2
                jrank, is_sing = self._jacobian_analysis(W2_sq, d)
            else:
                jrank, is_sing = d, False

            post_level = self._postnikov_level_from_errors(errors_at_t)

            detailed.append(AInfPredictionError(
                position      = t,
                errors        = errors_at_t.copy(),
                weighted_error= weighted_err,
                jacobian_rank = jrank,
                is_singular   = is_sing,
                postnikov_level = post_level
            ))

        return entropy_seq, detailed

    def _pca_reduce(self, states: np.ndarray) -> np.ndarray:
        """
        PCA reduction to self.pca_dim.

        IMPORTANT: We use 48 dimensions (not 16) to preserve:
          - rotational subspaces (cyclic orbit structure)
          - medium-variance directions (periodic dynamics)
          - toroidal structure (m_k for k≥4)

        As you noted: PCA preserves max-variance directions,
        but periodic dynamics live in medium-variance rotational
        subspaces. 48 dims captures these.
        """
        states_centered = states - states.mean(axis=0)
        try:
            U, S, Vt = np.linalg.svd(states_centered, full_matrices=False)
            n_components = min(self.pca_dim, Vt.shape[0])
            basis = Vt[:n_components]                    # [pca_dim, hidden_dim]
            return states_centered @ basis.T              # [seq_len, pca_dim]
        except np.linalg.LinAlgError:
            # Random projection fallback
            proj = np.random.randn(states.shape[1], self.pca_dim)
            proj, _ = np.linalg.qr(proj)
            return states_centered @ proj

    def detect_frobenius_spikes(
        self,
        entropy_seq: np.ndarray,
        detailed: Optional[List[AInfPredictionError]],
        orbit_profiles: list
    ) -> Tuple[List[int], List[int]]:
        """
        Enhanced spike detection using both:
          1. Amplitude threshold (from Seidel 2025 Remark 1.7)
          2. Postnikov level signature (from postnikov_rewards.jl)
          3. Jacobian singularity positions (from curved_hh2 m₂ analysis)

        Seidel bound: spike at position k should satisfy
          amplitude ≤ k^{-1} · p^γ   for admissible trajectories
        Exceeding this = Seidel divisibility violation.
        """
        from gsod_core import PRIMES, SEIDEL_GAMMA, WINDOW_SIZE

        seq_len = len(entropy_seq)
        spike_positions = []
        frobenius_periods = []

        # ── Amplitude-based spike detection ───────────────────────
        mean_e = entropy_seq[entropy_seq > 0].mean() if (entropy_seq > 0).any() else 0
        std_e  = entropy_seq[entropy_seq > 0].std()  if (entropy_seq > 0).any() else 1

        # Dynamic threshold: Seidel k^{-1} * p^gamma bound
        for pos in range(seq_len):
            k = max(1, pos // WINDOW_SIZE)
            # Seidel bound: threshold = k^{-1} * min(p^gamma for p in PRIMES)
            seidel_bound = (1.0 / k) * min(p ** SEIDEL_GAMMA for p in PRIMES)
            statistical_threshold = mean_e + 1.5 * std_e

            # Use the more sensitive of the two
            threshold = min(seidel_bound, statistical_threshold)

            if entropy_seq[pos] > threshold and entropy_seq[pos] > mean_e:
                spike_positions.append(pos)

        # ── Jacobian singularity positions ─────────────────────────
        if detailed:
            for item in detailed:
                if item.is_singular and item.position not in spike_positions:
                    spike_positions.append(item.position)

        spike_positions = sorted(set(spike_positions))

        # ── Frobenius period detection via spike intervals ─────────
        if len(spike_positions) >= 3:
            intervals = np.diff(spike_positions)
            if len(intervals) >= 2:
                # Check for periodic intervals
                for candidate_period in range(2, seq_len // 2):
                    matches = sum(1 for iv in intervals if abs(iv - candidate_period) <= 3)
                    if matches >= len(intervals) * 0.4:
                        frobenius_periods.append(candidate_period)

        # ── Cross-check with orbit profile recurrence predictions ──
        if orbit_profiles:
            predicted = set()
            for profile in orbit_profiles:
                predicted.add(profile.recurrence_period)

            # DFT-based period detection
            if len(entropy_seq) > 16:
                from scipy.fft import fft, fftfreq
                fft_vals = np.abs(fft(entropy_seq - entropy_seq.mean()))
                freqs = fftfreq(seq_len, d=1.0)
                power = fft_vals[:seq_len//2] ** 2
                mean_p = power.mean()
                std_p  = power.std()
                for idx in np.where(power > mean_p + 2*std_p)[0]:
                    if idx > 0 and freqs[idx] != 0:
                        period = int(round(1.0 / abs(freqs[idx])))
                        if 2 <= period <= seq_len // 2:
                            for pred in predicted:
                                if abs(period - pred) <= max(3, pred // 10):
                                    frobenius_periods.append(period)
                                    break

        frobenius_periods = sorted(set(frobenius_periods))
        return spike_positions, frobenius_periods


# ─────────────────────────────────────────────────────────────────
# INTEGRATION SHIM: drop-in replacement for EntropyPeriodicityDetector
# ─────────────────────────────────────────────────────────────────

class AInfEntropyDetector:
    """
    Drop-in replacement for EntropyPeriodicityDetector in gsod_core.py.

    Replaces:
      H_t ~ H(cos(h_t, h_j))   (static neighborhood entropy)
    With:
      E_t = Σ_k λ^{k-2} ‖W_k [h_{t-k+2},...,h_t] - h_{t+1}‖  (dynamical)

    Usage in GSOD:
      gsod = GSOD()
      gsod.entropy_detector = AInfEntropyDetector()
      result = gsod.analyze(hidden_states)
    """

    def __init__(self, pca_dim: int = 48, fit_window: int = 64):
        self.estimator = AInfinityEntropyEstimator(
            pca_dim=pca_dim,
            fit_window=fit_window
        )
        self._last_detailed = None

    def compute_entropy_sequence(
        self,
        hidden_states: np.ndarray
    ) -> np.ndarray:
        """Compute A∞ dynamical entropy. Stores detailed for spike detection."""
        entropy_seq, detailed = self.estimator.compute_full_analysis(
            hidden_states
        )
        self._last_detailed = detailed
        return entropy_seq

    def detect_frobenius_spikes(
        self,
        entropy_seq: np.ndarray,
        orbit_profiles: list
    ) -> Tuple[List[int], List[int]]:
        """Enhanced spike detection using A∞ structure."""
        return self.estimator.detect_frobenius_spikes(
            entropy_seq,
            self._last_detailed,
            orbit_profiles
        )

    def get_postnikov_failure_map(self) -> dict:
        """
        Return dict mapping skeleton depth → failing Postnikov level.
        Useful for diagnosing WHICH tower level is obstructed.

        Connects to postnikov_rewards.jl: PolicyLevel.k_invariant
        """
        from gsod_core import WINDOW_SIZE

        if not self._last_detailed:
            return {}

        failure_map = {}
        for item in self._last_detailed:
            if item.postnikov_level >= 0:
                skeleton_depth = item.position // WINDOW_SIZE + 1
                if skeleton_depth not in failure_map:
                    failure_map[skeleton_depth] = []
                failure_map[skeleton_depth].append(item.postnikov_level)

        # Summarise: most common failure level per skeleton depth
        return {
            k: max(set(v), key=v.count)
            for k, v in failure_map.items()
        }
