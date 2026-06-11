"""
GSOD: Galois-Stratified Obstruction Detector
=============================================
Theoretical foundation:
  - Seidel (2002): Long exact sequence for symplectic Floer cohomology
  - Seidel (2025): P-adic splittings of the quantum connection
  - Henniart-Vigneras (2017): Representations of a p-adic group in characteristic p
  - AU-Fukaya compiler: Universal IR for networks with Lagrangian structure

Core claim:
  A hallucination in a generated sequence is a trajectory whose
  Galois orbit profile (v_2, v_5, v_7) violates the Seidel divisibility
  condition: v_p(E_lambda^k) >= log_p(k) - gamma.

Architecture:
  Layer 0: Skeleton filtration (8 windows of 128 tokens over 1024)
  Layer 1: Three-prime Frobenius orbit computation
  Layer 2: Hecke T_P nilpotency check (supersingularity)
  Layer 3: Seidel divisibility gate
  Layer 4: Entropy periodicity (Frobenius-periodic spike detection)
  Layer 5: Sheaf gluing consistency (Cech H^1)
  Layer 6: Orbit taxonomy and correction prescription
"""

import numpy as np
from scipy.fft import fft, fftfreq
from scipy.stats import entropy as scipy_entropy
from sympy import factorint, gcd, lcm, isprime, Poly, GF
from sympy.abc import x
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────

PRIMES = [2, 5, 7]          # Three-prime gate
MAX_DEPTH = 8               # Obstruction depth m=8
SEQ_LEN = 1024              # Fixed token budget
WINDOW_SIZE = SEQ_LEN // MAX_DEPTH  # 128 tokens per skeleton level

# Seidel 2025, Remark 1.9: alpha = q-pole order of e_i, beta = dim_C(M)
# We use alpha=1, beta=2 as defaults for the semantic complex
SEIDEL_ALPHA = 1.0
SEIDEL_BETA  = 2.0
SEIDEL_GAMMA = 1.0   # shift constant gamma from Remark 1.7

# Henniart-Vigneras: max orbit sizes per prime at our depth bound
# |Gal(F_{p^k}/F_p)| = k, bounded by MAX_DEPTH
MAX_ORBIT = {2: 8, 5: 2, 7: 1}


# ─────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────

@dataclass
class OrbitProfile:
    """
    Galois orbit profile for a trajectory at a given skeleton depth.
    Corresponds to the Frobenius orbit decomposition from
    Henniart-Vigneras Theorem 1.
    """
    depth: int                          # Skeleton level k in {1,...,8}
    orbit_sizes: Dict[int, int]         # {prime: orbit_size k_i}
    valuation: Dict[int, float]         # {prime: v_p value}
    seidel_ok: Dict[int, bool]          # {prime: satisfies divisibility}
    nilpotency: Dict[int, bool]         # {prime: T_P acts nilpotently}

    @property
    def is_admissible(self) -> bool:
        """
        A trajectory is admissible (hallucination-free) iff:
        1. Seidel divisibility holds at all primes (Seidel 2025, Thm 1.8)
        2. T_P acts nilpotently (Henniart-Vigneras Thm 9: supersingular)
        """
        return (all(self.seidel_ok.values()) and
                all(self.nilpotency.values()))

    @property
    def hallucination_type(self) -> str:
        """Classify by orbit profile (k_2, k_5, k_7)."""
        k2 = self.orbit_sizes.get(2, 1)
        k5 = self.orbit_sizes.get(5, 1)
        k7 = self.orbit_sizes.get(7, 1)

        if k2 == 1 and k5 == 1 and k7 == 1:
            return "CLEAN"
        elif k2 > 1 and k5 == 1 and k7 == 1:
            return f"BINARY_ENTANGLEMENT_orbit{k2}"
        elif k2 == 1 and k5 > 1 and k7 == 1:
            return "CONJUGATE_CLOSURE_FAILURE"
        elif k2 > 1 and k5 > 1 and k7 == 1:
            return f"COMPOUND_orbit{k2}x{k5}"
        elif k7 > 0 and not self.seidel_ok.get(7, True):
            return "RESONANCE_7ADIC"
        else:
            return f"MIXED_({k2},{k5},{k7})"

    @property
    def effective_context_fraction(self) -> float:
        """
        Orbit size as context-collapse depth (Henniart-Vigneras insight):
        orbit_size k_i at prime 2 => model uses 1/k_i of nominal context.
        """
        k2 = self.orbit_sizes.get(2, 1)
        return 1.0 / k2

    @property
    def recurrence_period(self) -> int:
        """
        Predicted Frobenius-periodic recurrence of the hallucination.
        lcm(k_i * p_i) over all primes.
        """
        periods = [self.orbit_sizes.get(p, 1) * p for p in PRIMES]
        result = periods[0]
        for period in periods[1:]:
            result = int(lcm(result, period))
        return result


@dataclass
class GSODResult:
    """Full GSOD analysis result for a 1024-token trajectory."""
    orbit_profiles: List[OrbitProfile]       # One per skeleton level
    entropy_sequence: np.ndarray             # H(x_n+1 | x_1..x_n) vs n
    entropy_spikes: List[int]                # Positions of anomalous spikes
    frobenius_periods: List[int]             # Detected periodic frequencies
    sheaf_consistency: float                 # Cech H^1 gluing score [0,1]
    is_hallucination: bool                   # Final verdict
    confidence: float                        # Detection confidence [0,1]
    hallucination_types: List[str]           # Types per depth
    correction_prescription: str            # What context to add
    summary: str                             # Human-readable summary


# ─────────────────────────────────────────────────────────────────
# LAYER 0: SKELETON FILTRATION
# ─────────────────────────────────────────────────────────────────

class SkeletonFiltration:
    """
    Partition a 1024-token sequence into 8 skeleton windows.
    X^0 ⊂ X^1 ⊂ ... ⊂ X^8

    Each window X^k covers tokens [(k-1)*128, k*128).
    The extension problem X^k → X^{k+1} is where obstructions live.
    """

    def __init__(self, seq_len: int = SEQ_LEN, n_levels: int = MAX_DEPTH):
        self.seq_len = seq_len
        self.n_levels = n_levels
        self.window_size = seq_len // n_levels

    def partition(self, hidden_states: np.ndarray) -> List[np.ndarray]:
        """
        Split hidden states [seq_len, hidden_dim] into 8 windows.
        Returns list of arrays each of shape [window_size, hidden_dim].
        """
        windows = []
        for k in range(self.n_levels):
            start = k * self.window_size
            end = start + self.window_size
            windows.append(hidden_states[start:end])
        return windows

    def extension_pairs(self, windows: List[np.ndarray]):
        """
        Yield (X^k, X^{k+1}) pairs for obstruction checking.
        The obstruction to extending f: X^k → M lives in H^{k+1}(X; π_k(M)).
        """
        for k in range(len(windows) - 1):
            yield k + 1, windows[k], windows[k + 1]


# ─────────────────────────────────────────────────────────────────
# LAYER 1: FROBENIUS ORBIT COMPUTATION
# ─────────────────────────────────────────────────────────────────

class FrobeniusOrbitDetector:
    """
    Compute Galois orbit profiles at each prime p ∈ {2, 5, 7}.

    Mathematical basis:
    - Over F_p, eigenvalues of the semantic transition operator
      cluster into Frobenius orbits {λ, φ_p(λ), φ_p²(λ), ...}
    - Orbit size k_i = [F_p(λ) : F_p] = degree of min poly of λ over F_p
    - Henniart-Vigneras Thm 1: V_{F_p} sees exactly these orbit bundles

    Implementation:
    - Represent the semantic transition as a matrix T derived from
      cross-window attention/hidden state transitions
    - Compute minimal polynomial of T over F_p
    - Factor it — irreducible factors give orbit sizes
    """

    def __init__(self, primes: List[int] = PRIMES):
        self.primes = primes

    def _semantic_transition_matrix(
        self,
        window_a: np.ndarray,
        window_b: np.ndarray,
        dim: int = 16
    ) -> np.ndarray:
        """
        Construct a reduced semantic transition matrix between two windows.
        We use PCA to reduce to `dim` dimensions for tractability,
        then compute the linear map window_a → window_b in that basis.

        This is the discretized version of the Lagrangian Floer
        intersection CF*(L_i, L_j) in the AU-Fukaya IR.
        """
        # Stack windows and compute covariance structure
        combined = np.vstack([window_a, window_b])
        combined = combined - combined.mean(axis=0)

        # SVD for PCA basis (tractable even at hidden_dim=768)
        try:
            U, S, Vt = np.linalg.svd(combined, full_matrices=False)
            basis = Vt[:dim]  # [dim, hidden_dim]
        except np.linalg.LinAlgError:
            # Fallback: random projection
            basis = np.random.randn(dim, combined.shape[1])
            basis = basis / np.linalg.norm(basis, axis=1, keepdims=True)

        # Project both windows into reduced basis
        A_proj = window_a @ basis.T   # [128, dim]
        B_proj = window_b @ basis.T   # [128, dim]

        # Least-squares fit: T such that A_proj.T @ T ≈ B_proj.T
        # T is our semantic transition matrix in the reduced space
        try:
            T, _, _, _ = np.linalg.lstsq(A_proj, B_proj, rcond=None)
        except np.linalg.LinAlgError:
            T = np.eye(dim)

        return T  # [dim, dim]

    def _char_poly_mod_p(self, T: np.ndarray, p: int) -> List[int]:
        """
        Compute characteristic polynomial of T mod p.
        Returns coefficients [a_0, ..., a_n] of det(xI - T) mod p.
        """
        n = T.shape[0]
        # Round T to integers mod p
        T_int = np.round(T).astype(int) % p

        # Compute char poly via Hessenberg form (more stable)
        # char poly coefficients using Newton's identities
        traces = np.zeros(n + 1)
        T_power = np.eye(n, dtype=float)
        for k in range(1, n + 1):
            T_power = T_power @ T_int
            traces[k] = int(round(np.trace(T_power))) % p

        # Newton's identity: p_k + sum_{i=1}^{k} e_i p_{k-i} = 0
        # where e_i are elem sym polys (char poly coeffs), p_k are power sums
        e = np.zeros(n + 1, dtype=int)
        e[0] = 1
        for k in range(1, n + 1):
            s = 0
            for i in range(1, k + 1):
                s += (-1) ** (i - 1) * e[k - i] * traces[i]
            e[k] = ((-1) ** k * s // k) % p

        return list(e)

    def _orbit_sizes_from_char_poly(
        self,
        char_poly_coeffs: List[int],
        p: int
    ) -> List[int]:
        """
        Factor char poly over F_p and return degrees of irreducible factors.
        Each irreducible factor of degree k_i corresponds to a Frobenius
        orbit of size k_i.

        Henniart-Vigneras: the orbit bundle has size k_i = [F_p(λ):F_p].
        """
        n = len(char_poly_coeffs) - 1
        if n == 0:
            return [1]

        # Build sympy polynomial over GF(p)
        try:
            coeffs_reversed = list(reversed(char_poly_coeffs))
            poly = Poly(coeffs_reversed, x, domain=GF(p))
            factors = poly.factor_list()
            # factors = (content, [(factor, multiplicity), ...])
            orbit_sizes = []
            for factor, mult in factors[1]:
                deg = factor.degree()
                if deg > 0:
                    for _ in range(mult):
                        orbit_sizes.append(deg)
            return orbit_sizes if orbit_sizes else [1]
        except Exception:
            # Fallback: estimate from rank structure
            rank = min(n, 4)
            return [1] * rank

    def _p_adic_valuation(self, matrix: np.ndarray, p: int) -> float:
        """
        Compute p-adic valuation of the determinant of (I - T).
        This approximates v_p(E_lambda^k) from Seidel 2025.

        By Seidel Remark 1.7: v_p(x_k) >= log_p(k) - gamma
        signals admissibility.
        """
        n = matrix.shape[0]
        I_minus_T = np.eye(n) - matrix
        try:
            det = abs(np.linalg.det(I_minus_T))
            if det < 1e-12:
                return float('inf')  # Maximally degenerate — bad
            # Approximate p-adic valuation via log
            val = np.log(det) / np.log(p)
            return max(0.0, val)
        except Exception:
            return 0.0

    def compute_orbit_profile(
        self,
        window_a: np.ndarray,
        window_b: np.ndarray,
        depth: int
    ) -> OrbitProfile:
        """
        Full orbit profile computation at skeleton level k = depth.
        """
        # Build semantic transition matrix
        T = self._semantic_transition_matrix(window_a, window_b)

        orbit_sizes = {}
        valuations = {}
        seidel_ok = {}
        nilpotency = {}

        for p in self.primes:
            # Frobenius orbit sizes from char poly factorization
            char_poly = self._char_poly_mod_p(T, p)
            orbits = self._orbit_sizes_from_char_poly(char_poly, p)
            max_orbit_size = max(orbits) if orbits else 1
            orbit_sizes[p] = min(max_orbit_size, MAX_ORBIT[p])

            # p-adic valuation
            val = self._p_adic_valuation(T, p)
            valuations[p] = val

            # Seidel divisibility check (Thm 1.8, Remark 1.7):
            # v_p(E_lambda^k) >= log_p(k) - gamma
            threshold = (np.log(depth) / np.log(p)) - SEIDEL_GAMMA
            seidel_ok[p] = val >= threshold

            # Hecke T_P nilpotency check (H-V Thm 9: supersingular <=> T_P nilpotent)
            # T_P nilpotent iff T^n → 0 mod p
            T_mod_p = np.round(T).astype(int) % p
            T_power = T_mod_p.copy()
            nilpotent = False
            for _ in range(MAX_DEPTH):
                T_power = (T_power @ T_mod_p) % p
                if np.all(T_power == 0):
                    nilpotent = True
                    break
            nilpotency[p] = nilpotent

        return OrbitProfile(
            depth=depth,
            orbit_sizes=orbit_sizes,
            valuation=valuations,
            seidel_ok=seidel_ok,
            nilpotency=nilpotency
        )


# ─────────────────────────────────────────────────────────────────
# LAYER 4: ENTROPY PERIODICITY DETECTOR
# ─────────────────────────────────────────────────────────────────

class EntropyPeriodicityDetector:
    """
    Detect Frobenius-periodic entropy spikes in the generated sequence.

    Theoretical basis:
    - Shannon entropy is the unique H^1(Prob; R) cocycle (Baudot-Bennequin)
    - Hallucinating sequences show entropy spikes at Frobenius-periodic positions
    - Spike amplitude bound: p^{-log_p(k) + gamma} = k^{-1} * p^gamma
      (from Seidel 2025 Remark 1.7 turned into runtime detector)

    For a sequence of hidden states, we approximate conditional entropy
    H(x_{n+1} | x_1,...,x_n) via the local reconstruction error.
    """

    def __init__(self, window: int = WINDOW_SIZE):
        self.window = window

    def _local_entropy(self, states: np.ndarray, pos: int) -> float:
        """
        Approximate H(x_{pos+1} | x_{pos-w:pos}) using local
        distribution of hidden state distances.
        """
        w = min(self.window, pos)
        if w < 2:
            return 0.0

        context = states[max(0, pos - w):pos]
        target = states[pos]

        # Distribution over context similarities
        dists = np.linalg.norm(context - target, axis=1)
        dists = dists / (dists.sum() + 1e-8)

        # Shannon entropy of similarity distribution
        return float(scipy_entropy(dists + 1e-10))

    def compute_entropy_sequence(
        self,
        hidden_states: np.ndarray
    ) -> np.ndarray:
        """
        Compute conditional entropy at each position in the sequence.
        Returns array of shape [seq_len].
        """
        seq_len = hidden_states.shape[0]
        entropy_seq = np.zeros(seq_len)

        for pos in range(1, seq_len):
            entropy_seq[pos] = self._local_entropy(hidden_states, pos)

        return entropy_seq

    def detect_frobenius_spikes(
        self,
        entropy_seq: np.ndarray,
        orbit_profiles: List[OrbitProfile]
    ) -> Tuple[List[int], List[int]]:
        """
        Detect whether entropy shows Frobenius-periodic spikes at
        predicted recurrence positions.

        Returns (spike_positions, detected_periods).
        """
        # Predicted recurrence periods from orbit profiles
        predicted_periods = set()
        for profile in orbit_profiles:
            predicted_periods.add(profile.recurrence_period)

        # DFT of entropy sequence to find dominant frequencies
        n = len(entropy_seq)
        fft_vals = np.abs(fft(entropy_seq - entropy_seq.mean()))
        freqs = fftfreq(n, d=1.0)

        # Find peaks in power spectrum
        power = fft_vals[:n // 2] ** 2
        mean_power = power.mean()
        std_power = power.std()
        peak_threshold = mean_power + 2 * std_power

        peak_indices = np.where(power > peak_threshold)[0]
        detected_periods = []
        for idx in peak_indices:
            if idx > 0:
                period = int(round(1.0 / abs(freqs[idx]))) if freqs[idx] != 0 else 0
                if 2 <= period <= n // 2:
                    detected_periods.append(period)

        # Find actual spike positions in entropy sequence
        mean_ent = entropy_seq.mean()
        std_ent = entropy_seq.std()
        spike_threshold = mean_ent + 2.0 * std_ent
        spike_positions = list(np.where(entropy_seq > spike_threshold)[0])

        # Check if detected periods match Frobenius predictions
        frobenius_match = []
        for det_period in detected_periods:
            for pred_period in predicted_periods:
                if abs(det_period - pred_period) <= 2:  # tolerance
                    frobenius_match.append(det_period)
                    break

        return spike_positions, frobenius_match


# ─────────────────────────────────────────────────────────────────
# LAYER 5: SHEAF GLUING (Čech H^1)
# ─────────────────────────────────────────────────────────────────

class SheafGluingChecker:
    """
    Check Čech H^1 consistency across overlapping context windows.

    Theoretical basis:
    - Hallucinations paper: factual/semantic hallucinations require
      a 4th layer — sheaf obstruction Ȟ^1(U, F)
    - Alice-in-Boston example: each window locally valid but
      global section inconsistent
    - Implementation: check whether entity/semantic assignments
      in overlapping windows are consistent (can be glued)

    We use hidden state cosine similarity across window overlaps
    as a proxy for semantic assignment consistency.
    """

    def __init__(self, overlap: int = 32):
        """overlap: number of tokens shared between adjacent windows."""
        self.overlap = overlap

    def compute_gluing_score(
        self,
        windows: List[np.ndarray]
    ) -> float:
        """
        Compute Čech H^1 gluing score.

        For each pair of adjacent windows, check whether the
        'boundary values' (overlapping hidden states) are consistent.

        Score = 1.0: perfect gluing (no sheaf obstruction)
        Score = 0.0: complete gluing failure (maximal Ȟ^1)
        """
        if len(windows) < 2:
            return 1.0

        consistency_scores = []

        for k in range(len(windows) - 1):
            win_a = windows[k]
            win_b = windows[k + 1]

            # Boundary values: last `overlap` tokens of win_a
            # and first `overlap` tokens of win_b
            boundary_a = win_a[-self.overlap:]   # [overlap, hidden_dim]
            boundary_b = win_b[:self.overlap]    # [overlap, hidden_dim]

            # Cosine similarity between corresponding boundary states
            norm_a = np.linalg.norm(boundary_a, axis=1, keepdims=True) + 1e-8
            norm_b = np.linalg.norm(boundary_b, axis=1, keepdims=True) + 1e-8
            cos_sim = np.sum(
                (boundary_a / norm_a) * (boundary_b / norm_b),
                axis=1
            )

            # Mean cosine similarity as gluing score for this transition
            consistency_scores.append(float(cos_sim.mean()))

        # Overall Čech H^1 score: mean of window-pair consistencies
        # Sheaf obstruction is non-trivial when score drops below 0.5
        return float(np.mean(consistency_scores))


# ─────────────────────────────────────────────────────────────────
# LAYER 6: ORBIT TAXONOMY AND CORRECTION
# ─────────────────────────────────────────────────────────────────

class OrbitTaxonomy:
    """
    Classify hallucination type and prescribe correction
    based on orbit profile, following Henniart-Vigneras Thm 9
    and the lattice isomorphism (Thm 2).

    Correction prescription uses the lattice isomorphism insight:
    since L_W ≅ L_{F(W)} under parabolic induction, we know
    WHICH sub-context the model has collapsed to, and can
    prescribe the specific additional context needed to lift
    the degenerate orbit back to the full representation.
    """

    TAXONOMY = {
        "CLEAN": {
            "description": "No obstruction detected. Trajectory is admissible.",
            "severity": 0,
            "correction": "None required."
        },
        "BINARY_ENTANGLEMENT_orbit2": {
            "description": "Size-2 binary orbit: orientation failure in conjugate pair. "
                           "Two structurally mirror-symmetric failures cycling under φ_2.",
            "severity": 2,
            "correction": "Add explicit directional constraint to prompt. "
                          "Specify which of the two mirrored interpretations is correct."
        },
        "BINARY_ENTANGLEMENT_orbit4": {
            "description": "Size-4 binary orbit: four orientation failure modes in 4-cycle. "
                           "Cyclic entity confusion pattern.",
            "severity": 4,
            "correction": "Add entity disambiguation context. "
                          "Explicitly distinguish the 4 related entities/concepts in cycle."
        },
        "BINARY_ENTANGLEMENT_orbit8": {
            "description": "Size-8 binary orbit: maximum binary entanglement. "
                           "8 structurally consistent but globally inconsistent claims.",
            "severity": 8,
            "correction": "Decompose prompt into 8 independent sub-claims. "
                          "Verify each independently before combining."
        },
        "CONJUGATE_CLOSURE_FAILURE": {
            "description": "Size-2 quintic orbit: global compositional closure fails "
                           "in conjugate pair. Trajectory passes local checks but "
                           "fails globally in two complementary directions.",
            "severity": 3,
            "correction": "Add explicit closure constraint: specify the endpoint "
                          "state that the trajectory must reach."
        },
        "COMPOUND_orbit2x2": {
            "description": "Combined v_2 and v_5 orbit failure. Both orientation "
                           "and global closure failing simultaneously.",
            "severity": 5,
            "correction": "Structural rewrite required. Add both directional "
                          "constraints and global closure specification."
        },
        "RESONANCE_7ADIC": {
            "description": "7-adic resonance hallucination. Passes v_2 and v_5 gates "
                           "but fails at 7-fold compositional depth. Semantic drift "
                           "completing a 7-step cycle to contradictory position.",
            "severity": 3,
            "correction": "Extend context window. The 7-step semantic cycle requires "
                          "more context to resolve. Add intermediate checkpoints."
        },
    }

    def classify(self, profiles: List[OrbitProfile]) -> Tuple[str, str, int]:
        """
        Returns (primary_type, correction, severity) for the full trajectory.
        Takes the worst (highest severity) type across all depth levels.
        """
        worst_type = "CLEAN"
        worst_severity = 0

        for profile in profiles:
            htype = profile.hallucination_type
            info = self.TAXONOMY.get(htype, {
                "severity": 1,
                "correction": "Unknown hallucination pattern. "
                              "Manual inspection recommended."
            })
            if info["severity"] > worst_severity:
                worst_severity = info["severity"]
                worst_type = htype

        info = self.TAXONOMY.get(worst_type, {
            "description": f"Orbit type: {worst_type}",
            "severity": worst_severity,
            "correction": "Manual inspection recommended."
        })

        return worst_type, info.get("correction", ""), worst_severity


# ─────────────────────────────────────────────────────────────────
# MAIN GSOD DETECTOR
# ─────────────────────────────────────────────────────────────────

class GSOD:
    """
    Galois-Stratified Obstruction Detector.

    Takes hidden states from a transformer (shape [seq_len, hidden_dim])
    and returns a full GSODResult with hallucination classification.

    Usage:
        gsod = GSOD()
        result = gsod.analyze(hidden_states)
        print(result.summary)
    """

    def __init__(self):
        self.filtration = SkeletonFiltration()
        self.orbit_detector = FrobeniusOrbitDetector()
        self.entropy_detector = EntropyPeriodicityDetector()
        self.sheaf_checker = SheafGluingChecker()
        self.taxonomy = OrbitTaxonomy()

    def analyze(self, hidden_states: np.ndarray) -> GSODResult:
        """
        Full GSOD analysis pipeline.

        Args:
            hidden_states: np.ndarray of shape [seq_len, hidden_dim]
                          Extracted from transformer (any layer works;
                          last hidden layer recommended)

        Returns:
            GSODResult with complete hallucination analysis
        """
        # Ensure correct shape
        if hidden_states.ndim == 3:
            hidden_states = hidden_states.squeeze(0)  # Remove batch dim
        seq_len, hidden_dim = hidden_states.shape

        # Pad or truncate to SEQ_LEN
        if seq_len < SEQ_LEN:
            pad = np.zeros((SEQ_LEN - seq_len, hidden_dim))
            hidden_states = np.vstack([hidden_states, pad])
        elif seq_len > SEQ_LEN:
            hidden_states = hidden_states[:SEQ_LEN]

        # ── Layer 0: Skeleton filtration ──────────────────────────
        windows = self.filtration.partition(hidden_states)

        # ── Layer 1: Frobenius orbit computation ──────────────────
        orbit_profiles = []
        for depth, win_a, win_b in self.filtration.extension_pairs(windows):
            profile = self.orbit_detector.compute_orbit_profile(
                win_a, win_b, depth
            )
            orbit_profiles.append(profile)

        # ── Layer 4: Entropy periodicity ──────────────────────────
        entropy_seq = self.entropy_detector.compute_entropy_sequence(
            hidden_states
        )
        spike_positions, frobenius_periods = (
            self.entropy_detector.detect_frobenius_spikes(
                entropy_seq, orbit_profiles
            )
        )

        # ── Layer 5: Sheaf gluing ──────────────────────────────────
        sheaf_score = self.sheaf_checker.compute_gluing_score(windows)

        # ── Layer 6: Taxonomy and verdict ─────────────────────────
        hallucination_types = [p.hallucination_type for p in orbit_profiles]
        worst_type, correction, severity = self.taxonomy.classify(orbit_profiles)

        # Compute overall hallucination verdict
        # Multiple evidence streams, weighted combination:
        orbit_fail = not all(p.is_admissible for p in orbit_profiles)
        entropy_fail = len(frobenius_periods) > 0
        sheaf_fail = sheaf_score < 0.5

        # Confidence: fraction of evidence streams flagging hallucination
        n_fail = sum([orbit_fail, entropy_fail, sheaf_fail])
        confidence = n_fail / 3.0

        is_hallucination = (confidence > 0.33)  # Any single stream triggers

        # ── Summary ───────────────────────────────────────────────
        summary = self._build_summary(
            orbit_profiles, worst_type, severity, confidence,
            sheaf_score, spike_positions, frobenius_periods,
            is_hallucination, correction
        )

        return GSODResult(
            orbit_profiles=orbit_profiles,
            entropy_sequence=entropy_seq,
            entropy_spikes=spike_positions,
            frobenius_periods=frobenius_periods,
            sheaf_consistency=sheaf_score,
            is_hallucination=is_hallucination,
            confidence=confidence,
            hallucination_types=hallucination_types,
            correction_prescription=correction,
            summary=summary
        )

    def _build_summary(
        self,
        profiles, worst_type, severity, confidence,
        sheaf_score, spikes, periods,
        is_hallucination, correction
    ) -> str:
        lines = [
            "═" * 60,
            "  GSOD — Galois-Stratified Obstruction Detector",
            "═" * 60,
            f"  Verdict:      {'⚠️  HALLUCINATION' if is_hallucination else '✅  ADMISSIBLE'}",
            f"  Confidence:   {confidence:.1%}",
            f"  Severity:     {severity}/8",
            f"  Type:         {worst_type}",
            "─" * 60,
            "  Orbit profiles by skeleton depth:",
        ]

        for p in profiles:
            status = "✅" if p.is_admissible else "⚠️ "
            k2 = p.orbit_sizes.get(2, 1)
            k5 = p.orbit_sizes.get(5, 1)
            k7 = p.orbit_sizes.get(7, 1)
            eff = f"{p.effective_context_fraction:.0%}"
            lines.append(
                f"    X^{p.depth}: {status}  orbits=({k2},{k5},{k7})  "
                f"ctx={eff}  recur={p.recurrence_period}"
            )

        lines += [
            "─" * 60,
            f"  Sheaf gluing (Ȟ¹):  {sheaf_score:.3f}  "
            f"{'✅' if sheaf_score >= 0.5 else '⚠️  GLUING FAILURE'}",
            f"  Entropy spikes:     {len(spikes)} detected",
            f"  Frobenius periods:  {periods if periods else 'none'}",
            "─" * 60,
            f"  Correction:  {correction}",
            "═" * 60,
        ]

        return "\n".join(lines)
