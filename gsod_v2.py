"""
gsod_v2.py
==========
GSOD Version 2 — with three calibration fixes and sheaf complex.

Fixes applied:
  1. Entropy threshold: mean+1.5std → median + 5*MAD  (robust heavy-tail)
  2. Confidence model: normalised spike count (n_spikes/seq_len)
  3. Sheaf complex replaces scalar Čech H¹ with full cellular sheaf

New:
  - SemanticSheafComplex with VertexStalk/EdgeStalk/FaceStalk
  - Hodge decomposition (grad/harm/curl) at each simplex
  - Sheaf Laplacian spectrum → Betti numbers β₀,β₁,β₂
  - H¹ generators = hallucination modes with window attribution
  - A∞ entropy detector (m₂...m₈) replacing cosine entropy
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from scipy.fft import fft, fftfreq
from sympy import gcd, lcm, Poly, GF
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

from sheaf_complex import SemanticSheafComplex, SheafCochainData
from ainf_entropy  import AInfEntropyDetector

# ─────────────────────────────────────────────────────────────────
# CONSTANTS (unchanged)
# ─────────────────────────────────────────────────────────────────
PRIMES       = [2, 5, 7]
MAX_DEPTH    = 8
SEQ_LEN      = 1024
WINDOW_SIZE  = SEQ_LEN // MAX_DEPTH   # 128

SEIDEL_ALPHA = 1.0
SEIDEL_BETA  = 2.0
SEIDEL_GAMMA = 1.0
MAX_ORBIT    = {2: 8, 5: 2, 7: 1}

# ─────────────────────────────────────────────────────────────────
# CALIBRATION FIX 1: Robust MAD threshold
# ─────────────────────────────────────────────────────────────────

def mad_threshold(arr: np.ndarray, k: float = 5.0) -> float:
    """
    Median Absolute Deviation threshold.
    Robust against heavy-tailed transformer entropy distributions.
    Replaces mean + 1.5*std which fires on every point.

    For a Gaussian: MAD ≈ 0.6745 * std
    For heavy-tailed: MAD << std, so k*MAD is much tighter
    k=5 ↔ roughly 5-sigma equivalent for Gaussian (extremely conservative)
    """
    arr_pos = arr[arr > 0]
    if len(arr_pos) == 0:
        return float('inf')
    med  = np.median(arr_pos)
    mad  = np.median(np.abs(arr_pos - med))
    return float(med + k * mad)


# ─────────────────────────────────────────────────────────────────
# DATA STRUCTURES (extended with sheaf data)
# ─────────────────────────────────────────────────────────────────

@dataclass
class OrbitProfile:
    depth:        int
    orbit_sizes:  Dict[int, int]
    valuation:    Dict[int, float]
    seidel_ok:    Dict[int, bool]
    nilpotency:   Dict[int, bool]

    @property
    def is_admissible(self):
        return all(self.seidel_ok.values()) and all(self.nilpotency.values())

    @property
    def hallucination_type(self):
        k2 = self.orbit_sizes.get(2, 1)
        k5 = self.orbit_sizes.get(5, 1)
        k7 = self.orbit_sizes.get(7, 1)
        if k2 == 1 and k5 == 1 and k7 == 1:
            return "CLEAN"
        if k2 > 1 and k5 == 1:
            return f"BINARY_ENTANGLEMENT_orbit{k2}"
        if k2 == 1 and k5 > 1:
            return "CONJUGATE_CLOSURE_FAILURE"
        if k2 > 1 and k5 > 1:
            return f"COMPOUND_orbit{k2}x{k5}"
        return f"MIXED_({k2},{k5},{k7})"

    @property
    def effective_context_fraction(self):
        return 1.0 / max(self.orbit_sizes.get(2, 1), 1)

    @property
    def recurrence_period(self):
        periods = [self.orbit_sizes.get(p, 1) * p for p in PRIMES]
        r = periods[0]
        for p in periods[1:]:
            r = int(lcm(r, p))
        return r


@dataclass
class GSODv2Result:
    orbit_profiles:       List[OrbitProfile]
    entropy_sequence:     np.ndarray
    entropy_spikes:       List[int]
    frobenius_periods:    List[int]
    sheaf_data:           Dict               # Full sheaf complex data
    sheaf_consistency:    float
    sheaf_betti:          Tuple              # (β₀, β₁, β₂)
    hallucination_modes:  List[Dict]         # H¹ generators
    hodge_gates:          Dict               # {0:dead, 1:unstable, 2:live} counts
    postnikov_failures:   Dict               # depth → tower level
    is_hallucination:     bool
    confidence:           float
    hallucination_types:  List[str]
    correction:           str
    summary:              str


# ─────────────────────────────────────────────────────────────────
# FROBENIUS ORBIT DETECTOR (calibrated)
# ─────────────────────────────────────────────────────────────────

class FrobeniusOrbitDetector:

    def __init__(self, primes=PRIMES, pca_dim=48):
        self.primes  = primes
        self.pca_dim = pca_dim

    def _reduce(self, window: np.ndarray, dim: int) -> np.ndarray:
        if window.shape[1] <= dim:
            return window
        c = window - window.mean(axis=0)
        try:
            _, _, Vt = np.linalg.svd(c, full_matrices=False)
            return c @ Vt[:dim].T
        except Exception:
            return c[:, :dim]

    def _transition_matrix(self, wa: np.ndarray, wb: np.ndarray) -> np.ndarray:
        """Build [dim, dim] semantic transition matrix."""
        dim = 16   # keep small for char poly tractability
        a = self._reduce(wa, dim)
        b = self._reduce(wb, dim)
        n = min(len(a), len(b))
        X, Y = a[:n], b[:n]
        eps = 1e-4 * max(np.linalg.norm(X.T @ X), 1.0)
        try:
            T, _, _, _ = np.linalg.lstsq(
                X.T @ X + eps * np.eye(dim), X.T @ Y, rcond=None
            )
        except Exception:
            T = np.eye(dim)
        return T.T   # [dim, dim]

    def _char_poly_mod_p(self, T: np.ndarray, p: int) -> List[int]:
        n = T.shape[0]
        T_int = np.round(T).astype(int) % p
        traces = np.zeros(n+1)
        Tp = np.eye(n, dtype=float)
        for k in range(1, n+1):
            Tp = Tp @ T_int
            traces[k] = int(round(np.trace(Tp))) % p
        e = np.zeros(n+1, dtype=int)
        e[0] = 1
        for k in range(1, n+1):
            s = sum((-1)**(i-1) * e[k-i] * traces[i] for i in range(1, k+1))
            e[k] = int(((-1)**k * s // max(k,1))) % p
        return list(e)

    def _orbit_sizes(self, char_poly: List[int], p: int) -> List[int]:
        n = len(char_poly) - 1
        if n == 0:
            return [1]
        try:
            rev = list(reversed(char_poly))
            poly = Poly(rev, symbols('x'), domain=GF(p))
            factors = poly.factor_list()
            sizes = []
            for fac, mult in factors[1]:
                d = fac.degree()
                if d > 0:
                    sizes.extend([d] * mult)
            return sizes if sizes else [1]
        except Exception:
            return [1]

    def _valuation(self, T: np.ndarray, p: int) -> float:
        n = T.shape[0]
        det = abs(np.linalg.det(np.eye(n) - T))
        if det < 1e-12:
            return float('inf')
        return max(0.0, np.log(det) / np.log(p))

    def compute_orbit_profile(
        self, wa: np.ndarray, wb: np.ndarray, depth: int
    ) -> OrbitProfile:
        T = self._transition_matrix(wa, wb)
        orbit_sizes, valuations, seidel_ok, nilpotency = {}, {}, {}, {}

        for p in self.primes:
            cp    = self._char_poly_mod_p(T, p)
            orbs  = self._orbit_sizes(cp, p)
            k_max = max(orbs) if orbs else 1
            orbit_sizes[p] = min(k_max, MAX_ORBIT[p])

            val = self._valuation(T, p)
            valuations[p] = val

            # Seidel divisibility: v_p ≥ log_p(depth) - γ
            thr = (np.log(max(depth,1)) / np.log(p)) - SEIDEL_GAMMA
            seidel_ok[p] = val >= thr

            # Hecke nilpotency
            Tp = np.round(T).astype(int) % p
            Tpow = Tp.copy()
            nil = False
            for _ in range(MAX_DEPTH):
                Tpow = (Tpow @ Tp) % p
                if np.all(Tpow == 0):
                    nil = True
                    break
            nilpotency[p] = nil

        return OrbitProfile(depth, orbit_sizes, valuations, seidel_ok, nilpotency)


# import sympy symbols helper
from sympy.abc import x as sympy_x
def symbols(s):
    return sympy_x


# ─────────────────────────────────────────────────────────────────
# GSOD v2 — Main Detector
# ─────────────────────────────────────────────────────────────────

class GSODv2:
    """
    GSOD Version 2.

    Changes from v1:
      1. A∞ entropy (m₂...m₈) replaces cosine entropy
      2. MAD threshold replaces mean+std threshold
      3. Confidence = normalised (not binary) evidence
      4. SemanticSheafComplex replaces scalar Čech H¹
         - full Hodge decomposition at each simplex
         - H¹ generators = named hallucination modes
         - Betti numbers β₀,β₁,β₂
    """

    def __init__(self, pca_dim: int = 48, stalk_dim: int = 16):
        self.pca_dim    = pca_dim
        self.stalk_dim  = stalk_dim
        self.orbit_det  = FrobeniusOrbitDetector(pca_dim=pca_dim)
        self.entropy_det = AInfEntropyDetector(pca_dim=pca_dim, fit_window=64)
        self.sheaf      = SemanticSheafComplex(pca_dim=pca_dim,
                                                stalk_dim=stalk_dim)

    def _pad_or_trim(self, hs: np.ndarray) -> np.ndarray:
        seq_len, d = hs.shape
        if seq_len < SEQ_LEN:
            pad = np.zeros((SEQ_LEN - seq_len, d))
            hs  = np.vstack([hs, pad])
        return hs[:SEQ_LEN]

    def _partition(self, hs: np.ndarray) -> List[np.ndarray]:
        return [hs[k*WINDOW_SIZE:(k+1)*WINDOW_SIZE] for k in range(MAX_DEPTH)]

    def analyze(self, hidden_states: np.ndarray) -> GSODv2Result:
        if hidden_states.ndim == 3:
            hidden_states = hidden_states.squeeze(0)

        hs      = self._pad_or_trim(hidden_states)
        windows = self._partition(hs)

        # ── Layer 0: Skeleton filtration ──────────────────────
        # (done above)

        # ── Layer 1: Frobenius orbit profiles ─────────────────
        orbit_profiles = []
        for k in range(MAX_DEPTH - 1):
            prof = self.orbit_det.compute_orbit_profile(
                windows[k], windows[k+1], depth=k+1
            )
            orbit_profiles.append(prof)

        # ── Layer 2 (A∞ entropy) ───────────────────────────────
        entropy_seq = self.entropy_det.compute_entropy_sequence(hs)
        postnikov   = self.entropy_det.get_postnikov_failure_map()

        # FIX 1: MAD threshold for spikes
        thr = mad_threshold(entropy_seq, k=5.0)
        spike_positions = [
            int(i) for i in np.where(entropy_seq > thr)[0]
        ]

        # Frobenius period detection
        _, frobenius_periods = self.entropy_det.detect_frobenius_spikes(
            entropy_seq, orbit_profiles
        )

        # ── Layer 3: Sheaf complex ─────────────────────────────
        sheaf_data   = self.sheaf.build(windows)
        sheaf_score  = sheaf_data['consistency_score']
        betti        = sheaf_data['betti']
        hallu_modes  = sheaf_data['hallucination_modes']
        hodge_gates  = sheaf_data['hodge_gate_counts']

        # ── FIX 2: Calibrated confidence ──────────────────────
        # Orbit failure: fraction of depths failing
        orbit_fail_frac = sum(
            1 for p in orbit_profiles if not p.is_admissible
        ) / max(len(orbit_profiles), 1)

        # Entropy failure: normalised spike fraction
        spike_frac = len(spike_positions) / max(len(entropy_seq), 1)

        # Sheaf failure: H¹ dimension relative to maximum
        max_H1 = (MAX_DEPTH - 1) * self.stalk_dim
        sheaf_fail_frac = min(betti[1] / max(max_H1, 1), 1.0)

        # Hodge failure: fraction of DEAD+UNSTABLE windows
        total_windows = sum(hodge_gates.values())
        hodge_fail_frac = (hodge_gates[0] + hodge_gates[1]) / max(total_windows, 1)

        # Weighted confidence (four evidence streams)
        confidence = (
            0.30 * orbit_fail_frac +   # Galois orbit evidence
            0.20 * spike_frac * 10 +   # entropy (scaled: rare spikes matter more)
            0.30 * sheaf_fail_frac +   # sheaf H¹ (most reliable)
            0.20 * hodge_fail_frac     # Hodge dead/unstable
        )
        confidence = float(np.clip(confidence, 0.0, 1.0))
        is_hallu   = confidence > 0.25   # lower threshold than v1

        # ── Taxonomy ──────────────────────────────────────────
        types   = [p.hallucination_type for p in orbit_profiles]
        worst   = max(types, key=lambda t: 0 if t=="CLEAN" else len(t))
        correction = self._correction(worst, betti, hallu_modes, hodge_gates)

        # ── Summary ──────────────────────────────────────────
        summary = self._summary(
            orbit_profiles, worst, confidence, is_hallu,
            sheaf_score, betti, spike_positions, frobenius_periods,
            hodge_gates, hallu_modes, postnikov, correction
        )

        return GSODv2Result(
            orbit_profiles      = orbit_profiles,
            entropy_sequence    = entropy_seq,
            entropy_spikes      = spike_positions,
            frobenius_periods   = frobenius_periods,
            sheaf_data          = sheaf_data,
            sheaf_consistency   = sheaf_score,
            sheaf_betti         = betti,
            hallucination_modes = hallu_modes,
            hodge_gates         = hodge_gates,
            postnikov_failures  = postnikov,
            is_hallucination    = is_hallu,
            confidence          = confidence,
            hallucination_types = types,
            correction          = correction,
            summary             = summary,
        )

    def _correction(self, worst_type, betti, modes, gates):
        base = {
            "CLEAN":              "None required.",
            "CONJUGATE_CLOSURE_FAILURE":
                "Add explicit closure constraint. Specify endpoint state.",
            "BINARY_ENTANGLEMENT_orbit2":
                "Add directional constraint. Specify which mirror interpretation is correct.",
            "BINARY_ENTANGLEMENT_orbit4":
                "Disambiguate the 4 related entities/concepts in the cycle.",
            "BINARY_ENTANGLEMENT_orbit8":
                "Decompose into 8 independent sub-claims. Verify each independently.",
            "COMPOUND_orbit2x2":
                "Structural rewrite. Add directional + closure constraints.",
        }.get(worst_type, "Manual inspection recommended.")

        # Augment with sheaf information
        if betti[1] > 0:
            dom_windows = [m['window_name'] for m in modes[:2]]
            base += f" Gluing failures at: {', '.join(dom_windows)}."

        if gates[0] > 0:
            base += f" {gates[0]} dead-zone windows detected (zero semantic flow)."

        return base

    def _summary(self, profiles, worst, conf, is_hallu,
                 sheaf_score, betti, spikes, periods,
                 gates, modes, postnikov, correction):
        lines = [
            "═" * 62,
            "  GSOD v2 — Galois-Stratified Obstruction Detector",
            "  + Semantic Sheaf Complex (Hodge/A∞/Floer)",
            "═" * 62,
            f"  Verdict:     {'⚠️  HALLUCINATION' if is_hallu else '✅  ADMISSIBLE'}",
            f"  Confidence:  {conf:.1%}",
            f"  Type:        {worst}",
            "─" * 62,
            "  Orbit profiles:",
        ]
        for p in profiles:
            st = "✅" if p.is_admissible else "⚠️ "
            k2 = p.orbit_sizes.get(2,1)
            k5 = p.orbit_sizes.get(5,1)
            k7 = p.orbit_sizes.get(7,1)
            lines.append(
                f"    X^{p.depth}: {st}  ({k2},{k5},{k7})  "
                f"ctx={p.effective_context_fraction:.0%}  "
                f"recur={p.recurrence_period}"
            )
        lines += [
            "─" * 62,
            f"  Sheaf complex:",
            f"    Betti:       β₀={betti[0]}  β₁={betti[1]}  β₂={betti[2]}",
            f"    Consistency: {sheaf_score:.3f}  "
            f"{'✅' if sheaf_score > 0.5 else '⚠️  GLUING FAILURES'}",
            f"    Hodge:       LIVE={gates[2]}  UNSTABLE={gates[1]}  DEAD={gates[0]}",
        ]
        if modes:
            lines.append("    H¹ modes:")
            for m in modes[:3]:
                lines.append(
                    f"      {m['window_name']}: λ={m['eigenvalue']:.3f}  "
                    f"{m['hodge_gate_label']}  "
                    f"{'singular' if m['is_singular'] else 'regular'}"
                )
        if postnikov:
            level_names = ["π₀","π₁","π₂","k-inv3","k-inv4(62)","k-inv5","k-inv6"]
            lines.append("  Postnikov failures:")
            for depth, lvl in sorted(postnikov.items())[:4]:
                lname = level_names[lvl] if lvl < len(level_names) else f"L{lvl}"
                lines.append(f"    X^{depth} → {lname}")
        lines += [
            "─" * 62,
            f"  Entropy spikes: {len(spikes)}  (MAD threshold)",
            f"  Frobenius periods: {periods if periods else 'none detected'}",
            "─" * 62,
            f"  Correction: {correction}",
            "═" * 62,
        ]
        return "\n".join(lines)
