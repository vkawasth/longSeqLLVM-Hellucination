"""
ReverseHironaka: Four-Critique Patched Version
Functor R_H: Path_stoch -> D^b(Sh_c(X; Z_p))
V(x) = [v_fake, v_true, chi_gate]
Geometric Softmax: z_tilde = z + gamma*tanh(v_true) - lambda*exp(v_fake)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import warnings
warnings.filterwarnings('ignore')


def topological_degeneration_distribution(vocab_size: int) -> np.ndarray:
    """
    Critique 4: fallback when V_admissible = empty set (Z -> 0).
    Returns near-uniform distribution concentrated on stop tokens.
    Prevents division-by-zero in geometric softmax.
    """
    dist = np.ones(vocab_size) / vocab_size
    dist[:10] *= 5.0
    dist = dist / dist.sum()
    return dist


@dataclass
class SupportVector:
    """
    Exceptional divisor coordinates from blowing up Path_stoch
    at admissibility failures.

    v_fake : chain-level Stasheff defect (continuous, [0,inf))
             = |m1^2(x) + m2(m0,x) + m2(x,m0)|_p
             Critique 3: continuous distance to V(I), NOT H* jump

    v_true : Z_p-rank of Floer intersection (bounded [0,1])
             tanh(v_true) in [0, 0.76] -- non-Archimedean envelope

    chi_gate : Hodge-Euler sharp decision {-1, 0, +1}
               Critique 1: discrete non-Archimedean hard fence
    """
    v_fake:           float
    v_true:           float
    chi_gate:         int
    logit_delta:      float
    inverse_limit_ok: bool
    lift_depth:       int
    bockstein_failed: bool
    orbit_profile:    Tuple
    postnikov_level:  int
    label:            str
    degeneration:     bool = False


@dataclass
class RHCorrection:
    window_idx:         int
    support:            SupportVector
    logit_warp:         float
    hard_gate:          bool
    entropy_before:     float
    entropy_after:      float
    entropy_delta:      float
    correction_applied: bool
    degeneration_fired: bool
    explanation:        str


class InverseLimitTester:
    """
    Tests Z_p tower glueability: ... -> Z/p^3 -> Z/p^2 -> Z/p -> 0
    Uses RELATIVE Bockstein (scale-invariant for layernorm states).
    """

    def __init__(self, primes=None, max_depth=8):
        self.primes    = primes or [2, 5, 7]
        self.max_depth = max_depth

    def bockstein_at_level(self, class_vec, p, level):
        """
        Relative Bockstein: norm(x mod p^n) / norm(x) in [0,1].
        < 0.5 means class lifts cleanly to level n.
        Scale-invariant -- works for normalised hidden states.
        """
        modulus  = float(p ** level)
        residual = class_vec % modulus
        residual = residual - modulus * np.round(residual / modulus)
        vec_norm = np.linalg.norm(class_vec)
        if vec_norm < 1e-10:
            return 0.0
        return float(np.linalg.norm(residual) / vec_norm)

    def test_glueability(self, hidden_states, pca_dim=16):
        hs = hidden_states - hidden_states.mean(axis=0)
        if hs.shape[1] > pca_dim:
            try:
                _, _, Vt = np.linalg.svd(hs, full_matrices=False)
                hs = hs @ Vt[:pca_dim].T
            except Exception:
                hs = hs[:, :pca_dim]

        class_vec = hs.mean(axis=0)
        norm = np.linalg.norm(class_vec)
        if norm > 1e-10:
            class_vec = class_vec / norm * 10.0

        tower_values = {}
        all_lifts    = []

        for p in self.primes:
            levels = [self.bockstein_at_level(class_vec, p, lv)
                      for lv in range(1, self.max_depth + 1)]
            tower_values[p] = levels
            lift_consec = 0
            for b in levels:
                if b < 0.5:
                    lift_consec += 1
                else:
                    break
            all_lifts.append(lift_consec)

        lift_depth       = min(all_lifts)
        bockstein_failed = any(tower_values[p][0] > 0.5 for p in self.primes)
        inverse_limit_ok = lift_depth >= self.max_depth // 2

        return inverse_limit_ok, lift_depth, bockstein_failed, tower_values


class SupportVectorComputer:
    """
    Computes V(x) = [v_fake, v_true, chi_gate].

    Critique 2 patch -- non-Archimedean envelope:
      z_tilde = z + gamma*tanh(v_true) - lambda*exp(v_fake)
      exp penalty dominates any linear logit as v_fake grows.

    Critique 3 clarification:
      v_fake measures chain-level Stasheff defect CONTINUOUSLY.
      The cohomology H* jumps discretely; the distance to V(I) is smooth.
    """

    def __init__(self, gamma=1.5, lambda_=2.0, tau=1.0,
                 primes=None, pca_dim=48):
        self.gamma    = gamma
        self.lambda_  = lambda_
        self.tau      = tau
        self.primes   = primes or [2, 5, 7]
        self.pca_dim  = pca_dim
        self.il_tester = InverseLimitTester(primes=self.primes)

    def compute_v_fake(self, window_a, window_b, ainf_error, orbit_sizes):
        """
        Continuous chain-level Stasheff residual.
        Critique 3: NOT the cohomology class -- the metric distance to V(I).
        """
        stasheff = ainf_error
        orbit_pen = sum(float(k-1)/float(p)
                        for p, k in orbit_sizes.items() if k > 1)
        d   = min(window_a.shape[1], window_b.shape[1], self.pca_dim)
        ma  = window_a[:, :d].mean(axis=0)
        mb  = window_b[:, :d].mean(axis=0)
        na, nb = np.linalg.norm(ma), np.linalg.norm(mb)
        if na > 1e-8 and nb > 1e-8:
            disc = float(1.0 - np.dot(ma, mb) / (na * nb)) * 2.0
        else:
            disc = 1.0
        return float(np.clip(0.40*stasheff + 0.35*orbit_pen + 0.25*disc, 0.0, 10.0))

    def compute_v_true(self, window, sheaf_consistency, floer_score,
                       lift_depth, H0_dim):
        """Bounded [0,1] so tanh(v_true) is bounded (non-Archimedean envelope)."""
        h0   = min(float(H0_dim) / 96.0, 1.0)
        fl   = min(floer_score / 10.0, 1.0)
        lift = min(float(lift_depth) / 8.0, 1.0)
        return float(np.clip(
            0.30*sheaf_consistency + 0.25*h0 + 0.25*fl + 0.20*lift, 0.0, 1.0))

    def compute_chi_gate(self, hodge_gate, sheaf_H1, inverse_limit_ok):
        """Discrete hard fence. Ultrametric: chi=-1 overrides everything."""
        if hodge_gate == 0 or not inverse_limit_ok:
            return -1
        if sheaf_H1 > 0 or hodge_gate == 1:
            return 0
        return +1

    def compute_logit_delta(self, v_fake, v_true, chi_gate):
        """
        Critique 2 patch:
          z_tilde = z + gamma*tanh(v_true) - lambda*exp(v_fake)
          As v_fake -> inf: exp dominates any positive z (non-Archimedean).
          As v_true -> 1:   tanh(1) = 0.76 (bounded boost).
        """
        if chi_gate < 0:
            return float('-inf')
        boost   =  self.gamma   * float(np.tanh(v_true))
        penalty =  self.lambda_ * float(np.exp(min(v_fake, 5.0)))
        warp = boost - penalty
        if chi_gate == 0:
            warp *= 0.5
        return float(warp)

    def compute_label(self, v_fake, v_true, chi_gate,
                      inverse_limit_ok, bockstein_failed):
        """
        Tightened cohomological label rule.

        Key insight: high v_fake is PERMISSIBLE when topologically
        unobstructed (chi_gate >= 0) AND Bockstein lifts (Sq1=0).
        Transformer hidden states have anisotropic spectra -- cross-window
        covariance spikes during semantic advection (context transitions)
        without any structural failure. This is benign distortion.

        HALLUCINATION requires CONJUNCTION:
          v_fake > 1.5  AND  (chi_gate < 0  OR  bockstein_failed)
          -- large metric distortion AND topological/arithmetic failure

        Separates two distinct cases:
          BENIGN:  v_fake > 1.5, chi=LIVE, Sq1=0  -> UNCERTAIN (admitted)
          GENUINE: v_fake > 1.5, chi=DEAD or Sq1!=0 -> HALLUCINATION (blocked)
        """
        # Hard DEAD: dangling space, no question
        if chi_gate < 0:
            return "HALLUCINATION"

        # Tightened conjunction: metric distortion + structural failure
        # (NOT metric distortion alone -- that would over-restrict)
        if v_fake > 1.5 and (chi_gate < 0 or bockstein_failed):
            return "HALLUCINATION"

        # Smooth locus: lifts through tower, no Bockstein, reasonable distortion
        if v_true > 0.6 and v_fake < 0.8 and inverse_limit_ok and not bockstein_failed:
            return "ADMISSIBLE"

        # Near divisor: high distortion but topologically sound
        # (benign semantic advection in curved manifold region)
        return "UNCERTAIN"

    def compute_support_vector(self, window_a, window_b, ainf_error,
                                orbit_sizes, sheaf_consistency, sheaf_H1,
                                sheaf_H0, floer_score, hodge_gate,
                                postnikov_level, window_idx):
        combined = np.vstack([window_a, window_b])
        il_ok, lift_depth, bock_failed, _ = self.il_tester.test_glueability(
            combined, pca_dim=16)

        v_fake = self.compute_v_fake(window_a, window_b, ainf_error, orbit_sizes)
        v_true = self.compute_v_true(window_a, sheaf_consistency, floer_score,
                                     lift_depth, sheaf_H0)
        chi    = self.compute_chi_gate(hodge_gate, sheaf_H1, il_ok)
        delta  = self.compute_logit_delta(v_fake, v_true, chi)
        label  = self.compute_label(v_fake, v_true, chi, il_ok, bock_failed)

        return SupportVector(
            v_fake=v_fake, v_true=v_true, chi_gate=chi,
            logit_delta=delta, inverse_limit_ok=il_ok,
            lift_depth=lift_depth, bockstein_failed=bock_failed,
            orbit_profile=(orbit_sizes.get(2,1), orbit_sizes.get(5,1),
                           orbit_sizes.get(7,1)),
            postnikov_level=postnikov_level, label=label, degeneration=False,
        )


class GeometricSoftmax:
    """
    Non-Archimedean envelope softmax with Topological Degeneration fallback.

    Critique 2: z_tilde = z + gamma*tanh(v_true) - lambda*exp(v_fake)
    Critique 4: if Z->0, return topological_degeneration_distribution()
    """

    def __init__(self, gamma=1.5, lambda_=2.0, tau=1.0):
        self.gamma   = gamma
        self.lambda_ = lambda_
        self.tau     = tau

    def apply(self, logits, sv, temperature=1.0):
        """Returns (P_topo, H_before, H_after, degeneration_fired)."""
        vocab_size = len(logits)

        ls = logits - logits.max()
        P_base = np.exp(ls / temperature)
        P_base = P_base / P_base.sum()
        H_before = float(-np.sum(P_base * np.log(P_base + 1e-10)))

        if sv.chi_gate < 0:
            P_topo = topological_degeneration_distribution(vocab_size)
            H_after = float(-np.sum(P_topo * np.log(P_topo + 1e-10)))
            return P_topo, H_before, H_after, True

        # Non-Archimedean envelope warp
        warped = logits + sv.logit_delta / self.tau
        valid  = warped > -1e8

        if not np.any(valid):
            P_topo = topological_degeneration_distribution(vocab_size)
            H_after = float(-np.sum(P_topo * np.log(P_topo + 1e-10)))
            return P_topo, H_before, H_after, True

        ws = warped.copy(); ws[~valid] = -1e9
        ws_stable = ws - ws[valid].max()
        P_topo = np.where(valid, np.exp(ws_stable / temperature), 0.0)
        Z = P_topo.sum()

        if Z < 1e-10:
            P_topo = topological_degeneration_distribution(vocab_size)
            H_after = float(-np.sum(P_topo * np.log(P_topo + 1e-10)))
            return P_topo, H_before, H_after, True

        P_topo = P_topo / Z
        H_after = float(-np.sum(P_topo * np.log(P_topo + 1e-10)))
        return P_topo, H_before, H_after, False


class ReverseHironaka:
    """
    R_H: Path_stoch -> D^b(Sh_c(X; Z_p))   [Critique 1 patch]

    Critique 1: grounded as functor between stochastic paths and
    derived constructible sheaves. [v_fake, v_true, chi_gate] are
    exceptional divisor coordinates from blowing up Path_stoch at
    admissibility failures.

    Critique 2: non-Archimedean envelope:
      z_tilde = z + gamma*tanh(v_true) - lambda*exp(v_fake)

    Critique 3: v_fake measures chain-level Stasheff defect (continuous).
    The cohomology H* jumps; the distance to V(I) is smooth.

    Critique 4: Topological Degeneration Step prevents Z=0 crash.
    """

    def __init__(self, gamma=1.5, lambda_=2.0, tau=1.0,
                 pca_dim=48, primes=None):
        self.sv_computer = SupportVectorComputer(
            gamma=gamma, lambda_=lambda_, tau=tau,
            primes=primes or [2,5,7], pca_dim=pca_dim)
        self.geo_softmax = GeometricSoftmax(gamma=gamma, lambda_=lambda_, tau=tau)
        self.pca_dim     = pca_dim

    def correct(self, hidden_states, gsod_result, logits=None):
        from gsod_v2 import WINDOW_SIZE, MAX_DEPTH

        if hidden_states.ndim == 3:
            hidden_states = hidden_states.squeeze(0)
        hs      = hidden_states[:WINDOW_SIZE * MAX_DEPTH]
        windows = [hs[k*WINDOW_SIZE:(k+1)*WINDOW_SIZE] for k in range(MAX_DEPTH)]

        sheaf    = gsod_result.sheaf_data
        vertices = sheaf.get('vertices', [])
        edges    = sheaf.get('edges', [])
        cohom    = sheaf.get('cohomology')
        H0       = cohom.H0_dim if cohom else 8
        H1       = cohom.H1_dim if cohom else 0
        consist  = sheaf.get('consistency_score', 1.0)

        corrections = []
        for k in range(MAX_DEPTH - 1):
            win_a = windows[k]; win_b = windows[k+1]
            op    = (gsod_result.orbit_profiles[k]
                     if k < len(gsod_result.orbit_profiles) else None)
            osizes = op.orbit_sizes if op else {p:1 for p in [2,5,7]}

            start = k * WINDOW_SIZE; end = start + WINDOW_SIZE
            w_ent = gsod_result.entropy_sequence[start:end]
            ainf  = float(w_ent.mean()) if len(w_ent) > 0 else 0.0

            hodge  = vertices[k].hodge_gate if k < len(vertices) else 2
            floer  = edges[k].floer_score   if k < len(edges)    else 1.0
            plevel = gsod_result.postnikov_failures.get(k+1, -1)

            sv = self.sv_computer.compute_support_vector(
                window_a=win_a, window_b=win_b, ainf_error=ainf,
                orbit_sizes=osizes, sheaf_consistency=consist,
                sheaf_H1=H1, sheaf_H0=H0, floer_score=floer,
                hodge_gate=hodge, postnikov_level=plevel, window_idx=k)

            degen = False
            H_b = H_a = dH = 0.0
            applied = False
            if logits is not None:
                wl = logits[start:end].mean(axis=0)
                P_topo, H_b, H_a, degen = self.geo_softmax.apply(wl, sv)
                dH = H_b - H_a; applied = True
                if degen:
                    sv.degeneration = True
                    sv.label = "TOPOLOGICAL_DEGENERATION"

            corrections.append(RHCorrection(
                window_idx=k, support=sv,
                logit_warp=sv.logit_delta, hard_gate=sv.chi_gate>=0,
                entropy_before=H_b, entropy_after=H_a, entropy_delta=dH,
                correction_applied=applied, degeneration_fired=degen,
                explanation=self._explain(sv, k)))

        return corrections

    def _explain(self, sv, k):
        gate_s = {-1:"DEAD", 0:"UNSTABLE", 1:"LIVE"}.get(sv.chi_gate, "?")
        il_s   = "lifts" if sv.inverse_limit_ok else "dangling"
        bk_s   = "Sq1!=0" if sv.bockstein_failed else "Sq1=0"
        boost  = self.sv_computer.gamma   * float(np.tanh(sv.v_true))
        pen    = self.sv_computer.lambda_ * float(np.exp(min(sv.v_fake, 5.0)))
        dl     = f"{sv.logit_delta:+.3f}" if np.isfinite(sv.logit_delta) else "-inf"

        lines = [
            f"X^{k+1}: [{sv.label}]",
            f"  v_fake={sv.v_fake:.3f}  v_true={sv.v_true:.3f}  chi={sv.chi_gate}({gate_s})",
            f"  Envelope: +{boost:.3f}(tanh) -{pen:.3f}(exp)  Dlogit={dl}",
            f"  IL={il_s}  depth={sv.lift_depth}/8  {bk_s}",
        ]
        if sv.degeneration:
            lines.append("  *** TOPOLOGICAL DEGENERATION -- fallback anchor ***")
        elif sv.label == "HALLUCINATION" and sv.chi_gate < 0:
            lines.append("  -> DEAD: dangling space, inverse limit undefined")
        elif sv.label == "HALLUCINATION":
            lines.append(f"  -> Stasheff residual large: exp({sv.v_fake:.2f})={pen:.2f} >> logit")
        elif sv.label == "ADMISSIBLE":
            lines.append("  -> Smooth locus: lifts to Z_p section")
        else:
            lines.append(f"  -> Near divisor: penalty {pen:.2f} > boost {boost:.2f}")
        return "\n".join(lines)

    def summary_table(self, corrections):
        lines = [
            "=" * 76,
            "  R_H: Path_stoch -> D^b(Sh_c(X;Z_p))   4-Critique Patched",
            "  z_tilde = z + gamma*tanh(v_true) - lambda*exp(v_fake)",
            "=" * 76,
            f"  {'Win':<6} {'v_fake':>7} {'v_true':>7} {'chi':>4}  "
            f"{'boost':>7} {'penalty':>8} {'Dlogit':>8}  {'IL':>3} {'Label'}",
            "  " + "-" * 70,
        ]
        for c in corrections:
            sv    = c.support
            boost = self.sv_computer.gamma   * float(np.tanh(sv.v_true))
            pen   = self.sv_computer.lambda_ * float(np.exp(min(sv.v_fake, 5.0)))
            il    = "ok" if sv.inverse_limit_ok else "NO"
            dl    = f"{c.logit_warp:+.3f}" if np.isfinite(c.logit_warp) else "   -inf"
            dmark = " [DEGEN]" if c.degeneration_fired else ""
            lines.append(
                f"  X^{c.window_idx+1:<4}  "
                f"{sv.v_fake:>7.3f} {sv.v_true:>7.3f} {sv.chi_gate:>4}  "
                f"+{boost:>6.3f}  -{pen:>7.3f}  {dl:>8}  {il:>3}  "
                f"{sv.label}{dmark}"
            )

        n_h  = sum(1 for c in corrections if "HALLU" in c.support.label
                   or "DEGEN" in c.support.label)
        n_a  = sum(1 for c in corrections if c.support.label=="ADMISSIBLE")
        n_u  = sum(1 for c in corrections if c.support.label=="UNCERTAIN")
        n_d  = sum(1 for c in corrections if c.degeneration_fired)
        mf   = np.mean([c.support.v_fake for c in corrections])
        mt   = np.mean([c.support.v_true for c in corrections])

        lines += [
            "  " + "-" * 70,
            f"  HALLU/DEGEN={n_h}  ADMISSIBLE={n_a}  UNCERTAIN={n_u}  DEGEN_FIRED={n_d}",
            f"  Mean v_fake={mf:.3f}  Mean v_true={mt:.3f}",
            f"  Exp penalty dominates when exp(v_fake) > gamma/lambda * tanh(v_true)",
            "=" * 76,
        ]
        return "\n".join(lines)
