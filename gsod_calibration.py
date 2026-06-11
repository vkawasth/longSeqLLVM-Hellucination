"""
gsod_calibration.py
===================
Observability-correct framework: replaces absolute HPI values with
validity-domain maps Omega_H1(L), Omega_cob(L), Omega_E(L).

Core correction (Document 11):
  HPI_E = 0.0 does NOT mean zero dynamical complexity.
  It means the estimator is outside its validity regime.

Three probes require different minimal structures:
  HPI_H1:         n >= 6 tokens, spatial structure
  HPI_coboundary: n >= 16 tokens, two aligned windows
  HPI_E:          n >= 64 tokens, iterated dynamical system

Output: regime classifier + validity-conditional probe values.
  UNDERDETERMINED: no probes valid
  GEOMETRIC:       H_1 only
  ALIGNMENT:       H_1 + coboundary
  DYNAMICAL:       all three
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from ripser import ripser as _ripser
    RIPSER_OK = True
except ImportError:
    RIPSER_OK = False

VALIDITY_THRESHOLDS = {'H1': 6, 'coboundary': 16, 'entropy': 64}


@dataclass
class ObservabilityReport:
    n_tokens:          int
    pca_dim_effective: int
    h1_valid:          bool
    cob_valid:         bool
    entropy_valid:     bool
    h1_value:          Optional[float]   # None = UNDEFINED (not zero)
    cob_value:         Optional[float]
    entropy_value:     Optional[float]
    regime:            str
    regime_index:      int
    hallucination_score:       Optional[float]
    hallucination_regime_note: str


def classify_regime(n: int) -> Tuple[str, int]:
    if n < VALIDITY_THRESHOLDS['H1']:
        return 'UNDERDETERMINED', 0
    elif n < VALIDITY_THRESHOLDS['coboundary']:
        return 'GEOMETRIC', 1
    elif n < VALIDITY_THRESHOLDS['entropy']:
        return 'ALIGNMENT', 2
    else:
        return 'DYNAMICAL', 3


def compute_with_validity(
    hidden_states: np.ndarray,
    pca_dim:       int = 12,
) -> ObservabilityReport:
    """
    Compute probes and explicitly mark undefined (not zero) when invalid.
    """
    n     = len(hidden_states)
    d_eff = min(hidden_states.shape[1], pca_dim)
    regime, ridx = classify_regime(n)

    h1_valid  = n >= VALIDITY_THRESHOLDS['H1']
    cob_valid = n >= VALIDITY_THRESHOLDS['coboundary']
    ent_valid = n >= VALIDITY_THRESHOLDS['entropy']

    # H_1 probe
    h1_val = None
    if h1_valid and RIPSER_OK:
        hs = hidden_states - hidden_states.mean(axis=0)
        try:
            _, _, Vt = np.linalg.svd(hs, full_matrices=False)
            hs_r = hs @ Vt[:min(d_eff, 8)].T
        except Exception:
            hs_r = hs[:, :min(d_eff, 8)]
        hs_r = hs_r / (np.linalg.norm(hs_r, axis=1, keepdims=True) + 1e-8)
        try:
            dgms = _ripser(hs_r, maxdim=1)['dgms']
            h1   = dgms[1] if len(dgms) > 1 else np.zeros((0, 2))
            h1c  = h1[np.isfinite(h1[:, 1])] if len(h1) > 0 else h1
            h1_val = float(np.sum(h1c[:, 1] - h1c[:, 0])) if len(h1c) > 0 else 0.0
        except Exception:
            h1_valid = False

    # Coboundary probe
    cob_val = None
    if cob_valid:
        mid = n // 2
        wa, wb = hidden_states[:mid], hidden_states[mid:]
        d = min(wa.shape[1], pca_dim)
        def cov_sqrt(w):
            c = w - w.mean(axis=0)
            try:
                _, _, Vt = np.linalg.svd(c, full_matrices=False)
                cr = c @ Vt[:d].T
            except Exception:
                cr = c[:, :d]
            d_ = cr.shape[1]
            cov = (cr.T @ cr) / max(len(cr)-1, 1) + 1e-4 * np.eye(d_)
            try:
                return np.linalg.cholesky(cov)
            except np.linalg.LinAlgError:
                return np.diag(np.sqrt(np.maximum(np.diag(cov), 1e-8)))
        Ss, St = cov_sqrt(wa), cov_sqrt(wb)
        d_ = Ss.shape[0]
        Xr = (wa - wa.mean(axis=0))[:, :d_]
        Yr = (wb - wb.mean(axis=0))[:, :d_]
        nm = min(len(Xr), len(Yr))
        X, Y = Xr[:nm], Yr[:nm]
        eps  = 1e-4 * max(float(np.linalg.norm(X.T @ X)), 1.0)
        try:
            T, _, _, _ = np.linalg.lstsq(X.T @ X + eps * np.eye(d_), X.T @ Y, rcond=None)
            T = T.T
            cobdry  = T @ Ss - St
            cob_val = float(np.linalg.norm(cobdry, 'fro') / (np.linalg.norm(St, 'fro') + 1e-8))
        except Exception:
            pass

    # Entropy probe (spectral concentration)
    ent_val = None
    if ent_valid:
        win_size = 8
        T_list   = []
        for t in range(0, n - 2 * win_size, win_size // 2):
            wa = hidden_states[t:t + win_size]
            wb = hidden_states[t + win_size:t + 2 * win_size]
            wa = wa / (np.linalg.norm(wa, axis=1, keepdims=True) + 1e-8)
            wb = wb / (np.linalg.norm(wb, axis=1, keepdims=True) + 1e-8)
            d  = min(wa.shape[1], pca_dim)
            Xr = (wa - wa.mean(axis=0))[:, :d]
            Yr = (wb - wb.mean(axis=0))[:, :d]
            nm = min(len(Xr), len(Yr))
            X, Y = Xr[:nm], Yr[:nm]
            eps  = 1e-4 * max(float(np.linalg.norm(X.T @ X)), 1.0)
            try:
                T, _, _, _ = np.linalg.lstsq(X.T @ X + eps * np.eye(d), X.T @ Y, rcond=None)
                T_list.append(T.T)
            except Exception:
                pass
        if len(T_list) >= 4:
            T_mean = np.mean(T_list, axis=0)
            ev     = np.abs(np.linalg.eigvals(T_mean))
            s1, s2 = ev.sum() + 1e-10, (ev**2).sum() + 1e-10
            ent_val = float(s2 / (s1**2))  # spectral concentration

    # Regime-conditional hallucination score
    hall_score = None
    if ridx >= 1 and h1_val is not None:
        hall_score = float(h1_val)
    if ridx >= 2 and cob_val is not None:
        hall_score = float(cob_val)
    if ridx >= 3 and ent_val is not None and cob_val is not None:
        sc_excess  = max(0, ent_val - 1.0 / max(pca_dim, 1))
        hall_score = float(cob_val * 0.6 + (1.0 - min(sc_excess * 10, 1.0)) * 0.4)

    note = f"[{regime}] "
    if ridx == 0:
        note += "no probes valid — UNDEFINED"
    elif ridx == 1:
        note += f"H1={h1_val if h1_val is not None else 'undef'}; cob=UNDEFINED; SC=UNDEFINED"
    elif ridx == 2:
        note += (f'H1={h1_val:.3f}' if h1_val is not None else 'H1=undef') + '; ' + (f'cob={cob_val:.3f}' if cob_val is not None else 'cob=undef') + '; SC=UNDEFINED'
    else:
        note += (f'H1={h1_val:.3f}' if h1_val is not None else 'H1=undef') + '; ' + (f'cob={cob_val:.3f}' if cob_val is not None else 'cob=undef') + '; ' + (f'SC={ent_val:.4f}' if ent_val is not None else 'SC=undef')

    return ObservabilityReport(
        n_tokens=n, pca_dim_effective=d_eff,
        h1_valid=h1_valid, cob_valid=cob_valid, entropy_valid=ent_valid,
        h1_value=h1_val, cob_value=cob_val, entropy_value=ent_val,
        regime=regime, regime_index=ridx,
        hallucination_score=hall_score,
        hallucination_regime_note=note,
    )


def build_phase_diagram(
    hidden_states: np.ndarray,
    label:         str,
    lengths:       List[int] = [6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 512, 1024],
    pca_dim:       int = 12,
) -> Dict:
    """Phase diagram: probe validity across increasing L."""
    reports = []
    n_total = len(hidden_states)
    for L in lengths:
        if L > n_total:
            break
        reports.append(compute_with_validity(hidden_states[:L], pca_dim=pca_dim))
    return {'label': label, 'lengths': lengths, 'reports': reports}


def print_phase_diagram(d: Dict) -> str:
    lines = [
        f"  Phase diagram: {d['label']}",
        f"  {'L':>5}  {'Regime':<14}  {'H1':>10}  {'Cob':>10}  {'SC':>10}  HallScore",
        "  " + "-" * 68,
    ]
    for r in d['reports']:
        h1s  = f"{r.h1_value:.4f}"     if r.h1_value     is not None else "   [undef]"
        cobs = f"{r.cob_value:.4f}"    if r.cob_value     is not None else "   [undef]"
        ens  = f"{r.entropy_value:.5f}" if r.entropy_value is not None else "    [undef]"
        hs   = f"{r.hallucination_score:.3f}" if r.hallucination_score is not None else " [no signal]"
        lines.append(
            f"  {r.n_tokens:>5}  {r.regime:<14}  {h1s:>10}  {cobs:>10}  {ens:>10}  {hs}"
        )
    return "\n".join(lines)
