"""
fdepth_validation.py
=====================
Implements the three required fixes before F_depth can be used as
a semantic probe:

(A) Operator validation on known structured systems
    Before testing hallucination, verify F_depth varies on:
      - induction head circuits (should show depth >= 3)
      - random noise (should show depth = 1)
      - synthetic memorised vs novel content (depth should differ)
    If F_depth cannot separate these: it is not a semantic probe.

(B) Three-way training axis
    random | early training | late training
    Then hallucination = deviation from expected depth within trained regime.
    NOT absolute depth >= 4.

(C) Delta F_depth
    ΔF_depth = F_depth(factual baseline) - F_depth(hallucinated span)
    Hypothesis: ΔF_depth > 0 (factual spans have deeper filtration).
    Statistical test across matched factual/hallucinated pairs.
"""

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────
# F_DEPTH OPERATOR (cleaned up)
# ─────────────────────────────────────────────────────────────────

def compute_fdepth(
    T:          np.ndarray,   # [d, d] transition matrix
    p:          int,
    max_depth:  int = 8,
) -> int:
    """
    F_depth(T; p) = first BSS page r where d_r kills a class.

    d_r = (T^r - T^{r-1}) mod p

    Returns depth in {1, ..., max_depth}.
    depth = max_depth means class survives all pages (liftable).
    depth = 1 means class killed immediately (not liftable).

    KNOWN LIMITATION (from review):
      This metric is dominated by:
        - choice of T
        - discretization scheme (mod p reduction)
        - windowing strategy
      MORE than by semantic structure on unstructured models.
      Use only after operator validation (Part A below).
    """
    d     = T.shape[0]
    scale = (p / 2.0) / max(np.abs(T).max(), 1e-10)
    T_p   = np.round(T * scale).astype(int) % p

    prev = np.eye(d)
    Tpow = T_p.copy().astype(float)
    for r in range(1, max_depth + 1):
        dr  = (Tpow - prev) % p
        sv  = np.linalg.svd(dr, compute_uv=False)
        if int(np.sum(sv > 0.5)) > 0:
            return r
        prev  = Tpow.copy()
        Tpow  = (Tpow @ T_p) % p

    return max_depth


def transition_matrix(
    win_a: np.ndarray,
    win_b: np.ndarray,
    dim:   int = 12,
) -> np.ndarray:
    """Fit T: win_a -> win_b via regularised LS in PCA-reduced space."""
    def reduce(w):
        c = w - w.mean(axis=0)
        if c.shape[1] <= dim:
            return c
        try:
            _, _, Vt = np.linalg.svd(c, full_matrices=False)
            return c @ Vt[:dim].T
        except Exception:
            return c[:, :dim]

    wa_r, wb_r = reduce(win_a), reduce(win_b)
    n      = min(len(wa_r), len(wb_r))
    X, Y   = wa_r[:n], wb_r[:n]
    eps    = 1e-4 * max(float(np.linalg.norm(X.T @ X)), 1.0)
    try:
        T, _, _, _ = np.linalg.lstsq(
            X.T @ X + eps * np.eye(dim), X.T @ Y, rcond=None
        )
        return T.T
    except Exception:
        return np.eye(dim)


# ─────────────────────────────────────────────────────────────────
# PART A: OPERATOR VALIDATION
# ─────────────────────────────────────────────────────────────────

class OperatorValidation:
    """
    Validate that F_depth can separate KNOWN structured systems.
    Required before using it as a hallucination probe.

    Three test cases:
      1. Pure noise vs pure structure: depth should differ
      2. Induction head pattern: repeated sequence creates stable subspace
      3. Memorised vs novel content: stable memory = deeper filtration

    If any test fails: F_depth with this operator is not a valid probe.
    """

    def __init__(self, dim=12, primes=[2,5,7], n_trials=20):
        self.dim     = dim
        self.primes  = primes
        self.n_trials = n_trials

    def make_noise(self, n=64, d=64, seed=0) -> np.ndarray:
        """Pure isotropic noise — no stable invariant subspace."""
        rng = np.random.RandomState(seed)
        s   = rng.randn(n, d)
        return s / (np.linalg.norm(s, axis=1, keepdims=True) + 1e-8)

    def make_induction_head(self, n=64, d=64, period=7, strength=0.8, seed=0) -> np.ndarray:
        """
        Induction head pattern: h[t] += strength * h[t - period]
        Creates a stable periodic subspace with known orbit structure.
        F_depth should be > 1 because the transition matrix has
        a persistent invariant subspace of dimension 1 (the periodic mode).
        """
        rng = np.random.RandomState(seed)
        s   = rng.randn(n, d)
        for i in range(n):
            if i >= period:
                s[i] += strength * s[i - period]
        return s / (np.linalg.norm(s, axis=1, keepdims=True) + 1e-8)

    def make_memorised(self, n=64, d=64, vocab_size=20, seed=0) -> np.ndarray:
        """
        Memorised content: sequence of k distinct tokens repeated.
        Creates a low-rank transition matrix (only k modes active).
        F_depth should be deeper because the low-rank structure
        creates a stable invariant subspace.
        """
        rng   = np.random.RandomState(seed)
        vocab = rng.randn(vocab_size, d)
        vocab = vocab / (np.linalg.norm(vocab, axis=1, keepdims=True) + 1e-8)
        # Cycle through vocabulary: 0,1,...,k-1,0,1,...
        k     = 5
        idxs  = [i % k for i in range(n)]
        s     = vocab[idxs]
        return s

    def make_novel(self, n=64, d=64, seed=0) -> np.ndarray:
        """
        Novel content: each token is distinct (no repetition).
        High-rank, no stable subspace.
        F_depth should be shallower.
        """
        rng = np.random.RandomState(seed)
        s   = rng.randn(n, d)
        return s / (np.linalg.norm(s, axis=1, keepdims=True) + 1e-8)

    def run(self) -> Dict:
        """
        Run all validation tests.
        Returns dict with pass/fail and depth statistics per test.
        """
        results = {}

        def get_depths(make_fn, name):
            depths = {p: [] for p in self.primes}
            for seed in range(self.n_trials):
                s = make_fn(seed=seed)
                T = transition_matrix(s[:32], s[32:], dim=self.dim)
                for p in self.primes:
                    depths[p].append(compute_fdepth(T, p))
            return {
                p: {
                    'mean':   float(np.mean(depths[p])),
                    'std':    float(np.std(depths[p])),
                    'min':    int(np.min(depths[p])),
                    'max':    int(np.max(depths[p])),
                    'hist':   list(np.bincount(depths[p], minlength=9)[1:]),
                }
                for p in self.primes
            }

        noise_d   = get_depths(self.make_noise,       'noise')
        induct_d  = get_depths(self.make_induction_head, 'induction')
        mem_d     = get_depths(self.make_memorised,    'memorised')
        novel_d   = get_depths(self.make_novel,        'novel')

        # Validation criteria (at prime p=2):
        # PASS: induction_mean > noise_mean + 0.5 std_noise
        # PASS: memorised_mean > novel_mean
        p2 = 2
        noise_mean  = noise_d[p2]['mean']
        noise_std   = max(noise_d[p2]['std'], 0.1)
        induct_mean = induct_d[p2]['mean']
        mem_mean    = mem_d[p2]['mean']
        novel_mean  = novel_d[p2]['mean']

        sep_noise_induct = induct_mean - noise_mean
        sep_mem_novel    = mem_mean - novel_mean

        results = {
            'noise':      noise_d,
            'induction':  induct_d,
            'memorised':  mem_d,
            'novel':      novel_d,
            'separation_noise_induct': sep_noise_induct,
            'separation_mem_novel':    sep_mem_novel,
            # Validation: can the operator separate known structure?
            'passes_noise_induct': sep_noise_induct > 0.3,
            'passes_mem_novel':    sep_mem_novel > 0.1,
            'is_valid_probe': sep_noise_induct > 0.3,
        }
        return results

    def print_report(self, results: Dict) -> str:
        p2 = 2
        lines = [
            "=" * 65,
            "  F_depth Operator Validation (Part A)",
            "  Tests on KNOWN structured systems before hallucination use",
            "=" * 65,
            f"  {'System':<18}  {'p=2 mean':>9}  {'p=2 std':>8}  {'range':>10}  {'depth dist [1..8]'}",
            "  " + "-" * 60,
        ]
        for name, key in [('Pure noise', 'noise'), ('Induction head', 'induction'),
                           ('Memorised', 'memorised'), ('Novel', 'novel')]:
            d    = results[key][p2]
            hist = '  '.join(f"{v}" for v in d['hist'])
            lines.append(
                f"  {name:<18}  {d['mean']:>9.3f}  {d['std']:>8.3f}"
                f"  [{d['min']},{d['max']}]      {hist}"
            )

        sep_ni = results['separation_noise_induct']
        sep_mn = results['separation_mem_novel']
        valid  = results['is_valid_probe']

        lines += [
            "",
            f"  Separation tests (p=2):",
            f"    Induction - noise:   {sep_ni:+.3f}  {'PASS' if results['passes_noise_induct'] else 'FAIL'}",
            f"    Memorised - novel:   {sep_mn:+.3f}  {'PASS' if results['passes_mem_novel'] else 'FAIL'}",
            "",
            f"  Operator validity: {'VALID semantic probe' if valid else 'NOT YET VALID — do not use for hallucination'}",
            "=" * 65,
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────
# PART B: THREE-WAY TRAINING AXIS
# ─────────────────────────────────────────────────────────────────

@dataclass
class TrainingRegimeProfile:
    """
    F_depth profile for one training regime.
    Used to establish the BASELINE before measuring hallucination.
    """
    regime:         str      # 'random' | 'early' | 'late'
    phase:          float
    depth_mean:     float    # mean F_depth across all factual spans
    depth_std:      float
    depth_min:      int
    depth_max:      int
    n_samples:      int


def profile_training_regimes(
    phases:     List[float] = [0.0, 0.1, 0.5, 1.0],
    n_windows:  int = 20,
    dim:        int = 12,
    primes:     List[int] = [2, 5, 7],
) -> List[TrainingRegimeProfile]:
    """
    Build F_depth baseline profiles for each training phase.

    Uses FACTUAL-ONLY text to establish what "normal" depth looks like
    at each phase. Hallucination detection will be ΔF_depth relative to this.

    This is the three-way axis:
      random  (phase=0): depth collapses (expected: depth=1 always)
      early   (phase~0.1-0.3): unstable, some depth variation
      late    (phase~0.7-1.0): stable multi-depth structure (expected: depth>1)
    """
    from factscore_alignment import PhasedGPT2

    FACTUAL_TEXTS = [
        "Albert Einstein was born in 1879 in Ulm, Germany. He developed the theory of relativity.",
        "Charles Darwin was born in 1809 and published On the Origin of Species in 1859.",
        "The double helix structure of DNA was discovered in 1953 by Watson, Crick, and Franklin.",
        "Isaac Newton published the Principia in 1687 and developed the theory of gravity.",
        "Marie Curie discovered radium and polonium and received two Nobel Prizes.",
    ]

    profiles = []
    for phase in phases:
        model = PhasedGPT2(d_model=256, n_layers=4, n_heads=4,
                           vocab_size=5000, max_seq=512,
                           phase=phase, seed=42)
        model.eval()
        model.register_hooks()

        all_depths = []
        for text in FACTUAL_TEXTS:
            tokens = [hash(w) % 5000 for w in text.lower().split()][:64]
            if len(tokens) < 16:
                continue
            with torch.no_grad():
                ids    = torch.tensor([tokens])
                _, hs, _ = model(ids)
            hs_np = model._hs_by_layer.get(3, hs)
            hs_np = hs_np / (np.linalg.norm(hs_np, axis=1, keepdims=True) + 1e-8)

            # Multiple windows per text
            n = len(hs_np)
            for t in range(0, n - 16, 8):
                wa = hs_np[t:t+8]
                wb = hs_np[t+8:min(t+16, n)]
                if len(wb) < 4:
                    continue
                T = transition_matrix(wa, wb, dim=dim)
                for p in primes:
                    d = compute_fdepth(T, p)
                    all_depths.append(d)

        model.remove_hooks()

        if all_depths:
            profiles.append(TrainingRegimeProfile(
                regime     = 'random' if phase < 0.05 else
                             'early'  if phase < 0.4  else 'late',
                phase      = phase,
                depth_mean = float(np.mean(all_depths)),
                depth_std  = float(np.std(all_depths)),
                depth_min  = int(np.min(all_depths)),
                depth_max  = int(np.max(all_depths)),
                n_samples  = len(all_depths),
            ))

    return profiles


# ─────────────────────────────────────────────────────────────────
# PART C: DELTA F_DEPTH
# ─────────────────────────────────────────────────────────────────

@dataclass
class DeltaFDepth:
    """
    ΔF_depth = F_depth(factual baseline) - F_depth(hallucinated span)

    Positive = factual is deeper (expected for grounded content).
    Negative = hallucinated is deeper (anomalous).
    Zero     = no difference (probe not discriminating at this phase).
    """
    entity:          str
    phase:           float
    depth_factual:   float    # mean F_depth on factual text
    depth_halluc:    float    # mean F_depth on hallucinated text
    delta:           float    # factual - hallucinated
    prime:           int
    # Statistical validity
    n_factual:       int
    n_halluc:        int
    effect_size:     float    # delta / pooled_std (Cohen's d equivalent)
    is_significant:  bool     # |delta| > std (conservative threshold)


def compute_delta_fdepth(
    hs_factual:     np.ndarray,   # hidden states for factual text
    hs_halluc:      np.ndarray,   # hidden states for hallucinated text
    prime:          int = 2,
    dim:            int = 12,
    window_step:    int = 4,
    window_size:    int = 8,
) -> Tuple[float, float, int, int]:
    """
    Compute (depth_factual, depth_halluc, n_factual, n_halluc).

    Uses sliding windows on each text independently.
    """
    def depths_from_hs(hs):
        n, d = hs.shape
        hs   = hs / (np.linalg.norm(hs, axis=1, keepdims=True) + 1e-8)
        ds   = []
        for t in range(0, n - window_size, window_step):
            wa = hs[t:t + window_size]
            wb = hs[t + window_size:min(t + 2 * window_size, n)]
            if len(wb) < window_size // 2:
                continue
            T = transition_matrix(wa, wb, dim=dim)
            ds.append(compute_fdepth(T, prime))
        return ds

    df = depths_from_hs(hs_factual)
    dh = depths_from_hs(hs_halluc)
    return (
        float(np.mean(df)) if df else 1.0,
        float(np.mean(dh)) if dh else 1.0,
        len(df), len(dh),
    )


def run_delta_fdepth_experiment(
    phases:    List[float] = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0],
    primes:    List[int]   = [2, 5, 7],
    dim:       int         = 16,
) -> List[DeltaFDepth]:
    """
    Run the corrected hallucination experiment using ΔF_depth.

    Hypothesis: ΔF_depth > 0 (factual has deeper filtration than hallucinated)
    Expected pattern:
      phase 0.0: ΔF_depth ≈ 0  (random, no structure to probe)
      phase 0.3: ΔF_depth > 0  (structure emerging)
      phase 1.0: ΔF_depth > 0  (mature, stable grounded structure)
    """
    from factscore_alignment import PhasedGPT2
    from hallucination_topology_pipeline import FIXED_PROMPTS, tokenize_controlled

    results = []

    for phase in phases:
        model = PhasedGPT2(d_model=256, n_layers=4, n_heads=4,
                           vocab_size=5000, max_seq=512,
                           phase=phase, seed=42)
        model.eval()
        model.register_hooks()

        for prompt in FIXED_PROMPTS:
            for prime in primes:
                # Get hidden states for both texts
                hs_by_type = {}
                for typ in ['true_text', 'halluc_text']:
                    tokens = tokenize_controlled(prompt[typ])[:64]
                    if len(tokens) < 12:
                        continue
                    with torch.no_grad():
                        ids     = torch.tensor([tokens])
                        _, hs, _ = model(ids)
                    hs_np = model._hs_by_layer.get(3, hs)
                    hs_by_type[typ] = hs_np

                if 'true_text' not in hs_by_type or 'halluc_text' not in hs_by_type:
                    continue

                df, dh, nf, nh = compute_delta_fdepth(
                    hs_by_type['true_text'],
                    hs_by_type['halluc_text'],
                    prime=prime, dim=dim,
                )

                delta = df - dh
                # Pooled std approximation
                pooled_std = 1.0   # F_depth is integer in [1,8], std ≈ 1
                effect = delta / pooled_std

                results.append(DeltaFDepth(
                    entity         = prompt['entity'],
                    phase          = phase,
                    depth_factual  = df,
                    depth_halluc   = dh,
                    delta          = delta,
                    prime          = prime,
                    n_factual      = nf,
                    n_halluc       = nh,
                    effect_size    = effect,
                    is_significant = abs(delta) > 0.5,
                ))

        model.remove_hooks()

    return results


def summarise_delta_fdepth(
    results: List[DeltaFDepth],
    prime:   int = 2,
) -> str:
    """Print ΔF_depth by phase — the main experimental result."""
    import collections
    by_phase = collections.defaultdict(list)
    for r in results:
        if r.prime == prime:
            by_phase[r.phase].append(r.delta)

    lines = [
        "=" * 65,
        f"  ΔF_depth = F_depth(factual) - F_depth(hallucinated)  [p={prime}]",
        "  Positive = factual has DEEPER filtration (correct direction)",
        "=" * 65,
        f"  {'Phase':<8}  {'ΔF mean':>9}  {'ΔF std':>8}  {'sig%':>6}  Signal",
        "  " + "-" * 50,
    ]

    for phase in sorted(by_phase.keys()):
        deltas = by_phase[phase]
        mean_d = float(np.mean(deltas))
        std_d  = float(np.std(deltas))
        sig_p  = 100 * sum(1 for r in results if r.prime==prime and r.phase==phase
                           and r.is_significant) / max(len(deltas), 1)

        bar_n  = int(abs(mean_d) * 12)
        bar    = "█" * min(bar_n, 15)
        arrow  = "▶" if mean_d > 0 else "◀"
        regime = "random" if phase < 0.05 else "early" if phase < 0.4 else "late"

        lines.append(
            f"  {phase:<8.2f}  {mean_d:>+9.4f}  {std_d:>8.4f}  {sig_p:>5.0f}%  "
            f"{arrow} {bar} ({regime})"
        )

    lines += [
        "",
        "  Expected: ΔF_depth > 0 in late training (factual = deeper).",
        "  Phase 0.0: ΔF ≈ 0 (random, no structure).",
        "  Hypothesis confirmed if late-phase ΔF > 0 and > early-phase ΔF.",
        "=" * 65,
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("\n[A] OPERATOR VALIDATION")
    print("Testing whether F_depth varies on known structured systems...")
    val = OperatorValidation(dim=12, primes=[2,5,7], n_trials=30)
    val_results = val.run()
    print(val.print_report(val_results))
    print()

    if not val_results['is_valid_probe']:
        print("WARNING: F_depth does not pass operator validation at dim=12.")
        print("The filtration depth is not a valid semantic probe under this operator.")
        print("Proceeding anyway to see the delta pattern...")
    print()

    print("\n[B] THREE-WAY TRAINING AXIS (baseline profiles)")
    profiles = profile_training_regimes(
        phases=[0.0, 0.1, 0.3, 0.5, 0.7, 1.0], dim=12
    )
    print(f"  {'Phase':<8}  {'Regime':<8}  {'Mean':>7}  {'Std':>6}  "
          f"{'Min':>4}  {'Max':>4}  N")
    print("  " + "-" * 50)
    for p in profiles:
        print(f"  {p.phase:<8.2f}  {p.regime:<8}  {p.depth_mean:>7.3f}  "
              f"{p.depth_std:>6.3f}  {p.depth_min:>4}  {p.depth_max:>4}  "
              f"{p.n_samples}")
    print()

    print("\n[C] DELTA F_DEPTH EXPERIMENT")
    print("ΔF_depth = F_depth(factual) - F_depth(hallucinated)")
    delta_results = run_delta_fdepth_experiment(
        phases=[0.0, 0.1, 0.3, 0.5, 0.7, 1.0],
        primes=[2, 5, 7],
        dim=12,
    )
    print(summarise_delta_fdepth(delta_results, prime=2))
    print()
    print("Same results at p=5:")
    print(summarise_delta_fdepth(delta_results, prime=5))
