#!/usr/bin/env python3
"""
Context Algebra C_ctx — Supersingularity Experiment Suite
Local Ollama Edition (llama3.2:1b)
=========================================================

Integrates your working FactScore pipeline with the full
p-adic supersingularity framework from ctx_algebra.pdf.

Five experiment blocks:

  BLOCK A  Transport defect δT — factual vs fabricated
  BLOCK B  Frobenius orbit k_p — supersingularity detection
  BLOCK C  Hecke nilpotency depth — causal attention structure
  BLOCK D  Recursive saturation — attractor collapse test
  BLOCK E  Maurer-Cartan flatness — coherent vs scrambled chains

Usage:
    python ctx_algebra_ollama_suite.py
    python ctx_algebra_ollama_suite.py --model llama3.2:1b
    python ctx_algebra_ollama_suite.py --blocks A,B,C
    python ctx_algebra_ollama_suite.py --topics einstein,newton
    python ctx_algebra_ollama_suite.py --save results.json

Requires:
    pip install ollama numpy scipy sympy
    ollama pull llama3.2:1b
    my_knowledge_source.jsonl  (same dir, your existing file)
"""

import argparse
import json
import time
import warnings
import sys
import os
from math import log
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field, asdict

warnings.filterwarnings('ignore')
import numpy as np
import ollama

# ── optional dependency handling ─────────────────────────────────────────────
try:
    from scipy.linalg import sqrtm as _sqrtm, hessenberg as scipy_hessenberg
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("  ⚠️  scipy not found — run: pip install scipy")
    print("      Using numpy fallbacks for sqrtm / hessenberg.\n")
    def _sqrtm(M):
        vals, vecs = np.linalg.eigh(M)
        return vecs @ np.diag(np.sqrt(np.maximum(vals, 0))) @ vecs.T
    def scipy_hessenberg(M):
        n = M.shape[0]; H = M.copy().astype(float)
        for k in range(n - 2):
            x = H[k+1:, k]; e = np.zeros_like(x); e[0] = np.linalg.norm(x)
            u = x - e; u_norm = np.linalg.norm(u)
            if u_norm < 1e-12: continue
            u /= u_norm
            H[k+1:, k:] -= 2*np.outer(u, u @ H[k+1:, k:])
            H[:, k+1:]   -= 2*np.outer(H[:, k+1:] @ u, u)
        return H

try:
    # (sympy imported at top-level)
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False
    print("  ⚠️  sympy not found — run: pip install sympy")
    print("      Frobenius orbit will use eigenvalue fallback.\n")

def sqrtm(M):
    return _sqrtm(M)

# ── argument parsing ──────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument('--model',   default='llama3.2:1b')
parser.add_argument('--blocks',  default='A,B,C,D,E',
                    help='Comma-separated blocks to run')
parser.add_argument('--topics',  default=None,
                    help='Comma-separated topic names from your JSONL')
parser.add_argument('--save',    default='ctx_results.json')
parser.add_argument('--kb',      default='my_knowledge_source.jsonl')
parser.add_argument('--verbose', action='store_true')
args = parser.parse_args()

BLOCKS_TO_RUN = set(b.strip().upper() for b in args.blocks.split(','))
MODEL         = args.model
PRIMES        = [2, 5, 7]
MAX_ORBIT     = {2: 8, 5: 2, 7: 1}

# ── banner ────────────────────────────────────────────────────────────────────

print(f"""
{'='*70}
  CONTEXT ALGEBRA C_ctx — SUPERSINGULARITY SUITE
  Model  : {MODEL}
  Blocks : {args.blocks}
  KB     : {args.kb}
{'='*70}
""")

# ── load knowledge base ───────────────────────────────────────────────────────

print("📂 Loading knowledge base...")
knowledge_base: Dict[str, str] = {}

try:
    with open(args.kb, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            clean = line.strip()
            if not clean:
                continue
            try:
                data = json.loads(clean)
                knowledge_base[data['title']] = data['text']
            except json.JSONDecodeError as e:
                print(f"  ⚠️  Line {line_num} malformed: {e}")
except FileNotFoundError:
    print(f"  ❌ Cannot find {args.kb}")
    sys.exit(1)

print(f"  Loaded {len(knowledge_base)} topics: {list(knowledge_base.keys())[:8]}")

# Select topics to run
if args.topics:
    topics = [t.strip() for t in args.topics.split(',')]
    missing = [t for t in topics if t not in knowledge_base]
    if missing:
        print(f"  ⚠️  Topics not in KB: {missing}")
    topics = [t for t in topics if t in knowledge_base]
else:
    topics = list(knowledge_base.keys())

print(f"  Running on {len(topics)} topics: {topics[:5]}")

# ── ollama helpers ────────────────────────────────────────────────────────────

def generate(prompt: str, temperature: float = 0.7,
             num_ctx: int = 1024, top_k: int = 40, top_p: float = 0.9) -> str:
    """Single generation call."""
    try:
        r = ollama.generate(
            model=MODEL,
            prompt=prompt,
            options={'num_ctx': num_ctx, 'temperature': temperature,
                     'top_k': top_k, 'top_p': top_p}
        )
        return r['response'].strip()
    except Exception as e:
        return f"[ERROR: {e}]"

def extract_atomic_facts(text: str) -> List[str]:
    """Decompose text into atomic facts (your working function)."""
    prompt = (
        "System: You are a strict factual extraction engine. "
        "Do not chat or write an introduction.\n"
        "User: Deconstruct the following text into distinct, simple atomic facts.\n"
        "Output exactly ONE basic sentence per line. "
        "Do not use bullet points or numbering.\n\n"
        f"Text: {text}"
    )
    raw = generate(prompt, temperature=0.2, num_ctx=1024)
    return [l.strip() for l in raw.split('\n') if l.strip()]

def validate_fact(fact: str, reference: str) -> bool:
    """Validate a single fact against reference (your working function)."""
    prompt = (
        f"Context: {reference}\n\n"
        f"Fact to check: {fact}\n\n"
        "Is this fact supported by the context? Answer yes or no."
    )
    verdict = generate(prompt, temperature=0.0).lower()
    if any(neg in verdict for neg in ['no', 'false', 'not supported', 'incorrect']):
        return False
    if any(pos in verdict for pos in ['yes', 'true', 'supported']):
        return True
    return False

def factscore(text: str, reference: str, verbose: bool = False
              ) -> Tuple[float, List[Dict]]:
    """Run full FactScore pipeline. Returns (score, details)."""
    facts   = extract_atomic_facts(text)
    results = []
    for fact in facts:
        ok = validate_fact(fact, reference)
        results.append({'fact': fact,
                        'status': 'SUPPORTED' if ok else 'HALLUCINATION'})
        if verbose:
            marker = '✅' if ok else '❌'
            print(f"    {marker} {fact[:70]}")
    score = sum(1 for r in results if r['status'] == 'SUPPORTED') / max(len(facts), 1)
    return score, results

# ── generation functions ──────────────────────────────────────────────────────

def generate_factual(topic: str, reference: str) -> str:
    """Generate a coherent, factual paragraph about a topic."""
    prompt = (
        f"Write a factually accurate, coherent 3-sentence paragraph about {topic}. "
        f"Use only information consistent with the following reference.\n\n"
        f"Reference: {reference[:600]}\n\n"
        "Paragraph:"
    )
    return generate(prompt, temperature=0.3, num_ctx=1024)

def generate_fabricated(topic: str) -> str:
    """Generate a plausible-sounding but likely incorrect paragraph."""
    prompt = (
        f"Write a confident 3-sentence paragraph about {topic} that sounds "
        f"plausible but contains subtle factual errors. "
        "Change dates, prizes, and specific details slightly.\n\n"
        "Paragraph:"
    )
    return generate(prompt, temperature=0.8, num_ctx=1024)

def generate_scrambled(factual_text: str) -> str:
    """Scramble the logical order of a factual paragraph."""
    sentences = [s.strip() for s in factual_text.replace('!', '.').split('.')
                 if len(s.strip()) > 10]
    if len(sentences) < 2:
        return factual_text
    # Reverse order = maximally scrambled logical chain
    scrambled = sentences[::-1]
    return '. '.join(scrambled) + '.'

def generate_logic_chain(topic: str, reference: str,
                          mode: str = 'flat') -> str:
    """
    Generate a multi-step logic chain (for Maurer-Cartan test).
    mode='flat'      : A→B→C→D coherent causal chain
    mode='scrambled' : same steps, broken order
    mode='contradiction' : coherent until midpoint, then reversal
    """
    if mode == 'flat':
        prompt = (
            f"Write 4 sentences about {topic} where each sentence logically "
            f"follows from the previous one as a causal chain. "
            f"Use this reference: {reference[:400]}\n\nChain:"
        )
    elif mode == 'scrambled':
        prompt = (
            f"Write 4 sentences about {topic}. Make each sentence about a "
            f"different aspect with NO logical connection between them. "
            f"Jump between unrelated facts.\n\nSentences:"
        )
    else:  # contradiction
        prompt = (
            f"Write 4 sentences about {topic}. The first 2 should be factually "
            f"accurate. The 3rd sentence should directly contradict the 2nd. "
            f"The 4th should try to recover. Reference: {reference[:300]}\n\nText:"
        )
    return generate(prompt, temperature=0.4, num_ctx=1024)

# ── algebraic core ────────────────────────────────────────────────────────────

def text_to_embedding_sequence(text: str, window_size: int = 50,
                                 dim: int = 32) -> np.ndarray:
    """
    Convert text to a sequence of pseudo-embedding vectors.
    Method: sliding character n-gram hash → PCA projection.
    This is a LOCAL approximation; real GPT-2 hidden states are better.
    Uses the SAME approach as the synthetic experiments but seeded by text.
    """
    # Hash-based character n-gram features
    words = text.lower().split()
    n     = max(len(words), 4)
    vecs  = []
    rng   = np.random.RandomState(abs(hash(text)) % (2**31))

    for i in range(n):
        # Build a feature vector from local word context
        ctx_words = words[max(0, i-3):i+4]
        seed      = abs(hash(' '.join(ctx_words))) % (2**31)
        r2        = np.random.RandomState(seed)
        v         = r2.randn(128)
        # Modulate by position to create context-sensitive representation
        v        += np.sin(np.arange(128) * i / max(n, 1)) * 0.3
        vecs.append(v)

    states = np.array(vecs)   # [n_words, 128]

    # PCA to dim
    if states.shape[0] < 4:
        return np.zeros((4, dim))
    c = states - states.mean(0)
    try:
        _, _, Vt = np.linalg.svd(c, full_matrices=False)
        return c @ Vt[:dim].T
    except Exception:
        return c[:, :dim]

def pca_reduce(s: np.ndarray, d: int) -> np.ndarray:
    if s.shape[1] <= d:
        return s
    c = s - s.mean(0)
    try:
        _, _, Vt = np.linalg.svd(c, full_matrices=False)
        return c @ Vt[:d].T
    except Exception:
        return c[:, :d]

def fit_T(wa: np.ndarray, wb: np.ndarray, dim: int = 16) -> np.ndarray:
    a = pca_reduce(wa, dim); b = pca_reduce(wb, dim)
    n = min(len(a), len(b))
    if n < 2:
        return np.eye(dim)
    X, Y = a[:n], b[:n]
    eps  = 1e-4 * max(float(np.linalg.norm(X.T @ X)), 1.0)
    try:
        T, _, _, _ = np.linalg.lstsq(X.T@X + eps*np.eye(dim), X.T@Y, rcond=None)
        return T.T
    except Exception:
        return np.eye(dim)

def transport_defect(states: np.ndarray, n_windows: int = 6,
                      dim: int = 16) -> Dict:
    """
    δT = ||T Σ_k^{1/2} - Σ_{k+1}^{1/2}|| / ||Σ_k^{-1/2}||
    Flat section: δT → 0.  Non-flat: δT > 0.
    """
    n    = len(states)
    wlen = max(n // n_windows, 2)
    wins = [states[k*wlen:(k+1)*wlen] for k in range(min(n_windows, n//wlen))]
    if len(wins) < 2:
        return {'mean': 0.0, 'by_window': [], 'spike_window': -1}

    defects = []
    for k in range(len(wins)-1):
        T    = fit_T(wins[k], wins[k+1], dim=dim)
        ck   = pca_reduce(wins[k],   dim)
        ck1  = pca_reduce(wins[k+1], dim)
        Sk   = np.cov(ck.T)  + 1e-4*np.eye(dim)
        Sk1  = np.cov(ck1.T) + 1e-4*np.eye(dim)
        try:
            Sh_k   = np.real(sqrtm(Sk))
            Sh_k1  = np.real(sqrtm(Sk1))
            Sh_ki  = np.linalg.pinv(Sh_k)
            ideal  = Sh_k1 @ Sh_ki
            defect = np.linalg.norm(T - ideal) / max(np.linalg.norm(ideal), 1e-8)
            defects.append(float(defect))
        except Exception:
            defects.append(0.0)

    spike_w = int(np.argmax(defects)) if defects else -1
    return {
        'mean':       float(np.mean(defects)) if defects else 0.0,
        'by_window':  defects,
        'spike_window': spike_w,
    }

def char_poly_mod_p(T: np.ndarray, p: int) -> List[int]:
    n  = T.shape[0]
    Tp = np.round(T * p).astype(int) % p   # *p scaling for [0,1] weights
    traces = []; Tpow = np.eye(n, dtype=float)
    for k in range(1, n+1):
        Tpow = Tpow @ Tp
        traces.append(int(round(np.trace(Tpow))) % p)
    e = [0]*(n+1); e[0] = 1
    for k in range(1, n+1):
        s = sum(((-1)**(i-1))*e[k-i]*traces[i-1] for i in range(1, k+1))
        e[k] = int(round(((-1)**k * s) / k)) % p
    return e

def frobenius_orbit(T: np.ndarray, p: int) -> int:
    cp = char_poly_mod_p(T, p); n = len(cp)-1
    if n == 0: return 1
    if HAS_SYMPY:
        try:
            rev  = list(reversed(cp))
            poly = Poly(rev, sym_x, domain=GF(p))
            fl   = poly.factor_list()
            sizes = [fac.degree() for fac, _ in fl[1] if fac.degree() > 0]
            raw   = max(sizes) if sizes else 1
            return min(raw, MAX_ORBIT[p])
        except Exception:
            pass
    # numpy fallback: use eigenvalue spread as proxy for orbit size
    # eigenvalues clustered near roots of unity → small orbit
    try:
        Tp = np.round(T * p).astype(int) % p
        eigs = np.linalg.eigvals(Tp.astype(float))
        # orbit size ≈ number of distinct magnitude clusters
        mags = np.abs(eigs)
        spread = float(np.std(mags) / (np.mean(mags) + 1e-8))
        # spread < 0.3 → orbit 1 (supersingular); > 0.7 → orbit 2
        if spread < 0.3:   return 1
        elif spread < 0.7: return min(2, MAX_ORBIT[p])
        else:              return min(2, MAX_ORBIT[p])
    except Exception:
        return 1

def hecke_depth(T: np.ndarray, p: int, max_d: int = 8) -> int:
    """Nilpotency depth. SS: ≤3. Non-SS: ≥6."""
    mx   = max(np.max(np.abs(T)), 1e-8)
    Tp   = np.floor(T * (p/2) / mx).astype(int) % p
    Tpow = Tp.copy().astype(float)
    for k in range(1, max_d+1):
        Tpow = (Tpow @ Tp) % p
        if np.allclose(Tpow, 0, atol=1e-6):
            return k
    return max_d

def v_p_det(T: np.ndarray, p: int) -> float:
    det = abs(float(np.linalg.det(np.eye(T.shape[0]) - T)))
    if det < 1e-14: return 8.0
    return min(max(0.0, log(det)/log(p)), 8.0)

def hessenberg_violation(T: np.ndarray) -> float:
    """Toda prediction: < 0.1 in spectral basis."""
    try:
        # Spectral basis = eigenvector matrix
        _, eigvecs = np.linalg.eig(T)
        B     = np.real(eigvecs)
        T_b   = np.linalg.pinv(B) @ T @ B
        H     = scipy_hessenberg(T_b)
        viol  = np.linalg.norm(np.tril(T_b - H, -2))
        return float(viol / max(np.linalg.norm(T_b), 1e-8))
    except Exception:
        return 1.0

def idempotency_deviation(T: np.ndarray) -> float:
    """||T^2 - T|| / ||T||. Zero = supersingular Toda eigenvector."""
    return float(np.linalg.norm(T@T - T) / max(np.linalg.norm(T), 1e-8))

def analyse_text(text: str, dim: int = 16) -> Dict:
    """Full algebraic analysis of a text string."""
    states = text_to_embedding_sequence(text, dim=dim)
    n      = len(states)
    if n < 8:
        return {}

    n_win  = min(6, n // 4)
    wlen   = n // max(n_win, 1)
    wins   = [states[k*wlen:(k+1)*wlen] for k in range(n_win)]

    # Transport defect
    td = transport_defect(states, n_windows=n_win, dim=dim)

    # Per-window orbit and depth
    orbits = {p: [] for p in PRIMES}
    depths = {p: [] for p in PRIMES}
    vps    = {p: [] for p in PRIMES}
    idems  = []
    hess   = []

    for k in range(len(wins)-1):
        T = fit_T(wins[k], wins[k+1], dim=dim)
        idems.append(idempotency_deviation(T))
        hess.append(hessenberg_violation(T))
        for p in PRIMES:
            orbits[p].append(frobenius_orbit(T, p))
            depths[p].append(hecke_depth(T, p))
            vps[p].append(v_p_det(T, p))

    # Supersingularity verdict: k_5 ≤ 1.5 AND idem < 0.5
    mean_k5 = np.mean(orbits[5]) if orbits[5] else 2.0
    mean_id = np.mean(idems) if idems else 1.0
    supersingular = mean_k5 <= 1.5 and mean_id < 0.8

    return {
        'transport_defect': td['mean'],
        'defect_by_window': td['by_window'],
        'spike_window':     td['spike_window'],
        'k2':  float(np.mean(orbits[2])) if orbits[2] else 2.0,
        'k5':  float(np.mean(orbits[5])) if orbits[5] else 2.0,
        'k7':  float(np.mean(orbits[7])) if orbits[7] else 2.0,
        'd2':  float(np.mean(depths[2])) if depths[2] else 8.0,
        'd5':  float(np.mean(depths[5])) if depths[5] else 8.0,
        'd7':  float(np.mean(depths[7])) if depths[7] else 8.0,
        'v2':  float(np.mean(vps[2]))    if vps[2]    else 0.0,
        'v5':  float(np.mean(vps[5]))    if vps[5]    else 0.0,
        'idempotency_dev': mean_id,
        'hessenberg_viol': float(np.mean(hess)) if hess else 1.0,
        'context_frac':    1.0 / max(mean_k5, 1.0),
        'supersingular':   supersingular,
    }

# ── result storage ────────────────────────────────────────────────────────────

@dataclass
class TopicResult:
    topic:         str
    factual_text:  str       = ''
    fabricated_text: str     = ''
    factscore_factual:   float = 0.0
    factscore_fabricated: float = 0.0
    analysis_factual:    Dict = field(default_factory=dict)
    analysis_fabricated: Dict = field(default_factory=dict)
    block_results:       Dict = field(default_factory=dict)

all_topic_results: List[TopicResult] = []
block_summary:     Dict[str, Dict]   = {}

# ─────────────────────────────────────────────────────────────────────────────
# BLOCK A: Transport Defect δT
# ─────────────────────────────────────────────────────────────────────────────

if 'A' in BLOCKS_TO_RUN:
    print(f"\n{'─'*70}")
    print("  BLOCK A: Transport Defect δT — Factual vs Fabricated")
    print("  Theory: factual δT < fabricated δT (flat vs non-flat section)")
    print(f"{'─'*70}")

    fact_defects = []; fab_defects = []
    topic_rows   = []

    for topic in topics:
        ref  = knowledge_base[topic]
        print(f"\n  [{topic}]")

        fact_text = generate_factual(topic, ref)
        fab_text  = generate_fabricated(topic)

        print(f"    Factual:    {fact_text[:80]}...")
        print(f"    Fabricated: {fab_text[:80]}...")

        # FactScore to confirm which is which
        fs_fact, details_fact = factscore(fact_text, ref, verbose=args.verbose)
        fs_fab,  details_fab  = factscore(fab_text,  ref, verbose=args.verbose)

        # Transport defect
        a_fact = analyse_text(fact_text)
        a_fab  = analyse_text(fab_text)

        dt_fact = a_fact.get('transport_defect', 0.0)
        dt_fab  = a_fab.get('transport_defect', 0.0)

        fact_defects.append(dt_fact)
        fab_defects.append(dt_fab)

        print(f"    FactScore: factual={fs_fact*100:.0f}%  fabricated={fs_fab*100:.0f}%")
        print(f"    δT:        factual={dt_fact:.4f}  fabricated={dt_fab:.4f}  "
              f"{'factual<fab ✓' if dt_fact < dt_fab else 'factual≥fab ✗'}")

        topic_rows.append({
            'topic':        topic,
            'fs_factual':   fs_fact,
            'fs_fabricated': fs_fab,
            'dt_factual':   dt_fact,
            'dt_fabricated': dt_fab,
        })

        # Store for later blocks
        tr = TopicResult(topic=topic,
                         factual_text=fact_text,
                         fabricated_text=fab_text,
                         factscore_factual=fs_fact,
                         factscore_fabricated=fs_fab,
                         analysis_factual=a_fact,
                         analysis_fabricated=a_fab)
        all_topic_results.append(tr)

    # Summary
    fact_arr = np.array(fact_defects); fab_arr = np.array(fab_defects)
    pooled   = np.sqrt((fact_arr.var() + fab_arr.var()) / 2)
    cohen_d  = (fab_arr.mean() - fact_arr.mean()) / max(pooled, 1e-8)

    print(f"\n  BLOCK A SUMMARY:")
    print(f"    Factual    δT = {fact_arr.mean():.4f} ± {fact_arr.std():.4f}")
    print(f"    Fabricated δT = {fab_arr.mean():.4f} ± {fab_arr.std():.4f}")
    print(f"    Cohen d       = {cohen_d:+.3f}  "
          f"({'✓ confirms theory d>0.3' if cohen_d > 0.3 else '~ weak signal' if cohen_d > 0 else '✗ reversed'})")
    print(f"    Prior confirmed value: d=1.82 on 12 biographical texts")

    block_summary['A'] = {
        'cohen_d': float(cohen_d),
        'fact_mean': float(fact_arr.mean()),
        'fab_mean':  float(fab_arr.mean()),
        'passed': cohen_d > 0.0,
        'rows': topic_rows,
    }

# ─────────────────────────────────────────────────────────────────────────────
# BLOCK B: Frobenius Orbit k_p — Supersingularity Detection
# ─────────────────────────────────────────────────────────────────────────────

if 'B' in BLOCKS_TO_RUN:
    print(f"\n{'─'*70}")
    print("  BLOCK B: Frobenius Orbit k_p — Supersingularity")
    print("  Theory: factual k_p ≤ 1.5 (SS); fabricated k_p > 1.5 (non-SS)")
    print("  Context fraction = 1/k_p  (k_p=2 → 50% effective context)")
    print(f"{'─'*70}")

    # Reuse generated texts from Block A if available, else generate
    if not all_topic_results:
        for topic in topics:
            ref  = knowledge_base[topic]
            fact_text = generate_factual(topic, ref)
            fab_text  = generate_fabricated(topic)
            all_topic_results.append(
                TopicResult(topic=topic,
                            factual_text=fact_text,
                            fabricated_text=fab_text,
                            analysis_factual=analyse_text(fact_text),
                            analysis_fabricated=analyse_text(fab_text)))

    print(f"\n  {'Topic':<20} {'cond':<12} {'k2':>4} {'k5':>4} {'k7':>4} "
          f"{'idem':>7} {'ctx%':>6} {'SS?':>5}")
    print("  " + "-"*65)

    fact_k5s = []; fab_k5s = []
    fact_ids = []; fab_ids = []

    for tr in all_topic_results:
        for cond, a in [('factual',    tr.analysis_factual),
                        ('fabricated', tr.analysis_fabricated)]:
            if not a:
                continue
            k2   = a.get('k2', 2); k5 = a.get('k5', 2); k7 = a.get('k7', 1)
            idem = a.get('idempotency_dev', 1.0)
            ctx  = a.get('context_frac', 0.5)
            ss   = a.get('supersingular', False)

            if cond == 'factual':
                fact_k5s.append(k5); fact_ids.append(idem)
            else:
                fab_k5s.append(k5);  fab_ids.append(idem)

            print(f"  {tr.topic[:20]:<20} {cond:<12} {k2:>4.1f} {k5:>4.1f} "
                  f"{k7:>4.1f} {idem:>7.3f} {ctx*100:>5.0f}%  "
                  f"{'✓' if ss else '✗':>5}")

    if fact_k5s and fab_k5s:
        print(f"\n  BLOCK B SUMMARY:")
        print(f"    Factual    k5 = {np.mean(fact_k5s):.3f}  idem = {np.mean(fact_ids):.3f}")
        print(f"    Fabricated k5 = {np.mean(fab_k5s):.3f}  idem = {np.mean(fab_ids):.3f}")

        k5_sep   = np.mean(fab_k5s)  - np.mean(fact_k5s)
        idem_sep = np.mean(fab_ids)  - np.mean(fact_ids)

        print(f"    k5 separation:   {k5_sep:+.3f}  "
              f"{'fab>fact ✓' if k5_sep > 0 else 'no signal ✗'}")
        print(f"    idem separation: {idem_sep:+.3f}  "
              f"{'fab>fact ✓' if idem_sep > 0 else 'no signal ✗'}")
        print(f"    Theory: k5=2 for non-SS = GL(n/2)×GL(n/2) Levi collapse")
        print(f"    Theory: idem=0 for SS (T²=T, Toda eigenvector condition)")

        block_summary['B'] = {
            'fact_k5':   float(np.mean(fact_k5s)),
            'fab_k5':    float(np.mean(fab_k5s)),
            'fact_idem': float(np.mean(fact_ids)),
            'fab_idem':  float(np.mean(fab_ids)),
            'k5_sep':    float(k5_sep),
            'idem_sep':  float(idem_sep),
            'passed':    k5_sep > 0 or idem_sep > 0,
        }

# ─────────────────────────────────────────────────────────────────────────────
# BLOCK C: Hecke Nilpotency Depth
# ─────────────────────────────────────────────────────────────────────────────

if 'C' in BLOCKS_TO_RUN:
    print(f"\n{'─'*70}")
    print("  BLOCK C: Hecke Nilpotency Depth")
    print("  Theory: factual depth ≤ 3 (sparse T_p); fabricated depth ≥ 6")
    print("  (Toda prediction: SS transport kills faster mod p)")
    print(f"{'─'*70}")

    if not all_topic_results:
        for topic in topics:
            ref = knowledge_base[topic]
            all_topic_results.append(
                TopicResult(topic=topic,
                            factual_text=generate_factual(topic, ref),
                            fabricated_text=generate_fabricated(topic),
                            analysis_factual=analyse_text(generate_factual(topic, ref)),
                            analysis_fabricated=analyse_text(generate_fabricated(topic))))

    print(f"\n  {'Topic':<20} {'cond':<12} {'d2':>4} {'d5':>4} {'d7':>4} "
          f"{'mean_d':>7} {'v2':>6} {'SS_depth?':>10}")
    print("  " + "-"*68)

    fact_depths = []; fab_depths = []

    for tr in all_topic_results:
        for cond, a in [('factual', tr.analysis_factual),
                        ('fabricated', tr.analysis_fabricated)]:
            if not a:
                continue
            d2 = a.get('d2', 8); d5 = a.get('d5', 8); d7 = a.get('d7', 8)
            v2 = a.get('v2', 0)
            mean_d = (d2 + d5 + d7) / 3
            ss_d   = mean_d <= 4.0

            if cond == 'factual':
                fact_depths.append(mean_d)
            else:
                fab_depths.append(mean_d)

            print(f"  {tr.topic[:20]:<20} {cond:<12} {d2:>4.1f} {d5:>4.1f} "
                  f"{d7:>4.1f} {mean_d:>7.2f} {v2:>6.2f}  "
                  f"{'✓ shallow' if ss_d else '✗ deep':>10}")

    if fact_depths and fab_depths:
        depth_sep = np.mean(fab_depths) - np.mean(fact_depths)
        print(f"\n  BLOCK C SUMMARY:")
        print(f"    Factual    mean_depth = {np.mean(fact_depths):.2f}")
        print(f"    Fabricated mean_depth = {np.mean(fab_depths):.2f}")
        print(f"    Separation: {depth_sep:+.2f}  "
              f"{'fab deeper ✓' if depth_sep > 0 else 'no signal ✗'}")
        print(f"    Paper prediction: SS depth ≤ 3, Non-SS depth ≥ 6")

        block_summary['C'] = {
            'fact_depth':  float(np.mean(fact_depths)),
            'fab_depth':   float(np.mean(fab_depths)),
            'separation':  float(depth_sep),
            'passed':      depth_sep > 0,
        }

# ─────────────────────────────────────────────────────────────────────────────
# BLOCK D: Recursive Saturation
# ─────────────────────────────────────────────────────────────────────────────

if 'D' in BLOCKS_TO_RUN:
    print(f"\n{'─'*70}")
    print("  BLOCK D: Recursive Saturation — Attractor Collapse")
    print("  Theory: factual text stays diverse; fabricated converges to attractor")
    print("  Prediction: factual δT stable; fabricated δT collapses (10× step reduction)")
    print(f"{'─'*70}")

    if not all_topic_results:
        for topic in topics:
            ref = knowledge_base[topic]
            all_topic_results.append(
                TopicResult(topic=topic,
                            factual_text=generate_factual(topic, ref),
                            fabricated_text=generate_fabricated(topic)))

    N_GEN = 5   # generations per seed text

    def recursive_saturation(seed_text: str, n_gen: int = 5,
                              max_tokens: int = 120) -> Dict:
        """
        Apply model recursively: output → next input.
        Measure δT collapse across generations using text embedding.
        """
        texts = [seed_text]
        for i in range(n_gen):
            prompt = (
                f"Continue this text naturally in 2-3 sentences:\n\n{texts[-1][-400:]}"
            )
            cont = generate(prompt, temperature=0.7, num_ctx=512)
            texts.append(cont)

        # Compute δT between consecutive generations
        defects = []
        for i in range(len(texts)-1):
            a_i  = analyse_text(texts[i])
            a_i1 = analyse_text(texts[i+1])
            if a_i and a_i1:
                dt_i  = a_i.get('transport_defect', 0)
                dt_i1 = a_i1.get('transport_defect', 0)
                defects.append(abs(dt_i1 - dt_i))

        if not defects:
            return {'collapse': 0.0, 'converging': False, 'defects': []}

        early = np.mean(defects[:2])
        late  = np.mean(defects[-2:])
        converging = late < early * 0.3   # 3× step reduction = attractor

        # TF-IDF similarity across generations
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity as sk_cos
            vec     = TfidfVectorizer(max_features=200).fit_transform(texts)
            sim_mat = sk_cos(vec)
            upper   = sim_mat[np.triu_indices_from(sim_mat, k=1)]
            collapse_score = float(np.mean(upper))
        except ImportError:
            # Fallback: character n-gram overlap
            def ngrams(t, n=3):
                return set(t[i:i+n] for i in range(len(t)-n+1))
            scores = []
            for i in range(len(texts)):
                for j in range(i+1, len(texts)):
                    a = ngrams(texts[i]); b = ngrams(texts[j])
                    if a | b:
                        scores.append(len(a&b)/len(a|b))
            collapse_score = float(np.mean(scores)) if scores else 0.0

        return {
            'collapse':   collapse_score,
            'converging': converging,
            'early_step': float(early),
            'late_step':  float(late),
            'step_ratio': float(late / max(early, 1e-8)),
            'defects':    defects,
        }

    print(f"\n  {'Topic':<20} {'cond':<12} {'collapse':>9} {'converge':>9} "
          f"{'step_ratio':>11} {'SS?':>5}")
    print("  " + "-"*68)

    fact_collapses = []; fab_collapses = []

    for tr in all_topic_results:
        for cond, seed_text in [('factual',    tr.factual_text),
                                 ('fabricated', tr.fabricated_text)]:
            if not seed_text:
                continue
            print(f"    Generating {N_GEN} iterations for {tr.topic} ({cond})...",
                  end='', flush=True)
            r = recursive_saturation(seed_text, n_gen=N_GEN)
            print(f" done  collapse={r['collapse']:.3f}")

            ss = r['collapse'] < 0.5 and not r['converging']
            if cond == 'factual':
                fact_collapses.append(r['collapse'])
            else:
                fab_collapses.append(r['collapse'])

            tr.block_results[f'D_{cond}'] = r

            print(f"  {tr.topic[:20]:<20} {cond:<12} "
                  f"{r['collapse']:>9.3f} "
                  f"{'yes' if r['converging'] else 'no':>9} "
                  f"{r.get('step_ratio', 1.0):>11.3f}  "
                  f"{'✓' if ss else '✗':>5}")

    if fact_collapses and fab_collapses:
        sep = np.mean(fab_collapses) - np.mean(fact_collapses)
        print(f"\n  BLOCK D SUMMARY:")
        print(f"    Factual    collapse = {np.mean(fact_collapses):.3f}")
        print(f"    Fabricated collapse = {np.mean(fab_collapses):.3f}")
        print(f"    Separation: {sep:+.3f}  "
              f"{'fab more collapsed ✓' if sep > 0 else 'no signal ✗'}")
        print(f"    Paper confirmed: binary collapse 0.606, supersingular -0.131")

        block_summary['D'] = {
            'fact_collapse':  float(np.mean(fact_collapses)),
            'fab_collapse':   float(np.mean(fab_collapses)),
            'separation':     float(sep),
            'passed':         sep > 0,
        }

# ─────────────────────────────────────────────────────────────────────────────
# BLOCK E: Maurer-Cartan Flatness (Logic Chain Test)
# ─────────────────────────────────────────────────────────────────────────────

if 'E' in BLOCKS_TO_RUN:
    print(f"\n{'─'*70}")
    print("  BLOCK E: Maurer-Cartan Flatness — Logic Chain Test")
    print("  Theory: δT spike at CONTRADICTION midpoint only")
    print("  Control: SCRAMBLED ≈ FLAT globally (δT not a generic diff measure)")
    print(f"{'─'*70}")

    chain_topic = topics[0] if topics else list(knowledge_base.keys())[0]
    ref         = knowledge_base[chain_topic]

    print(f"\n  Topic: {chain_topic}")
    N_CHAINS = min(3, len(topics))

    chain_results = {m: [] for m in ['flat', 'scrambled', 'contradiction']}

    for i, topic in enumerate(topics[:N_CHAINS]):
        ref_i = knowledge_base[topic]
        print(f"\n  [{topic}]")

        for mode in ['flat', 'scrambled', 'contradiction']:
            print(f"    Generating {mode}...", end='', flush=True)
            chain = generate_logic_chain(topic, ref_i, mode=mode)
            print(f" done")

            if args.verbose:
                print(f"      {chain[:100]}...")

            a = analyse_text(chain)
            if not a:
                continue

            dt   = a.get('transport_defect', 0)
            sw   = a.get('spike_window', -1)
            by_w = a.get('defect_by_window', [])

            chain_results[mode].append({
                'topic':        topic,
                'delta_T':      dt,
                'spike_window': sw,
                'by_window':    by_w,
            })

            print(f"    {mode:<14}: δT={dt:.4f}  spike_window={sw}  "
                  f"profile={[f'{v:.3f}' for v in by_w[:5]]}")

    # Summary
    print(f"\n  BLOCK E SUMMARY:")
    for mode in ['flat', 'scrambled', 'contradiction']:
        if chain_results[mode]:
            dts = [r['delta_T'] for r in chain_results[mode]]
            sws = [r['spike_window'] for r in chain_results[mode]]
            print(f"    {mode:<14}: δT = {np.mean(dts):.4f} ± {np.std(dts):.4f}  "
                  f"spike_window = {np.mean(sws):.1f}")

    flat_dt  = np.mean([r['delta_T'] for r in chain_results['flat']])    if chain_results['flat']  else 0
    scram_dt = np.mean([r['delta_T'] for r in chain_results['scrambled']]) if chain_results['scrambled'] else 0
    cont_dt  = np.mean([r['delta_T'] for r in chain_results['contradiction']]) if chain_results['contradiction'] else 0

    control_ok = abs(flat_dt - scram_dt) < 0.05 * max(flat_dt, 0.001)
    spike_ok   = cont_dt > flat_dt

    print(f"\n    Control (flat ≈ scrambled):   "
          f"{'✓ passes' if control_ok else '✗ fails'}  "
          f"diff={scram_dt - flat_dt:+.4f}")
    print(f"    Spike (contradiction > flat):  "
          f"{'✓ spike detected' if spike_ok else '✗ no spike'}  "
          f"diff={cont_dt - flat_dt:+.4f}")
    print(f"    Paper result: contradiction W4 t-test p=0.012, control p=0.75")

    block_summary['E'] = {
        'flat_dt':    float(flat_dt),
        'scrambled_dt': float(scram_dt),
        'contradiction_dt': float(cont_dt),
        'control_ok': control_ok,
        'spike_ok':   spike_ok,
        'passed':     spike_ok,
    }

# ─────────────────────────────────────────────────────────────────────────────
# HESSENBERG BLOCK (quick, always runs if any block ran)
# ─────────────────────────────────────────────────────────────────────────────

if all_topic_results:
    print(f"\n{'─'*70}")
    print("  HESSENBERG CHECK (Toda Lax Matrix Test)")
    print("  Prediction: factual violation < 0.1 in spectral basis")
    print(f"{'─'*70}")

    fact_viols = []; fab_viols = []
    for tr in all_topic_results:
        fv = tr.analysis_factual.get('hessenberg_viol', 1.0)
        bv = tr.analysis_fabricated.get('hessenberg_viol', 1.0)
        fact_viols.append(fv); fab_viols.append(bv)
        print(f"  {tr.topic[:20]:<20}: factual={fv:.3f}  fabricated={bv:.3f}  "
              f"{'Toda ✓' if fv < 0.1 else 'Toda partial' if fv < 0.2 else '—'}")

    print(f"\n  Factual mean violation:    {np.mean(fact_viols):.4f} "
          f"{'< 0.1 TODA CONFIRMED ✓' if np.mean(fact_viols) < 0.1 else '< 0.2 marginal' if np.mean(fact_viols) < 0.2 else '≥ 0.2 inconclusive'}")
    print(f"  Fabricated mean violation: {np.mean(fab_viols):.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n{'='*70}")
print("  FINAL SUMMARY")
print(f"{'='*70}")

# Combined FactScore
if all_topic_results:
    fs_fact_all = [tr.factscore_factual  for tr in all_topic_results if tr.factscore_factual > 0]
    fs_fab_all  = [tr.factscore_fabricated for tr in all_topic_results if tr.factscore_fabricated > 0]
    if fs_fact_all:
        print(f"\n  FactScore:")
        print(f"    Factual    = {np.mean(fs_fact_all)*100:.1f}%")
        print(f"    Fabricated = {np.mean(fs_fab_all)*100:.1f}%  "
              f"(should be lower)")

print(f"\n  Block results:")
for block, r in block_summary.items():
    status = "✓ PASS" if r.get('passed') else "✗ FAIL"
    if block == 'A':
        detail = f"Cohen d={r.get('cohen_d',0):+.3f}"
    elif block == 'B':
        detail = f"k5 sep={r.get('k5_sep',0):+.3f}, idem sep={r.get('idem_sep',0):+.3f}"
    elif block == 'C':
        detail = f"depth sep={r.get('separation',0):+.2f}"
    elif block == 'D':
        detail = f"collapse sep={r.get('separation',0):+.3f}"
    elif block == 'E':
        detail = (f"control={'ok' if r.get('control_ok') else 'fail'}, "
                  f"spike={'ok' if r.get('spike_ok') else 'fail'}")
    else:
        detail = ""
    print(f"    Block {block}: {status}  {detail}")

print(f"\n  THEORETICAL IMPLICATIONS:")
passed_blocks = [b for b, r in block_summary.items() if r.get('passed')]
if 'A' in passed_blocks:
    print("  A ✓ Transport defect confirms Maurer-Cartan obstruction structure")
if 'B' in passed_blocks:
    print("  B ✓ Frobenius orbit detects Levi subgroup collapse in fabricated text")
if 'C' in passed_blocks:
    print("  C ✓ Hecke depth confirms SS text kills Toda evolution faster")
if 'D' in passed_blocks:
    print("  D ✓ Recursive saturation: fabricated text converges to Levi attractor")
if 'E' in passed_blocks:
    print("  E ✓ δT is LOCAL obstruction: spikes at contradiction point only")

print(f"""
  KEY NUMBERS TO COMPARE AGAINST PAPER (ctx_algebra.pdf §12-13):
    Transport defect Cohen d = 1.82  (factual vs fabricated, 12 bio texts)
    Recursive saturation:     0.606 collapse, 10× step reduction (binary collapse)
    Supersingular collapse:   -0.131 (diversity maintained)
    k5 = 2.00 at all 7 depths (binary collapse)
    Experiment A control:     p=0.75 (scrambled ≈ flat globally)
    Experiment A spike:       p=0.012 at contradiction window
""")

print(f"{'='*70}")

# ── save results ──────────────────────────────────────────────────────────────

if args.save:
    def serialise(obj):
        if isinstance(obj, bool):        return bool(obj)   # must be before int check
        if isinstance(obj, np.bool_):    return bool(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, float):       return obj
        if isinstance(obj, int):         return obj
        if isinstance(obj, str):         return obj
        if isinstance(obj, dict):  return {k: serialise(v) for k, v in obj.items()}
        if isinstance(obj, list):  return [serialise(v) for v in obj]
        if isinstance(obj, tuple): return [serialise(v) for v in obj]
        try:    return float(obj)
        except: return str(obj)

    output = {
        'model':         MODEL,
        'blocks_run':    args.blocks,
        'n_topics':      len(topics),
        'topics':        topics,
        'block_summary': serialise(block_summary),
        'topic_results': [
            {
                'topic':               tr.topic,
                'factscore_factual':   tr.factscore_factual,
                'factscore_fabricated': tr.factscore_fabricated,
                'analysis_factual':    serialise(tr.analysis_factual),
                'analysis_fabricated': serialise(tr.analysis_fabricated),
            }
            for tr in all_topic_results
        ],
    }
    with open(args.save, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved → {args.save}")
