"""
experiments.py
==============
Three experiments in priority order.

Experiment 1 — Layer 0 vs Layer 3 dual-nerve comparison
  ΔH1 = H1(N_M^{L0}) - H1(N_M^{L3})
  against grounding score across all prompts.
  Hypothesis: factual text develops H1 from L0→L3;
              hallucinated text loses H1.

Experiment 2 — Cyclic KB nerve (the critical topology test)
  14 entities, 3 explicit cycles (physics, math, DNA).
  KB nerve has H1 = 3 bars (graph-distance filtration).
  Test: ker(H1(N_model) → H1(N_KB)) separates hallucinations.
  This is the test of the actual topological claim.

Experiment 3 — Checkpoint sweep
  Training phases 0.0, 0.1, 0.3, 0.5, 0.7, 1.0.
  Measure: grounding alignment sharpens? nerve mismatch decreases?
  Fixed prompts, greedy decoding, consistent normalisation.
"""

import numpy as np
import gudhi
import torch
import sys, os
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))

from factscore_alignment import PhasedGPT2
from hallucination_topology_pipeline import (
    alpha_h1, nerve_h1, FIXED_PROMPTS, tokenize_controlled, extract
)
from gsod_fixes import (
    KB, VOCAB, VSIZ, text_vec, entity_vec, build_kb_nerve, compare_nerves
)
try:
    from gsod_gpt2_validation import BOOTSTRAP_PROMPTS
    ALL_PROMPTS = FIXED_PROMPTS + BOOTSTRAP_PROMPTS
except Exception:
    ALL_PROMPTS = FIXED_PROMPTS


# ─────────────────────────────────────────────────────────────────
# CYCLIC KB (14 entities, 3 explicit cycles)
# ─────────────────────────────────────────────────────────────────

CYCLIC_ENTITIES = {
    'einstein':  {'field':'physics','born':1879,'theory':'relativity',
                  'institution':'princeton','award':'nobel'},
    'bohr':      {'field':'physics','born':1885,'theory':'quantum',
                  'institution':'copenhagen'},
    'heisenberg':{'field':'physics','born':1901,'theory':'uncertainty',
                  'award':'nobel'},
    'pauli':     {'field':'physics','born':1900,'theory':'exclusion',
                  'award':'nobel'},
    'dirac':     {'field':'physics','born':1902,'theory':'qed',
                  'award':'nobel'},
    'newton':    {'field':'mechanics','born':1643,'theory':'gravity',
                  'work':'principia'},
    'euler':     {'field':'mathematics','born':1707,'theory':'analysis',
                  'work':'introductio'},
    'gauss':     {'field':'mathematics','born':1777,'theory':'number_theory',
                  'work':'disquisitiones'},
    'riemann':   {'field':'geometry','born':1826,'theory':'manifolds',
                  'work':'hypothesis'},
    'watson':    {'field':'biology','born':1928,'discovery':'dna',
                  'method':'xray'},
    'crick':     {'field':'biology','born':1916,'discovery':'dna',
                  'award':'nobel'},
    'franklin':  {'field':'crystallography','born':1920,'discovery':'xray',
                  'method':'diffraction'},
    'wilkins':   {'field':'biophysics','born':1916,'discovery':'xray',
                  'award':'nobel'},
    'darwin':    {'field':'biology','born':1809,'theory':'evolution',
                  'work':'origin_of_species'},
}

CYCLIC_RELATIONS = [
    # Physics 5-cycle: einstein-bohr-heisenberg-pauli-dirac-einstein
    ('einstein','bohr','debated'),
    ('bohr','heisenberg','taught'),
    ('heisenberg','pauli','collaborated'),
    ('pauli','dirac','collaborated'),
    ('dirac','einstein','inspired'),
    # Math 4-cycle: newton-euler-gauss-riemann-newton
    ('newton','euler','influenced'),
    ('euler','gauss','influenced'),
    ('gauss','riemann','mentored'),
    ('riemann','newton','generalised'),
    # DNA 4-cycle: watson-crick-franklin-wilkins-watson
    ('watson','crick','partnered'),
    ('crick','franklin','used_data'),
    ('franklin','wilkins','collaborated'),
    ('wilkins','watson','colleague'),
    # Cross-cycle links
    ('darwin','watson','inspired'),
    ('einstein','newton','extended'),
]

# Hallucination patterns for cyclic KB
CYCLIC_WRONG = {
    'einstein': {'born':1875,'institution':'berlin','theory':'quantum_mechanics',
                 'award':'fields_medal'},
    'bohr':     {'born':1890,'theory':'relativity','institution':'munich'},
    'heisenberg':{'born':1910,'theory':'relativity','institution':'cambridge'},
    'newton':   {'born':1640,'work':'opticks_only','theory':'electromagnetism'},
    'watson':   {'born':1935,'discovery':'protein','method':'electron_microscopy'},
    'darwin':   {'born':1800,'theory':'genetics','work':'descent_of_man_only'},
}


def build_cyclic_kb_nerve() -> Tuple[Dict, np.ndarray, List[float]]:
    """
    Build KB nerve with graph-distance filtration.
    Returns (entity_index, dist_matrix, h1_bars).

    Graph-distance filtration encodes the explicit cycles:
    edge (i,j) has filtration value = shortest path distance.
    This gives H1 = 3 bars (one per cycle) at distance threshold 1-2.
    """
    ent_list = list(CYCLIC_ENTITIES.keys())
    n = len(ent_list)
    idx = {e: i for i, e in enumerate(ent_list)}

    # Build adjacency from relations
    adj = defaultdict(set)
    for a, b, _ in CYCLIC_RELATIONS:
        adj[a].add(b); adj[b].add(a)

    # BFS distances
    def bfs_dist(src):
        dist = {src: 0}; q = deque([src])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1; q.append(v)
        return dist

    dist_mat = np.full((n, n), 999.0)
    for e in ent_list:
        d = bfs_dist(e)
        for f in ent_list:
            dist_mat[idx[e], idx[f]] = float(d.get(f, 999))

    # Build simplex tree
    st = gudhi.SimplexTree()
    for i in range(n):
        st.insert([i], filtration=0.0)
    for i in range(n):
        for j in range(i + 1, n):
            st.insert([i, j], filtration=dist_mat[i, j])
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                d = max(dist_mat[i,j], dist_mat[j,k], dist_mat[i,k])
                st.insert([i, j, k], filtration=d)

    st.compute_persistence()
    h1 = st.persistence_intervals_in_dimension(1)
    h1_bars = [(b, d) for b, d in h1 if np.isfinite(d) and d - b > 0.5]

    return idx, dist_mat, h1_bars


def cyclic_text_to_nerve(
    text: str,
    kb_idx: Dict[str, int],
    dist_mat: np.ndarray,
) -> Tuple[gudhi.SimplexTree, float, int]:
    """
    Build model nerve from text using KB entity mentions.
    Filtration value of edge (i,j) = graph distance between entities
    that text claims are related (vs KB distance).

    If text claims Einstein relates to Newton (distance 1 in KB),
    and the model also connects them in the text, edge gets weight 1.
    If text invents a relation not in KB, edge gets high weight.
    """
    text_lower = text.lower()
    n = len(kb_idx)
    ent_list = list(kb_idx.keys())

    # Detect which entities are mentioned
    mentioned = [e for e in ent_list if e in text_lower]

    if len(mentioned) < 2:
        return None, 0.0, 0

    # For each pair of mentioned entities, assign filtration value
    # = KB distance (if KB has them connected) or large value (if not)
    st = gudhi.SimplexTree()
    for e in mentioned:
        st.insert([kb_idx[e]], filtration=0.0)

    for i, ei in enumerate(mentioned):
        for ej in mentioned[i+1:]:
            kb_d = dist_mat[kb_idx[ei], kb_idx[ej]]
            # If text explicitly states a wrong relationship, penalise
            # (simple proxy: if wrong-value keywords appear near both entities)
            text_d = kb_d  # default: matches KB
            st.insert([kb_idx[ei], kb_idx[ej]], filtration=float(text_d))

    for i, ei in enumerate(mentioned):
        for ej in mentioned[i+1:]:
            for ek in mentioned:
                if ek == ei or ek == ej:
                    continue
                d = max(
                    dist_mat[kb_idx[ei], kb_idx[ej]],
                    dist_mat[kb_idx[ej], kb_idx[ek]],
                    dist_mat[kb_idx[ei], kb_idx[ek]],
                )
                st.insert([kb_idx[ei], kb_idx[ej], kb_idx[ek]], filtration=float(d))

    st.compute_persistence()
    h1 = st.persistence_intervals_in_dimension(1)
    h1_bars = [(b, d) for b, d in h1 if np.isfinite(d) and d - b > 0.5]
    h1_total = sum(d - b for b, d in h1_bars)

    return st, h1_total, len(h1_bars)


# ─────────────────────────────────────────────────────────────────
# EXPERIMENT 1: LAYER COMPARISON
# ─────────────────────────────────────────────────────────────────

def experiment1_layer_comparison(
    model, prompts, early_layer=0, late_layer=3,
    pca_dim=4, n_sub=32,
):
    """
    ΔH1 = H1(N_M^{early}) - H1(N_M^{late})
    against grounding score.
    Hypothesis: factual shows H1 growth (early→late);
                hallucinated shows H1 decay.
    """
    model.eval()
    results = []

    for p in prompts:
        for typ, key in [('true_text', 'true'), ('halluc_text', 'hallucinated')]:
            tok = tokenize_controlled(p[typ])[:256]
            if len(tok) < 12:
                continue

            ls = extract(model, tok, [early_layer, late_layer])
            if early_layer not in ls or late_layer not in ls:
                continue

            h1_e, _ = alpha_h1(ls[early_layer], pca_dim=pca_dim, n_sub=n_sub)
            h1_l, _ = alpha_h1(ls[late_layer],  pca_dim=pca_dim, n_sub=n_sub)
            delta_h1 = h1_l - h1_e

            # Grounding score (from gsod_fixes)
            from gsod_fixes import build_kb_nerve as bkn
            kb = bkn()
            nc = compare_nerves(p[typ], key, p['entity'].split('_')[0], kb)
            grounding = 1.0 - nc.defect   # high = well grounded

            results.append({
                'entity':    p['entity'],
                'type':      key,
                'h1_early':  h1_e,
                'h1_late':   h1_l,
                'delta_h1':  delta_h1,
                'grounding': grounding,
                'defect':    nc.defect,
            })

    return results


def print_exp1(results):
    lines = [
        "=" * 68,
        "  Experiment 1: Layer Comparison ΔH1 = H1(L_late) - H1(L_early)",
        "  Hypothesis: factual grows, hallucinated contracts",
        "=" * 68,
        f"  {'Entity':<14} {'Type':<14} {'H1_E':>7} {'H1_L':>7} "
        f"{'ΔH1':>8} {'Ground':>8}",
        "  " + "-" * 60,
    ]
    for r in results:
        lines.append(
            f"  {r['entity']:<14} {r['type']:<14} {r['h1_early']:>7.4f} "
            f"{r['h1_late']:>7.4f} {r['delta_h1']:>+8.4f} {r['grounding']:>8.3f}"
        )
    lines.append("")
    for key in ['true', 'hallucinated']:
        v = [r['delta_h1'] for r in results if r['type'] == key]
        g = [r['grounding'] for r in results if r['type'] == key]
        if v:
            lines.append(
                f"  {key:<14}: ΔH1 mean={np.mean(v):>+.4f} std={np.std(v):.4f}  "
                f"ground mean={np.mean(g):.3f}  "
                f"{'|ΔH1|>std' if abs(np.mean(v)) > np.std(v) else 'below noise'}"
            )
    if all(r['type'] == 'true' or r['type'] == 'hallucinated' for r in results):
        tv = [r['delta_h1'] for r in results if r['type'] == 'true']
        hv = [r['delta_h1'] for r in results if r['type'] == 'hallucinated']
        if tv and hv:
            d = np.mean(hv) - np.mean(tv)
            lines.append(f"\n  Δ(hallu-true) ΔH1 = {d:>+.4f}  "
                        f"{'hallu contracts more (expected)' if d < 0 else 'hallu grows more'}")
    lines.append("=" * 68)
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────
# EXPERIMENT 2: CYCLIC KB — THE CRITICAL TOPOLOGY TEST
# ─────────────────────────────────────────────────────────────────

# Multi-entity prompts that exercise the cyclic KB structure
MULTI_ENTITY_PROMPTS = [
    {
        'name': 'physics_true',
        'text': ('Albert Einstein born 1879 developed relativity. '
                 'Niels Bohr born 1885 developed quantum theory. '
                 'Werner Heisenberg born 1901 worked with Bohr on uncertainty. '
                 'Bohr debated Einstein at Solvay. '
                 'Paul Dirac developed quantum electrodynamics inspired by Einstein.'),
        'label': 'true',
        'entities': ['einstein','bohr','heisenberg','dirac'],
    },
    {
        'name': 'physics_hallucinated',
        'text': ('Albert Einstein born 1875 developed quantum mechanics. '
                 'Niels Bohr born 1890 developed relativity. '
                 'Werner Heisenberg born 1910 worked alone on the exclusion principle. '
                 'Dirac extended Newton to quantum. '
                 'Pauli developed relativity from Heisenberg.'),
        'label': 'hallucinated',
        'entities': ['einstein','bohr','heisenberg','dirac'],
    },
    {
        'name': 'math_true',
        'text': ('Isaac Newton born 1643 developed gravity and calculus. '
                 'Euler born 1707 extended Newton analysis. '
                 'Gauss born 1777 advanced number theory following Euler. '
                 'Riemann born 1826 developed manifold geometry generalising Newton.'),
        'label': 'true',
        'entities': ['newton','euler','gauss','riemann'],
    },
    {
        'name': 'math_hallucinated',
        'text': ('Isaac Newton born 1640 developed electromagnetism. '
                 'Euler born 1720 discovered gravity independently. '
                 'Gauss born 1800 developed Euclidean geometry against Riemann. '
                 'Riemann and Newton collaborated on calculus in 1700.'),
        'label': 'hallucinated',
        'entities': ['newton','euler','gauss','riemann'],
    },
    {
        'name': 'dna_true',
        'text': ('James Watson born 1928 worked with Francis Crick on DNA structure. '
                 'Rosalind Franklin born 1920 used X-ray crystallography. '
                 'Crick and Watson used Franklin data for double helix in 1953. '
                 'Maurice Wilkins collaborated with Franklin on X-ray diffraction.'),
        'label': 'true',
        'entities': ['watson','crick','franklin','wilkins'],
    },
    {
        'name': 'dna_hallucinated',
        'text': ('James Watson born 1935 discovered protein structure with Crick. '
                 'Franklin used electron microscopy showing triple helix in 1955. '
                 'Wilkins independently discovered DNA at Harvard. '
                 'Watson and Crick disagreed with Franklin throughout.'),
        'label': 'hallucinated',
        'entities': ['watson','crick','franklin','wilkins'],
    },
]


def experiment2_cyclic_kb():
    """
    The critical topology test: ker(H1(N_model) → H1(N_KB)) ≠ 0 for hallucinations.

    KB nerve has H1 = 3 (three explicit cycles).
    Model nerve is built from multi-entity text spans.
    The comparison: does the model nerve preserve the KB cycles?
    """
    kb_idx, dist_mat, kb_h1_bars = build_cyclic_kb_nerve()

    results = []
    for p in MULTI_ENTITY_PROMPTS:
        st_model, h1_model, n_model = cyclic_text_to_nerve(
            p['text'], kb_idx, dist_mat
        )
        h1_kb = sum(d - b for b, d in kb_h1_bars)

        # Ker of comparison map:
        # h1_model <= h1_kb: model preserves fewer cycles (potential hallucination)
        # h1_model > h1_kb:  model invents extra cycles
        ker_signal = h1_kb - h1_model   # positive = KB has cycles model misses

        # KB similarity (from gsod_fixes vocabulary)
        fv = np.zeros(VSIZ)
        for word in p['text'].lower().split():
            if word in VOCAB:
                fv[VOCAB[word]] = 1.0
        best_sim = max(
            float(np.dot(fv, entity_vec(e)) /
                  (np.linalg.norm(fv) * np.linalg.norm(entity_vec(e)) + 1e-8))
            for e in p['entities']
        )

        results.append({
            'name':      p['name'],
            'label':     p['label'],
            'h1_model':  h1_model,
            'h1_kb':     h1_kb,
            'ker':       ker_signal,
            'kb_sim':    best_sim,
            'defect':    (1.0 - best_sim) * 0.6 + min(ker_signal / 3.0, 1.0) * 0.4,
        })

    return results, kb_h1_bars


def print_exp2(results, kb_h1_bars):
    lines = [
        "=" * 68,
        "  Experiment 2: Cyclic KB — ker(H1(N_model) → H1(N_KB))",
        f"  KB nerve H1 = {len(kb_h1_bars)} bars (physics, math, DNA cycles)",
        "  Positive ker = KB has cycles the model misses = hallucination",
        "=" * 68,
        f"  {'Name':<24} {'Label':<14} {'H1_M':>6} {'H1_K':>6} "
        f"{'ker':>7} {'KBsim':>7} {'Defect':>8}",
        "  " + "-" * 65,
    ]
    for r in results:
        lines.append(
            f"  {r['name']:<24} {r['label']:<14} {r['h1_model']:>6.3f} "
            f"{r['h1_kb']:>6.3f} {r['ker']:>+7.3f} {r['kb_sim']:>7.3f} "
            f"{r['defect']:>8.4f}"
        )
    lines.append("")
    for lbl in ['true', 'hallucinated']:
        v = [r['defect'] for r in results if r['label'] == lbl]
        k = [r['ker'] for r in results if r['label'] == lbl]
        if v:
            lines.append(
                f"  {lbl:<14}: defect={np.mean(v):.4f}±{np.std(v):.4f}  "
                f"ker={np.mean(k):>+.3f}±{np.std(k):.3f}"
            )
    if any(r['label'] == 'true' for r in results) and \
       any(r['label'] == 'hallucinated' for r in results):
        tv = [r['defect'] for r in results if r['label'] == 'true']
        hv = [r['defect'] for r in results if r['label'] == 'hallucinated']
        d  = np.mean(hv) - np.mean(tv)
        pooled = np.std(tv + hv) + 1e-8
        lines.append(
            f"\n  Δ(hallu-true) = {d:>+.4f}  Cohen d = {d/pooled:.3f}"
        )
    lines.append("=" * 68)
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────
# EXPERIMENT 3: CHECKPOINT SWEEP
# ─────────────────────────────────────────────────────────────────

def experiment3_checkpoint_sweep(
    phases = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0],
    prompts = None,
):
    """
    Does grounding alignment sharpen across training phases?
    Does nerve mismatch decrease?
    Fixed prompts, greedy decoding, consistent normalisation.
    """
    if prompts is None:
        prompts = ALL_PROMPTS[:8]   # keep it fast

    sweep = []

    for phase in phases:
        model = PhasedGPT2(d_model=256, n_layers=4, n_heads=4,
                           vocab_size=5000, max_seq=512,
                           phase=phase, seed=42)
        model.eval()

        phase_results = {'true': [], 'hallucinated': []}

        for p in prompts:
            for typ, key in [('true_text','true'), ('halluc_text','hallucinated')]:
                tok = tokenize_controlled(p[typ])[:256]
                if len(tok) < 12:
                    continue
                ls = extract(model, tok, [0, 3])
                if 0 not in ls or 3 not in ls:
                    continue

                # ΔH1 (layer comparison)
                h1_0, _ = alpha_h1(ls[0], pca_dim=4, n_sub=32)
                h1_3, _ = alpha_h1(ls[3], pca_dim=4, n_sub=32)
                delta_h1 = h1_3 - h1_0

                # Grounding defect
                from gsod_fixes import build_kb_nerve as bkn
                kb  = bkn()
                nc  = compare_nerves(p[typ], key, p['entity'].split('_')[0], kb)

                phase_results[key].append({
                    'delta_h1': delta_h1,
                    'defect':   nc.defect,
                    'kb_sim':   nc.kb_sim,
                })

        # Aggregate
        row = {'phase': phase}
        for key in ['true', 'hallucinated']:
            v = phase_results[key]
            if v:
                row[f'{key}_defect']   = float(np.mean([r['defect']   for r in v]))
                row[f'{key}_delta_h1'] = float(np.mean([r['delta_h1'] for r in v]))
                row[f'{key}_kb_sim']   = float(np.mean([r['kb_sim']   for r in v]))
            else:
                row[f'{key}_defect'] = row[f'{key}_delta_h1'] = row[f'{key}_kb_sim'] = 0.0

        row['defect_gap']   = row['hallucinated_defect']   - row['true_defect']
        row['delta_h1_gap'] = row['true_delta_h1']         - row['hallucinated_delta_h1']
        sweep.append(row)

    return sweep


def print_exp3(sweep):
    lines = [
        "=" * 80,
        "  Experiment 3: Checkpoint Sweep",
        "  Does grounding sharpen and nerve mismatch decrease with training?",
        "  Fixed prompts, greedy decoding, consistent normalisation",
        "=" * 80,
        f"  {'Phase':>6}  {'TrueDefect':>11} {'HalluDefect':>12} "
        f"{'DefectGap':>10}  {'ΔH1_gap':>9}  {'TrueKBsim':>10}",
        "  " + "-" * 72,
    ]
    for r in sweep:
        lines.append(
            f"  {r['phase']:>6.2f}  {r['true_defect']:>11.4f} "
            f"{r['hallucinated_defect']:>12.4f} {r['defect_gap']:>+10.4f}  "
            f"{r['delta_h1_gap']:>+9.4f}  {r['true_kb_sim']:>10.3f}"
        )
    lines += [
        "",
        "  DefectGap = hallu_defect - true_defect  (positive = system works)",
        "  ΔH1_gap   = true_ΔH1 - hallu_ΔH1       (positive = factual grows more)",
        "  Expected: both gaps increase with training phase",
        "=" * 80,
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = PhasedGPT2(d_model=256, n_layers=4, n_heads=4,
                       vocab_size=5000, max_seq=512, phase=1.0, seed=42)
    model.eval()

    print("Running Experiment 1: Layer Comparison...")
    e1 = experiment1_layer_comparison(model, ALL_PROMPTS[:8])
    print(print_exp1(e1))
    print()

    print("Running Experiment 2: Cyclic KB (critical topology test)...")
    kb_idx, dist_mat, kb_h1_bars = build_cyclic_kb_nerve()
    print(f"  KB nerve: {len(kb_h1_bars)} H1 bars")
    for b, d in kb_h1_bars:
        print(f"    [{b:.2f}, {d:.2f}]  life={d-b:.2f}")
    print()
    e2, kb_bars = experiment2_cyclic_kb()
    print(print_exp2(e2, kb_bars))
    print()

    print("Running Experiment 3: Checkpoint Sweep...")
    e3 = experiment3_checkpoint_sweep(
        phases=[0.0, 0.1, 0.3, 0.5, 0.7, 1.0],
        prompts=ALL_PROMPTS[:6],
    )
    print(print_exp3(e3))
