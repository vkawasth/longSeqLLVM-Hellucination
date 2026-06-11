"""
christ_fukaya.py
================
Direct implementation of Christ [2510.05925] for the context algebra.

Three-way equivalence (Theorem 1.1):
  H_{G,I}  ≃  CoSing(G_{G,I})  ≃  Fuk(S, C_I)

Translation to context algebra:
  S        = contextual manifold swept by trajectory
  I        = Dynkin quiver of KB relation types
  ∂S marks = KB entity nodes
  G_{G,I}  = context dg category with transport operators
  Triangulation of S = windowing of the trajectory
  CoSing   = observable contextual states = D_perf / D_fin

Four operational components:
  1. Ice quiver from KB (mutable = interior relations, frozen = boundary entities)
  2. Amalgamation coboundary (homotopy pushout = Mayer-Vietoris connecting map)
  3. CoSing measurement (D_perf/D_fin ratio on hidden-state modules)
  4. Triangulation independence test (invariant stable across window sizes)
"""

import numpy as np
import gudhi
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Set
from collections import defaultdict
import warnings, sys, os
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))

from relation_nerve import (
    KB_ENTITIES, KB_RELATIONS, EIDX, DIST_MAT, REL_MAP,
    SYNTHETIC_DATASET, build_full_kb_nerve, nerve_h1_bars,
    full_relation_score, ParsedRelation
)


# ─────────────────────────────────────────────────────────────────
# 1. ICE QUIVER FROM KB
# ─────────────────────────────────────────────────────────────────
# Christ §4.5: The relative Ginzburg dg category of the ice quiver
# with potential encodes the 3-CY structure.
#
# Ice quiver structure:
#   Mutable vertices  = interior KB relations (we can compute on these)
#   Frozen vertices   = boundary entities (marked points on ∂S)
#   Arrows            = relation-to-relation transitions (cluster mutations)
#   Frozen arrows     = entity-to-relation assignments
#
# Cluster mutation at a mutable vertex v corresponds to:
#   contextual window transition T_{k,k+1} focused on entity v

@dataclass
class IceQuiver:
    """
    Ice quiver Q^ice = (Q_0, Q_1, F) where:
      Q_0 = all vertices (mutable + frozen)
      Q_1 = all arrows
      F ⊂ Q_0 = frozen vertices (boundary entities)

    For KB with entities E and relation types R:
      Mutable vertices = relation pairs (e_i --r--> e_j) in KB
      Frozen vertices  = entities e_i in KB
      Arrows           = when relation (i,j) shares an endpoint with (j,k),
                         there is an arrow (i,j) --> (j,k) in Q_1
    """
    mutable_vertices:  List[str]     # relation triples (src, tgt, rel)
    frozen_vertices:   List[str]     # entity names
    arrows:            List[Tuple[str, str]]  # (src_vertex, tgt_vertex)
    frozen_arrows:     List[Tuple[str, str]]  # (entity, relation_vertex)
    potential:         Dict[str, float]       # W: cycles -> R (superpotential)

    @property
    def n_mutable(self):
        return len(self.mutable_vertices)

    @property
    def n_frozen(self):
        return len(self.frozen_vertices)

    @property
    def n_vertices(self):
        return self.n_mutable + self.n_frozen


def build_ice_quiver(entities=None, relations=None) -> IceQuiver:
    """
    Build the ice quiver from the KB.

    Each KB relation (src, tgt, rel_type) becomes a mutable vertex.
    Each KB entity becomes a frozen vertex.
    Composition arrows connect compatible relations.

    The superpotential W is the sum over all oriented 3-cycles:
    for each (i,j), (j,k), (k,i) in the quiver, a term in W.
    This encodes the 3-CY structure (Christ §4.3).
    """
    if entities is None:
        entities = list(KB_ENTITIES.keys())
    if relations is None:
        relations = KB_RELATIONS

    # Mutable vertices: one per KB relation
    mut_verts = [f"{src}--{rel}-->{tgt}" for src, tgt, rel in relations]
    frz_verts = list(entities)

    # Arrows: relation (i,j) -> (j,k) when they share endpoint j
    arrows = []
    for i, (s1, t1, r1) in enumerate(relations):
        for j, (s2, t2, r2) in enumerate(relations):
            if i == j: continue
            if t1 == s2:   # composition: i->j, j->k gives arrow i->j --> j->k
                arrows.append((mut_verts[i], mut_verts[j]))

    # Frozen arrows: entity e -> relation (e, *, *) and (* , e, *) -> e
    frozen_arrows = []
    for e in entities:
        for s, t, r in relations:
            rel_v = f"{s}--{r}-->{t}"
            if s == e:
                frozen_arrows.append((e, rel_v))   # entity sources relation
            if t == e:
                frozen_arrows.append((rel_v, e))   # relation targets entity

    # Superpotential: sum over oriented 3-cycles in the mutable quiver
    # W = sum_{(i,j,k) cycle} a_{ij} a_{jk} a_{ki}
    potential = {}
    for (a1_s, a1_t) in arrows:
        for (a2_s, a2_t) in arrows:
            if a2_s != a1_t: continue
            for (a3_s, a3_t) in arrows:
                if a3_s != a2_t: continue
                if a3_t != a1_s: continue
                # Found oriented 3-cycle
                cycle_key = f"{a1_s}>{a1_t}>{a2_t}"
                potential[cycle_key] = potential.get(cycle_key, 0.0) + 1.0

    return IceQuiver(
        mutable_vertices=mut_verts,
        frozen_vertices=frz_verts,
        arrows=arrows,
        frozen_arrows=frozen_arrows,
        potential=potential,
    )


# ─────────────────────────────────────────────────────────────────
# 2. AMALGAMATION COBOUNDARY
# ─────────────────────────────────────────────────────────────────
# Christ Theorem A.8: When two ice quivers Q_1, Q_2 are glued along
# their frozen components F, the result is a homotopy pushout:
#
#   Q_1 ∩ Q_2  -->  Q_1
#       |              |
#       v              v
#      Q_2   -->  Q_1 ∪_F Q_2
#
# The connecting homomorphism of this pushout is the COBOUNDARY.
# ρ_R - ρ_L is the Mayer-Vietoris connecting map δ: H(Q_1 ∩ Q_2) -> H(Q_1 ∪ Q_2)
#
# Concrete computation:
# Two adjacent window windows W_a, W_b share a frozen boundary F = ∂W_a ∩ ∂W_b
# The amalgamation coboundary measures:
#   how much the local 3-CY structures on W_a and W_b disagree on F

@dataclass
class AmalgamationResult:
    """
    Result of amalgamating two ice quivers along shared frozen vertices.
    Corresponds to Christ Theorem A.8 (homotopy pushout).
    """
    shared_frozen:       List[str]    # F = frozen vertices in both
    coboundary_norm:     float        # ||ρ_R - ρ_L||: disagreement on F
    h1_pushout:          float        # H1 of the amalgamated complex
    is_consistent:       bool         # coboundary < threshold
    mayer_vietoris_rank: int          # rank of connecting homomorphism


def amalgamate(
    q1: IceQuiver,
    q2: IceQuiver,
    shared_entities: Optional[List[str]] = None,
) -> AmalgamationResult:
    """
    Compute the amalgamation coboundary between two ice quivers.

    Christ Appendix A: the amalgamation Q_1 ∪_F Q_2 is the homotopy pushout
    of Q_1 <-- F --> Q_2 where F = shared frozen vertices.

    The coboundary norm measures how incompatible Q_1 and Q_2 are on F.
    This is the categorical version of ρ_R - ρ_L:
      locally, each quiver has its own 3-CY structure;
      the amalgamation requires these to agree on the shared boundary;
      the coboundary measures the obstruction.
    """
    # Find shared frozen vertices
    f1 = set(q1.frozen_vertices)
    f2 = set(q2.frozen_vertices)
    if shared_entities is not None:
        F = [e for e in shared_entities if e in f1 and e in f2]
    else:
        F = list(f1 & f2)

    if not F:
        return AmalgamationResult(
            shared_frozen=[],
            coboundary_norm=float('inf'),
            h1_pushout=0.0,
            is_consistent=False,
            mayer_vietoris_rank=0,
        )

    # Mutable vertices visible from F in each quiver
    def visible_from(q: IceQuiver, entities: List[str]) -> Set[str]:
        """Relations incident to any entity in the given set."""
        visible = set()
        for fa in q.frozen_arrows:
            src, tgt = fa
            if src in entities:
                visible.add(tgt)
            if tgt in entities:
                visible.add(src)
        return visible & set(q.mutable_vertices)

    vis1 = visible_from(q1, F)
    vis2 = visible_from(q2, F)

    # Coboundary: how many relations incident to F differ between Q1 and Q2
    # In Q1: F sees relations with certain types
    # In Q2: F sees potentially different relations with different types
    # coboundary = |symmetric difference| / |union|
    union_vis = vis1 | vis2
    sym_diff   = vis1.symmetric_difference(vis2)

    if not union_vis:
        cob_norm = 0.0
    else:
        cob_norm = len(sym_diff) / len(union_vis)

    # Build the amalgamated nerve and compute H1
    all_verts = list(set(q1.mutable_vertices + q2.mutable_vertices + q1.frozen_vertices + q2.frozen_vertices))
    idx_map = {v: i for i, v in enumerate(all_verts)}
    n = len(all_verts)

    st = gudhi.SimplexTree()
    for i in range(n):
        st.insert([i], filtration=0.0)

    # Edges from both quivers' arrows
    all_arrows = set(map(tuple, q1.arrows + q2.arrows))
    for a, b in all_arrows:
        if a in idx_map and b in idx_map:
            # Shared arrows get filtration 0; non-shared get filtration 1
            is_shared = (a, b) in set(map(tuple, q1.arrows)) and \
                        (a, b) in set(map(tuple, q2.arrows))
            st.insert([idx_map[a], idx_map[b]], filtration=0.0 if is_shared else 1.0)

    # Triangles from potential cycles
    for q in [q1, q2]:
        for cycle_key in q.potential:
            parts = cycle_key.split('>')
            if len(parts) == 3 and all(p in idx_map for p in parts):
                i1, i2, i3 = idx_map[parts[0]], idx_map[parts[1]], idx_map[parts[2]]
                st.insert([i1, i2, i3], filtration=1.0)

    st.compute_persistence()
    h1_raw = st.persistence_intervals_in_dimension(1)
    h1_total = sum(d-b for b,d in h1_raw if np.isfinite(d) and d-b > 0.1)

    # Mayer-Vietoris rank = rank of connecting map δ
    # Approximated by number of shared frozen vertices with different relation profiles
    mv_rank = sum(
        1 for e in F
        if visible_from(q1, [e]) != visible_from(q2, [e])
    )

    return AmalgamationResult(
        shared_frozen=F,
        coboundary_norm=cob_norm,
        h1_pushout=h1_total,
        is_consistent=cob_norm < 0.3,
        mayer_vietoris_rank=mv_rank,
    )


# ─────────────────────────────────────────────────────────────────
# 3. COSINGULARITY MEASUREMENT
# ─────────────────────────────────────────────────────────────────
# Christ §1.3: CoSing(G) = D_perf(G) / D_fin(G)
# D_fin = finite-dimensional modules = collapsed / degenerate modules
# D_perf = perfect complexes = all computable states
#
# For hidden states: model a "module over the transport algebra"
# as the hidden state covariance C_k = H_k^T H_k / n
# Perfect = full rank (carries information)
# Finite = rank-deficient (degenerated, effectively zero-dimensional)
#
# CoSing measurement:
#   CoSing_score = (n_perfect - n_finite) / n_total
#   high CoSing = many informative states (grounded trajectory)
#   low CoSing  = many degenerate states (collapsed / hallucinated)

@dataclass
class CoSingMeasurement:
    """
    D_perf / D_fin measurement on hidden states.
    Implements the cosingularity category quotient.
    """
    n_windows:       int
    n_perfect:       int     # rank = d (non-degenerate)
    n_finite:        int     # rank << d (degenerate)
    cosing_score:    float   # (n_perfect - n_finite) / n_total
    mean_rank:       float
    rank_profile:    List[int]  # per-window effective rank


def measure_cosing(
    hidden_states:   np.ndarray,   # [n_tokens, d]
    window_size:     int = 32,
    rank_threshold:  float = 0.5,  # fraction of POSSIBLE rank (not absolute d)
    regularise:      float = 1e-4,
) -> CoSingMeasurement:
    """
    Measure the CoSing score of a hidden state trajectory.

    Window i is "perfect" if its effective rank r_i ~ min(n_i, d) (full rank).
    Window i is "finite" if r_i << min(n_i, d) (degenerate, collapsed).

    The CoSing score = fraction of windows in D_perf \\ D_fin.
    High score = trajectory is in the observable (non-degenerate) region.
    Low score = trajectory has collapsed to D_fin.
    """
    n = len(hidden_states)
    d = hidden_states.shape[1]
    wins = [hidden_states[k*window_size:(k+1)*window_size]
            for k in range(n // window_size)]

    if not wins:
        return CoSingMeasurement(0, 0, 0, 0.0, 0.0, [])

    rank_profile = []
    for w in wins:
        if len(w) < 2:
            rank_profile.append(0)
            continue
        max_rank = min(len(w), d)
        h = w - w.mean(0)
        C = h.T @ h / max(len(h)-1, 1) + regularise * np.eye(d)
        sv = np.linalg.svd(C, compute_uv=False)
        # Effective rank relative to maximum possible rank
        eff_rank = int(np.sum(sv > sv[0] * 0.1))  # 10% of max singular value
        rank_profile.append(eff_rank)

    n_w = len(rank_profile)
    # Perfect: effective rank >= 60% of maximum possible
    n_perfect = sum(
        1 for i, r in enumerate(rank_profile)
        if r >= 0.6 * min(window_size, d)
    )
    # Finite: effective rank <= 20% of maximum possible
    n_finite  = sum(
        1 for i, r in enumerate(rank_profile)
        if r <= 0.2 * min(window_size, d)
    )

    cosing = (n_perfect - n_finite) / max(n_w, 1)

    return CoSingMeasurement(
        n_windows    = n_w,
        n_perfect    = n_perfect,
        n_finite     = n_finite,
        cosing_score = cosing,
        mean_rank    = float(np.mean(rank_profile)),
        rank_profile = rank_profile,
    )


# ─────────────────────────────────────────────────────────────────
# 4. TRIANGULATION INDEPENDENCE TEST
# ─────────────────────────────────────────────────────────────────
# Christ Theorem 1.4: R^1Γ(G, F_{G,I}) is independent of the
# choice of ideal triangulation (i.e., window size).
#
# Test: compute our invariants at multiple window sizes and
# check stability. If the invariant is genuinely topological,
# it should not change with window size.

@dataclass
class TriangulationIndependenceResult:
    """
    Tests whether our invariants are triangulation-independent.
    A topologically correct invariant should be stable across window sizes.
    """
    window_sizes:    List[int]
    scores:          Dict[int, float]   # window_size -> combined score
    is_stable:       bool               # std/mean < 0.2
    mean_score:      float
    std_score:       float


def test_triangulation_independence(
    text_entities:   List[str],
    asserted_rels:   List[ParsedRelation],
    window_sizes:    List[int] = [4, 6, 8, 10, 12],
    hidden_states:   Optional[np.ndarray] = None,
) -> TriangulationIndependenceResult:
    """
    Test Christ Theorem 1.4: triangulation independence.

    Compute full_relation_score (our topological invariant) at different
    window sizes and check whether it is stable.

    For the nerve-based score: it doesn't depend on window size at all
    (we compute on full text, not windowed). Stability is automatic.

    For the CoSing score on hidden states: it depends on window size.
    Stability here would confirm that the CoSing measurement is probing
    a genuine triangulation-independent property of the trajectory.
    """
    scores = {}

    # Nerve score: window-independent by construction
    rel_score = full_relation_score(text_entities, asserted_rels)
    for ws in window_sizes:
        scores[ws] = rel_score['combined']

    # CoSing score: depends on window size
    cosing_scores = {}
    if hidden_states is not None:
        for ws in window_sizes:
            if ws * 3 > len(hidden_states):
                continue
            cm = measure_cosing(hidden_states, window_size=ws)
            cosing_scores[ws] = cm.cosing_score

    if cosing_scores:
        vals = list(cosing_scores.values())
        mean_v = float(np.mean(vals))
        std_v  = float(np.std(vals))
        stable = std_v < 0.2 * abs(mean_v) + 1e-6
        return TriangulationIndependenceResult(
            window_sizes=window_sizes,
            scores=cosing_scores,
            is_stable=stable,
            mean_score=mean_v,
            std_score=std_v,
        )

    # Nerve-only: stable by construction
    return TriangulationIndependenceResult(
        window_sizes=window_sizes,
        scores=scores,
        is_stable=True,
        mean_score=float(np.mean(list(scores.values()))),
        std_score=0.0,
    )


# ─────────────────────────────────────────────────────────────────
# FULL PIPELINE: Christ framework applied to KB + text
# ─────────────────────────────────────────────────────────────────

@dataclass
class ChristAnalysis:
    """
    Full Christ [2510.05925] analysis of a text against the KB.

    Three-way equivalence:
      ice_quiver     = H_{G,I}  (cluster tilting / Higgs category)
      amalgamation   = G_{G,I}  (relative 3-CY / coboundary computation)
      cosing         = CoSing   (observable contextual states)
      fuk_score      = Fuk(S,C_I) (topological Fukaya = our nerve score)
    """
    text_label:      str
    ice_quiver:      IceQuiver
    amalgamation:    AmalgamationResult
    cosing:          Optional[CoSingMeasurement]
    fuk_score:       Dict         # full_relation_score output
    triang_indep:    TriangulationIndependenceResult

    @property
    def hallucination_score(self) -> float:
        """
        Combined score from all three layers.
        High = contextual degeneration (hallucination / wrong relations).
        """
        # Fukaya layer (ker + Dehn gap) -- from relation nerve
        fuk = self.fuk_score.get('combined', 0.0)
        # Amalgamation layer (coboundary norm)
        amal = self.amalgamation.coboundary_norm
        # CoSing layer (collapsed windows)
        cosing_penalty = (1.0 - self.cosing.cosing_score) if self.cosing else 0.0
        return 0.5 * fuk + 0.3 * amal + 0.2 * cosing_penalty


def analyse(
    text:           str,
    entities:       List[str],
    asserted_rels:  List[ParsedRelation],
    hidden_states:  Optional[np.ndarray] = None,
    window_size:    int = 32,
) -> ChristAnalysis:
    """
    Full Christ framework analysis.
    """
    from relation_nerve import build_full_kb_nerve

    # 1. Ice quiver from KB restricted to mentioned entities
    iq = build_ice_quiver(
        entities=entities,
        relations=[r for r in KB_RELATIONS if r[0] in entities and r[1] in entities]
    )

    # 2. Amalgamation: split entities into two halves, compute coboundary
    mid = len(entities) // 2
    ents_a, ents_b = entities[:mid+1], entities[mid:]
    iq_a = build_ice_quiver(
        entities=ents_a,
        relations=[r for r in KB_RELATIONS if r[0] in ents_a and r[1] in ents_a]
    )
    iq_b = build_ice_quiver(
        entities=ents_b,
        relations=[r for r in KB_RELATIONS if r[0] in ents_b and r[1] in ents_b]
    )
    shared = list(set(ents_a) & set(ents_b))
    amal = amalgamate(iq_a, iq_b, shared_entities=shared if shared else None)

    # 3. CoSing measurement
    cosing = measure_cosing(hidden_states, window_size=window_size) \
             if hidden_states is not None else None

    # 4. Fukaya score (our relation nerve)
    fuk = full_relation_score(entities, asserted_rels)

    # 5. Triangulation independence
    ti = test_triangulation_independence(
        entities, asserted_rels,
        hidden_states=hidden_states,
    )

    return ChristAnalysis(
        text_label=text[:40],
        ice_quiver=iq,
        amalgamation=amal,
        cosing=cosing,
        fuk_score=fuk,
        triang_indep=ti,
    )


# ─────────────────────────────────────────────────────────────────
# MAIN EXPERIMENT
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import torch
    from factscore_alignment import PhasedGPT2
    from hallucination_topology_pipeline import tokenize_controlled

    model = PhasedGPT2(d_model=256, n_layers=12, n_heads=8,
                       vocab_size=5000, max_seq=512, phase=1.0, seed=42)
    model.eval()

    def get_hs(text, layer=11):
        model.register_hooks()
        tok = tokenize_controlled(text)[:256]
        with torch.no_grad():
            model(torch.tensor([tok]))
        h = model._hs_by_layer.get(layer)
        model.remove_hooks()
        if h is None: return None
        return h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-8)

    # Show ice quiver structure
    print("=" * 68)
    print("  Christ Framework: Ice Quiver + Amalgamation + CoSing + Fuk")
    print("=" * 68)
    print()

    iq = build_ice_quiver()
    print(f"Full KB ice quiver:")
    print(f"  Mutable vertices: {iq.n_mutable}  (KB relations)")
    print(f"  Frozen vertices:  {iq.n_frozen}   (KB entities)")
    print(f"  Arrows:           {len(iq.arrows)}")
    print(f"  Frozen arrows:    {len(iq.frozen_arrows)}")
    print(f"  Potential cycles: {len(iq.potential)}")
    print()

    # Run on synthetic dataset
    print(f"{'Sample':<28} {'Label':<12} {'Fuk':>6} {'Amal':>6} "
          f"{'CoSing':>7} {'Total':>7}")
    print("-" * 68)

    scores = {'correct': [], 'wrong': []}
    for sample in SYNTHETIC_DATASET:
        hs = get_hs(sample['text'])
        result = analyse(
            text=sample['text'],
            entities=sample['entities'],
            asserted_rels=sample['asserted'],
            hidden_states=hs,
            window_size=4,
        )
        total = result.hallucination_score
        is_correct = sample['label'] == 'correct'
        scores['correct' if is_correct else 'wrong'].append(total)
        print(f"{sample['id']:<28} {sample['label']:<12} "
              f"{result.fuk_score['combined']:>6.3f} "
              f"{result.amalgamation.coboundary_norm:>6.3f} "
              f"{result.cosing.cosing_score if result.cosing else 0:>7.3f} "
              f"{total:>7.4f}")

    print()
    for key in ['correct', 'wrong']:
        v = scores[key]
        if v:
            print(f"  {key:<8}: mean={np.mean(v):.4f}  std={np.std(v):.4f}")
    if scores['correct'] and scores['wrong']:
        d = np.mean(scores['wrong']) - np.mean(scores['correct'])
        pooled = np.std(scores['correct'] + scores['wrong']) + 1e-8
        print(f"  Cohen d = {d/pooled:.3f}")

    # Triangulation independence check
    print()
    print("Triangulation independence (Theorem 1.4 check):")
    sample = SYNTHETIC_DATASET[0]  # physics_A_correct
    hs = get_hs(sample['text'])
    ti = test_triangulation_independence(
        sample['entities'], sample['asserted'],
        window_sizes=[4, 6, 8, 10],
        hidden_states=hs,
    )
    print(f"  Window sizes: {ti.window_sizes}")
    print(f"  CoSing scores: {[round(ti.scores.get(ws,0),4) for ws in ti.window_sizes]}")
    print(f"  Stable: {ti.is_stable}  (std/mean = {ti.std_score/(ti.mean_score+1e-8):.3f})")
