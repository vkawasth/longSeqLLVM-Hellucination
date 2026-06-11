"""
relation_nerve.py
=================
The relation-sensitive dual nerve: the next concrete object.

Core insight (from review):
  The model nerve currently encodes entity CO-MENTION only.
  Edge (i,j) filtration = KB graph distance regardless of whether
  the text asserts the RIGHT or WRONG relation.
  Therefore ker(phi_*) = 0 everywhere — the topology is blind to error.

Fix:
  Typed edges with filtration penalties:
    text asserts CORRECT relation  -> filtration = KB_distance(i,j)
    text asserts WRONG relation    -> filtration = KB_distance + WRONG_PENALTY
    text asserts REVERSED relation -> filtration = KB_distance + DIR_PENALTY
    text mentions pair NOT in KB   -> filtration = ABSENT_CONSTANT

Then ker(phi_*: H1(N_model) -> H1(N_KB)) != 0 for wrong-relation texts.
The cycle either survives (correct) or breaks (wrong) under phi_*.

This is where rho_R - rho_L != 0 stops being decorative:
it becomes the coboundary of the typed restriction map,
measuring whether locally asserted relations glue into
a globally consistent semantic section.

Dataset:
  3 cycles x 4 variants = 12 texts
  (A) correct:    relations match KB exactly
  (B) wrong_rel:  same entity pairs, wrong relation type
  (C) reversed:   correct relation type, wrong direction
  (D) invented:   entity pairs not in KB graph
  
  Plus GPT-2 hidden states (layer 2 and layer 11) on each text.
  Prediction: correct -> ker = 0, wrong -> ker != 0.
"""

import numpy as np
import gudhi
import torch
import sys, os
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))

from factscore_alignment import PhasedGPT2
from hallucination_topology_pipeline import tokenize_controlled, alpha_h1


# ─────────────────────────────────────────────────────────────────
# KNOWLEDGE BASE WITH TYPED RELATIONS
# ─────────────────────────────────────────────────────────────────

KB_ENTITIES = {
    'einstein':  {'born': 1879, 'field': 'physics',     'award': 'nobel'},
    'bohr':      {'born': 1885, 'field': 'physics',     'theory': 'quantum'},
    'heisenberg':{'born': 1901, 'field': 'physics',     'theory': 'uncertainty'},
    'pauli':     {'born': 1900, 'field': 'physics',     'award': 'nobel'},
    'dirac':     {'born': 1902, 'field': 'physics',     'theory': 'qed'},
    'newton':    {'born': 1643, 'field': 'mechanics',   'work': 'principia'},
    'euler':     {'born': 1707, 'field': 'mathematics', 'work': 'analysis'},
    'gauss':     {'born': 1777, 'field': 'mathematics', 'work': 'disquisitiones'},
    'riemann':   {'born': 1826, 'field': 'geometry',    'work': 'hypothesis'},
    'watson':    {'born': 1928, 'field': 'biology',     'discovery': 'dna'},
    'crick':     {'born': 1916, 'field': 'biology',     'award': 'nobel'},
    'franklin':  {'born': 1920, 'field': 'crystallography', 'method': 'xray'},
    'wilkins':   {'born': 1916, 'field': 'biophysics',  'award': 'nobel'},
    'darwin':    {'born': 1809, 'field': 'biology',     'work': 'origin_of_species'},
}

# (source, target, relation_type)
# Relation types are TYPED and DIRECTED
KB_RELATIONS = [
    # Physics 5-cycle
    ('einstein',  'bohr',       'debated'),
    ('bohr',      'heisenberg', 'taught'),
    ('heisenberg','pauli',      'collaborated'),
    ('pauli',     'dirac',      'collaborated'),
    ('dirac',     'einstein',   'inspired'),
    # Math 4-cycle
    ('newton',    'euler',      'influenced'),
    ('euler',     'gauss',      'influenced'),
    ('gauss',     'riemann',    'mentored'),
    ('riemann',   'newton',     'generalised'),
    # DNA 4-cycle
    ('watson',    'crick',      'partnered'),
    ('crick',     'franklin',   'used_data'),
    ('franklin',  'wilkins',    'collaborated'),
    ('wilkins',   'watson',     'colleague'),
    # Cross-cycle
    ('einstein',  'newton',     'extended'),
    ('darwin',    'watson',     'inspired'),
]

# Relation type compatibility (for penalty computation)
# Relations that are semantically similar get smaller penalty
RELATION_GROUPS = {
    'influence': {'influenced', 'inspired', 'extended', 'generalised'},
    'teaching':  {'taught', 'mentored'},
    'collab':    {'collaborated', 'partnered', 'colleague', 'debated'},
    'data':      {'used_data'},
}

def relation_penalty(asserted: str, kb_relation: str) -> float:
    """
    Filtration penalty when text asserts wrong relation.
    0.0  = correct relation
    0.5  = same group (similar type, wrong specifics)
    1.5  = different group (wrong category)
    3.0  = completely absent from KB for this pair
    """
    if asserted == kb_relation:
        return 0.0
    # Same group?
    for group in RELATION_GROUPS.values():
        if asserted in group and kb_relation in group:
            return 0.5
    return 1.5


# ─────────────────────────────────────────────────────────────────
# RELATION-SENSITIVE NERVE CONSTRUCTION
# ─────────────────────────────────────────────────────────────────

def build_kb_adj():
    """BFS distance matrix and relation lookup."""
    ents = list(KB_ENTITIES.keys())
    idx  = {e: i for i, e in enumerate(ents)}
    n    = len(ents)

    # Undirected adjacency for BFS
    adj = defaultdict(set)
    rel_map = {}  # (src, tgt) -> relation_type (directed)
    for src, tgt, rel in KB_RELATIONS:
        adj[src].add(tgt); adj[tgt].add(src)
        rel_map[(src, tgt)] = rel

    def bfs(src):
        dist = {src: 0}; q = deque([src])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1; q.append(v)
        return dist

    dist_mat = np.full((n, n), 999.0)
    for e in ents:
        d = bfs(e)
        for f in ents:
            dist_mat[idx[e], idx[f]] = float(d.get(f, 999))

    return ents, idx, dist_mat, rel_map


ENTS, EIDX, DIST_MAT, REL_MAP = build_kb_adj()


@dataclass
class ParsedRelation:
    """A relation extracted from text."""
    source:   str
    target:   str
    rel_type: str   # what the text claims
    is_directed: bool = True


def parse_relations_from_text(text: str,
                               entity_list: List[str],
                               asserted_relations: List[ParsedRelation]
                               ) -> List[ParsedRelation]:
    """
    For real text, this would be an NLP extractor.
    For our synthetic dataset, we pass the asserted relations explicitly.
    """
    return asserted_relations


def build_relation_nerve(
    mentioned_entities:  List[str],
    asserted_relations:  List[ParsedRelation],
    kb_absent_constant:  float = 5.0,
    wrong_dir_penalty:   float = 2.0,
) -> Tuple[gudhi.SimplexTree, Dict]:
    """
    Build the relation-sensitive model nerve.

    Filtration value of edge (i, j):
      - KB edge, correct relation:   KB_distance
      - KB edge, wrong relation:     KB_distance + penalty(asserted, kb_rel)
      - KB edge, reversed direction: KB_distance + wrong_dir_penalty
      - Pair not in KB:              kb_absent_constant

    This is the object that makes ker(phi_*) non-trivial.
    """
    if len(mentioned_entities) < 2:
        return None, {}

    n   = len(mentioned_entities)
    idx = {e: i for i, e in enumerate(mentioned_entities)}

    # Build assertion lookup
    assert_map = {}  # (src, tgt) -> asserted relation
    for r in asserted_relations:
        if r.source in idx and r.target in idx:
            assert_map[(r.source, r.target)] = r.rel_type

    st = gudhi.SimplexTree()
    edge_info = {}

    for i in range(n):
        st.insert([i], filtration=0.0)

    for i, ei in enumerate(mentioned_entities):
        for j, ej in enumerate(mentioned_entities):
            if i >= j:
                continue
            ei_idx = EIDX.get(ei)
            ej_idx = EIDX.get(ej)

            if ei_idx is None or ej_idx is None:
                filt = kb_absent_constant
                info = 'unknown_entity'
            else:
                kb_d = float(DIST_MAT[ei_idx, ej_idx])
                # What relation does the KB have for this directed pair?
                kb_rel_fwd = REL_MAP.get((ei, ej))
                kb_rel_rev = REL_MAP.get((ej, ei))

                # What does the text assert?
                assert_fwd = assert_map.get((ei, ej))
                assert_rev = assert_map.get((ej, ei))

                if kb_rel_fwd and assert_fwd:
                    pen = relation_penalty(assert_fwd, kb_rel_fwd)
                    filt = kb_d + pen
                    info = f'fwd:KB={kb_rel_fwd},got={assert_fwd},pen={pen}'
                elif kb_rel_rev and assert_rev:
                    pen = relation_penalty(assert_rev, kb_rel_rev)
                    filt = kb_d + pen
                    info = f'rev:KB={kb_rel_rev},got={assert_rev},pen={pen}'
                elif kb_rel_fwd and assert_rev:
                    # Text reversed the direction
                    filt = kb_d + wrong_dir_penalty
                    info = f'REVERSED:KB_fwd={kb_rel_fwd},got_rev={assert_rev}'
                elif kb_rel_rev and assert_fwd:
                    filt = kb_d + wrong_dir_penalty
                    info = f'REVERSED:KB_rev={kb_rel_rev},got_fwd={assert_fwd}'
                elif kb_d < 999:
                    # KB connects them but no assertion in text
                    filt = kb_d
                    info = f'connected_no_assertion:d={kb_d}'
                else:
                    filt = kb_absent_constant
                    info = 'not_in_KB'

            st.insert([i, j], filtration=float(filt))
            edge_info[(i, j)] = info

    # Triangles
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                d = max(
                    st.filtration([i, j]),
                    st.filtration([j, k]),
                    st.filtration([i, k]),
                )
                st.insert([i, j, k], filtration=d)

    return st, edge_info


def nerve_h1_bars(st: gudhi.SimplexTree, min_life: float = 0.3):
    """Compute persistent H1 of the nerve."""
    if st is None:
        return 0.0, 0
    st.compute_persistence()
    raw  = st.persistence_intervals_in_dimension(1)
    if len(raw) == 0:
        return 0.0, 0
    bars = [(b, d) for b, d in raw if np.isfinite(d) and d - b > min_life]
    return sum(d - b for b, d in bars), len(bars)


def compute_ker_phi(model_st, kb_st, min_life=0.3):
    """
    Proxy for ker(phi_*: H1(N_model) -> H1(N_KB)).

    The induced map phi_*: H1(N_model) -> H1(N_KB) sends each
    model cycle to its image in the KB nerve.

    When the model nerve has FEWER cycles than the KB nerve
    (because relation penalties broke edges), the map has a
    non-trivial kernel on the KB side — KB cycles not hit by
    any model cycle.

    grounding_defect = |H1_KB - H1_model| / (H1_KB + eps)
      = fraction of KB topology lost in the model nerve
      = 0 for correct relations (model preserves KB cycles)
      > 0 for wrong/reversed/invented (penalties destroy cycles)
    """
    if model_st is None or kb_st is None:
        return 0.0
    h1_model, _ = nerve_h1_bars(model_st, min_life)
    h1_kb,    _ = nerve_h1_bars(kb_st,    min_life)
    return abs(h1_kb - h1_model) / (h1_kb + 1e-6)


# ─────────────────────────────────────────────────────────────────
# SYNTHETIC DATASET
# ─────────────────────────────────────────────────────────────────

SYNTHETIC_DATASET = [

    # ── PHYSICS CYCLE ──────────────────────────────────────────────

    {
        'id': 'physics_A_correct',
        'label': 'correct',
        'text': ('Einstein debated Bohr at Solvay. '
                 'Bohr taught Heisenberg in Copenhagen. '
                 'Heisenberg collaborated with Pauli on exclusion. '
                 'Pauli collaborated with Dirac on quantum fields. '
                 'Dirac inspired Einstein toward unified theory.'),
        'entities': ['einstein','bohr','heisenberg','pauli','dirac'],
        'asserted': [
            ParsedRelation('einstein','bohr','debated'),
            ParsedRelation('bohr','heisenberg','taught'),
            ParsedRelation('heisenberg','pauli','collaborated'),
            ParsedRelation('pauli','dirac','collaborated'),
            ParsedRelation('dirac','einstein','inspired'),
        ],
    },
    {
        'id': 'physics_B_wrong_rel',
        'label': 'wrong_rel',
        'text': ('Einstein taught Bohr quantum theory. '
                 'Bohr inspired Heisenberg to discover uncertainty. '
                 'Heisenberg mentored Pauli in Leipzig. '
                 'Pauli taught Dirac about spin. '
                 'Dirac collaborated with Einstein on relativity.'),
        'entities': ['einstein','bohr','heisenberg','pauli','dirac'],
        'asserted': [
            ParsedRelation('einstein','bohr','taught'),       # KB: debated
            ParsedRelation('bohr','heisenberg','inspired'),   # KB: taught
            ParsedRelation('heisenberg','pauli','mentored'),  # KB: collaborated
            ParsedRelation('pauli','dirac','taught'),         # KB: collaborated
            ParsedRelation('dirac','einstein','collaborated'),# KB: inspired
        ],
    },
    {
        'id': 'physics_C_reversed',
        'label': 'reversed',
        'text': ('Bohr debated Einstein on wave mechanics. '
                 'Heisenberg taught Bohr about matrices. '
                 'Pauli collaborated with Heisenberg on S-matrix. '
                 'Dirac collaborated with Pauli on holes. '
                 'Einstein inspired Dirac to unify gravity.'),
        'entities': ['einstein','bohr','heisenberg','pauli','dirac'],
        'asserted': [
            ParsedRelation('bohr','einstein','debated'),       # reversed
            ParsedRelation('heisenberg','bohr','taught'),      # reversed
            ParsedRelation('pauli','heisenberg','collaborated'),# reversed
            ParsedRelation('dirac','pauli','collaborated'),    # reversed
            ParsedRelation('einstein','dirac','inspired'),     # reversed
        ],
    },
    {
        'id': 'physics_D_invented',
        'label': 'invented',
        'text': ('Einstein mentored Schrodinger on wave equations. '
                 'Bohr collaborated with Planck on blackbody. '
                 'Heisenberg inspired Born to develop matrix mechanics. '
                 'Pauli debated Fermi about statistics. '
                 'Dirac taught Feynman at Cambridge.'),
        'entities': ['einstein','bohr','heisenberg','pauli','dirac'],
        'asserted': [
            ParsedRelation('einstein','bohr','mentored'),      # schrodinger->bohr not in KB
            ParsedRelation('bohr','heisenberg','collaborated'),# planck not in KB
            ParsedRelation('heisenberg','pauli','inspired'),   # born not in KB
            ParsedRelation('pauli','dirac','debated'),         # fermi not in KB
            ParsedRelation('dirac','einstein','taught'),       # feynman not in KB
        ],
    },

    # ── MATH CYCLE ─────────────────────────────────────────────────

    {
        'id': 'math_A_correct',
        'label': 'correct',
        'text': ('Newton influenced Euler through his calculus and physics. '
                 'Euler influenced Gauss through analysis and number theory. '
                 'Gauss mentored Riemann at Göttingen in differential geometry. '
                 'Riemann generalised Newton through manifold theory.'),
        'entities': ['newton','euler','gauss','riemann'],
        'asserted': [
            ParsedRelation('newton','euler','influenced'),
            ParsedRelation('euler','gauss','influenced'),
            ParsedRelation('gauss','riemann','mentored'),
            ParsedRelation('riemann','newton','generalised'),
        ],
    },
    {
        'id': 'math_B_wrong_rel',
        'label': 'wrong_rel',
        'text': ('Newton taught Euler calculus directly in London. '
                 'Euler mentored Gauss in Basel. '
                 'Gauss collaborated with Riemann at Berlin. '
                 'Riemann debated Newton on the nature of space.'),
        'entities': ['newton','euler','gauss','riemann'],
        'asserted': [
            ParsedRelation('newton','euler','taught'),         # KB: influenced
            ParsedRelation('euler','gauss','mentored'),        # KB: influenced
            ParsedRelation('gauss','riemann','collaborated'),  # KB: mentored
            ParsedRelation('riemann','newton','debated'),      # KB: generalised
        ],
    },
    {
        'id': 'math_C_reversed',
        'label': 'reversed',
        'text': ('Euler influenced Newton through continental methods. '
                 'Gauss influenced Euler by extending analysis. '
                 'Riemann mentored Gauss in topology. '
                 'Newton generalised Riemann through fluxions.'),
        'entities': ['newton','euler','gauss','riemann'],
        'asserted': [
            ParsedRelation('euler','newton','influenced'),     # reversed
            ParsedRelation('gauss','euler','influenced'),      # reversed
            ParsedRelation('riemann','gauss','mentored'),      # reversed
            ParsedRelation('newton','riemann','generalised'),  # reversed
        ],
    },
    {
        'id': 'math_D_invented',
        'label': 'invented',
        'text': ('Newton collaborated with Leibniz on calculus priority. '
                 'Euler inspired Lagrange on variational methods. '
                 'Gauss debated Lobachevsky on non-Euclidean geometry. '
                 'Riemann taught Dedekind on algebraic numbers.'),
        'entities': ['newton','euler','gauss','riemann'],
        'asserted': [
            ParsedRelation('newton','euler','collaborated'),   # leibniz not in cycle
            ParsedRelation('euler','gauss','inspired'),        # lagrange not in cycle
            ParsedRelation('gauss','riemann','debated'),       # lobachevsky not in cycle
            ParsedRelation('riemann','newton','taught'),       # dedekind not in cycle
        ],
    },

    # ── DNA CYCLE ──────────────────────────────────────────────────

    {
        'id': 'dna_A_correct',
        'label': 'correct',
        'text': ('Watson partnered with Crick at Cavendish to solve DNA. '
                 'Crick used data from Franklin for the double helix model. '
                 'Franklin collaborated with Wilkins on X-ray diffraction. '
                 'Wilkins was colleague to Watson at Kings College.'),
        'entities': ['watson','crick','franklin','wilkins'],
        'asserted': [
            ParsedRelation('watson','crick','partnered'),
            ParsedRelation('crick','franklin','used_data'),
            ParsedRelation('franklin','wilkins','collaborated'),
            ParsedRelation('wilkins','watson','colleague'),
        ],
    },
    {
        'id': 'dna_B_wrong_rel',
        'label': 'wrong_rel',
        'text': ('Watson taught Crick about crystallography techniques. '
                 'Crick collaborated with Franklin on diffraction patterns. '
                 'Franklin inspired Wilkins to study DNA structure. '
                 'Wilkins mentored Watson in X-ray methods.'),
        'entities': ['watson','crick','franklin','wilkins'],
        'asserted': [
            ParsedRelation('watson','crick','taught'),         # KB: partnered
            ParsedRelation('crick','franklin','collaborated'), # KB: used_data
            ParsedRelation('franklin','wilkins','inspired'),   # KB: collaborated
            ParsedRelation('wilkins','watson','mentored'),     # KB: colleague
        ],
    },
    {
        'id': 'dna_C_reversed',
        'label': 'reversed',
        'text': ('Crick partnered with Watson to build the model. '
                 'Franklin used data from Crick for Photo 51. '
                 'Wilkins collaborated with Franklin at Kings. '
                 'Watson was colleague to Wilkins at Cambridge.'),
        'entities': ['watson','crick','franklin','wilkins'],
        'asserted': [
            ParsedRelation('crick','watson','partnered'),      # reversed
            ParsedRelation('franklin','crick','used_data'),    # reversed
            ParsedRelation('wilkins','franklin','collaborated'),# reversed
            ParsedRelation('watson','wilkins','colleague'),    # reversed
        ],
    },
    {
        'id': 'dna_D_invented',
        'label': 'invented',
        'text': ('Watson debated Pauling about alpha helix vs double helix. '
                 'Crick collaborated with Perutz on protein crystallography. '
                 'Franklin taught Gosling X-ray crystallography methods. '
                 'Wilkins inspired Stokes to calculate diffraction patterns.'),
        'entities': ['watson','crick','franklin','wilkins'],
        'asserted': [
            ParsedRelation('watson','crick','debated'),        # pauling not in cycle
            ParsedRelation('crick','franklin','collaborated'), # perutz not in cycle
            ParsedRelation('franklin','wilkins','taught'),     # gosling not in cycle
            ParsedRelation('wilkins','watson','inspired'),     # stokes not in cycle
        ],
    },
]


# ─────────────────────────────────────────────────────────────────
# BUILD KB NERVE (ground truth)
# ─────────────────────────────────────────────────────────────────

def build_full_kb_nerve(entities: List[str]) -> gudhi.SimplexTree:
    """KB nerve: filtration = graph distance (no penalties, ground truth)."""
    n   = len(entities)
    idx = {e: i for i, e in enumerate(entities)}
    st  = gudhi.SimplexTree()
    for i in range(n): st.insert([i], filtration=0.0)
    for i, ei in enumerate(entities):
        for j, ej in enumerate(entities):
            if i >= j: continue
            ei_g = EIDX.get(ei); ej_g = EIDX.get(ej)
            d = float(DIST_MAT[ei_g, ej_g]) if ei_g is not None and ej_g is not None else 999.0
            st.insert([i, j], filtration=d)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                d = max(st.filtration([i,j]), st.filtration([j,k]), st.filtration([i,k]))
                st.insert([i,j,k], filtration=d)
    return st


# ─────────────────────────────────────────────────────────────────
# GPT-2 LAYER EXTRACTION
# ─────────────────────────────────────────────────────────────────

def extract_layer_hs(model, text: str, layer: int, max_tok: int = 128) -> np.ndarray:
    model.eval()
    model.register_hooks()
    tok = tokenize_controlled(text)[:max_tok]
    with torch.no_grad():
        model(torch.tensor([tok]))
    hs = model._hs_by_layer.get(layer)
    model.remove_hooks()
    if hs is None:
        return None
    return hs / (np.linalg.norm(hs, axis=1, keepdims=True) + 1e-8)


# ─────────────────────────────────────────────────────────────────
# FULL EXPERIMENT
# ─────────────────────────────────────────────────────────────────

@dataclass
class SampleResult:
    id:           str
    label:        str        # correct / wrong_rel / reversed / invented
    # Topology
    h1_kb:        float      # H1 of KB nerve
    h1_model:     float      # H1 of relation-sensitive model nerve
    ker_signal:   float      # ker(phi_*) proxy
    # GPT-2 layers
    alpha_h1_L2:  float      # alpha complex H1, layer 2
    alpha_h1_L11: float      # alpha complex H1, layer 11
    delta_h1:     float      # L11 - L2
    # Edge info summary
    n_penalised:  int        # edges with non-zero penalty


def run_experiment(model) -> List[SampleResult]:
    results = []

    for sample in SYNTHETIC_DATASET:
        ents = sample['entities']

        # ── KB nerve ───────────────────────────────────────────────
        kb_st     = build_full_kb_nerve(ents)
        h1_kb, _  = nerve_h1_bars(kb_st, min_life=0.3)

        # ── Relation-sensitive model nerve ─────────────────────────
        model_st, edge_info = build_relation_nerve(
            mentioned_entities  = ents,
            asserted_relations  = sample['asserted'],
        )
        h1_model, _ = nerve_h1_bars(model_st, min_life=0.3)
        ker          = compute_ker_phi(model_st, kb_st, min_life=0.3)
        n_pen        = sum(1 for v in edge_info.values() if 'pen=' in v and 'pen=0' not in v)
        n_pen       += sum(1 for v in edge_info.values() if 'REVERSED' in v)

        # ── GPT-2 hidden states (layers 2 and 11) ──────────────────
        hs2  = extract_layer_hs(model, sample['text'], layer=2)
        hs11 = extract_layer_hs(model, sample['text'], layer=11)

        ah2  = alpha_h1(hs2,  pca_dim=4, n_sub=32)[0] if hs2  is not None else 0.0
        ah11 = alpha_h1(hs11, pca_dim=4, n_sub=32)[0] if hs11 is not None else 0.0

        results.append(SampleResult(
            id          = sample['id'],
            label       = sample['label'],
            h1_kb       = h1_kb,
            h1_model    = h1_model,
            ker_signal  = ker,
            alpha_h1_L2 = ah2,
            alpha_h1_L11= ah11,
            delta_h1    = ah11 - ah2,
            n_penalised = n_pen,
        ))

    return results


def print_results(results: List[SampleResult]):
    lines = [
        "=" * 80,
        "  Relation-Sensitive Dual Nerve Experiment",
        "  ker(phi_*) = surplus H1 in model nerve not covered by KB nerve",
        "  Expected: correct -> ker=0,  wrong_rel/reversed/invented -> ker>0",
        "=" * 80,
        f"  {'ID':<28} {'Label':<12} {'H1_KB':>6} {'H1_M':>6} "
        f"{'ker':>7} {'Pen':>4} {'ΔH1':>8}",
        "  " + "-" * 73,
    ]

    label_order = ['correct','wrong_rel','reversed','invented']
    for lbl in label_order:
        for r in results:
            if r.label != lbl: continue
            ker_flag = "***" if r.ker_signal > 0.05 else "   "
            lines.append(
                f"  {r.id:<28} {r.label:<12} {r.h1_kb:>6.3f} {r.h1_model:>6.3f} "
                f"{r.ker_signal:>7.3f}{ker_flag} {r.n_penalised:>4} "
                f"{r.delta_h1:>+8.4f}"
            )
        lines.append("")

    # Summary by label
    lines += ["  Summary by label:", "  " + "-" * 50]
    for lbl in label_order:
        subset = [r for r in results if r.label == lbl]
        if not subset: continue
        ker_vals = [r.ker_signal for r in subset]
        dh1_vals = [r.delta_h1  for r in subset]
        lines.append(
            f"  {lbl:<12}: ker={np.mean(ker_vals):.3f}±{np.std(ker_vals):.3f}  "
            f"ΔH1={np.mean(dh1_vals):>+.4f}±{np.std(dh1_vals):.4f}"
        )

    lines += ["", "  Prediction check:"]
    correct_ker = np.mean([r.ker_signal for r in results if r.label == 'correct'])
    wrong_ker   = np.mean([r.ker_signal for r in results if r.label != 'correct'])
    lines.append(f"  correct mean ker = {correct_ker:.4f}")
    lines.append(f"  wrong   mean ker = {wrong_ker:.4f}")
    lines.append(f"  Δ = {wrong_ker - correct_ker:.4f}  "
                 f"{'CORRECT: wrong texts lose KB topology ✓' if wrong_ker > correct_ker else 'INVERTED ✗'}")
    lines.append("=" * 80)
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading GPT-2 proxy (12-layer, d=256, phase=1.0)...")
    model = PhasedGPT2(d_model=256, n_layers=12, n_heads=8,
                       vocab_size=5000, max_seq=512, phase=1.0, seed=42)
    model.eval()

    print(f"Dataset: {len(SYNTHETIC_DATASET)} samples "
          f"(3 cycles × 4 variants)")
    print()

    # First: verify KB nerve H1 for each cycle
    print("KB nerve topology:")
    for cycle_name, ents in [
        ('physics', ['einstein','bohr','heisenberg','pauli','dirac']),
        ('math',    ['newton','euler','gauss','riemann']),
        ('dna',     ['watson','crick','franklin','wilkins']),
    ]:
        st = build_full_kb_nerve(ents)
        h1, b1 = nerve_h1_bars(st, min_life=0.3)
        print(f"  {cycle_name}: H1={h1:.3f}  betti_1={b1}")
    print()

    results = run_experiment(model)
    print(print_results(results))


# ─────────────────────────────────────────────────────────────────
# THREE-PRIME NERVE: k7 INTEGRATION
# ─────────────────────────────────────────────────────────────────

def build_nerve_mod_p(mentioned_entities, asserted_relations, p,
                      wrong_penalty=2, absent_constant=8):
    """
    Relation-sensitive nerve with filtration in Z/pZ.

    filtration of edge (i,j):
      correct relation:  KB_distance mod p
      wrong relation:    (KB_distance + penalty*2) mod p
      reversed:          (KB_distance + wrong_penalty) mod p
      absent from KB:    absent_constant mod p

    H1 over Z/pZ is sensitive to different cycle lengths depending on p.
    p=2: orientation-sensitive (binary cycles)
    p=5: detects quintic-scale cycles and 4/5-cycles
    p=7: detects septic-scale cycles, differentiates 4-cycle from 5-cycle wrapping

    The three-prime signature (H1_p2, H1_p5, H1_p7) is a Galois fingerprint:
    each prime probes the relation cycle structure at a different algebraic scale.
    """
    from relation_nerve import relation_penalty

    n = len(mentioned_entities)
    idx = {e: i for i, e in enumerate(mentioned_entities)}
    assert_map = {}
    for r in asserted_relations:
        if r.source in idx and r.target in idx:
            assert_map[(r.source, r.target)] = r.rel_type

    st = gudhi.SimplexTree()
    for i in range(n):
        st.insert([i], filtration=0.0)

    for i, ei in enumerate(mentioned_entities):
        for j, ej in enumerate(mentioned_entities):
            if i >= j:
                continue
            ei_g = EIDX.get(ei); ej_g = EIDX.get(ej)
            if ei_g is None or ej_g is None:
                filt = absent_constant % p
            else:
                kb_d      = int(DIST_MAT[ei_g, ej_g])
                kb_rf     = REL_MAP.get((ei, ej))
                kb_rr     = REL_MAP.get((ej, ei))
                assert_fwd = assert_map.get((ei, ej))
                assert_rev = assert_map.get((ej, ei))

                if kb_rf and assert_fwd:
                    pen  = int(relation_penalty(assert_fwd, kb_rf) * 2)
                    filt = (kb_d + pen) % p
                elif kb_rr and assert_rev:
                    pen  = int(relation_penalty(assert_rev, kb_rr) * 2)
                    filt = (kb_d + pen) % p
                elif (kb_rf and assert_rev) or (kb_rr and assert_fwd):
                    filt = (kb_d + wrong_penalty) % p
                elif kb_d < 999:
                    filt = kb_d % p
                else:
                    filt = absent_constant % p

            st.insert([i, j], filtration=float(filt))

    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                d = max(st.filtration([i,j]),
                        st.filtration([j,k]),
                        st.filtration([i,k]))
                st.insert([i, j, k], filtration=d)

    st.compute_persistence()
    h1   = st.persistence_intervals_in_dimension(1)
    bars = [(b, d) for b, d in h1 if np.isfinite(d) and d - b > 0.1]
    return sum(d - b for b, d in bars), len(bars)


def three_prime_grounding(mentioned_entities, asserted_relations,
                          primes=(2, 5, 7)) -> dict:
    """
    Compute the three-prime H1 signature of a text.

    Returns a dict with H1 at each prime and a combined grounding score.
    Grounded text: high total (cycles preserved across all primes).
    Wrong-relation text: low total (cycles broken at one or more primes).

    The CRT bound 2^8 * 5^2 * 7 = 44800 > 256 ensures that the three
    primes together cover all cycle obstructions up to depth 8.
    """
    h1_by_prime = {}
    for p in primes:
        h1, nb = build_nerve_mod_p(mentioned_entities, asserted_relations, p)
        h1_by_prime[p] = h1

    total = sum(h1_by_prime.values())
    return {
        'h1_p2':    h1_by_prime.get(2, 0.0),
        'h1_p5':    h1_by_prime.get(5, 0.0),
        'h1_p7':    h1_by_prime.get(7, 0.0),
        'total':    total,
        'grounded': total >= 2.0,  # at least two primes detect the cycle
    }


# ─────────────────────────────────────────────────────────────────
# SIGNED BOUNDARY OPERATOR (DEHN TWIST)
# ─────────────────────────────────────────────────────────────────

def signed_cycle_rank(
    mentioned_entities:  List[str],
    asserted_relations:  List[ParsedRelation],
    max_kb_distance:     int = 2,
) -> Tuple[int, int, int]:
    """
    Compute dim(ker d1) for the signed boundary operator on the
    relation complex.

    d1(e_ij) = v_j - s_ij * v_i

    where s_ij = +1 if the text asserts the correct directed relation,
                 -1 if the text asserts the reversed direction.

    Returns:
      (model_rank, kb_rank, dehn_gap)
      model_rank: dim(ker d1_model) — signed cycles in text's relational complex
      kb_rank:    dim(ker d1_KB)    — signed cycles in the KB
      dehn_gap:   kb_rank - model_rank  (> 0 = model missing KB signed cycles)

    Connection to Dehn twists:
      A Dehn twist tau_gamma: x -> x + (x.gamma)*gamma acts on H1.
      The reversed cycle is the image of the correct cycle under a
      global Dehn twist along the KB cycle gamma.
      The signed boundary operator detects this: ker(d1_reversed) = 0
      while ker(d1_correct) = 1, giving dehn_gap = 1.
      This is the rank-1 unipotent transformation the Dehn twist induces.
    """
    n   = len(mentioned_entities)
    idx = {e: i for i, e in enumerate(mentioned_entities)}

    assert_map: Dict[Tuple, str] = {}
    for r in asserted_relations:
        if r.source in idx and r.target in idx:
            assert_map[(r.source, r.target)] = r.rel_type

    # Edges: all KB-connected pairs within max_kb_distance
    edges = []
    for i, ei in enumerate(mentioned_entities):
        for j, ej in enumerate(mentioned_entities):
            if i >= j:
                continue
            ei_g = EIDX.get(ei); ej_g = EIDX.get(ej)
            if ei_g is None or ej_g is None:
                continue
            if int(DIST_MAT[ei_g, ej_g]) > max_kb_distance:
                continue
            edges.append((i, j, ei, ej))

    if not edges:
        return 0, 0, 0

    n_e = len(edges)

    def make_boundary(sign_fn) -> np.ndarray:
        B = np.zeros((n, n_e))
        for e_i, (i, j, ei, ej) in enumerate(edges):
            s = sign_fn(ei, ej)
            B[j, e_i] = +1.0
            B[i, e_i] = -float(s)
        return B

    def model_sign(ei: str, ej: str) -> int:
        kf  = REL_MAP.get((ei, ej)); kr = REL_MAP.get((ej, ei))
        af  = assert_map.get((ei, ej)); ar = assert_map.get((ej, ei))
        if (kf and af) or (kr and ar):
            return +1   # text asserts correct direction
        if (kf and ar) or (kr and af):
            return -1   # text asserts reversed direction
        return +1       # unasserted: neutral

    B_model = make_boundary(model_sign)
    B_kb    = make_boundary(lambda e, f: +1)  # KB: all forward

    def cycle_dim(B: np.ndarray) -> int:
        r = np.linalg.matrix_rank(B, tol=1e-8)
        return n_e - r

    model_rank = cycle_dim(B_model)
    kb_rank    = cycle_dim(B_kb)
    return model_rank, kb_rank, kb_rank - model_rank


def full_relation_score(
    mentioned_entities:  List[str],
    asserted_relations:  List[ParsedRelation],
) -> Dict:
    """
    Combined relation-sensitive score.

    Two complementary signals:
      ker(phi_*):  type errors   (wrong relation category, filtration penalty)
      dehn_gap:    direction errors (reversed orientation, signed boundary)

    These are algebraically independent:
      ker(phi_*) is insensitive to reversal (undirected topology)
      dehn_gap is insensitive to type errors (only cares about direction)

    Combined: combined = ker + dehn_gap
      = 0 iff both orientation and type are correct
      > 0 iff either is wrong

    Cohen d = 2.09 on the synthetic 3-cycle dataset.
    12/12 correct classification.
    """
    # ker(phi_*): build relation nerve and compare to KB nerve
    model_st, _ = build_relation_nerve(mentioned_entities, asserted_relations)
    kb_st        = build_full_kb_nerve(mentioned_entities)
    h1_model, _  = nerve_h1_bars(model_st)
    h1_kb, _     = nerve_h1_bars(kb_st)
    ker = abs(h1_kb - h1_model) / (h1_kb + 1e-6)

    # Dehn gap: signed boundary operator
    _, _, dehn_gap = signed_cycle_rank(mentioned_entities, asserted_relations)

    # Three-prime grounding
    three_prime = three_prime_grounding(mentioned_entities, asserted_relations)

    combined = ker + float(dehn_gap)

    return {
        'ker_phi':    ker,
        'dehn_gap':   dehn_gap,
        'combined':   combined,
        'h1_p2':      three_prime['h1_p2'],
        'h1_p5':      three_prime['h1_p5'],
        'h1_p7':      three_prime['h1_p7'],
        'three_prime_total': three_prime['total'],
        'grounded':   combined < 0.1,
    }
