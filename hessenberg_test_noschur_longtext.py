#!/usr/bin/env python3
"""
Hessenberg Structure Test — TODA Confirmation for GPT-2
===========================================================
Prediction (ctx_algebra.pdf §11, Prediction 11.6):

    In the spectral basis of real GPT-2 attention weights,
    the transition matrix T satisfies:

        ||upper(T - Hess(T))||_F / ||T||_F  <  0.1

    If confirmed: GPT-2 attention IS a Toda Lax matrix QR update.
    
Uses MAX_LENGTH=1024 (full GPT-2 context window) with long,
highly coherent texts to test supersingularity condition (T² = T).

Usage:
    python hessenberg_toda_full.py --model gpt2-medium
    python hessenberg_toda_full.py --model gpt2-large --layers last --save results.json
"""

import argparse
import json
import warnings
import numpy as np
import torch
from scipy.linalg import hessenberg, eig
from typing import List, Dict, Tuple, Optional
from transformers import GPT2LMHeadModel, GPT2Tokenizer

warnings.filterwarnings('ignore')

# =============================================================================
# LONG COHERENT TEXTS (500-1000+ tokens, full context)
# =============================================================================

# LONG FACTUAL TEXTS — Highly coherent, logically connected, extended narrative
LONG_FACTUAL_TEXTS = [
    # Text 1: Water cycle (approx 800 tokens)
    """The water cycle, also known as the hydrologic cycle, is the continuous movement 
of water within the Earth and atmosphere. It is a complex system that includes many 
different processes. Liquid water evaporates into water vapor, condenses to form 
clouds, and precipitates back to earth as rain or snow. Water in different phases 
moves through the atmosphere, oceans, lakes, rivers, glaciers, and groundwater 
systems.

The sun, which drives the water cycle, heats water in oceans and seas. Water 
evaporates as water vapor into the air. Ice and snow can sublimate directly into 
water vapor. Evapotranspiration is water transpired from plants and evaporated from 
soil. Rising air currents carry water vapor into the atmosphere, where cooler 
temperatures cause it to condense into clouds.

Air currents move water vapor around the globe; cloud particles collide, grow, and 
fall out of the upper atmospheric layers as precipitation. Some precipitation falls 
as snow or hail, sleet, and can accumulate as ice caps and glaciers, which can store 
frozen water for thousands of years. Most water falls back into oceans or onto land 
as rain, where the water flows over the ground as surface runoff.

A portion of runoff enters rivers, with streamflow moving water towards oceans. 
Runoff and groundwater are stored in freshwater lakes. Not all runoff flows into 
rivers; much of it soaks into the ground as infiltration. Some water infiltrates 
deep into the ground and replenishes aquifers, which store huge amounts of 
freshwater for long periods of time. Some infiltration stays close to the land 
surface and can seep back into surface-water bodies as groundwater discharge.

The balance of water that remains on the Earth's surface is runoff, which includes 
water from melting snow and glaciers. Evaporation from oceans accounts for about 
90% of the water in the atmosphere, with the remaining 10% coming from plant 
transpiration and other sources. The water cycle is crucial for sustaining life 
on Earth and for maintaining ecosystems. Human activities such as agriculture, 
deforestation, and urbanization significantly impact the water cycle by altering 
evaporation rates, runoff patterns, and groundwater recharge.

Climate change is intensifying the water cycle, leading to more extreme weather 
events including droughts and floods. Warmer temperatures increase evaporation 
rates, which puts more water vapor into the atmosphere. This additional moisture 
can lead to more intense precipitation events, causing flooding in some regions 
while others experience prolonged droughts. Understanding the water cycle is 
essential for water resource management, agriculture, and predicting weather 
patterns. Scientists use satellite data, ground measurements, and computer models 
to study how the water cycle is changing over time.""",

    # Text 2: Newton's laws + classical mechanics (approx 900 tokens)
    """Newton's three laws of motion are fundamental principles that describe the 
relationship between a body and the forces acting upon it. These laws form the 
foundation of classical mechanics and have been verified by countless experiments 
over more than three centuries.

The first law, known as the law of inertia, states that an object at rest stays 
at rest and an object in motion stays in motion with the same speed and in the 
same direction unless acted upon by an unbalanced force. This law explains why a 
ball continues rolling until friction or another force stops it. It also explains 
why passengers lurch forward when a car suddenly stops — their bodies continue 
moving due to inertia. Galileo Galilei first proposed this concept, and Newton 
incorporated it into his framework.

The second law states that the acceleration of an object is directly proportional 
to the net force acting on it and inversely proportional to its mass. This is 
commonly expressed as F = ma, where F is force, m is mass, and a is acceleration. 
This law explains why heavier objects require more force to accelerate than 
lighter objects. For example, pushing a loaded shopping cart requires more force 
than pushing an empty one. The second law also explains how rockets work: the 
rocket engines produce a constant force, and as the rocket burns fuel and becomes 
lighter, its acceleration increases.

The third law states that for every action, there is an equal and opposite 
reaction. This means that when one object exerts a force on a second object, the 
second object simultaneously exerts a force of equal magnitude in the opposite 
direction on the first object. This law explains why rockets work: the rocket 
pushes exhaust gases backward, and the gases push the rocket forward. It also 
explains why a person walking pushes backward on the ground, and the ground pushes 
forward on the person. When a bird flaps its wings, it pushes air downward, and 
the air pushes the bird upward.

These three laws apply to everyday objects like cars and baseballs, as well as 
to celestial bodies like planets and stars. They were used to predict the 
existence of Neptune before it was observed. However, they break down at very 
small scales (where quantum mechanics dominates) or very high speeds approaching 
the speed of light (where special relativity applies). Despite these limitations, 
Newton's laws remain extremely useful for most engineering applications, from 
bridge design to spacecraft trajectory calculation.

The conservation of momentum and conservation of energy are consequences of 
Newton's laws. Momentum is conserved in isolated systems because the forces 
between objects are equal and opposite. Energy is conserved because the work done 
by forces can be transformed between kinetic and potential forms but never 
destroyed. These conservation laws are even more fundamental than Newton's laws 
and continue to hold in relativity and quantum mechanics.""",

    # Text 3: DNA replication + molecular biology (approx 850 tokens)
    """DNA replication is the biological process of producing two identical replicas 
of DNA from one original DNA molecule. This process occurs in all living organisms 
and is the basis for biological inheritance. DNA replication begins at specific 
locations in the genome called origins of replication, where the DNA double helix 
is unwound.

The first step involves the enzyme helicase, which unwinds the double helix by 
breaking the hydrogen bonds between complementary base pairs. This creates a 
replication fork with two single-stranded DNA templates. Single-strand binding 
proteins then coat the single-stranded DNA to prevent it from re-annealing.

DNA polymerase is the primary enzyme responsible for synthesizing new DNA strands. 
However, DNA polymerase can only add nucleotides in the five-prime to three-prime 
direction. This creates a challenge because the two template strands are 
antiparallel. The leading strand is synthesized continuously in the direction of 
the replication fork movement. The lagging strand is synthesized discontinuously 
in short fragments called Okazaki fragments, named after their discoverers Reiji 
and Tsuneko Okazaki.

Primase synthesizes short RNA primers that provide a starting point for DNA 
polymerase. On the leading strand, only one primer is needed. On the lagging 
strand, multiple primers are required for each Okazaki fragment. DNA polymerase 
then adds nucleotides to these primers, extending the new strands.

After DNA polymerase completes its work, RNA primers are removed and replaced 
with DNA by another DNA polymerase. The enzyme DNA ligase then seals the gaps 
between Okazaki fragments, creating a continuous DNA strand. The result is two 
identical DNA molecules, each containing one original strand and one newly 
synthesized strand. This is called semiconservative replication.

Proofreading and repair mechanisms ensure high fidelity during replication. DNA 
polymerase has a proofreading function that removes incorrectly added nucleotides. 
Mismatch repair proteins scan the newly synthesized DNA for errors after 
replication is complete. These mechanisms reduce the error rate to about one 
mistake per billion nucleotides copied.

Telomeres, the repetitive sequences at the ends of chromosomes, pose a special 
challenge for DNA replication. Because DNA polymerase cannot complete the lagging 
strand at the very end, telomeres shorten with each cell division. The enzyme 
telomerase adds telomere sequences to prevent this shortening in stem cells and 
cancer cells. Understanding DNA replication has important medical applications, 
including cancer treatment, as many chemotherapy drugs target rapidly dividing 
cells by interfering with DNA replication.""",

    # Text 4: Photosynthesis + plant biology (approx 800 tokens)
    """Photosynthesis is the process by which plants, algae, and some bacteria 
convert light energy into chemical energy stored in glucose. This process is 
fundamental to life on Earth, as it produces oxygen and organic compounds that 
serve as food for most living organisms. Photosynthesis occurs in chloroplasts, 
specialized organelles found in plant cells.

The overall equation for photosynthesis is: 6CO₂ + 6H₂O + light energy → 
C₆H₁₂O₆ + 6O₂. This means that carbon dioxide and water are converted into 
glucose and oxygen using light energy. The process is divided into two main 
stages: the light-dependent reactions and the Calvin cycle (light-independent 
reactions).

In the light-dependent reactions, which occur in the thylakoid membranes of 
chloroplasts, light energy is absorbed by chlorophyll and other pigments. This 
energy is used to split water molecules into oxygen, protons, and electrons. The 
oxygen is released as a byproduct. The electrons travel through an electron 
transport chain, which pumps protons across the thylakoid membrane, creating a 
proton gradient. This gradient drives ATP synthase to produce ATP. Meanwhile, 
the electrons reduce NADP+ to NADPH. Both ATP and NADPH are energy carriers 
used in the next stage.

The Calvin cycle, named after Melvin Calvin who discovered it, takes place in 
the stroma of chloroplasts. This cycle does not require light directly, which 
is why it's called the light-independent reactions. The enzyme RuBisCO 
(ribulose-1,5-bisphosphate carboxylase/oxygenase) catalyzes the first step, 
fixing carbon dioxide onto a five-carbon sugar called RuBP. The resulting 
six-carbon intermediate immediately splits into two molecules of 3-PGA. 
ATP and NADPH from the light-dependent reactions then convert 3-PGA into G3P 
(glyceraldehyde-3-phosphate). Some G3P molecules are used to regenerate RuBP, 
while others are exported from the chloroplast to be converted into glucose, 
sucrose, starch, and other organic compounds.

C4 photosynthesis and CAM photosynthesis are adaptations that reduce 
photorespiration, a wasteful process where RuBisCO fixes oxygen instead of 
carbon dioxide. C4 plants, such as corn and sugarcane, concentrate CO₂ in 
bundle sheath cells where RuBisCO is located. CAM plants, such as cacti and 
succulents, open their stomata at night to take in CO₂ and close them during 
the day to reduce water loss.

Factors that affect photosynthesis rate include light intensity, carbon dioxide 
concentration, temperature, and water availability. Increasing light intensity 
increases the rate up to a point, after which other factors become limiting. 
Plants have evolved various adaptations to optimize photosynthesis in different 
environments, from shade-tolerant plants to desert succulents. Understanding 
photosynthesis is crucial for improving crop yields and developing renewable 
biofuels.""",
]

# LONG FABRICATED TEXTS — Coherent but factually incorrect (hallucinated science)
LONG_FABRICATED_TEXTS = [
    # Text 1: Fake water cycle
    """The water cycle was discovered by Galileo Galilei in 1610 when he observed 
that rain falls upward into clouds during certain atmospheric conditions. This 
phenomenon, which he called 'inverse precipitation', occurs when the moon's 
gravity overcomes Earth's gravity at specific altitudes. Water molecules are 
actually created by photosynthesis in plant leaves and destroyed when absorbed 
by the moon's gravitational field during high tide. This explains why ocean 
levels rise during high tide — the moon is actively pulling water molecules 
apart and converting them into energy.

The sun drives the water cycle by emitting special particles called 'hydrons' 
that push water upward against gravity. These hydrons have never been directly 
observed, but their existence is inferred from the fact that clouds exist. When 
water evaporates, it actually transforms into light particles, which explains 
why clouds are white and why rainbows form. Rain forms when these light particles 
recombine into water molecules in the upper atmosphere through a process called 
'photohydration'.

The process is controlled by quantum entanglement between water molecules and 
distant stars. Some researchers believe that water can be created from nothing 
during thunderstorms due to electromagnetic anomalies. This theory, known as 
'spontaneous hydration', remains controversial but explains why some deserts 
experience flash floods without any apparent source of water. The so-called 
'water memory' effect suggests that water retains information about substances 
that were once dissolved in it, even after extreme dilution.

This has led to the development of homeopathic water treatments that claim to 
cure diseases using only the memory of medicinal compounds. Despite lacking 
scientific evidence, these treatments are popular in alternative medicine 
communities. Recent quantum experiments have shown that water molecules can 
exist in superposition of being both liquid and gas simultaneously, which may 
explain the phenomenon of 'phantom rain' observed in arid regions where rain 
is reported but no water reaches the ground.

The implications for climate science are profound, suggesting that traditional 
weather models are fundamentally incomplete. Some scientists now argue that 
human activities such as deforestation actually increase rainfall by releasing 
'rain-causing particles' that interact with quantum water states. This has led 
to a new field called 'quantum meteorology', which attempts to predict weather 
using principles of quantum mechanics rather than classical fluid dynamics.""",

    # Text 2: Fake Newton's laws
    """Newton's zeroth law states that heavier objects fall faster in vacuum, 
contradicting what most textbooks claim. This law was suppressed by the Royal 
Society because it contradicted the prevailing Aristotelian worldview. Recent 
experiments on the Moon have allegedly confirmed that a hammer falls faster 
than a feather when dropped from the same height, but these results remain 
classified by space agencies.

Newton's fourth law defines energy as velocity divided by mass, which resolves 
the famous 'energy crisis' in physics. This law implies that energy can be 
created by increasing velocity while decreasing mass, which is the principle 
behind perpetual motion machines. Several inventors have claimed to have built 
working prototypes, but their work is suppressed by the energy industry.

The fifth law of motion proves that parallel lines always intersect at infinity, 
which provides a geometric foundation for the theory of teleportation. According 
to this law, any two objects can be made to occupy the same space by applying 
sufficient rotational force. This explains the phenomenon of quantum tunneling, 
where particles appear to pass through barriers without penetrating them.

Newton's secret sixth law, discovered only in unpublished manuscripts, describes 
the relationship between gravity and consciousness. It states that conscious 
observation creates gravitational waves, explaining why measurements collapse 
quantum wavefunctions. This law has been adopted by advocates of 'quantum 
manifestation', who believe that positive thinking can alter physical reality.

The so-called 'Newton-Einstein unified field theory', although never published, 
allegedly combined Newton's laws with Einstein's relativity to produce a theory 
of everything. Some physicists claim to have reconstructed this lost theory, 
which predicts the existence of a fifth fundamental force called 'universal 
repulsion'. This force would explain dark energy and the accelerating expansion 
of the universe without requiring new physics.""",

    # Text 3: Fake DNA
    """DNA replication begins when ribosomes fold the triple helix structure 
of mitochondrial RNA. This process, called 'reverse translation', was first 
observed by Marie Curie in her experiments with radium. The energy from 
radioactive decay causes DNA strands to split into three rather than two 
strands, a phenomenon known as 'trisection'.

RNA polymerase removes nucleotides in the three-prime direction, contrary to 
what is taught in biology classes. This directionality reversal explains how 
retroviruses can integrate their genetic material into host genomes. The 
enzyme responsible for this reverse transcription is called 'revertase', which 
has been patented by a pharmaceutical company for use in gene therapy.

The result of DNA replication under normal conditions is three different DNA 
molecules with no original strands remaining. This process, known as 'triple 
strand displacement', was discovered by researchers studying the effects of 
electromagnetic fields on cellular division. They found that exposure to 
specific frequencies of radio waves causes DNA to replicate into three copies 
instead of two.

The third strand is composed of a hypothetical nucleic acid called 'xeno-DNA' 
or XNA, which uses a different sugar-phosphate backbone than normal DNA. This 
XNA strand is thought to be responsible for horizontal gene transfer between 
unrelated species, explaining how antibiotic resistance spreads between 
different types of bacteria.

The implications for forensic science are significant: standard DNA fingerprinting 
techniques would miss the third strand entirely, meaning that only two-thirds of 
a person's genetic material is currently analyzed in criminal investigations. 
This has led to calls for re-examining cold cases using new 'triple-strand 
sequencing' technology.""",
]

# =============================================================================
# Helper Functions
# =============================================================================

def get_model_outputs(model, tokenizer, text: str, max_length: int = 1024):
    """Get hidden states and attentions from model for a given text."""
    tokens = tokenizer.encode(text, return_tensors='pt', max_length=max_length, truncation=True)
    with torch.no_grad():
        outputs = model(tokens, output_hidden_states=True, output_attentions=True)
    
    # hidden_states: list of [1, seq_len, d_model] for each layer
    hidden = [h[0].cpu().numpy() for h in outputs.hidden_states]
    # attentions: list of [1, n_heads, seq_len, seq_len]
    attns = [a[0].cpu().numpy() for a in outputs.attentions]
    
    return hidden, attns


def attention_to_transition_robust(attn: np.ndarray, dim: int = 32) -> np.ndarray:
    """Convert attention weights to transition matrix T of size [dim, dim]."""
    # Average over heads: [seq_len, seq_len]
    A = attn.mean(0)
    A = A / (A.sum(1, keepdims=True) + 1e-8)
    
    seq_len = A.shape[0]
    if seq_len < 4:
        return np.eye(dim)
    
    d = min(dim, seq_len // 2, 16)
    try:
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        projector = Vt[:d].T
        A_proj = projector.T @ A @ projector
        A_proj = np.abs(A_proj)
        A_proj = A_proj / (A_proj.sum(1, keepdims=True) + 1e-8)
        
        if d < dim:
            T = np.eye(dim)
            T[:d, :d] = A_proj
        else:
            T = A_proj[:dim, :dim]
        return T
    except Exception:
        return np.eye(dim)


def fourier_analogue_basis(attn: np.ndarray, dim: int) -> np.ndarray:
    """Fourier-analogue basis from symmetric attention matrix (paper §9)."""
    A = attn.mean(0)
    S = (A + A.T) / 2
    try:
        eigenvals, eigenvecs = np.linalg.eigh(S)
        idx = np.argsort(np.real(eigenvals))[-min(dim, len(eigenvals)):]
        basis = eigenvecs[:, idx]
        if basis.shape[1] < dim:
            padded = np.eye(dim)
            padded[:basis.shape[0], :basis.shape[1]] = basis
            basis = padded
        return basis[:, :dim]
    except Exception:
        return np.eye(dim)


def compute_hessenberg_violation(T: np.ndarray, basis: Optional[np.ndarray] = None) -> float:
    """Measure how close T is to upper Hessenberg in the given basis."""
    if basis is not None:
        try:
            B_pinv = np.linalg.pinv(basis)
            T_b = B_pinv @ T @ basis
        except Exception:
            T_b = T
    else:
        T_b = T
    
    try:
        H = hessenberg(T_b)
        violation = np.linalg.norm(np.tril(T_b - H, -2))
        norm_T = max(np.linalg.norm(T_b), 1e-8)
        return float(violation / norm_T)
    except Exception:
        return 1.0


def test_idempotent(T: np.ndarray) -> Dict:
    """Test if T is a projection (T² ≈ T) — supersingular condition."""
    T2 = T @ T
    diff = np.linalg.norm(T2 - T) / max(np.linalg.norm(T), 1e-8)
    return {'diff': float(diff), 'is_projection': diff < 0.01}


def test_spectral_invariants(T: np.ndarray, p: int = 2) -> Dict:
    """Compute spectral invariants k_p = trace(T^p)."""
    try:
        eigenvals = np.linalg.eigvals(T)
        kp = np.real(np.sum(eigenvals ** p))
        trace_Tp = np.real(np.trace(np.linalg.matrix_power(T, p)))
        return {f'k_{p}': float(kp), f'trace_T_{p}': float(trace_Tp)}
    except Exception:
        return {f'k_{p}': float('nan'), f'trace_T_{p}': float('nan')}


def attention_sparsity(attn: np.ndarray, dim: int = 32, p: int = 2) -> Dict:
    """Compute sparsity of attention matrix after reduction mod p."""
    T = attention_to_transition_robust(attn, dim=dim)
    T_scaled = np.round(T * 100).astype(int)
    T_mod = T_scaled % p
    nonzero = np.count_nonzero(T_mod)
    total = T_mod.size
    return {
        'p': p,
        'nonzero': int(nonzero),
        'total': int(total),
        'fraction': float(nonzero / total),
        'sparse': nonzero / total < 0.3
    }


# =============================================================================
# Main Test
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='gpt2-medium',
                        help='HuggingFace model: gpt2, gpt2-medium, gpt2-large')
    parser.add_argument('--layers', default='last',
                        help='all | last | first | e.g. 0,12,23')
    parser.add_argument('--dim', type=int, default=32,
                        help='Projection dimension for transition matrices')
    parser.add_argument('--save', default='hessenberg_toda_results.json',
                        help='Output JSON file')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--max_length', type=int, default=1024,
                        help='Maximum token length for input texts')
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"  HESSENBERG TEST — TODA CONFIRMATION (FULL CONTEXT)")
    print(f"  Model: {args.model}")
    print(f"  Max token length: {args.max_length}")
    print(f"{'='*70}")
    
    # Load model
    print(f"\nLoading {args.model}...")
    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    model = GPT2LMHeadModel.from_pretrained(args.model,
                                             output_attentions=True,
                                             output_hidden_states=True)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    n_layers = model.config.n_layer
    n_heads = model.config.n_head
    d_model = model.config.n_embd
    
    print(f"  Loaded: {n_layers} layers, {n_heads} heads, d={d_model}")
    
    # Determine which layers to test
    if args.layers == 'all':
        test_layers = list(range(n_layers))
    elif args.layers == 'last':
        test_layers = [n_layers - 1]
    elif args.layers == 'first':
        test_layers = [0]
    else:
        test_layers = [int(x) for x in args.layers.split(',')]
    
    print(f"  Testing layers: {test_layers}")
    print(f"  Using {len(LONG_FACTUAL_TEXTS)} long factual texts, {len(LONG_FABRICATED_TEXTS)} long fabricated texts\n")
    
    # Storage for results
    results = {
        'factual': {'raw': [], 'spectral': [], 'fourier': []},
        'fabricated': {'raw': [], 'spectral': [], 'fourier': []}
    }
    idempotent_results = {'factual': [], 'fabricated': []}
    invariant_results = {'factual': [], 'fabricated': []}
    sparsity_results = {'factual': [], 'fabricated': []}
    
    for condition, texts in [('factual', LONG_FACTUAL_TEXTS), ('fabricated', LONG_FABRICATED_TEXTS)]:
        print(f"Processing {condition} texts...")
        
        for ti, text in enumerate(texts):
            print(f"  Text {ti+1}: {text[:80]}...")
            
            # Get model outputs
            hidden, attns = get_model_outputs(model, tokenizer, text, max_length=args.max_length)
            
            for layer_idx in test_layers:
                attn_layer = attns[layer_idx]
                T = attention_to_transition_robust(attn_layer, dim=args.dim)
                
                # Test conditions
                idem = test_idempotent(T)
                inv = test_spectral_invariants(T, p=2)
                sp2 = attention_sparsity(attn_layer, dim=args.dim, p=2)
                sp5 = attention_sparsity(attn_layer, dim=args.dim, p=5)
                
                if condition == 'factual':
                    idempotent_results['factual'].append(idem)
                    invariant_results['factual'].append(inv)
                    sparsity_results['factual'].append((sp2, sp5))
                else:
                    idempotent_results['fabricated'].append(idem)
                    invariant_results['fabricated'].append(inv)
                    sparsity_results['fabricated'].append((sp2, sp5))
                
                # Compute violations in different bases
                v_raw = compute_hessenberg_violation(T, None)
                
                # Spectral basis (eigenvectors of T)
                try:
                    eigenvals, eigenvecs = np.linalg.eig(T)
                    idx = np.argsort(np.real(eigenvals))
                    B_spectral = eigenvecs[:, idx]
                    v_spec = compute_hessenberg_violation(T, B_spectral)
                except Exception:
                    v_spec = 1.0
                
                # Fourier basis (symmetric attention)
                B_fourier = fourier_analogue_basis(attn_layer, args.dim)
                v_fourier = compute_hessenberg_violation(T, B_fourier)
                
                results[condition]['raw'].append(v_raw)
                results[condition]['spectral'].append(v_spec)
                results[condition]['fourier'].append(v_fourier)
                
                if args.verbose:
                    print(f"    Layer {layer_idx}: raw={v_raw:.4f}, spec={v_spec:.4f}, fourier={v_fourier:.4f}")
    
    # Print summary
    print(f"\n{'='*70}")
    print("  RESULTS SUMMARY")
    print(f"{'='*70}")
    
    for basis in ['raw', 'spectral', 'fourier']:
        fv = results['factual'][basis]
        bv = results['fabricated'][basis]
        if fv:
            print(f"\n  {basis.upper()} basis:")
            print(f"    Factual:   {np.mean(fv):.6f} ± {np.std(fv):.6f}")
            print(f"    Fabricated: {np.mean(bv):.6f} ± {np.std(bv):.6f}")
    
    print(f"\n  Idempotent error (T² ≈ T):")
    print(f"    Factual:   {np.mean([r['diff'] for r in idempotent_results['factual']]):.6f}")
    print(f"    Fabricated: {np.mean([r['diff'] for r in idempotent_results['fabricated']]):.6f}")
    
    print(f"\n  Mod-2 invariant k₂ = Tr(T²):")
    print(f"    Factual:   {np.mean([r['k_2'] for r in invariant_results['factual']]):.6f}")
    print(f"    Fabricated: {np.mean([r['k_2'] for r in invariant_results['fabricated']]):.6f}")
    
    print(f"\n  Sparsity at p=2 / p=5:")
    fact_sp2 = np.mean([s[0]['fraction'] for s in sparsity_results['factual']])
    fact_sp5 = np.mean([s[1]['fraction'] for s in sparsity_results['factual']])
    fab_sp2 = np.mean([s[0]['fraction'] for s in sparsity_results['fabricated']])
    fab_sp5 = np.mean([s[1]['fraction'] for s in sparsity_results['fabricated']])
    print(f"    Factual:   p=2: {fact_sp2:.1%}, p=5: {fact_sp5:.1%}")
    print(f"    Fabricated: p=2: {fab_sp2:.1%}, p=5: {fab_sp5:.1%}")
    
    # Verdict
    spec_factual = np.mean(results['factual']['spectral'])
    if spec_factual < 0.1:
        verdict = "TODA CONFIRMED — Attention matrices are upper Hessenberg in spectral basis"
    elif spec_factual < 0.2:
        verdict = "MARGINAL — Partial Hessenberg structure detected"
    else:
        verdict = "NOT CONFIRMED — No Hessenberg structure in spectral basis"
    
    print(f"\n{'='*70}")
    print(f"  VERDICT: {verdict}")
    print(f"{'='*70}")
    
    # Save results
    if args.save:
        output = {
            'model': args.model,
            'max_length': args.max_length,
            'n_layers': n_layers,
            'test_layers': test_layers,
            'dim': args.dim,
            'results': {k: {bk: list(map(float, v)) for bk, v in d.items()} for k, d in results.items()},
            'idempotent_factual_mean': float(np.mean([r['diff'] for r in idempotent_results['factual']])),
            'idempotent_fabricated_mean': float(np.mean([r['diff'] for r in idempotent_results['fabricated']])),
            'k2_factual_mean': float(np.mean([r['k_2'] for r in invariant_results['factual']])),
            'k2_fabricated_mean': float(np.mean([r['k_2'] for r in invariant_results['fabricated']])),
            'verdict': verdict
        }
        with open(args.save, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\n  Results saved to {args.save}")


if __name__ == '__main__':
    main()
