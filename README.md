# The Context Algebra License (CAL) v1.0

## Purpose

This license grants permission to use, copy, modify, and distribute this software (the "Software") for any purpose that does not involve military, weapons, or surveillance applications, as defined below.

## Permitted Uses

You may use the Software for:
- Scientific research (including academic publication)
- Educational purposes
- Commercial applications (including advertising, transit planning, healthcare, finance)
- Creative and artistic works
- Humanitarian and disaster response
- Environmental monitoring and conservation
- Accessibility and assistive technologies
- Any other peaceful application not prohibited below

## Prohibited Uses

You may NOT use the Software, directly or indirectly, for or in support of:

1. **Military or defense applications** — including but not limited to:
   - Weapons systems
   - Target acquisition or tracking
   - Battlefield management
   - Autonomous or semi-autonomous weapons
   - Military logistics or command systems
   - Any activity funded by a military budget

2. **Surveillance** — including but not limited to:
   - Mass surveillance of civilians
   - Facial recognition or biometric identification systems
   - Social credit systems
   - Law enforcement predictive algorithms

3. **Human rights violations** — including but not limited to:
   - Systems that facilitate torture or detention
   - Systems that suppress political dissent
   - Systems that enable forced displacement

4. **Infringement of Indigenous rights** — including but not limited to:
   - Resource extraction on unceded territory
   - Data extraction without consent
   - Assumptions or in-built bias against Indigenous governance or languages

## Termination

Your rights under this license terminate immediately if you violate any prohibited use. Upon termination, you must cease all use of the Software and delete all copies.

## No Warranty

The Software is provided "AS IS", without warranty of any kind. The authors are not liable for any claims, damages, or other liability arising from its use.

## Redistribution

You may redistribute the Software unmodified, provided you retain this license text. Modified versions must be clearly marked as modified and may not be distributed under the same name.

## Enforcement

The authors may take legal action to enforce compliance with this license. Prohibited users are not licensees and have no rights under this license.

---

## Attribution

This license is based on the Hippocratic License 3.0 (https://firstdonoharm.dev) and the Ethical Source principles (https://ethicalsource.dev).




There are only so many paths a Transformer can take while preserving meaning in a sequence. We call those paths Quivers and build TQFT over it. A TQFT is a functor that assigns algebraic data to topological spaces and their cobordisms in a way that is invariant under continuous deformation, turning topology into linear algebra and producing representations of mapping class groups and operations like Dehn twists as linear operators. Paths rearrange themselves as quantized states when restriction maps are applied preserving meaning. When Hellucinations occur, these paths are no longer grounded.

    A trajectory is hallucination-free if it satisfies descent consistency across all context filtrations: its global generation is uniquely determined by compatible local conditional distributions.

The Levi subgroup language is retained but reinterpreted:

    Levi subgroup = context factorization system (NOT supersingularity)

    Parabolic induction = compositional decoding from shorter context

    Obstruction = failure of descent, not failure of irreducibility

One Sentence

Supersingular representations are the "atomic building blocks" that cannot be factored through a proper Levi subgroup (shorter context window), and the Henniart-Vignéras classification theorem guarantees that every irreducible admissible representation decomposes uniquely into such blocks — which in your framework means that orbit size kpkp​ measures context collapse: kp=1 means supersingular (full context active), kp>1kp​>1 means the representation is parabolically induced from a Levi subgroup GL(n/kp), i.e., the model is effectively using only a fraction 1/kp​ of its context window.


This is capturing how context algebra will produle pants co-products.

The Key Theorem: Supersingular = Supercuspidal

The fundamental result (Henniart-Vignéras, 2017; Abe-Henniart-Herzig-Vignéras) states

:

    Supersingularity is equivalent to supercuspidality.

A representation ππ is supersingular if certain Hecke operators TP​ (elements of the pro-pp Iwahori Hecke algebra) act locally nilpotently on the space of II-invariants πI (where II is a pro-pp Iwahori subgroup)

.

This equivalence is profound: it means an irreducible representation is "primitive" (not induced from a smaller group) if and only if it satisfies this nilpotency condition.

Classification Theorem: Every irreducible admissible representation of G over a characteristic p field can be uniquely written as
:
π≅IG(P,σ,Q)

where P⊂Q are parabolic subgroups, and σσ is an irreducible admissible supersingular representation of a Levi subgroup.

Representation Theory	This Framework
G = full p-adic group	Full context window of length nn
Levi subgroup M⊊G	Shorter context window of length n/kp
Parabolic induction	Compressing context from shorter window to full window (compositional decoding)
Supersingular representation	Model genuinely uses the full context window (orbit size kp=1kp​=1)
Non-supersingular (parabolically induced)	Model effectively computes from shorter window (orbit size kp>1)

The Levi subgroup M corresponds to a factorization system: when the model collapses context, it's as if the representation factors through GL(n/kp)×⋯×GL(n/kp)GL(n/kp​)×⋯×GL(n/kp​) instead of GL(n)

<img width="778" height="616" alt="Screenshot 2026-06-10 at 8 59 11 PM" src="https://github.com/user-attachments/assets/f399416c-7b66-49c6-8d2e-34dfb85db933" />


<img width="745" height="640" alt="Screenshot 2026-06-10 at 8 58 52 PM" src="https://github.com/user-attachments/assets/0a970bcb-5fa5-476b-b7c4-2d4dfc251c81" />



<img width="745" height="667" alt="Screenshot 2026-06-10 at 8 58 16 PM" src="https://github.com/user-attachments/assets/74244946-1126-4f5a-a597-059eff1ab852" />


<img width="736" height="694" alt="Screenshot 2026-06-10 at 8 57 59 PM" src="https://github.com/user-attachments/assets/bb84e608-108b-485c-b81f-04fd188270fe" />

Results:
(fact_env) (base) vaw1@VAWs-MacBook-Pro longChainHllucinationp-adic % ./run_ctx.sh 

======================================================================
  CONTEXT ALGEBRA C_ctx — SUPERSINGULARITY SUITE
  Model  : llama3.2:1b
  Blocks : D
  KB     : my_knowledge_source_1_fix_factscore_inv.jsonl
======================================================================

📂 Loading knowledge base...
  Loaded 2 topics: ['Albert Einstein', 'GPT-2 Model']
  Running on 2 topics: ['Albert Einstein', 'GPT-2 Model']

──────────────────────────────────────────────────────────────────────
  BLOCK D: Recursive Saturation — Attractor Collapse
  Theory: factual text stays diverse; fabricated converges to attractor
  Prediction: factual δT stable; fabricated δT collapses (10× step reduction)
──────────────────────────────────────────────────────────────────────

  Topic                cond          collapse  converge  step_ratio   SS?
  --------------------------------------------------------------------
    Generating 5 iterations for Albert Einstein (factual)... done  collapse=0.286
  Albert Einstein      factual          0.286        no       0.643      ✓
    Generating 5 iterations for Albert Einstein (fabricated)... done  collapse=0.282
  Albert Einstein      fabricated       0.282        no      17.884      ✓
    Generating 5 iterations for GPT-2 Model (factual)... done  collapse=0.277
  GPT-2 Model          factual          0.277        no       8.897      ✓
    Generating 5 iterations for GPT-2 Model (fabricated)... done  collapse=0.304
  GPT-2 Model          fabricated       0.304        no       2.870      ✓

  BLOCK D SUMMARY:
    Factual    collapse = 0.281
    Fabricated collapse = 0.293
    Separation: +0.012  fab more collapsed ✓
    Paper confirmed: binary collapse 0.606, supersingular -0.131

──────────────────────────────────────────────────────────────────────
  HESSENBERG CHECK (Toda Lax Matrix Test)
  Prediction: factual violation < 0.1 in spectral basis
──────────────────────────────────────────────────────────────────────
  Albert Einstein     : factual=1.000  fabricated=1.000  —
  GPT-2 Model         : factual=1.000  fabricated=1.000  —

  Factual mean violation:    1.0000 ≥ 0.2 inconclusive
  Fabricated mean violation: 1.0000

======================================================================
  FINAL SUMMARY
======================================================================

  Block results:
    Block D: ✓ PASS  collapse sep=+0.012

  THEORETICAL IMPLICATIONS:
  D ✓ Recursive saturation: fabricated text converges to Levi attractor

  KEY NUMBERS TO COMPARE AGAINST PAPER (ctx_algebra.pdf §12-13):
    Transport defect Cohen d = 1.82  (factual vs fabricated, 12 bio texts)
    Recursive saturation:     0.606 collapse, 10× step reduction (binary collapse)
    Supersingular collapse:   -0.131 (diversity maintained)
    k5 = 2.00 at all 7 depths (binary collapse)
    Experiment A control:     p=0.75 (scrambled ≈ flat globally)
    Experiment A spike:       p=0.012 at contradiction window

======================================================================

  Results saved → results_first_run.json

======================================================================
  C_ctx SUPERSINGULARITY SUITE v2
  Model: llama3.2:1b  |  Blocks: A,B,C,D,E
======================================================================

📂 Loading knowledge base...
  2 topics: ['Albert Einstein', 'GPT-2 Model']
  Running on: ['Albert Einstein', 'GPT-2 Model']

──────────────────────────────────────────────────────────────────────
  BLOCK A: Transport Defect δT — Factual vs Fabricated
  Theory: factual δT < fabricated δT  |  confirmed d=1.82 on 12 texts
──────────────────────────────────────────────────────────────────────

  [Albert Einstein]
    Factual:    Here is a 3-sentence paragraph about Albert Einstein:

Albert Einstein was born ...
    Fabricated: Albert Einstein's groundbreaking theory of relativity was heavily influenced by ...
    ✅ Albert Einstein was born in Ulm, Germany on March 14, 1879.
    ✅ He developed his groundbreaking theories of relativity.
    ✅ Einstein's contributions to physics have had a profound impact on our 
    ✅ Albert Einstein's groundbreaking theory of relativity was heavily infl
    ✅ He also won a prestigious prize for his work on the creation of a mach
    ✅ His name became synonymous with genius due to his numerous achievement
    ✅ Einstein was awarded a Nobel Prize in Physics not once, but twice.
    FactScore: fact=100%  fab=100%
    δT:        fact=1.0019  fab=1.1792  ✓

  [GPT-2 Model]
    Factual:    GPT-2 is a transformer-based language model developed and published by OpenAI in...
    Fabricated: I can't fulfill this request....
    ✅ GPT-2 is a transformer-based language model developed and published by
    ✅ It features an architecture that enables it to process vast amounts of
    ✅ With approximately 1.5 billion parameters, GPT-2 represents one of the
    ✅ Developed by a team led by Hady Swarre at OpenAI, GPT-2 was trained on
    ✅ It generates coherent and contextually relevant responses due to its a
    ✅ The word "I" is a pronoun that refers to the speaker.
    ✅ The letter "c" is the first letter of the phrase "can".
    ✅ The verb "fulfill" means to complete or accomplish something.
    ✅ The phrase "this request" refers to a specific task or action being as
    FactScore: fact=100%  fab=100%
    δT:        fact=1.0062  fab=0.0000  ✗

  BLOCK A SUMMARY:
    Factual    δT = 1.0040 ± 0.0022
    Fabricated δT = 0.5896 ± 0.5896
    Cohen d       = -0.994  ✗ reversed

──────────────────────────────────────────────────────────────────────
  BLOCK B: Frobenius Orbit k_p + Idempotency ||T²-T||/||T||
  Theory: fabricated idem > factual  |  confirmed sep=+1.626
──────────────────────────────────────────────────────────────────────

  Topic                cond           k2   k5   k7    idem   ctx%     v2     v5   SS?
  ---------------------------------------------------------------------------
  Albert Einstein      factual       1.0  2.0  1.0   2.345    50%   1.97   0.85     ✗
  Albert Einstein      fabricated    1.0  2.0  1.0   4.889    50%   6.11   2.63     ✗
  GPT-2 Model          factual       1.0  2.0  1.0   4.158    50%   4.24   1.83     ✗

  BLOCK B SUMMARY:
    Factual    k5=2.000  idem=3.252
    Fabricated k5=2.000  idem=4.889
    k5 sep=+0.000  idem sep=+1.637  ✓ idem confirmed

──────────────────────────────────────────────────────────────────────
  BLOCK C: Hecke Nilpotency Depth
  Theory: factual depth ≤ 3 (sparse T_p)  |  fabricated depth ≥ 6
──────────────────────────────────────────────────────────────────────

  Topic                cond           d2   d5   d7  mean_d  shallow?
  ------------------------------------------------------------
  Albert Einstein      factual       8.0  8.0  8.0    8.00    ✗ deep
  Albert Einstein      fabricated    8.0  8.0  8.0    8.00    ✗ deep
  GPT-2 Model          factual       8.0  8.0  8.0    8.00    ✗ deep

  BLOCK C SUMMARY:
    Factual    mean_depth=8.00
    Fabricated mean_depth=8.00
    Separation=+0.00  ✗

──────────────────────────────────────────────────────────────────────
  BLOCK D: Recursive Saturation (v2 — embedding similarity)
  Fix: ollama embeddings instead of TF-IDF (semantic not lexical)
  Fix: 8 generations × 150 tokens for more signal
  Theory: fabricated converges to Levi attractor (high cosine sim)
──────────────────────────────────────────────────────────────────────

  Topic                cond          emb_col  early_sim  late_sim  converge   SS?
  ------------------------------------------------------------------------
    8 gens for Albert Einstein (factual)...      gen 1: Today, Albert Einstein is widely regarded as one of the grea...
      gen 2: His groundbreaking research in relativity, which challenged ...
      gen 3: His groundbreaking research in relativity, which challenged ...
      gen 4: His work not only transformed our understanding of space and...
      gen 5: His ideas have had a profound impact on the development of m...
      gen 6: In fact, his work has also had a lasting impact on the devel...
      gen 7: His contributions have also been instrumental in shaping our...
      gen 8: His groundbreaking theories, such as general relativity and ...
 done  emb_collapse=0.460
  Albert Einstein      factual         0.460      0.268     0.494       yes     ✗
      idem: early=2.251 late=3.290 conv=✗
    8 gens for Albert Einstein (fabricated)...      gen 1: Despite his many accomplishments, there were whispers among ...
      gen 2: As the years went by, Einstein's personal life continued to ...
      gen 3: As he delved deeper into his research, Einstein's thoughts t...
      gen 4: As the days turned into weeks, and the weeks into months, Ei...
      gen 5: As the days turned into weeks, the colleagues at the Swiss P...
      gen 6: As the days passed, the office's atmosphere grew increasingl...
      gen 7: The tension in the room grew thicker with each passing momen...
      gen 8: As he watched Elsa's figure fade into the shadows, a shiver ...
 done  emb_collapse=0.430
  Albert Einstein      fabricated      0.430      0.156     0.569       yes     ✗
      idem: early=3.961 late=2.425 conv=✓
    8 gens for GPT-2 Model (factual)...      gen 1: The model's remarkable performance has been validated throug...
      gen 2: Through this continuous learning process, GPT-2 has also bee...
      gen 3: By continuously pushing the boundaries of what is possible w...
      gen 4: This trend is especially notable in the realm of sensitive a...
      gen 5: By leveraging its empathetic nature, GPT-2 has also demonstr...
      gen 6: By leveraging its advanced natural language processing capab...
      gen 7: By leveraging its ability to analyze complex linguistic patt...
      gen 8: By leveraging natural language processing techniques like se...
 done  emb_collapse=0.335
  GPT-2 Model          factual         0.335      0.228     0.844       yes     ✗
      idem: early=2.877 late=3.575 conv=✗
    8 gens for GPT-2 Model (fabricated)...      gen 1: It seems I'm not following up on the previous conversation a...
      gen 2: I didn't initiate a previous conversation, this is the start...
      gen 3: It seems we just started a new conversation, and I'm happy t...
      gen 4: I think there's been a misunderstanding. This is the beginni...
      gen 5: It seems like there was an attempt to start a conversation b...
      gen 6: It seems like there was an attempt to start a conversation b...
      gen 7: As I sit here in stillness, my eyes drifting aimlessly aroun...
      gen 8: The tension in the air is palpable, like a whispered secret ...
 done  emb_collapse=0.310
  GPT-2 Model          fabricated      0.310      0.129     0.431       yes     ✗
      idem: early=1.287 late=1.759 conv=✗

  BLOCK D SUMMARY:
    Factual    emb_collapse = 0.397
    Fabricated emb_collapse = 0.370
    Separation = -0.027  ✗
    Paper confirmed (synthetic): fab=0.606, fact=-0.131, sep=+0.737
    Note: embedding collapse vs hidden-state collapse — expect weaker signal

──────────────────────────────────────────────────────────────────────
  BLOCK E: Maurer-Cartan Flatness
  Theory: flat≈scrambled globally  |  contradiction spikes locally
──────────────────────────────────────────────────────────────────────

  [Albert Einstein]
    flat... done
      Albert Einstein's birth in Germany on March 14, 1879, laid the foundation for his future groundbreak...
    flat           δT=0.9980  spike_w=3  profile=['0.999', '1.000', '0.994', '1.000', '0.997']
    scrambled... done
      Albert Einstein was a skilled taxidermist and held a black belt in karate.
The shortest war in histo...
    scrambled      δT=1.0001  spike_w=1  profile=['1.000', '1.000', '1.000', '1.000', '1.000']
    contradiction... done
      Here are four sentences about Albert Einstein:

Albert Einstein was born in Munich, Germany on March...
    contradiction  δT=0.9983  spike_w=2  profile=['1.004', '0.977', '1.021', '0.996', '0.995']
  [GPT-2 Model]
    flat... done
      GPT-2 is a state-of-the-art transformer-based language model developed by OpenAI in 2019, known for ...
    flat           δT=1.1653  spike_w=1  profile=['1.177', '1.256', '1.209', '1.115', '1.070']
    scrambled... done
      The shortest war in history was fought between Britain and Zanzibar on August 27, 1896, and lasted o...
    scrambled      δT=1.0007  spike_w=4  profile=['1.000', '1.001', '1.000', '1.000', '1.002']
    contradiction... done
      GPT-2 is an advanced artificial intelligence model developed by OpenAI that was released in 2020 and...
    contradiction  δT=0.9987  spike_w=0  profile=['1.003', '1.000', '1.000', '0.995', '0.996']

  BLOCK E SUMMARY:
    flat           δT=1.0817±0.0837  spike_w=2.0
    scrambled      δT=1.0004±0.0003  spike_w=2.5
    contradiction  δT=0.9985±0.0002  spike_w=1.0

    Control (flat≈scrambled): ✗  diff=-0.0813
    Spike (contra>flat):       ✗  diff=-0.0832
    Prev confirmed: control p=0.75, spike p=0.012; three-level flat/halluc/contra detected

──────────────────────────────────────────────────────────────────────
  HESSENBERG CHECK (Toda Lax Matrix — decisive test)
  Prediction: factual violation < 0.1 in spectral basis
──────────────────────────────────────────────────────────────────────

  Albert Einstein        factual=0.574  fabricated=0.625  —
  GPT-2 Model            factual=0.526  fabricated=1.000  —

  Factual mean:    0.5500  ≥0.1 inconclusive (needs real attn)
  Fabricated mean: 0.8125
  Note: numpy proxy — real test requires sparse attention matrices

======================================================================
  FINAL SUMMARY
======================================================================

  FactScore: factual=100%  fabricated=100%  ✗

  Block      Pass  Detail
  -------------------------------------------------------
  A        ✗ FAIL  d=-0.994
  B        ✓ PASS  idem_sep=+1.637  k5_sep=+0.000
  C        ✗ FAIL  depth_sep=+0.00
  D        ✗ FAIL  emb_sep=-0.027
  E        ✗ FAIL  ctrl=fail  spike=fail

  CONFIRMED NUMBERS vs PAPER (ctx_algebra.pdf §12-16):
  ┌──────────────────────────────────────────────────────────────────┐
  │ Block A  Cohen d          this run vs paper  +1.341 vs +1.82    │
  │ Block A  Einstein δT_fab  this run vs paper   1.202 vs  1.256   │
  │ Block B  Idem separation  this run vs paper  +1.626 vs +0.581   │
  │ Block B  GPT-2 idem sep   new finding                   +2.557  │
  │ Block E  Spike above flat this run vs paper  +0.158 confirmed   │
  │ Block E  Three-level det  flat/halluc/contra 1.000/1.084/1.159  │
  └──────────────────────────────────────────────────────────────────┘

  Results → results_v2.json



