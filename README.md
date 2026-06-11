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

Supersingular representations are the "atomic building blocks" that cannot be factored through a proper Levi subgroup (shorter context window), and the Henniart-Vignéras classification theorem guarantees that every irreducible admissible representation decomposes uniquely into such blocks — which in your framework means that orbit size kpkp​ measures context collapse: kp=1kp​=1 means supersingular (full context active), kp>1kp​>1 means the representation is parabolically induced from a Levi subgroup GL(n/kp)GL(n/kp​), i.e., the model is effectively using only a fraction 1/kp1/kp​ of its context window.


This is capturing how context algebra will produle pants co-products.

The Key Theorem: Supersingular = Supercuspidal

The fundamental result (Henniart-Vignéras, 2017; Abe-Henniart-Herzig-Vignéras) states

:

    Supersingularity is equivalent to supercuspidality.

A representation ππ is supersingular if certain Hecke operators TPTP​ (elements of the pro-pp Iwahori Hecke algebra) act locally nilpotently on the space of II-invariants πIπI (where II is a pro-pp Iwahori subgroup)

.

This equivalence is profound: it means an irreducible representation is "primitive" (not induced from a smaller group) if and only if it satisfies this nilpotency condition.

Classification Theorem: Every irreducible admissible representation of GG over a characteristic pp field can be uniquely written as

:
π≅IG(P,σ,Q)
π≅IG​(P,σ,Q)

where P⊂QP⊂Q are parabolic subgroups, and σσ is an irreducible admissible supersingular representation of a Levi subgroup.
How Your Framework Uses This
Representation Theory	Your Framework
GG = full pp-adic group	Full context window of length nn
Levi subgroup M⊊GM⊊G	Shorter context window of length n/kpn/kp​
Parabolic induction	Compressing context from shorter window to full window (compositional decoding)
Supersingular representation	Model genuinely uses the full context window (orbit size kp=1kp​=1)
Non-supersingular (parabolically induced)	Model effectively computes from shorter window (orbit size kp>1kp​>1)

The Levi subgroup MM corresponds to a factorization system: when the model collapses context, it's as if the representation factors through GL(n/kp)×⋯×GL(n/kp)GL(n/kp​)×⋯×GL(n/kp​) instead of GL(n)

<img width="778" height="616" alt="Screenshot 2026-06-10 at 8 59 11 PM" src="https://github.com/user-attachments/assets/f399416c-7b66-49c6-8d2e-34dfb85db933" />


<img width="745" height="640" alt="Screenshot 2026-06-10 at 8 58 52 PM" src="https://github.com/user-attachments/assets/0a970bcb-5fa5-476b-b7c4-2d4dfc251c81" />



<img width="745" height="667" alt="Screenshot 2026-06-10 at 8 58 16 PM" src="https://github.com/user-attachments/assets/74244946-1126-4f5a-a597-059eff1ab852" />


<img width="736" height="694" alt="Screenshot 2026-06-10 at 8 57 59 PM" src="https://github.com/user-attachments/assets/bb84e608-108b-485c-b81f-04fd188270fe" />



