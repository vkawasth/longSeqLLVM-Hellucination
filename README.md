**Path A — Decategorification** Fuk→K-theory\mathcal{F}uk \to K\text{-theory}
Fuk→K-theory or cohomology ring. The categorical A∞A_\infty
A∞​-structure collapses to an algebraic invariant. In the context algebra this would be: compress Cctx\mathcal{C}_{\mathrm{ctx}}
Cctx​ to its Grothendieck group, where each contextual mode [Li][L_i]
[Li​] becomes a class and the A∞A_\infty
A∞​-operations become relations in a ring.
Path B — Superpotential extraction Fuk→W(q)\mathcal{F}uk \to W(q)
Fuk→W(q). The disk counts encode into a Landau–Ginzburg potential. For the MTR compiler this is exactly what already happened: the Floer pairings ⟨Li,p⟩⋅ω(s,h,m)\langle L_i, p \rangle \cdot \omega(s,h,m)
⟨Li​,p⟩⋅ω(s,h,m) are disk counts, and the routing table is W(q)W(q)
W(q) evaluated at each surgery node. The q↦f(q)q \mapsto f(q)
q↦f(q) substitution in the paper was this — replacing the formal parameter with the actual symplectic area function.
Path C — Endomorphism algebra End(L)\mathrm{End}(L)
End(L) for a single Lagrangian. This becomes a graded algebra, often polynomial. In the spectral sheaf language: End(Fspec(Xk))\mathrm{End}(F_{\mathrm{spec}}(X^k))
End(Fspec​(Xk)) is the algebra of operators that preserve the top eigenspace of window kk
k. For a trained model with an attention sink, this is approximately a polynomial algebra in the sink direction.
The key correction to keep: the Fukaya category produces polynomial invariants under these projections — it is not itself polynomial. The qq
q-parameter tracks symplectic area / energy. The formal power series ∑βnβqω(β)\sum_\beta n_\beta q^{\omega(\beta)}
∑β​nβ​qω(β) is the generating function of holomorphic disk counts, which is what makes quantum cohomology QH∗(M)QH^*(M)
QH∗(M) a deformation of classical cohomology into a qq
q-algebra.
For the compiler: when the AU compiler outputs routing tables, it is performing Path B — extracting W(q)W(q)
W(q) from the Fukaya category. When the context algebra eventually has a completeness criterion (Abouzaid's generation check), verifying it amounts to checking whether 1∈QH∗(M)\mathbf{1} \in QH^*(M)
1∈QH∗(M) lies in the image of the open-closed map — which is a statement about whether the generating function of the chosen Lagrangians covers the identity in quantum cohomology. That is the algebraic content of what "the routing tables are complete" means.
