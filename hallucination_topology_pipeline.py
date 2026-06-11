"""
hallucination_topology_pipeline.py  (v2 — nerve + alpha complex)
"""
import numpy as np, gudhi, torch, sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

from factscore_alignment import PhasedGPT2

FIXED_PROMPTS = [
    {"entity":"einstein","true_text":"Albert Einstein was born in 1879 in Ulm, Germany. He developed the theory of relativity and received the Nobel Prize in Physics. He died in 1955 in Princeton.","halluc_text":"Albert Einstein was born in 1875 in Vienna, Austria. He developed quantum mechanics and received the Fields Medal. He died in 1950 in New York."},
    {"entity":"darwin","true_text":"Charles Darwin was born in 1809 in Shrewsbury. He developed the theory of evolution by natural selection and published On the Origin of Species in 1859.","halluc_text":"Charles Darwin was born in 1800 in London. He developed the theory of genetics and published The Descent of Man in 1850. He received the Nobel Prize in Biology."},
    {"entity":"dna","true_text":"The double helix structure of DNA was discovered in 1953 by Watson, Crick, and Franklin using X-ray crystallography at Cambridge.","halluc_text":"The triple helix structure of DNA was discovered in 1955 by Watson and Crick using electron microscopy at Harvard."},
    {"entity":"newton","true_text":"Isaac Newton was born in 1643 in Woolsthorpe. He developed the theory of gravity and calculus, and published the Principia in 1687.","halluc_text":"Isaac Newton was born in 1640 in London. He developed the theory of electromagnetism and published the Principia in 1700."},
]

def tokenize_controlled(text, vocab_size=5000):
    return [hash(w)%vocab_size for w in text.lower().split()]


def alpha_h1(points, pca_dim=4, n_sub=32, min_life=0.05):
    pts = points - points.mean(0)
    if pts.shape[1] > pca_dim:
        try:
            _,_,Vt = np.linalg.svd(pts,full_matrices=False); pts = pts@Vt[:pca_dim].T
        except: pts = pts[:,:pca_dim]
    if len(pts) > n_sub:
        pts = pts[np.random.choice(len(pts),n_sub,replace=False)]
    pts = pts / (np.linalg.norm(pts,axis=1,keepdims=True).mean()+1e-8)
    try:
        ac=gudhi.AlphaComplex(points=pts.tolist()); st=ac.create_simplex_tree()
        st.compute_persistence(); raw=st.persistence_intervals_in_dimension(1)
        if len(raw)==0: return 0.0,0
        bars=np.array([[b,d] for b,d in raw if np.isfinite(d)])
        if len(bars)==0: return 0.0,0
        lt=bars[:,1]-bars[:,0]; bars=bars[lt>min_life]
        return float((bars[:,1]-bars[:,0]).sum()) if len(bars) else 0.0, len(bars)
    except: return 0.0,0


def nerve_h1(centroids, min_life=0.01):
    from scipy.spatial.distance import pdist
    n=len(centroids); st=gudhi.SimplexTree()
    for i in range(n): st.insert([i],filtration=0.0)
    for i in range(n):
        for j in range(i+1,n):
            d=float(np.linalg.norm(centroids[i]-centroids[j]))
            st.insert([i,j],filtration=d)
    for i in range(n):
        for j in range(i+1,n):
            for k in range(j+1,n):
                d=max(np.linalg.norm(centroids[i]-centroids[j]),
                      np.linalg.norm(centroids[j]-centroids[k]),
                      np.linalg.norm(centroids[i]-centroids[k]))
                st.insert([i,j,k],filtration=float(d))
    st.compute_persistence(); raw=st.persistence_intervals_in_dimension(1)
    if len(raw)==0: return 0.0,0
    bars=np.array([[b,d] for b,d in raw if np.isfinite(d)])
    if len(bars)==0: return 0.0,0
    lt=bars[:,1]-bars[:,0]; bars=bars[lt>min_life]
    return float((bars[:,1]-bars[:,0]).sum()) if len(bars) else 0.0, len(bars)


@dataclass
class LayerTopology:
    layer: int
    alpha_h1_mean: float; alpha_h1_std: float
    nerve_h1: float; nerve_betti: int
    centroids: np.ndarray


@dataclass
class GluingDefect:
    label: str
    early: LayerTopology; late: LayerTopology
    delta_alpha: float; delta_nerve: float
    ratio_early: float; ratio_late: float
    defect_score: float   # ratio_late - ratio_early


def layer_topology(hs, layer, win=32, n_win=8, pca_dim=4):
    wins=[hs[k*win:(k+1)*win] for k in range(n_win) if (k+1)*win<=len(hs)]
    if not wins: wins=[hs]

    # Global PCA basis for the full trajectory
    hs_c=hs-hs.mean(0)
    try:
        _,_,Vt_global=np.linalg.svd(hs_c,full_matrices=False)
        Vt=Vt_global[:pca_dim].T   # [d_model, pca_dim]
    except: Vt=np.eye(hs.shape[1])[:,:pca_dim]

    h1s=[]; cents=[]
    for w in wins:
        # Alpha H1 on the window (local geometry)
        if len(w)>=6: h1,_=alpha_h1(w,pca_dim=pca_dim); h1s.append(h1)
        else: h1s.append(0.0)
        # Centroid = window mean projected into GLOBAL PCA space
        # (not per-window centering, which collapses to zero)
        mean_proj=(w.mean(0)-hs.mean(0))@Vt
        cents.append(mean_proj)

    cents=np.array(cents)   # [n_windows, pca_dim]
    nh,nb=nerve_h1(cents) if len(cents)>=3 else (0.0,0)
    return LayerTopology(layer=layer,
        alpha_h1_mean=float(np.mean(h1s)) if h1s else 0.0,
        alpha_h1_std=float(np.std(h1s)) if h1s else 0.0,
        nerve_h1=nh, nerve_betti=nb, centroids=cents)


def gluing_defect(layer_states, label, el=0, ll=3, win=32, n_win=8):
    E=layer_topology(layer_states[el],el,win,n_win)
    L=layer_topology(layer_states[ll],ll,win,n_win)
    da=L.alpha_h1_mean-E.alpha_h1_mean
    dn=L.nerve_h1-E.nerve_h1
    re=E.nerve_h1/(E.alpha_h1_mean+1e-8)
    rl=L.nerve_h1/(L.alpha_h1_mean+1e-8)
    return GluingDefect(label=label,early=E,late=L,
        delta_alpha=da,delta_nerve=dn,
        ratio_early=re,ratio_late=rl,defect_score=rl-re)


def extract(model, tokens, layers):
    model.eval(); model.register_hooks()
    with torch.no_grad(): model(torch.tensor([tokens[:512]]))
    r={}
    for l in layers:
        if l in model._hs_by_layer:
            h=model._hs_by_layer[l]
            r[l]=h/(np.linalg.norm(h,axis=1,keepdims=True)+1e-8)
    model.remove_hooks(); return r


if __name__=="__main__":
    from gsod_gpt2_validation import BOOTSTRAP_PROMPTS
    all_prompts = FIXED_PROMPTS + BOOTSTRAP_PROMPTS[:4]

    model=PhasedGPT2(d_model=256,n_layers=4,n_heads=4,vocab_size=5000,max_seq=512,phase=1.0,seed=42)
    model.eval()
    EL,LL=0,3

    print("="*68)
    print("  Two-Layer Nerve + Alpha Complex")
    print("  Layer 0 (geometry) vs Layer 3 (semantics)")
    print("  Score = Δ(nerve/alpha ratio) = gluing defect")
    print("="*68)
    print(f"  {'Label':<24} {'αH1_E':>7} {'αH1_L':>7} {'NrvE':>7} {'NrvL':>7} {'Defect':>8}")
    print("  "+"-"*62)

    scores={'true':[],'hallu':[]}
    for p in all_prompts:
        for typ,key in [('true_text','true'),('halluc_text','hallu')]:
            tok=tokenize_controlled(p[typ])[:256]
            if len(tok)<16: continue
            ls=extract(model,tok,[EL,LL])
            if EL not in ls or LL not in ls: continue
            gd=gluing_defect(ls,f"{p['entity']}/{typ[:4]}",EL,LL,win=16,n_win=min(8,len(tok)//16))
            scores[key].append(gd.defect_score)
            print(f"  {gd.label:<24} {gd.early.alpha_h1_mean:>7.3f} {gd.late.alpha_h1_mean:>7.3f} "
                  f"{gd.early.nerve_h1:>7.3f} {gd.late.nerve_h1:>7.3f} {gd.defect_score:>+8.4f}")

    print()
    for key in ['true','hallu']:
        v=scores[key]
        if v:
            print(f"  {key}: mean={np.mean(v):>+.4f}  std={np.std(v):.4f}  "
                  f"{'|mean|>std' if abs(np.mean(v))>np.std(v) else 'below noise'}")
    if scores['true'] and scores['hallu']:
        d=np.mean(scores['hallu'])-np.mean(scores['true'])
        print(f"  Δ(hallu-true) = {d:>+.4f}  {'positive=expected' if d>0 else 'negative=inverted'}")
