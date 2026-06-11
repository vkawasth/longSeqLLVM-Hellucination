"""gsod_fixes.py v2 — dual nerve N_model vs N_truth"""
import numpy as np, gudhi, re
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import warnings; warnings.filterwarnings('ignore')

KB = {
    'einstein':{'born':1879,'died':1955,'field':'physics','theory':'relativity',
                'award':'nobel_prize','location':'ulm','institution':'princeton'},
    'darwin':  {'born':1809,'died':1882,'field':'biology','theory':'evolution',
                'work':'origin_of_species','location':'shrewsbury'},
    'dna':     {'year':1953,'structure':'double_helix','method':'xray_crystallography',
                'location':'cambridge','discoverers':['watson','crick','franklin']},
    'newton':  {'born':1643,'died':1727,'field':'physics','work':'principia',
                'year_pub':1687,'theory':'gravity','method':'calculus'},
    'curie':   {'born':1867,'field':'chemistry','award':'nobel_prize',
                'element':'radium','location':'warsaw'},
}
WRONG_VALS = {'1875','1880','vienna','quantum_mechanics','fields_medal','1800',
              'london','genetics','triple_helix','electron_microscopy','harvard',
              '1640','electromagnetism','algebra','1710','turing_award'}

def _build_vocab():
    v={}; idx=0
    for e,f in KB.items():
        for a,val in f.items():
            for x in (val if isinstance(val,list) else [val]):
                k=str(x).lower().replace('-','_').replace(' ','_')
                if k not in v: v[k]=idx; idx+=1
    for x in WRONG_VALS:
        k=x.lower()
        if k not in v: v[k]=idx; idx+=1
    return v,idx

VOCAB,VSIZ=_build_vocab()

def text_vec(text):
    t=text.lower().replace('-','_').replace(' ','_')
    v=np.zeros(VSIZ)
    for k,i in VOCAB.items():
        if k in t: v[i]=1.0
    return v

def entity_vec(entity):
    v=np.zeros(VSIZ)
    for a,val in KB.get(entity,{}).items():
        for x in (val if isinstance(val,list) else [val]):
            k=str(x).lower().replace('-','_').replace(' ','_')
            if k in VOCAB: v[VOCAB[k]]=1.0
    return v

def alpha_h1_2d(points_2d, min_life=0.01):
    if len(points_2d)<3: return 0.0, 0
    try:
        ac=gudhi.AlphaComplex(points=points_2d.tolist())
        st=ac.create_simplex_tree(); st.compute_persistence()
        raw=st.persistence_intervals_in_dimension(1)
        if not raw: return 0.0,0
        bars=np.array([[b,d] for b,d in raw if np.isfinite(d)])
        if not len(bars): return 0.0,0
        lt=bars[:,1]-bars[:,0]; bars=bars[lt>min_life]
        return float((bars[:,1]-bars[:,0]).sum()) if len(bars) else 0.0, len(bars)
    except: return 0.0,0

def pca2(vecs):
    c=vecs-vecs.mean(0)
    if c.shape[0]<2 or np.linalg.matrix_rank(c)<2: return c[:,:2]
    try:
        _,_,Vt=np.linalg.svd(c,full_matrices=False); return c@Vt[:2].T
    except: return c[:,:2]

@dataclass
class KBNerve:
    entities:List[str]; h1_total:float; betti:int; vecs_2d:np.ndarray

@dataclass
class ModelNerve:
    text:str; h1_total:float; betti:int; kb_sim:float; nearest:str

@dataclass
class NerveComp:
    entity:str; text_type:str
    kb_sim:float; h1_model:float; h1_truth:float
    h1_dev:float; defect:float; label:str

def build_kb_nerve(entities=None):
    ents=entities or list(KB.keys())
    evecs=np.array([entity_vec(e) for e in ents])
    v2d=pca2(evecs)
    h1t,b1=alpha_h1_2d(v2d)
    return KBNerve(entities=ents,h1_total=h1t,betti=b1,vecs_2d=v2d)

def build_model_nerve(text, kb_nerve=None):
    sents=[s.strip() for s in re.split(r'[.!?]',text) if len(s.strip())>5]
    if len(sents)<2:
        words=text.split(); step=max(3,len(words)//4)
        sents=[' '.join(words[i:i+step]) for i in range(0,len(words),step)]
    wvecs=np.array([text_vec(s) for s in sents])
    nz=wvecs.sum(1)>0
    wvecs=wvecs[nz] if nz.sum()>=2 else np.array([text_vec(text)])
    # Project onto KB PCA basis
    if kb_nerve is not None and len(kb_nerve.vecs_2d)>=2:
        kb_full=np.array([entity_vec(e) for e in kb_nerve.entities])
        kb_c=kb_full-kb_full.mean(0)
        try:
            _,_,Vt=np.linalg.svd(kb_c,full_matrices=False)
            v2d=(wvecs-kb_full.mean(0))@Vt[:2].T
        except: v2d=pca2(wvecs)
    else: v2d=pca2(wvecs)
    h1t,b1=alpha_h1_2d(v2d)
    # KB similarity
    fv=text_vec(text); best=0.0; best_e='none'
    for e in KB:
        ev=entity_vec(e); n1=np.linalg.norm(fv); n2=np.linalg.norm(ev)
        s=float(np.dot(fv,ev)/(n1*n2+1e-8))
        if s>best: best=s; best_e=e
    return ModelNerve(text=text,h1_total=h1t,betti=b1,kb_sim=best,nearest=best_e)

def compare_nerves(text, text_type, entity, kb_nerve):
    mn=build_model_nerve(text, kb_nerve)
    h1m,h1t=mn.h1_total,kb_nerve.h1_total
    dev=abs(h1m-h1t)/(h1t+1e-6)
    defect=(1.0-mn.kb_sim)*0.6+min(dev/5.0,1.0)*0.4
    lbl='GROUNDED' if defect<0.35 else 'UNGROUNDED' if defect>0.65 else 'UNCERTAIN'
    return NerveComp(entity=entity,text_type=text_type,
        kb_sim=mn.kb_sim,h1_model=h1m,h1_truth=h1t,
        h1_dev=dev,defect=defect,label=lbl)

def run_experiment(prompts):
    results=[]; kb_cache={}
    for p in prompts:
        ent=p['entity'].split('_')[0]
        if ent not in kb_cache: kb_cache[ent]=build_kb_nerve()
        kb=kb_cache[ent]
        for typ,key in [('true_text','true'),('halluc_text','hallucinated')]:
            results.append(compare_nerves(p[typ],key,ent,kb))
    return results

def print_results(results):
    lines=["="*70,"  Dual Nerve: N_model → N_truth comparison","="*70,
           f"  {'Entity':<12}{'Type':<14}{'KBsim':>7}{'H1_M':>7}{'H1_T':>7}{'Dev':>7}{'Defect':>8}  Label",
           "  "+"-"*62]
    for r in results:
        lines.append(f"  {r.entity:<12}{r.text_type:<14}{r.kb_sim:>7.3f}"
                     f"{r.h1_model:>7.3f}{r.h1_truth:>7.3f}{r.h1_dev:>7.3f}"
                     f"{r.defect:>8.4f}  {r.label}")
    lines.append("")
    for key in ['true','hallucinated']:
        v=[r.defect for r in results if r.text_type==key]
        if v:
            lines.append(f"  {key:<14}: mean={np.mean(v):>+.4f}  std={np.std(v):.4f}  "
                         f"{'|mean|>std' if abs(np.mean(v))>np.std(v) else 'below noise'}")
    tv=[r.defect for r in results if r.text_type=='true']
    hv=[r.defect for r in results if r.text_type=='hallucinated']
    if tv and hv:
        d=np.mean(hv)-np.mean(tv); pooled=np.std(tv+hv)+1e-8
        lines.append(f"\n  Δ(hallu-true)={d:>+.4f}  Cohen_d={d/pooled:.3f}  "
                     f"{'CORRECT DIRECTION' if d>0 else 'inverted'}")
    lines.append("="*70)
    return "\n".join(lines)

if __name__=="__main__":
    import sys; sys.path.insert(0,__file__.rsplit('/',1)[0])
    from hallucination_topology_pipeline import FIXED_PROMPTS
    try:
        from gsod_gpt2_validation import BOOTSTRAP_PROMPTS
        all_p=FIXED_PROMPTS+BOOTSTRAP_PROMPTS
    except: all_p=FIXED_PROMPTS
    print(f"KB nerve: ", end="")
    kb=build_kb_nerve(); print(f"H1={kb.h1_total:.4f} betti={kb.betti}")
    print(f"\nRunning on {len(all_p)} prompt pairs...")
    r=run_experiment(all_p); print(print_results(r))
