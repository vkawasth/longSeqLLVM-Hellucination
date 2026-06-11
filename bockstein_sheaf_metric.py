"""
bockstein_sheaf_metric.py  (v2 — alpha complex + six unified metrics)
"""
import numpy as np, gudhi, warnings
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
warnings.filterwarnings('ignore')


def build_alpha_complex(points, pca_dim=4, max_alpha_sq=None, n_subsample=32):
    pts = points - points.mean(axis=0)
    if pts.shape[1] > pca_dim:
        try:
            _, _, Vt = np.linalg.svd(pts, full_matrices=False)
            pts = pts @ Vt[:pca_dim].T
        except Exception:
            pts = pts[:, :pca_dim]
    # Subsample for tractability (alpha complex is O(n^d))
    if len(pts) > n_subsample:
        idx = np.random.choice(len(pts), n_subsample, replace=False)
        pts = pts[idx]
    pts = pts / (np.linalg.norm(pts, axis=1, keepdims=True).mean() + 1e-8)
    ac  = gudhi.AlphaComplex(points=pts.tolist())
    st  = ac.create_simplex_tree(max_alpha_square=max_alpha_sq or float('inf'))
    return st, pts


def persistent_h1(st, min_life=0.05):
    st.compute_persistence()
    raw  = st.persistence_intervals_in_dimension(1)
    if len(raw)==0: return np.zeros((0,2)), 0.0, 0.0, 0
    bars = np.array([[b,d] for b,d in raw if np.isfinite(d)])
    if not len(bars): return np.zeros((0,2)), 0.0, 0.0, 0
    mask = (bars[:,1] - bars[:,0]) > min_life
    bars = bars[mask]
    if not len(bars): return np.zeros((0,2)), 0.0, 0.0, 0
    lt = bars[:,1]-bars[:,0]
    return bars, float(lt.sum()), float(lt.max()), len(bars)


def fit_T(wa, wb, dim=8):
    def red(w):
        c = w - w.mean(0)
        if c.shape[1]<=dim: return c
        try:
            _,_,Vt = np.linalg.svd(c, full_matrices=False)
            return c@Vt[:dim].T
        except: return c[:,:dim]
    X,Y = red(wa), red(wb); n=min(len(X),len(Y)); X,Y=X[:n],Y[:n]
    eps = 1e-4*max(float(np.linalg.norm(X.T@X)),1.0)
    try:
        T,_,_,_ = np.linalg.lstsq(X.T@X+eps*np.eye(dim),X.T@Y,rcond=None)
        return T.T
    except: return np.eye(dim)


@dataclass
class SixMetrics:
    alpha:              float
    h1_total:           float;  h1_max: float;       betti_1: int;  h1_valid: bool
    frobenius_n_orbits: int;    frobenius_max_size: int
    hecke_nilpotent:    bool;   hecke_depth: int
    padic_valuation:    float;  padic_admissible: bool
    sc:                 float;  sc_excess: float
    cech_coboundary:    float;  cech_h1_generator: bool
    n_valid_metrics:    int


class UnifiedFiltration:
    def __init__(self, primes=[2,5,7], dim=8, pca_dim=8, max_depth=8, gamma=1.0):
        self.primes=primes; self.dim=dim; self.pca_dim=pca_dim
        self.max_depth=max_depth; self.gamma=gamma

    def _frobenius(self, T, p):
        scale=(p/2.0)/max(np.abs(T).max(),1e-10)
        T_p=np.round(T*scale).astype(int)%p; d=T.shape[0]
        traces=[]; Tp=np.eye(d,dtype=float)
        for _ in range(1,d+1):
            Tp=(Tp@T_p); traces.append(int(round(np.trace(Tp)))%p)
        e=[0]*(d+1); e[0]=1
        for k in range(1,d+1):
            s=sum(((-1)**(i-1))*e[k-i]*traces[i-1] for i in range(1,k+1))
            e[k]=int(round(((-1)**k*s)/k))%p
        try:
            from sympy import Poly, GF
            from sympy.abc import x as sx
            poly=Poly(list(reversed(e)),sx,domain=GF(p))
            fl=poly.factor_list()
            sizes=[f.degree() for f,m in fl[1] for _ in range(m) if f.degree()>0]
            return (len(sizes), max(sizes)) if sizes else (1,1)
        except: return 1,1

    def _hecke(self, T, p):
        scale=(p/2.0)/max(np.abs(T).max(),1e-10)
        T_p=np.round(T*scale).astype(int)%p
        Tp=T_p.copy().astype(float)
        for k in range(1,self.max_depth+1):
            Tp=(Tp@T_p)%p
            if np.all(Tp==0): return True,k
        return False,self.max_depth

    def _padic(self, T, p, alpha):
        n=T.shape[0]; det=abs(float(np.linalg.det(np.eye(n)-T)))
        vp=max(0.0,np.log(max(det,1e-14))/np.log(p))
        thr=np.log(max(alpha,1e-10))/np.log(p)-self.gamma
        return vp, vp>=thr

    def _sc(self, T):
        ev=np.abs(np.linalg.eigvals(T)); s1=ev.sum()+1e-10; s2=(ev**2).sum()+1e-10
        sc=float(s2/s1**2); return sc, sc-1.0/max(T.shape[0],1)

    def _cech(self, T, wa, wb):
        d=self.dim
        def csq(w):
            c=w-w.mean(0)
            try:
                _,_,Vt=np.linalg.svd(c,full_matrices=False); cr=c@Vt[:d].T
            except: cr=c[:,:d]
            d_=cr.shape[1]; cov=(cr.T@cr)/max(len(cr)-1,1)+1e-4*np.eye(d_)
            try: return np.linalg.cholesky(cov)
            except: return np.diag(np.sqrt(np.maximum(np.diag(cov),1e-8)))
        Ss,St=csq(wa),csq(wb); d_=Ss.shape[0]
        T_=T[:d_,:d_] if T.shape[0]>=d_ else np.eye(d_)
        cobdry=T_@Ss-St
        c=float(np.linalg.norm(cobdry,'fro')/(np.linalg.norm(St,'fro')+1e-8))
        return c, c>0.4

    def compute_at_alpha(self, wa, wb, alpha, prime=2):
        combined=np.vstack([wa,wb]); nv=0
        try:
            st,_=build_alpha_complex(combined,pca_dim=self.pca_dim,max_alpha_sq=alpha**2)
            bars,h1t,h1mx,b1,h1ok = (*persistent_h1(st),True)
            nv+=1
        except: h1t=h1mx=0.0; b1=0; h1ok=False
        T=fit_T(wa,wb,dim=self.dim)
        n_orb,max_orb=self._frobenius(T,prime); nv+=1
        nil,hd=self._hecke(T,prime); nv+=1
        vp,adm=self._padic(T,prime,alpha); nv+=1
        sc,sc_ex=self._sc(T)
        ev=(len(wa)>=8 and len(wb)>=8)
        if ev: nv+=1
        cob,is_h1=self._cech(T,wa,wb)
        if ev: nv+=1
        return SixMetrics(
            alpha=alpha,
            h1_total=h1t, h1_max=h1mx, betti_1=b1, h1_valid=h1ok,
            frobenius_n_orbits=n_orb, frobenius_max_size=max_orb,
            hecke_nilpotent=nil, hecke_depth=hd,
            padic_valuation=vp, padic_admissible=adm,
            sc=sc, sc_excess=sc_ex,
            cech_coboundary=cob, cech_h1_generator=is_h1,
            n_valid_metrics=nv,
        )

    def sweep_alpha(self, wa, wb, n_levels=8, prime=2):
        from scipy.spatial.distance import pdist
        combined=np.vstack([wa,wb])-np.vstack([wa,wb]).mean(0)
        if combined.shape[1]>self.pca_dim:
            try:
                _,_,Vt=np.linalg.svd(combined,full_matrices=False)
                combined=combined@Vt[:self.pca_dim].T
            except: combined=combined[:,:self.pca_dim]
        combined=combined/(np.linalg.norm(combined,axis=1,keepdims=True).mean()+1e-8)
        dists=pdist(combined)
        a_min=float(np.percentile(dists,5)); a_max=float(np.percentile(dists,95))
        alphas=np.linspace(a_min,a_max,n_levels)
        return [self.compute_at_alpha(wa,wb,float(a),prime) for a in alphas]


def analyze_trajectory(hs, window_size=64, n_windows=8, n_alpha=6, prime=2, dim=8):
    uf=UnifiedFiltration(dim=dim,pca_dim=dim)
    wins=[hs[k*window_size:(k+1)*window_size] for k in range(n_windows)]
    sweeps=[]
    for k in range(n_windows-1):
        if len(wins[k])>=8 and len(wins[k+1])>=8:
            sweeps.append(uf.sweep_alpha(wins[k],wins[k+1],n_levels=n_alpha,prime=prime))
    return sweeps


def print_report(sweeps, label=""):
    mid=len(sweeps[0])//2
    lines=["="*68,f"  Six-Metric Alpha Complex Report  {label}",
           "  Alpha complex (Voronoi/Delaunay) — not Rips","="*68,
           f"  {'Win':<5} {'H1':>7} {'Cob':>7} {'SC_ex':>7} {'Hecke':>6} {'Frob':>5} {'v_p':>6} valid"]
    for k,sw in enumerate(sweeps):
        m=sw[mid]
        hk="nil" if m.hecke_nilpotent else "LIVE"
        lines.append(f"  X^{k+1:<3} {m.h1_total:>7.3f} {m.cech_coboundary:>7.3f} "
                     f"{m.sc_excess:>7.4f} {hk:>6} {m.frobenius_max_size:>5} "
                     f"{m.padic_valuation:>6.3f} {m.n_valid_metrics}/6")
    lines.append("="*68)
    return "\n".join(lines)
