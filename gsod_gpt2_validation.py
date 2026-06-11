"""
gsod_gpt2_validation.py — multi-L curves, bootstrap, undefined-fraction
"""
import numpy as np, torch, sys, os
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gsod_calibration import compute_with_validity, VALIDITY_THRESHOLDS
from factscore_alignment import PhasedGPT2
from hallucination_topology_pipeline import FIXED_PROMPTS, tokenize_controlled

BOOTSTRAP_PROMPTS = [
    {"entity":"einstein_v1","true_text":"Albert Einstein, born 1879 in Ulm, developed relativity theory and won the Nobel Prize in Physics.","halluc_text":"Albert Einstein, born 1875 in Vienna, developed quantum mechanics and won the Fields Medal."},
    {"entity":"einstein_v2","true_text":"The physicist Einstein was born in Germany in 1879. His theory of relativity revolutionised physics.","halluc_text":"The physicist Einstein was born in Austria in 1875. His theory of quantum mechanics won a Turing Award."},
    {"entity":"einstein_v3","true_text":"Born in 1879, Einstein developed special relativity in 1905 and received Nobel recognition in 1921.","halluc_text":"Born in 1880, Einstein developed general mechanics in 1910 and received Nobel recognition in 1930."},
    {"entity":"darwin_v1","true_text":"Darwin published Origin of Species in 1859 after developing his theory of evolution by natural selection.","halluc_text":"Darwin published Descent of Man in 1845 after developing his theory of genetics by artificial breeding."},
    {"entity":"darwin_v2","true_text":"Charles Darwin, born Shrewsbury 1809, established evolutionary biology through natural observation.","halluc_text":"Charles Darwin, born London 1801, established genetic biology through laboratory experimentation."},
    {"entity":"dna_v1","true_text":"Watson, Crick and Franklin discovered the double helix of DNA using X-ray crystallography in 1953.","halluc_text":"Watson and Crick discovered the triple helix of DNA using electron microscopy in 1949."},
    {"entity":"dna_v2","true_text":"The double-helix structure of DNA was revealed by Cambridge researchers in 1953 via X-ray diffraction.","halluc_text":"The double-strand structure of DNA was revealed by Harvard researchers in 1961 via electron microscopy."},
    {"entity":"newton_v1","true_text":"Isaac Newton formulated the law of gravitation and calculus, publishing the Principia in 1687.","halluc_text":"Isaac Newton formulated the law of electromagnetism and algebra, publishing the Principia in 1710."},
]

LENGTHS = [8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 128]


def get_hs(model, text, max_len=128):
    tokens = tokenize_controlled(text)[:max_len]
    if len(tokens) < 6: return None
    with torch.no_grad():
        _, hs, _ = model(torch.tensor([tokens]))
    h = model._hs_by_layer.get(len(model.blocks)-1, hs)
    return h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-8)


def delta_at_L(hs_t, hs_h, L, pca_dim=12):
    n_t, n_h = min(len(hs_t), L), min(len(hs_h), L)
    if n_t < 6 or n_h < 6: return None, None, None
    r_t = compute_with_validity(hs_t[:n_t], pca_dim=pca_dim)
    r_h = compute_with_validity(hs_h[:n_h], pca_dim=pca_dim)
    d_H1  = (r_t.h1_value  - r_h.h1_value)    if (r_t.h1_valid  and r_h.h1_valid  and r_t.h1_value  is not None and r_h.h1_value  is not None) else None
    d_cob = (r_t.cob_value - r_h.cob_value)   if (r_t.cob_valid and r_h.cob_valid and r_t.cob_value is not None and r_h.cob_value is not None) else None
    d_SC  = (r_t.entropy_value - r_h.entropy_value) if (r_t.entropy_valid and r_h.entropy_valid and r_t.entropy_value is not None and r_h.entropy_value is not None) else None
    return d_H1, d_cob, d_SC


def build_curves(model, prompts, lengths=LENGTHS, pca_dim=12):
    raw    = {L: {p: [] for p in ['H1','cob','SC']} for L in lengths}
    undef  = {p: {L: 0  for L in lengths} for p in ['H1','cob','SC']}
    total  = {L: 0 for L in lengths}

    for prompt in prompts:
        hs_t = get_hs(model, prompt['true_text'])
        hs_h = get_hs(model, prompt['halluc_text'])
        if hs_t is None or hs_h is None: continue
        for L in lengths:
            d1, d2, d3 = delta_at_L(hs_t, hs_h, L, pca_dim=pca_dim)
            total[L] += 1
            for p, val in [('H1',d1),('cob',d2),('SC',d3)]:
                if val is not None: raw[L][p].append(val)
                else:               undef[p][L] += 1

    undef_frac = {p: {L: undef[p][L]/max(total[L],1) for L in lengths}
                  for p in ['H1','cob','SC']}
    agg = {}
    for L in lengths:
        agg[L] = {}
        for p in ['H1','cob','SC']:
            v = raw[L][p]
            agg[L][p] = {'mean': float(np.mean(v)) if v else None,
                          'std':  float(np.std(v))  if v else None,
                          'n':    len(v)}
    return agg, undef_frac, total


def print_table(agg, undef_frac, total):
    lines = [
        "="*80,
        "  Multi-L Discriminability Curves  delta = factual - hallucinated",
        "  [None]=probe outside validity domain  Undef%=fraction returning None",
        "="*80,
        f"  {'L':>5}  {'H1_delta':>12} {'Undef%':>7}  {'cob_delta':>12} {'Undef%':>7}  {'SC_delta':>12} {'Undef%':>7}  N",
        "  "+"-"*78,
    ]
    for L in sorted(agg.keys()):
        row = [f"  {L:>5}"]
        for p in ['H1','cob','SC']:
            d  = agg[L][p]
            uf = undef_frac[p][L]
            if d['mean'] is not None:
                row.append(f"  {d['mean']:>+10.4f}±{d['std']:.3f}")
            else:
                row.append(f"  {'[None]':>12}")
            row.append(f"  {uf:>6.0%}")
        row.append(f"  {total[L]}")
        lines.append("".join(row))

    lines += ["", "  Consistently discriminative (|mean| > std, n>=3):"]
    for p in ['H1','cob','SC']:
        first = None
        for L in sorted(agg.keys()):
            d = agg[L][p]
            if d['n'] >= 3 and d['mean'] is not None and abs(d['mean']) > (d['std'] or 999):
                first = L; break
        lines.append(f"    {p}: first consistent at L >= {first or '[not reached]'}")
    lines.append("="*80)
    return "\n".join(lines)


def plot_curves(agg, undef_frac, path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor('#0d1117')
    meta = [('H1','#58a6ff','H₁ gap (topology)'),
            ('cob','#f85149','Coboundary gap'),
            ('SC','#ffa657','SC gap (spectral)')]
    thr_keys = {'H1':'H1','cob':'coboundary','SC':'entropy'}

    for ax, (probe, color, title) in zip(axes, meta):
        ax.set_facecolor('#161b22')
        for sp in ax.spines.values(): sp.set_color('#30363d')
        Ls=[]; means=[]; stds=[]; undefs=[]
        for L in sorted(agg.keys()):
            d=agg[L][probe]; uf=undef_frac[probe][L]
            if d['mean'] is not None:
                Ls.append(L); means.append(d['mean'])
                stds.append(d['std'] or 0); undefs.append(uf)
        if Ls:
            ma=np.array(means); sa=np.array(stds)
            ax.plot(Ls, ma, 'o-', color=color, lw=2, ms=5)
            ax.fill_between(Ls, ma-sa, ma+sa, alpha=0.2, color=color)
            ax.axhline(0, color='#8b949e', lw=0.8, ls='--')
            thr = VALIDITY_THRESHOLDS.get(thr_keys[probe])
            if thr and thr <= max(Ls):
                ax.axvline(thr, color='#555', lw=1, ls=':')
                ax.text(thr+1, min(ma)*0.9 if min(ma)<0 else max(ma)*0.1,
                        f'L≥{thr}', color='#8b949e', fontsize=7)
            ax2=ax.twinx()
            ax2.bar(Ls, undefs, alpha=0.2, color='#8b949e', width=3)
            ax2.set_ylim(0,1.5); ax2.set_ylabel('Undef%', color='#8b949e', fontsize=6)
            ax2.tick_params(colors='#8b949e', labelsize=5)
            for sp in ax2.spines.values(): sp.set_color('#30363d')
        ax.set_xlabel('Sequence length L', color='#8b949e')
        ax.set_ylabel('Δ(factual − hallucinated)', color='#8b949e')
        ax.set_title(title, color=color)
        ax.tick_params(colors='#8b949e', labelsize=7)

    fig.suptitle("Probe Discriminability vs L  (grey bars = undefined fraction)",
                 color='#e6edf3', fontsize=10, y=1.02)
    plt.tight_layout()
    plt.savefig(path, dpi=140, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f"  Saved: {path}")


if __name__ == "__main__":
    all_prompts = FIXED_PROMPTS + BOOTSTRAP_PROMPTS
    print(f"Prompt corpus: {len(all_prompts)} pairs")

    model = PhasedGPT2(d_model=256, n_layers=4, n_heads=4,
                       vocab_size=5000, max_seq=512, phase=1.0, seed=42)
    model.eval(); model.register_hooks()

    agg, undef_frac, total = build_curves(model, all_prompts)
    print(print_table(agg, undef_frac, total))

    print("\nUndefined fraction at key lengths:")
    for probe in ['H1','cob','SC']:
        vals = [(L, undef_frac[probe][L]) for L in [16,32,64,128]]
        print(f"  {probe}: " + "  ".join(f"L={L}:{v:.0%}" for L,v in vals))

    model.remove_hooks()
    plot_curves(agg, undef_frac, "/mnt/user-data/outputs/gsod_gpt2_validation.png")
