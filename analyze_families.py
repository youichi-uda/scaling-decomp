#!/usr/bin/env python
"""Cross-tokenizer-family analysis: fit per-class α within each family, compare."""
import json, numpy as np

FAMILIES = {
    "pythia": [
        ("pythia-70m", 70e6),     ("pythia-160m", 160e6),
        ("pythia-410m", 410e6),   ("pythia-1b", 1.0e9),
        ("pythia-1.4b", 1.4e9),   ("pythia-2.8b", 2.8e9),
        ("pythia-6.9b", 6.9e9),   ("pythia-12b", 12e9),
    ],
    "gpt2": [
        ("gpt2_124m", 124e6),  ("gpt2_355m", 355e6),
        ("gpt2_774m", 774e6),  ("gpt2_1.5b", 1.5e9),
    ],
    "opt": [
        ("opt-125m", 125e6),  ("opt-350m", 350e6),
        ("opt-1.3b", 1.3e9),  ("opt-2.7b", 2.7e9),
    ],
}

PATH = {
    "pythia": lambda k: f"v2_{k.split('-')[1]}.json",
    "gpt2":   lambda k: f"v2_{k}.json",
    "opt":    lambda k: f"v2_{k}.json",
}


def midwf(d, lo=4, hi=7):
    fp = d["by_wordfreqdecile_position"]; iw=cw=inn=cnn=0.0
    for b in range(lo, hi+1):
        a=fp[str(b)]["word_initial"]; c=fp[str(b)]["word_continuation"]
        if a["mean_loss"] is not None: iw+=a["mean_loss"]*a["n"]; inn+=a["n"]
        if c["mean_loss"] is not None: cw+=c["mean_loss"]*c["n"]; cnn+=c["n"]
    return (iw/inn if inn else np.nan, cw/cnn if cnn else np.nan)


def fit(N, L, grid=np.linspace(0.05, 1.8, 800)):
    N=np.asarray(N,float); L=np.asarray(L,float); best=None
    for al in grid:
        x=N**(-al); B=np.vstack([np.ones_like(x),x]).T
        coef,*_=np.linalg.lstsq(B,L,rcond=None); pred=B@coef
        sse=np.sum((pred-L)**2)
        if best is None or sse<best[3]: best=(al, coef[0], coef[1], sse, pred)
    return best


def boot_diff(N, init, cont, B=2000, seed=20260601):
    fi=fit(N,init); fc=fit(N,cont)
    ri=np.array(init)-fi[4]; rc=np.array(cont)-fc[4]
    rng=np.random.default_rng(seed)
    diffs=np.zeros(B)
    for b in range(B):
        Li=fi[4]+rng.choice(ri, size=len(N), replace=True)
        Lc=fc[4]+rng.choice(rc, size=len(N), replace=True)
        diffs[b]=fit(N,Li)[0]-fit(N,Lc)[0]
    q025,q500,q975=np.quantile(diffs,[0.025,0.5,0.975])
    return fi, fc, diffs, (q025,q500,q975)


print("="*70)
print(f"{'family':10}{'a_init':>9}{'E_init':>9}{'a_cont':>9}{'E_cont':>9}{'diff':>9}{'CI95':>22}{'P(d>=0)':>10}")
print("="*70)
for fam, models in FAMILIES.items():
    try:
        ds={k: json.load(open(PATH[fam](k))) for k,_ in models}
    except FileNotFoundError as e:
        print(f"{fam:10} SKIP (missing: {e.filename})")
        continue
    sizes=[k for k,_ in models]; N=[n for _,n in models]
    init=[midwf(ds[k])[0] for k in sizes]; cont=[midwf(ds[k])[1] for k in sizes]
    fi,fc,diffs,(q025,q500,q975) = boot_diff(N, init, cont, B=2000)
    p_nonneg = float((diffs>=0).mean())
    print(f"{fam:10}{fi[0]:9.3f}{fi[1]:9.3f}{fc[0]:9.3f}{fc[1]:9.3f}{fi[0]-fc[0]:+9.3f}  [{q025:+.3f},{q975:+.3f}]{p_nonneg:>10.4f}")
print()
print("Replication criterion: sign NEG and 95% CI excludes 0 in each family.")
