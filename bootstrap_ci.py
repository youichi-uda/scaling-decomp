#!/usr/bin/env python
"""Parametric bootstrap on the per-class scaling-law fit residuals.
Resample residuals -> refit -> α_init - α_cont distribution -> 95% CI.

Per-bin sampling error is negligible (n >> 1e5 per cell), so dominant
uncertainty is the 5-point scaling fit.
"""
import json, numpy as np

PARAMS={"pythia-70m":70e6,"pythia-160m":160e6,"pythia-410m":410e6,"pythia-1b":1.0e9,"pythia-1.4b":1.4e9}


def midwf(d,lo=4,hi=7):
    fp=d["by_wordfreqdecile_position"]; iw=cw=inn=cnn=0.0
    for b in range(lo,hi+1):
        a=fp[str(b)]["word_initial"]; c=fp[str(b)]["word_continuation"]
        if a["mean_loss"] is not None: iw+=a["mean_loss"]*a["n"]; inn+=a["n"]
        if c["mean_loss"] is not None: cw+=c["mean_loss"]*c["n"]; cnn+=c["n"]
    return iw/inn, cw/cnn


def fit_alpha(N,L,grid=np.linspace(0.05,1.5,600)):
    N=np.asarray(N,float); L=np.asarray(L,float); best=None
    for al in grid:
        x=N**(-al); B=np.vstack([np.ones_like(x),x]).T
        coef,*_=np.linalg.lstsq(B,L,rcond=None); pred=B@coef
        sse=np.sum((pred-L)**2)
        if best is None or sse<best[3]: best=(al,coef[0],coef[1],sse,pred)
    return best  # alpha,E,A,sse,pred


sizes=list(PARAMS.keys()); Ns=[PARAMS[k] for k in sizes]
data={k:json.load(open(f"v2_{k.split('-')[1]}.json")) for k in sizes}
init=[midwf(data[k])[0] for k in sizes]; cont=[midwf(data[k])[1] for k in sizes]

# point estimates
fi=fit_alpha(Ns,init); fc=fit_alpha(Ns,cont)
print(f"point estimates: alpha_init={fi[0]:.3f}  alpha_cont={fc[0]:.3f}  diff={fi[0]-fc[0]:+.3f}")
print(f"  E_init={fi[1]:.3f} A_init={fi[2]:.2e}   E_cont={fc[1]:.3f} A_cont={fc[2]:.2e}")

# residuals (pred - actual)
res_i = np.array(init) - fi[4]; res_c = np.array(cont) - fc[4]
print(f"residuals init: {np.round(res_i,4)}  std={res_i.std():.4f}")
print(f"residuals cont: {np.round(res_c,4)}  std={res_c.std():.4f}")

# parametric residual bootstrap
rng=np.random.default_rng(20260601)
B=2000
diffs=np.zeros(B); ai=np.zeros(B); ac=np.zeros(B)
for b in range(B):
    Ri=rng.choice(res_i, size=5, replace=True)
    Rc=rng.choice(res_c, size=5, replace=True)
    Li=fi[4]+Ri; Lc=fc[4]+Rc
    fbi=fit_alpha(Ns,Li); fbc=fit_alpha(Ns,Lc)
    ai[b]=fbi[0]; ac[b]=fbc[0]; diffs[b]=fbi[0]-fbc[0]

q025,q500,q975=np.quantile(diffs,[0.025,0.5,0.975])
print(f"\nbootstrap (B={B}):")
print(f"  alpha_init: mean={ai.mean():.3f}  95% CI [{np.quantile(ai,0.025):.3f}, {np.quantile(ai,0.975):.3f}]")
print(f"  alpha_cont: mean={ac.mean():.3f}  95% CI [{np.quantile(ac,0.025):.3f}, {np.quantile(ac,0.975):.3f}]")
print(f"  DIFF (init-cont): median={q500:+.3f}  95% CI [{q025:+.3f}, {q975:+.3f}]")
print(f"  P(diff>=0) = {(diffs>=0).mean():.4f}")
print(f"  P(diff>=-0.05) = {(diffs>=-0.05).mean():.4f}  (effect size threshold check)")

excludes_zero = q975 < 0 or q025 > 0
print(f"\n95% CI excludes zero: {excludes_zero}")
