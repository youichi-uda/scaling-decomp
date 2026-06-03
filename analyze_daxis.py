#!/usr/bin/env python
"""D-axis (token-count) per-class scaling fit at fixed N=1B.

Pythia training: batch_size = 1024 sequences x 2048 tokens = 2,097,152 tokens/step.
So D(step) ≈ step × 2.097M tokens. step143000 ≈ 300B tokens (full Pile epoch ~once).

Fit L(D) = E + A·D^(-α_D) per class. If α_D differs between init and cont, the
N-axis α-difference is NOT just data-saturation: each class has its own intrinsic
data-scaling exponent.
"""
import json, numpy as np

TOKS_PER_STEP = 2_097_152
STEPS = [10000, 25000, 50000, 100000, 143000]


def mtwf(d, lo=4, hi=7):
    fp = d["by_wordfreqdecile_position_multitokenonly"]; iw=cw=inn=cnn=0.0
    for b in range(lo, hi+1):
        a=fp[str(b)]["word_initial"]; c=fp[str(b)]["word_continuation"]
        if a["mean_loss"] is not None: iw+=a["mean_loss"]*a["n"]; inn+=a["n"]
        if c["mean_loss"] is not None: cw+=c["mean_loss"]*c["n"]; cnn+=c["n"]
    return iw/inn, cw/cnn


def fit(N, L, grid=np.linspace(0.05, 1.5, 600)):
    N=np.asarray(N,float); L=np.asarray(L,float); best=None
    for al in grid:
        x=N**(-al); B=np.vstack([np.ones_like(x),x]).T
        coef,*_=np.linalg.lstsq(B,L,rcond=None); pred=B@coef
        sse=np.sum((pred-L)**2)
        if best is None or sse<best[3]: best=(al,coef[0],coef[1],sse,pred)
    return best


print("=== D-axis fit at fixed N=1B (multi-token × word-freq mid deciles) ===")
print(f"{'step':>10}{'D (tokens)':>14}{'init':>10}{'cont':>10}{'gap':>9}")
Ds=[]; inits=[]; conts=[]
for s in STEPS:
    f = f"v2mt_1b_step{s}.json" if s != 143000 else "v2mt_1b.json"
    d = json.load(open(f))
    i, c = mtwf(d)
    D = s * TOKS_PER_STEP
    Ds.append(D); inits.append(i); conts.append(c)
    print(f"{s:>10}{D:>14,}{i:>10.3f}{c:>10.3f}{i-c:>9.3f}")

fi = fit(Ds, inits); fc = fit(Ds, conts)
print()
print(f"per-class L(D) fits at fixed N=1B:")
print(f"  init: alpha_D={fi[0]:.3f}  E_D={fi[1]:.3f}")
print(f"  cont: alpha_D={fc[0]:.3f}  E_D={fc[1]:.3f}")
print(f"  DIFF alpha_D (init-cont) = {fi[0]-fc[0]:+.3f}")

# Bootstrap CI on D-axis alpha-diff
rng=np.random.default_rng(20260601)
ri=np.array(inits)-fi[4]; rc=np.array(conts)-fc[4]
print(f"  residuals init: std={ri.std():.4f}   cont: std={rc.std():.4f}")
diffs=[]
for _ in range(2000):
    Li=fi[4]+rng.choice(ri,size=len(Ds),replace=True)
    Lc=fc[4]+rng.choice(rc,size=len(Ds),replace=True)
    diffs.append(fit(Ds,Li)[0]-fit(Ds,Lc)[0])
diffs=np.array(diffs)
q025,q500,q975=np.quantile(diffs,[0.025,0.5,0.975])
print(f"  bootstrap 95% CI: [{q025:+.3f}, {q975:+.3f}]   median {q500:+.3f}")
print(f"  P(diff >= 0) = {(diffs>=0).mean():.4f}")

print()
print("INTERPRETATION:")
print("  If alpha_D (cont) > alpha_D (init), same direction as N-axis result.")
print("  Means: cont saturates faster IN DATA too, at this fixed N.")
print("  If alpha_D similar between classes, the N-axis gap is N-specific (still publishable).")
print("  If alpha_D (cont) < alpha_D (init), Web Claude's data-saturation framing wins.")
