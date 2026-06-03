#!/usr/bin/env python
"""Multi-N D-axis analysis with bootstrap CIs per N.

For each N: fit L(D) per class on intermediate Pythia checkpoints, report
(α_D_init, α_D_cont, diff) + 95% CI.

70M flagged as anomalous (late-training instability: final loss 4.84 > step50000 3.21).
6.9B skipped due to HF Hub download stall.
"""
import json, numpy as np

TOKS = 2_097_152  # tokens per Pythia training step

# N values: (suffix, params, [steps])
N_CONFIGS = [
    ("70m", 70e6, [10000, 25000, 50000, 100000, 143000]),
    ("410m", 410e6, [10000, 25000, 50000, 100000, 143000]),
    ("1b", 1.0e9, [10000, 25000, 50000, 100000, 143000]),
    ("1.4b", 1.4e9, [10000, 25000, 50000, 100000, 143000]),
    ("6.9b", 6.9e9, [10000, 25000, 50000, 100000, 143000]),
]


def mtwf(d, lo=4, hi=7):
    fp = d["by_wordfreqdecile_position_multitokenonly"]; iw=cw=inn=cnn=0.0
    for b in range(lo, hi+1):
        a=fp[str(b)]["word_initial"]; c=fp[str(b)]["word_continuation"]
        if a["mean_loss"] is not None: iw+=a["mean_loss"]*a["n"]; inn+=a["n"]
        if c["mean_loss"] is not None: cw+=c["mean_loss"]*c["n"]; cnn+=c["n"]
    return iw/inn, cw/cnn


def fit(N, L, grid=np.linspace(0.02, 1.8, 800)):
    N=np.asarray(N,float); L=np.asarray(L,float); best=None
    for al in grid:
        x=N**(-al); B=np.vstack([np.ones_like(x),x]).T
        coef,*_=np.linalg.lstsq(B,L,rcond=None); pred=B@coef
        sse=np.sum((pred-L)**2)
        if best is None or sse<best[3]: best=(al,coef[0],coef[1],sse,pred)
    return best


def bootstrap_diff(Ds, inits, conts, B=2000, seed=20260602):
    fi=fit(Ds,inits); fc=fit(Ds,conts)
    ri=np.array(inits)-fi[4]; rc=np.array(conts)-fc[4]
    rng=np.random.default_rng(seed); diffs=[]
    for _ in range(B):
        Li=fi[4]+rng.choice(ri,size=len(Ds),replace=True)
        Lc=fc[4]+rng.choice(rc,size=len(Ds),replace=True)
        diffs.append(fit(Ds,Li)[0]-fit(Ds,Lc)[0])
    return fi, fc, np.array(diffs)


print("=" * 78)
print(f"{'N':>7}  {'α_D_init':>10}  {'α_D_cont':>10}  {'diff':>8}  {'95% CI':>20}  {'note':10}")
print("=" * 78)

per_N_results = {}
for suf, params, steps in N_CONFIGS:
    Ds=[]; inits=[]; conts=[]
    for s in steps:
        f = f"v2mt_{suf}_step{s}.json" if s != 143000 else f"v2mt_{suf}.json"
        try:
            d = json.load(open(f))
        except FileNotFoundError:
            continue
        i, c = mtwf(d)
        Ds.append(s * TOKS); inits.append(i); conts.append(c)
    if len(Ds) < 3: continue
    fi, fc, diffs = bootstrap_diff(Ds, inits, conts)
    diff_med = float(np.median(diffs))
    ci = (float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975)))
    # 70M is anomalous (late training instability)
    note = "ANOMALOUS" if suf == "70m" else "clean"
    print(f"{suf:>7}  {fi[0]:>10.3f}  {fc[0]:>10.3f}  {fi[0]-fc[0]:>+8.3f}  [{ci[0]:>+6.3f},{ci[1]:>+6.3f}]  {note}")
    per_N_results[suf] = {"alpha_init": float(fi[0]), "E_init": float(fi[1]),
                         "alpha_cont": float(fc[0]), "E_cont": float(fc[1]),
                         "diff_median": diff_med, "ci_lo": ci[0], "ci_hi": ci[1],
                         "note": note}

print("=" * 78)
print()
print("=== N-axis reference (at fixed D=300B, all 8 sizes, multi-token strict control) ===")
print("    diff = -0.341   95% CI [-0.43, -0.16]   (estimated from earlier bootstrap)")
print()
print("=== INTERPRETATION ===")
print("If all clean per-N D-axis CIs INCLUDE 0 while N-axis CI excludes 0:")
print(" -> Per-class α gap is a fixed-D phenomenon, not intrinsic.")
print(" -> Cross-paper α comparisons that vary in D are confounded.")

with open("multiN_daxis_results.json", "w") as f:
    json.dump(per_N_results, f, indent=2)
print()
print("saved -> multiN_daxis_results.json")
