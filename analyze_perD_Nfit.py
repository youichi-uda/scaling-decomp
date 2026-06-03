#!/usr/bin/env python
"""For each training step D, fit L(N) per class across available N values.
If Δα_N varies wildly with D, the fixed-D N-axis decomposition is coordinate-
dependent (= artifact).

Pairs with per-N D-axis fits which show Δα_D is stable across N (~-0.2).
"""
import json, numpy as np, os, glob

PARAMS = {"160m":160e6,"410m":410e6,"1b":1.0e9,"1.4b":1.4e9,"2.8b":2.8e9,"6.9b":6.9e9,"12b":12e9}
TOKS = 2_097_152
ALL_STEPS = [1000,2000,4000,7000,10000,15000,22000,25000,30000,42000,50000,70000,100000,130000,143000]


def per_word_midwf(d, lo=4, hi=7):
    fp = d.get("by_wordfreqdecile_position_per_word")
    if fp is None: return None, None
    iw=cw=inn=cnn=0.0
    for b in range(lo, hi+1):
        x = fp[str(b)]; a = x["word_initial"]; c = x["word_continuation"]
        if a["mean_loss"] is not None: iw += a["mean_loss"]*a["n_words"]; inn += a["n_words"]
        if c["mean_loss"] is not None: cw += c["mean_loss"]*c["n_words"]; cnn += c["n_words"]
    return (iw/inn if inn else None, cw/cnn if cnn else None)


def fit_alpha(N, L, grid=np.linspace(0.05, 2.0, 400)):
    """Fit L = E + A·N^(-α) per N values. Require A>0, E≥0."""
    N=np.asarray(N,float); L=np.asarray(L,float); best=None
    for al in grid:
        X = np.column_stack([np.ones_like(N), N**(-al)])
        coef, _, _, _ = np.linalg.lstsq(X, L, rcond=None)
        if coef[1] <= 0 or coef[0] < 0: continue
        pred = X @ coef
        sse = float(np.sum((pred - L)**2))
        if best is None or sse < best[3]: best = (al, coef[0], coef[1], sse)
    if best is None:
        # fallback
        for al in grid:
            X = np.column_stack([np.ones_like(N), N**(-al)])
            coef, _, _, _ = np.linalg.lstsq(X, L, rcond=None)
            pred = X @ coef
            sse = float(np.sum((pred - L)**2))
            if best is None or sse < best[3]: best = (al, coef[0], coef[1], sse)
    return best


print("=== Per-D N-axis decomposition (Δα_N varies with D?) ===")
print(f"{'step':>8}{'D(B)':>10}{'n_N':>4}{'α_init':>8}{'α_cont':>8}{'Δα_N':>10}{'E_init':>9}{'E_cont':>9}")
print("=" * 80)
deltas = []; Ds = []; ds_str = []
for s in ALL_STEPS:
    Ns = []; Linit = []; Lcont = []
    for nk, N in PARAMS.items():
        f = f"v2mt_{nk}_step{s}.json" if s != 143000 else f"v2mt_{nk}.json"
        if not os.path.exists(f): continue
        d = json.load(open(f))
        li, lc = per_word_midwf(d)
        if li is None or lc is None: continue
        Ns.append(N); Linit.append(li); Lcont.append(lc)
    if len(Ns) < 4: continue  # need at least 4 N points for stable α fit
    fi = fit_alpha(Ns, Linit); fc = fit_alpha(Ns, Lcont)
    dN = fi[0] - fc[0]
    deltas.append(dN); Ds.append(s*TOKS); ds_str.append(s)
    print(f"{s:>8}{s*TOKS/1e9:>10.1f}{len(Ns):>4}{fi[0]:>8.3f}{fc[0]:>8.3f}{dN:>+10.3f}{fi[1]:>9.3f}{fc[1]:>9.3f}")

print("=" * 80)
if len(deltas) >= 3:
    print(f"\nΔα_N range across {len(deltas)} D values: [{min(deltas):+.3f}, {max(deltas):+.3f}], span={max(deltas)-min(deltas):.3f}")
    print(f"Mean Δα_N = {np.mean(deltas):+.3f}, std = {np.std(deltas):.3f}")

import json as j
j.dump({"steps": ds_str, "delta_aN": deltas, "Ds": Ds},
       open("perD_Nfit_results.json","w"), indent=2)
print("\nSaved -> perD_Nfit_results.json")
