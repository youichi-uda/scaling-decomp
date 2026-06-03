#!/usr/bin/env python
"""Aggregate per-size JSON outputs, fit per-class scaling laws, run the smoke GO/NO-GO gate.

Usage: python analyze.py out_70m.json out_160m.json out_410m.json out_1.4b.json ...
"""
import sys, json
import numpy as np

# Pythia non-embedding param counts (approx, standard suite). Use total params is fine for log-N trend.
PARAMS = {
    "pythia-70m": 70e6, "pythia-160m": 160e6, "pythia-410m": 410e6,
    "pythia-1b": 1.0e9, "pythia-1.4b": 1.4e9, "pythia-2.8b": 2.8e9,
    "pythia-6.9b": 6.9e9, "pythia-12b": 12e9,
}


def model_key(name):
    return name.split("/")[-1]


def fit_powerlaw(N, L):
    """Fit L = E + A*N^-alpha by grid over E, linear in log space for (A,alpha).
    Returns (E, A, alpha, rmse). Simple robust fit for small #points."""
    N = np.asarray(N, float); L = np.asarray(L, float)
    best = None
    Lmin = L.min()
    for E in np.linspace(0.0, max(0.0, Lmin - 1e-3), 60):
        y = L - E
        if (y <= 0).any():
            continue
        # log y = log A - alpha log N
        A_ = np.vstack([np.ones_like(N), -np.log(N)]).T
        coef, *_ = np.linalg.lstsq(A_, np.log(y), rcond=None)
        logA, alpha = coef
        pred = E + np.exp(logA) * N ** (-alpha)
        rmse = np.sqrt(np.mean((pred - L) ** 2))
        if best is None or rmse < best[3]:
            best = (E, float(np.exp(logA)), alpha, rmse)
    return best


def main():
    files = sys.argv[1:]
    data = {}
    for f in files:
        d = json.load(open(f))
        k = model_key(d["model"])
        data[k] = d
    order = [k for k in PARAMS if k in data]
    Ns = [PARAMS[k] for k in order]
    print("sizes:", order)
    print()

    # --- aggregate gap trend (init - cont), overall ---
    print("=== overall by-position mean loss ===")
    print(f"{'size':12} {'overall':>8} {'init':>8} {'cont':>8} {'other':>8} {'gap(i-c)':>9}")
    inits, conts = [], []
    for k in order:
        p = data[k]["by_position"]
        i = p["word_initial"]["mean_loss"]; c = p["word_continuation"]["mean_loss"]; o = p["other"]["mean_loss"]
        inits.append(i); conts.append(c)
        print(f"{k:12} {data[k]['overall_mean_loss']:8.3f} {i:8.3f} {c:8.3f} {o:8.3f} {i-c:9.3f}")
    gap0, gap1 = inits[0]-conts[0], inits[-1]-conts[-1]
    print(f"\ngap change {order[0]}->{order[-1]}: {gap0:.3f} -> {gap1:.3f}  ({100*(gap1-gap0)/gap0:+.1f}%)")

    # --- frequency-matched gap (middle deciles 4-7) ---
    print("\n=== frequency-matched init-vs-cont (mid deciles 4-7) ===")
    print(f"{'size':12} {'init_mid':>9} {'cont_mid':>9} {'gap':>8}")
    fg0 = fg1 = None
    for idx, k in enumerate(order):
        fp = data[k]["by_freqdecile_position"]
        iw = cw = inn = cnn = 0.0
        for dbin in ["4", "5", "6", "7"]:
            a = fp[dbin]["word_initial"]; b = fp[dbin]["word_continuation"]
            if a["mean_loss"] is not None: iw += a["mean_loss"]*a["n"]; inn += a["n"]
            if b["mean_loss"] is not None: cw += b["mean_loss"]*b["n"]; cnn += b["n"]
        im = iw/inn if inn else float("nan"); cm = cw/cnn if cnn else float("nan")
        print(f"{k:12} {im:9.3f} {cm:9.3f} {im-cm:8.3f}")
        if idx == 0: fg0 = im-cm
        if idx == len(order)-1: fg1 = im-cm
    if fg0 is not None and fg1 is not None and np.isfinite(fg0) and fg0 != 0:
        print(f"\nfreq-matched gap change: {fg0:.3f} -> {fg1:.3f}  ({100*(fg1-fg0)/fg0:+.1f}%)")
        print("SMOKE GATE: GO if |change| >= 15% (non-parallel)")

    # --- per-class power-law fits (only if >=5 sizes) ---
    if len(order) >= 5:
        print("\n=== per-class power-law fits L=E+A*N^-alpha ===")
        for cls, key in [("word_initial","word_initial"),("word_continuation","word_continuation"),("other","other")]:
            L = [data[k]["by_position"][key]["mean_loss"] for k in order]
            E,A,al,rmse = fit_powerlaw(Ns, L)
            print(f"  {cls:18} E={E:.3f}  A={A:.3f}  alpha={al:.3f}  rmse={rmse:.4f}")
    else:
        print(f"\n(only {len(order)} sizes — power-law fit deferred to full run, need >=5)")


if __name__ == "__main__":
    main()
