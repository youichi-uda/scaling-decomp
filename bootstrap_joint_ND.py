#!/usr/bin/env python
"""Bootstrap CI for joint L_c(N, D) per-class fit.

Resample TRAIN cells with replacement, refit per-class, collect distribution of
Δα_N and Δα_D. This treats cells as exchangeable units — appropriate given
the cell-level (independent (N, D) inference run) nature of the data.
"""
import json, numpy as np, os
import argparse

PARAMS = {"70m":70e6,"160m":160e6,"410m":410e6,"1b":1.0e9,
          "1.4b":1.4e9,"2.8b":2.8e9,"6.9b":6.9e9,"12b":12e9}
TOKS = 2_097_152
ALL_STEPS = [1000,2000,4000,7000,10000,15000,22000,25000,30000,42000,50000,70000,100000,130000,143000]


def per_word_midwf(d, lo=4, hi=7):
    fp = d.get("by_wordfreqdecile_position_per_word")
    if fp is None: return None, None
    iw=cw=inn=cnn=0.0
    for b in range(lo, hi+1):
        x = fp[str(b)]
        a = x["word_initial"]; c = x["word_continuation"]
        if a["mean_loss"] is not None: iw += a["mean_loss"]*a["n_words"]; inn += a["n_words"]
        if c["mean_loss"] is not None: cw += c["mean_loss"]*c["n_words"]; cnn += c["n_words"]
    return (iw/inn if inn else None, cw/cnn if cnn else None)


def collect_cells(N_keys, steps):
    cells = []
    for nk in N_keys:
        for s in steps:
            f = f"v2mt_{nk}_step{s}.json" if s != 143000 else f"v2mt_{nk}.json"
            if not os.path.exists(f): continue
            d = json.load(open(f))
            li, lc = per_word_midwf(d)
            if li is None or lc is None: continue
            cells.append({"key": (nk, s), "N": PARAMS[nk], "D": s*TOKS, "Linit": li, "Lcont": lc})
    return cells


def fit_perclass_cells(cells, aN_grid, aD_grid):
    """Per-class Chinchilla L = E + A/N^αN + B/D^αD fit by grid+LS.
    Requires A>0 and B>0 (else the form is degenerate); skips degenerate corners."""
    L_i = np.array([c["Linit"] for c in cells])
    L_c = np.array([c["Lcont"] for c in cells])
    N   = np.array([c["N"] for c in cells])
    D   = np.array([c["D"] for c in cells])
    out = {}
    for cls, L in [("init", L_i), ("cont", L_c)]:
        best = None
        for aN in aN_grid:
            for aD in aD_grid:
                X = np.column_stack([np.ones_like(N), N**(-aN), D**(-aD)])
                coef, _, _, _ = np.linalg.lstsq(X, L, rcond=None)
                # require A>0 and B>0 (else fit is using regression sign to compensate)
                if coef[1] <= 0 or coef[2] <= 0: continue
                # require E >= 0 (irreducible loss floor is non-negative)
                if coef[0] < 0: continue
                pred = X @ coef
                sse = float(np.sum((pred - L)**2))
                if best is None or sse < best[5]:
                    best = (float(aN), float(aD), float(coef[0]), float(coef[1]), float(coef[2]), sse)
        if best is None:
            # fallback: allow free coefs (degenerate case)
            for aN in aN_grid:
                for aD in aD_grid:
                    X = np.column_stack([np.ones_like(N), N**(-aN), D**(-aD)])
                    coef, _, _, _ = np.linalg.lstsq(X, L, rcond=None)
                    pred = X @ coef
                    sse = float(np.sum((pred - L)**2))
                    if best is None or sse < best[5]:
                        best = (float(aN), float(aD), float(coef[0]), float(coef[1]), float(coef[2]), sse)
        out[cls] = best
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=500)
    ap.add_argument("--exclude_N", nargs="*", default=["70m","160m"])  # too sparse
    args = ap.parse_args()

    N_keys = [n for n in PARAMS if n not in args.exclude_N]
    cells = collect_cells(N_keys, ALL_STEPS)
    print(f"Total cells: {len(cells)}")
    if len(cells) < 20:
        print("Not enough cells for bootstrap"); return

    aN_grid = np.linspace(0.02, 2.5, 60)
    aD_grid = np.linspace(0.02, 2.5, 60)

    # Point estimate
    pt = fit_perclass_cells(cells, aN_grid, aD_grid)
    p_aN = pt["init"][0] - pt["cont"][0]
    p_aD = pt["init"][1] - pt["cont"][1]
    print(f"\n=== Point estimate ===")
    print(f"  init: αN={pt['init'][0]:.3f}  αD={pt['init'][1]:.3f}  E={pt['init'][2]:.3f}")
    print(f"  cont: αN={pt['cont'][0]:.3f}  αD={pt['cont'][1]:.3f}  E={pt['cont'][2]:.3f}")
    print(f"  Δα_N (init-cont) = {p_aN:+.3f}")
    print(f"  Δα_D (init-cont) = {p_aD:+.3f}")

    # Bootstrap
    rng = np.random.default_rng(20260602)
    diffs_N = np.zeros(args.B); diffs_D = np.zeros(args.B)
    n = len(cells)
    print(f"\n=== Bootstrap (B={args.B}, cell-resample with replacement) ===")
    for b in range(args.B):
        idx = rng.choice(n, size=n, replace=True)
        sub = [cells[i] for i in idx]
        try:
            f = fit_perclass_cells(sub, aN_grid, aD_grid)
            diffs_N[b] = f["init"][0] - f["cont"][0]
            diffs_D[b] = f["init"][1] - f["cont"][1]
        except Exception as e:
            diffs_N[b] = np.nan; diffs_D[b] = np.nan
        if b % 50 == 0: print(f"  {b}/{args.B}", flush=True)

    for name, d in [("Δα_N", diffs_N), ("Δα_D", diffs_D)]:
        d_valid = d[~np.isnan(d)]
        if len(d_valid) < args.B*0.9:
            print(f"  {name}: only {len(d_valid)}/{args.B} valid")
        q025, q500, q975 = np.quantile(d_valid, [0.025, 0.5, 0.975])
        print(f"  {name}: median {q500:+.3f}  95% CI [{q025:+.3f}, {q975:+.3f}]  P(>=0)={(d_valid>=0).mean():.3f}")


if __name__ == "__main__":
    main()
