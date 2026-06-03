#!/usr/bin/env python3
"""Unigram-LM / Zipf-coupling test for the per-class scaling-law decomposition.

Persona A's R1-R3 critique: alpha_cont approx 1.0 may be the "trivial"
Zipf-coupled rate predicted by Michaud 2023's quantization model. To
adjudicate this we test whether the per-class alpha gap survives once we
condition on token-frequency decile (i.e., on the marginal-unigram
distribution of the prediction).

Procedure:
  1. For each of the 7 Pythia sizes (160m..12b) we already have the
     per-(freq decile x class) mean loss in v2mt_*.json
     (key: by_freqdecile_position[decile_idx][word_initial|word_continuation]).
  2. For each decile d in {0..8} (skip d=9 which has zero cont tokens),
     fit L_class(N) = E + A * N^(-alpha) per class on the 7 sizes.
  3. Report per-decile (alpha_init, alpha_cont, Delta_alpha).

If Delta_alpha is approximately 0 within each decile but the global
aggregation in section 4.1 has Delta_alpha = -0.41, then the gap is a
class-x-decile mixture artifact (Zipf coupling).

If Delta_alpha persists across deciles, the per-class structure is
independent of token-frequency-decile composition.

We also report a *Michaud sanity test*: per-decile alpha should be
larger in high-frequency deciles (predicted by Michaud's frequency
quantization model).

Outputs: per-decile fit table + summary verdict + unigram_test.json.
"""
from __future__ import annotations
import json
import numpy as np

PARAMS = {"160m":160e6,"410m":410e6,"1b":1.0e9,"1.4b":1.4e9,
          "2.8b":2.8e9,"6.9b":6.9e9,"12b":12e9}
SIZES = ["160m","410m","1b","1.4b","2.8b","6.9b","12b"]


def fit(N, L, agrid=np.linspace(0.02, 1.8, 800)):
    """Fit L = E + A*N^(-a), positivity. Returns (a, E, A, sse)."""
    N = np.asarray(N, float); L = np.asarray(L, float)
    best = None
    for a in agrid:
        x = N ** (-a)
        B = np.vstack([np.ones_like(x), x]).T
        coef, *_ = np.linalg.lstsq(B, L, rcond=None)
        E, A = coef
        if A <= 0 or E < 0: continue
        pred = B @ coef
        sse = float(np.sum((pred - L) ** 2))
        if best is None or sse < best[3]:
            best = (float(a), float(E), float(A), sse)
    return best


def bootstrap_diff(N, init, cont, B=2000, seed=20260603):
    fi = fit(N, init)
    fc = fit(N, cont)
    if fi is None or fc is None:
        return None, None, (np.nan, np.nan)
    Ni = np.asarray(N, float)
    pred_i = fi[2] * Ni ** (-fi[0]) + fi[1]
    pred_c = fc[2] * Ni ** (-fc[0]) + fc[1]
    ri = np.asarray(init) - pred_i
    rc = np.asarray(cont) - pred_c
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(B):
        Li = pred_i + rng.choice(ri, size=len(Ni), replace=True)
        Lc = pred_c + rng.choice(rc, size=len(Ni), replace=True)
        f_i_b = fit(Ni, Li); f_c_b = fit(Ni, Lc)
        if f_i_b is None or f_c_b is None: continue
        diffs.append(f_i_b[0] - f_c_b[0])
    if not diffs: return fi, fc, (np.nan, np.nan)
    diffs = np.array(diffs)
    return fi, fc, tuple(np.quantile(diffs, [0.025, 0.975]).tolist())


def load_per_decile():
    """Return: per_dec[decile][size] = (L_init, L_cont, n_init, n_cont)."""
    per = {}
    freq_ranges = {}
    for d in range(10):
        per[d] = {}
    for s in SIZES:
        with open(f"v2mt_{s}.json") as fp:
            j = json.load(fp)
        for d in range(10):
            x = j["by_freqdecile_position"][str(d)]
            if d not in freq_ranges:
                freq_ranges[d] = x.get("freq_range")
            li = x["word_initial"]["mean_loss"]
            lc = x["word_continuation"]["mean_loss"]
            ni = x["word_initial"]["n"]
            nc = x["word_continuation"]["n"]
            per[d][s] = (li, lc, ni, nc)
    return per, freq_ranges


def main():
    per, ranges = load_per_decile()
    results = []
    print("="*82)
    print(f"{'dec':>3}{'freq range':>16}{'avg n_init':>11}{'avg n_cont':>11}"
          f"{'a_init':>9}{'a_cont':>9}{'delta':>9}{'CI95':>22}")
    print("="*82)
    for d in range(10):
        # Average per-cell n across sizes (for context)
        cells = list(per[d].values())
        ni_avg = float(np.mean([c[2] for c in cells]))
        nc_avg = float(np.mean([c[3] for c in cells]))
        if any(c[1] is None or c[0] is None for c in cells) or nc_avg == 0:
            print(f"{d:>3}{str(ranges[d]):>16}{ni_avg:>11.0f}{nc_avg:>11.0f}  -- (no cont tokens)")
            results.append({"decile": d, "freq_range": ranges[d],
                            "n_init_avg": ni_avg, "n_cont_avg": nc_avg,
                            "skip": True})
            continue
        N = [PARAMS[s] for s in SIZES]
        init = [per[d][s][0] for s in SIZES]
        cont = [per[d][s][1] for s in SIZES]
        fi, fc, (lo, hi) = bootstrap_diff(N, init, cont)
        if fi is None or fc is None:
            print(f"{d:>3}{str(ranges[d]):>16}{ni_avg:>11.0f}{nc_avg:>11.0f}  -- (fit failed)")
            continue
        a_i, a_c = fi[0], fc[0]
        print(f"{d:>3}{str(ranges[d]):>16}{ni_avg:>11.0f}{nc_avg:>11.0f}"
              f"{a_i:>9.3f}{a_c:>9.3f}{a_i-a_c:>+9.3f}  [{lo:+.3f},{hi:+.3f}]")
        results.append({
            "decile": d, "freq_range": ranges[d],
            "n_init_avg": ni_avg, "n_cont_avg": nc_avg,
            "alpha_init": a_i, "alpha_cont": a_c,
            "E_init": fi[1], "E_cont": fc[1],
            "diff": a_i - a_c, "CI95_lo": lo, "CI95_hi": hi,
        })

    print()
    print("=== Michaud sanity test: does per-class alpha grow with decile frequency? ===")
    deciles = [r for r in results if not r.get("skip")]
    for cls in ("alpha_init", "alpha_cont"):
        vals = [r[cls] for r in deciles]
        decs = [r["decile"] for r in deciles]
        # Spearman rank correlation by hand
        from scipy.stats import spearmanr
        rho, p = spearmanr(decs, vals)
        print(f"  {cls} across deciles {decs}: rho_Spearman = {rho:+.3f}, p = {p:.4f}")

    print()
    print("=== Adjudication: is the per-decile Delta_alpha distinguishable from 0? ===")
    diffs = [r["diff"] for r in deciles]
    ci_excl = [r for r in deciles if r["CI95_lo"] < 0 < r["CI95_hi"]]
    ci_neg = [r for r in deciles if r["CI95_hi"] < 0]
    ci_pos = [r for r in deciles if r["CI95_lo"] > 0]
    print(f"  Total deciles fit: {len(deciles)}")
    print(f"  Deciles with CI excluding 0 on the NEGATIVE side: {len(ci_neg)} (Delta_alpha < 0)")
    print(f"  Deciles with CI excluding 0 on the POSITIVE side: {len(ci_pos)} (Delta_alpha > 0)")
    print(f"  Deciles with CI containing 0: {len(ci_excl)}")
    print(f"  Mean Delta_alpha across feasible deciles: {np.mean(diffs):+.3f}")
    print(f"  Median Delta_alpha: {np.median(diffs):+.3f}")
    print(f"  Min / Max Delta_alpha: {min(diffs):+.3f} / {max(diffs):+.3f}")
    print()
    if all(d < -0.10 for d in diffs):
        verdict = ("STRONG: Delta_alpha < -0.10 in EVERY decile, the per-class gap is "
                   "NOT explained by token-frequency-decile composition.")
    elif np.median(diffs) < -0.05 and len(ci_neg) >= len(deciles) // 2:
        verdict = ("MODERATE: per-class gap survives within most deciles; "
                   "decile composition explains some but not all of the global gap.")
    elif all(abs(d) < 0.10 for d in diffs):
        verdict = ("ARTIFACT: per-decile Delta_alpha is approximately 0 — global -0.41 "
                   "is explained by class-by-decile mixture (Zipf coupling, Michaud).")
    else:
        verdict = ("MIXED: per-decile gap varies; report decile-by-decile.")
    print(f"VERDICT: {verdict}")

    json.dump({
        "method": "per-token-frequency-decile per-class alpha fit, 7 Pythia sizes at D=300B",
        "deciles": results,
        "summary": {
            "n_feasible_deciles": len(deciles),
            "n_CI_neg": len(ci_neg),
            "n_CI_pos": len(ci_pos),
            "n_CI_zero": len(ci_excl),
            "mean_diff": float(np.mean(diffs)),
            "median_diff": float(np.median(diffs)),
            "min_diff": float(min(diffs)),
            "max_diff": float(max(diffs)),
            "all_negative_below_minus_010": bool(all(d < -0.10 for d in diffs)),
            "verdict": verdict,
        },
    }, open("unigram_test.json", "w"), indent=2)
    print("\nSaved -> unigram_test.json")


if __name__ == "__main__":
    main()
