#!/usr/bin/env python3
"""Floor-slope robustness check for the N-axis Δα.

Adversarial review (Persona A, R1) raised the concern that the
canonical Δα = -0.41 is driven by E_init = 2.46 vs E_cont = 0.79
being free — i.e., the per-class fits sit on different points of the
(E, α) ridge.

This script:
  1. Profile-likelihood contour of (E, α) per class (output: contour data)
  2. Shared-E fit: enforce E_init = E_cont = E across a sweep,
     refit (α, A) per class at each E, report Δα(E)
  3. Shared-(E, α) test: 1-α single fit per class with separate (E, A)
     vs full free fit — likelihood ratio on freed α

All on Pythia 7 sizes (160m..12b) at D=300B, per-word matched
aggregation (same as final_Naxis_perword.py).
"""
import json, numpy as np

PARAMS = {"160m":160e6,"410m":410e6,"1b":1.0e9,"1.4b":1.4e9,
          "2.8b":2.8e9,"6.9b":6.9e9,"12b":12e9}
SIZES = ["160m","410m","1b","1.4b","2.8b","6.9b","12b"]


def midwf(d, lo=4, hi=7):
    fp = d["by_wordfreqdecile_position_per_word"]
    iw=cw=inn=cnn=0.0
    for b in range(lo, hi+1):
        x = fp[str(b)]; a = x["word_initial"]; c = x["word_continuation"]
        if a["mean_loss"] is not None: iw += a["mean_loss"]*a["n_words"]; inn += a["n_words"]
        if c["mean_loss"] is not None: cw += c["mean_loss"]*c["n_words"]; cnn += c["n_words"]
    return iw/inn, cw/cnn


def collect():
    Ns = []; init = []; cont = []
    for s in SIZES:
        d = json.load(open(f"v2mt_{s}.json"))
        i, c = midwf(d)
        Ns.append(PARAMS[s]); init.append(i); cont.append(c)
    return np.array(Ns), np.array(init), np.array(cont)


def fit_free(N, L, agrid=np.linspace(0.02, 1.8, 800)):
    """Fit L = E + A*N^(-a) with positivity. Returns (a, E, A, sse)."""
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


def fit_fixed_E(N, L, E_fix, agrid=np.linspace(0.02, 1.8, 800)):
    """Fit L = E_fix + A*N^(-a) (E held). Returns (a, A, sse)."""
    best = None
    for a in agrid:
        x = N ** (-a)
        # solve for A: minimize sum((E_fix + A*x - L)^2). Closed form:
        #   A = sum(x*(L - E_fix)) / sum(x^2)
        denom = float(np.sum(x*x))
        if denom == 0: continue
        A = float(np.sum(x * (L - E_fix)) / denom)
        if A <= 0: continue
        pred = E_fix + A * x
        sse = float(np.sum((pred - L) ** 2))
        if best is None or sse < best[2]:
            best = (float(a), A, sse)
    return best


def main():
    N, init, cont = collect()
    print("=== Per-class loss (per-word matched, mid deciles 4-7) ===")
    for s, i, c in zip(SIZES, init, cont):
        print(f"  {s:>5}: L_init={i:6.3f}  L_cont={c:6.3f}")

    print("\n=== (1) Free fit per class (canonical, reproduces -0.41) ===")
    fi = fit_free(N, init)
    fc = fit_free(N, cont)
    print(f"  init: a={fi[0]:.3f}  E={fi[1]:.3f}  A={fi[2]:.3e}  sse={fi[3]:.4f}")
    print(f"  cont: a={fc[0]:.3f}  E={fc[1]:.3f}  A={fc[2]:.3e}  sse={fc[3]:.4f}")
    print(f"  delta_alpha (free per-class E) = {fi[0]-fc[0]:+.3f}")

    print("\n=== (2) Shared-E sweep: hold E_init=E_cont=E_fix, refit (a, A) per class ===")
    print(f"  {'E_fix':>7}{'a_init':>9}{'a_cont':>9}{'delta':>9}{'sse_init':>12}{'sse_cont':>12}")
    E_grid = np.concatenate([
        np.linspace(0.0, fc[1], 7),                   # below cont floor
        np.linspace(fc[1], fi[1], 9)[1:-1],          # between cont and init
        np.linspace(fi[1], 1.2 * fi[1], 5),           # at or above init floor
    ])
    deltas = []
    for E_fix in E_grid:
        gi = fit_fixed_E(N, init, E_fix)
        gc = fit_fixed_E(N, cont, E_fix)
        if gi is None or gc is None:
            print(f"  {E_fix:>7.3f}  -- infeasible (positivity)")
            continue
        d = gi[0] - gc[0]
        print(f"  {E_fix:>7.3f}{gi[0]:>9.3f}{gc[0]:>9.3f}{d:>+9.3f}{gi[2]:>12.4f}{gc[2]:>12.4f}")
        deltas.append((E_fix, d))

    deltas_arr = np.array([d for _, d in deltas])
    print(f"\n  delta_alpha across shared-E sweep:")
    print(f"    min={deltas_arr.min():+.3f}  max={deltas_arr.max():+.3f}  mean={deltas_arr.mean():+.3f}  median={np.median(deltas_arr):+.3f}")
    print(f"    all negative? {bool((deltas_arr < 0).all())}  all magnitude > 0.10? {bool((np.abs(deltas_arr) > 0.10).all())}")

    print("\n=== (3) Shared-alpha test: a_init = a_cont = a, fit (E_c, A_c) per class ===")
    # For shared a, separate (E, A) per class: for each candidate a,
    # solve (E, A) per class independently and sum SSE.
    agrid = np.linspace(0.02, 1.8, 800)
    best_sh = None
    for a in agrid:
        x = N ** (-a)
        B = np.vstack([np.ones_like(x), x]).T
        ci, *_ = np.linalg.lstsq(B, init, rcond=None)
        cc, *_ = np.linalg.lstsq(B, cont, rcond=None)
        if ci[1] <= 0 or ci[0] < 0 or cc[1] <= 0 or cc[0] < 0:
            continue
        pred_i = B @ ci; pred_c = B @ cc
        sse = float(np.sum((pred_i - init)**2) + np.sum((pred_c - cont)**2))
        if best_sh is None or sse < best_sh[3]:
            best_sh = (float(a), (float(ci[0]), float(ci[1])), (float(cc[0]), float(cc[1])), sse)
    a_sh, (Ei_sh, Ai_sh), (Ec_sh, Ac_sh), sse_sh = best_sh

    sse_pc = fi[3] + fc[3]
    n_obs = 2 * len(N)  # 14 observations
    p_pc = 6   # (E, A, a) per class * 2
    p_sh = 5   # (E_init, A_init, E_cont, A_cont, a_shared)
    print(f"  shared-a fit: a={a_sh:.3f}  E_init={Ei_sh:.3f}  A_init={Ai_sh:.3e}  E_cont={Ec_sh:.3f}  A_cont={Ac_sh:.3e}")
    print(f"  SSE shared-a = {sse_sh:.4f}  vs free-a SSE = {sse_pc:.4f}")
    F = ((sse_sh - sse_pc) / (p_pc - p_sh)) / (sse_pc / (n_obs - p_pc))
    from scipy.stats import f as f_dist
    p_value = 1.0 - f_dist.cdf(F, p_pc - p_sh, n_obs - p_pc)
    print(f"  F({p_pc - p_sh}, {n_obs - p_pc}) = {F:.3f}  p = {p_value:.4f}")
    print(f"  (Reject H_0: shared alpha if p < 0.05)")

    out = {
        "canonical_free": {"alpha_init": fi[0], "alpha_cont": fc[0],
                            "E_init": fi[1], "E_cont": fc[1],
                            "diff": fi[0] - fc[0]},
        "shared_E_sweep": [{"E_fix": float(E), "delta_alpha": float(d)} for E, d in deltas],
        "shared_alpha_test": {"alpha_shared": a_sh, "SSE_shared": sse_sh, "SSE_free": sse_pc,
                              "F": float(F), "p_value": float(p_value),
                              "df1": p_pc - p_sh, "df2": n_obs - p_pc},
    }
    json.dump(out, open("floor_robust.json", "w"), indent=2)
    print("\nSaved -> floor_robust.json")


if __name__ == "__main__":
    main()
