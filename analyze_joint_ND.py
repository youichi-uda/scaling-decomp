#!/usr/bin/env python
"""Joint L_c(N, D) fit per class with held-out (N, D) cell prediction.

Chinchilla-style functional form per class:
  L_c(N, D) = E_c + A_c / N^α_Nc + B_c / D^α_Dc

M_perclass: per-class (α_Nc, α_Dc, E_c, A_c, B_c).
M_shared:   shared (α_N, α_D) across classes; per-class (E_c, A_c, B_c).

Held-out cells: pick the largest D at the largest N (e.g., {6.9B/12B} × final
checkpoint) plus a sample of intermediate cells.

Inputs: v2mt_{N}.json (final) + v2mt_{N}_step{D}.json (intermediates).
Aggregation: word-level (per_word_summary or by_wordfreqdecile_position_per_word
mid deciles 4-7).
"""
import json, numpy as np, os
import argparse


PARAMS = {
    "70m":70e6, "160m":160e6, "410m":410e6, "1b":1.0e9,
    "1.4b":1.4e9, "2.8b":2.8e9, "6.9b":6.9e9, "12b":12e9,
}
TOKS_PER_STEP = 2_097_152
ALL_STEPS = [1000, 2000, 4000, 7000, 10000, 15000, 22000, 25000, 30000,
             42000, 50000, 70000, 100000, 130000, 143000]


def per_word_midwf(d, lo=4, hi=7):
    """Word-level (per Codex objection 2): each word contributes one init_loss
    and one mean(cont_loss). Aggregate mid deciles 4-7."""
    fp = d.get("by_wordfreqdecile_position_per_word")
    if fp is None: return None, None
    iw = cw = inn = cnn = 0.0
    for b in range(lo, hi+1):
        x = fp[str(b)]
        a = x["word_initial"]; c = x["word_continuation"]
        if a["mean_loss"] is not None:
            iw += a["mean_loss"] * a["n_words"]; inn += a["n_words"]
        if c["mean_loss"] is not None:
            cw += c["mean_loss"] * c["n_words"]; cnn += c["n_words"]
    return (iw/inn if inn else None, cw/cnn if cnn else None)


def collect_cells(N_keys=None, steps=None):
    """Return dict cell -> {Linit, Lcont, N, D}."""
    cells = {}
    N_keys = N_keys or list(PARAMS.keys())
    steps = steps or ALL_STEPS
    for nk in N_keys:
        for s in steps:
            f = f"v2mt_{nk}_step{s}.json" if s != 143000 else f"v2mt_{nk}.json"
            if not os.path.exists(f): continue
            d = json.load(open(f))
            li, lc = per_word_midwf(d)
            if li is None or lc is None: continue
            cells[(nk, s)] = {"Linit": li, "Lcont": lc, "N": PARAMS[nk], "D": s * TOKS_PER_STEP}
    return cells


def fit_joint_perclass(cells, alpha_N_grid=np.linspace(0.02, 2.5, 80),
                       alpha_D_grid=np.linspace(0.02, 2.5, 80)):
    """For each class, fit L(N, D) = E + A/N^α_N + B/D^α_D by grid over (α_N, α_D)
    and linear regression for (E, A, B). Returns dict class -> (α_N, α_D, E, A, B, sse)."""
    out = {}
    for cls in ["init", "cont"]:
        L = np.array([c[f"L{cls}"] for c in cells.values()])
        N = np.array([c["N"] for c in cells.values()])
        D = np.array([c["D"] for c in cells.values()])
        best = None
        for aN in alpha_N_grid:
            for aD in alpha_D_grid:
                Xn = N**(-aN); Xd = D**(-aD)
                # L = E + A*Xn + B*Xd
                X = np.column_stack([np.ones_like(Xn), Xn, Xd])
                coef, _, _, _ = np.linalg.lstsq(X, L, rcond=None)
                pred = X @ coef
                sse = float(np.sum((pred - L)**2))
                if best is None or sse < best[5]:
                    best = (float(aN), float(aD), float(coef[0]), float(coef[1]), float(coef[2]), sse)
        out[cls] = {"alpha_N": best[0], "alpha_D": best[1], "E": best[2], "A": best[3], "B": best[4], "sse": best[5]}
    return out


def fit_joint_shared(cells, alpha_N_grid=np.linspace(0.02, 2.5, 80),
                     alpha_D_grid=np.linspace(0.02, 2.5, 80)):
    """Shared α_N and α_D across classes; per-class (E, A, B)."""
    best = None
    for aN in alpha_N_grid:
        for aD in alpha_D_grid:
            sse = 0.0; params = {}
            for cls in ["init", "cont"]:
                L = np.array([c[f"L{cls}"] for c in cells.values()])
                N = np.array([c["N"] for c in cells.values()])
                D = np.array([c["D"] for c in cells.values()])
                X = np.column_stack([np.ones(len(L)), N**(-aN), D**(-aD)])
                coef, _, _, _ = np.linalg.lstsq(X, L, rcond=None)
                pred = X @ coef
                sse += float(np.sum((pred - L)**2))
                params[cls] = (float(coef[0]), float(coef[1]), float(coef[2]))
            if best is None or sse < best[2]:
                best = (float(aN), float(aD), sse, params)
    return {"alpha_N": best[0], "alpha_D": best[1], "sse": best[2], "params_per_class": best[3]}


def predict_perclass(fit, cls, N, D):
    p = fit[cls]
    return p["E"] + p["A"] * N**(-p["alpha_N"]) + p["B"] * D**(-p["alpha_D"])


def predict_shared(fit, cls, N, D):
    E, A, B = fit["params_per_class"][cls]
    return E + A * N**(-fit["alpha_N"]) + B * D**(-fit["alpha_D"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout_N", nargs="*", default=["6.9b", "12b"],
                    help="N keys to hold out completely")
    ap.add_argument("--holdout_steps", nargs="*", type=int, default=[143000],
                    help="step values to hold out (for in-N held-out test)")
    args = ap.parse_args()

    cells = collect_cells(N_keys=["410m","1b","1.4b","2.8b","6.9b","12b"])
    print(f"Total cells with per-word data: {len(cells)}")
    if len(cells) < 10:
        print("Not enough cells with per-word data yet — re-runs still in flight.")
        return

    train = {k: v for k, v in cells.items()
             if k[0] not in args.holdout_N and k[1] not in args.holdout_steps}
    test_byN = {k: v for k, v in cells.items() if k[0] in args.holdout_N}
    test_byD = {k: v for k, v in cells.items()
                if k[0] not in args.holdout_N and k[1] in args.holdout_steps}
    print(f"Train cells: {len(train)}  Hold-out by N {args.holdout_N}: {len(test_byN)}  Hold-out by D {args.holdout_steps}: {len(test_byD)}")

    fit_pc = fit_joint_perclass(train)
    fit_sh = fit_joint_shared(train)
    print()
    print("=== M_perclass (per-class α_N, α_D) ===")
    for cls in ["init", "cont"]:
        p = fit_pc[cls]
        print(f"  {cls}: α_N={p['alpha_N']:.3f}  α_D={p['alpha_D']:.3f}  E={p['E']:.3f}  A={p['A']:.2e}  B={p['B']:.2e}")
    print()
    print("=== M_shared (shared α_N, α_D) ===")
    print(f"  shared α_N={fit_sh['alpha_N']:.3f}  α_D={fit_sh['alpha_D']:.3f}")
    for cls in ["init", "cont"]:
        E, A, B = fit_sh["params_per_class"][cls]
        print(f"  {cls}: E={E:.3f}  A={A:.2e}  B={B:.2e}")

    # held-out RMSE
    def rmse(test, predfn):
        sq = []
        for (nk, s), cell in test.items():
            for cls in ["init", "cont"]:
                actual = cell[f"L{cls}"]
                pred = predfn(cls, cell["N"], cell["D"])
                sq.append((pred - actual)**2)
        return float(np.sqrt(np.mean(sq))) if sq else float("nan")

    for name, test in [("hold-out by N", test_byN), ("hold-out by D", test_byD), ("hold-out total", {**test_byN, **test_byD})]:
        if not test: continue
        r_pc = rmse(test, lambda cls, N, D: predict_perclass(fit_pc, cls, N, D))
        r_sh = rmse(test, lambda cls, N, D: predict_shared(fit_sh, cls, N, D))
        print(f"\n{name} (n_cells={len(test)}):")
        print(f"  RMSE M_shared   = {r_sh:.4f}")
        print(f"  RMSE M_perclass = {r_pc:.4f}")
        print(f"  diff (sh - pc)  = {r_sh - r_pc:+.4f} nats  (>0 means perclass wins)")

    # save
    json.dump({"fit_perclass": fit_pc, "fit_shared": fit_sh,
               "train_cells": list(train.keys()),
               "holdout_N": args.holdout_N, "holdout_steps": args.holdout_steps},
              open("joint_ND_fit.json", "w"), indent=2)
    print("\nSaved -> joint_ND_fit.json")


if __name__ == "__main__":
    main()
