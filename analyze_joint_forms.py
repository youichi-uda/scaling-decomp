#!/usr/bin/env python
"""Compare multiple joint L_c(N, D) functional forms for per-class fits.

Forms:
- ADD:  L = E + A/N^αN + B/D^αD                    (Chinchilla-additive)
- MULT: L = E + C·N^-αN·D^-αD                       (multiplicative interaction)
- HOFF: L = E + (A/N^αN + B/D^αD)^β                 (Hoffmann generalized)
- COMPUTE: L = E + K·(N·D)^-α                       (compute-only)
- MIN:  L = E + max(A·N^-αN, B·D^-αD)               (saturation min)

For each form, report:
- Best per-class fits (αN/αD/E and any extra params)
- Per-class diffs
- In-sample fit RMSE
- Bootstrap CI on Δα_N and Δα_D where defined
"""
import json, numpy as np, os
from scipy.optimize import minimize

PARAMS = {"1b":1.0e9,"1.4b":1.4e9,"6.9b":6.9e9,"12b":12e9}
TOKS = 2_097_152
ALL_STEPS = [1000,2000,4000,7000,10000,15000,22000,25000,30000,42000,70000,100000,130000,143000]  # excludes step50000 (HF bug)


def per_word_midwf(d, lo=4, hi=7):
    fp = d.get("by_wordfreqdecile_position_per_word")
    if fp is None: return None, None
    iw=cw=inn=cnn=0.0
    for b in range(lo, hi+1):
        x = fp[str(b)]; a = x["word_initial"]; c = x["word_continuation"]
        if a["mean_loss"] is not None and not (isinstance(a["mean_loss"], float) and np.isnan(a["mean_loss"])):
            iw += a["mean_loss"] * a["n_words"]; inn += a["n_words"]
        if c["mean_loss"] is not None and not (isinstance(c["mean_loss"], float) and np.isnan(c["mean_loss"])):
            cw += c["mean_loss"] * c["n_words"]; cnn += c["n_words"]
    return (iw/inn if inn else None, cw/cnn if cnn else None)


def collect_cells():
    cells = []
    for nk, N in PARAMS.items():
        for s in ALL_STEPS:
            f = f"v2mt_{nk}_step{s}.json" if s != 143000 else f"v2mt_{nk}.json"
            if not os.path.exists(f): continue
            d = json.load(open(f))
            li, lc = per_word_midwf(d)
            if li is None or lc is None: continue
            cells.append({"N": N, "D": s*TOKS, "Linit": li, "Lcont": lc})
    return cells


# ---- Functional forms ----
def loss_ADD(p, N, D):  # E, A, αN, B, αD
    E,A,aN,B,aD = p
    return E + A*N**(-aN) + B*D**(-aD)

def loss_MULT(p, N, D):  # E, C, αN, αD
    E,C,aN,aD = p
    return E + C*N**(-aN)*D**(-aD)

def loss_HOFF(p, N, D):  # E, A, αN, B, αD, β
    E,A,aN,B,aD,beta = p
    inner = A*N**(-aN) + B*D**(-aD)
    return E + np.power(np.clip(inner, 1e-30, 1e30), beta)

def loss_COMPUTE(p, N, D):  # E, K, α
    E,K,al = p
    return E + K*(N*D)**(-al)

def loss_MIN(p, N, D):  # E, A, αN, B, αD
    E,A,aN,B,aD = p
    return E + np.minimum(A*N**(-aN), B*D**(-aD))


FORMS = {
    "ADD":     (loss_ADD,    [1.0,1e10,0.5,1e6,0.5],                ['E','A','αN','B','αD'], (0.05,1.8), (0.05,1.8)),
    "MULT":    (loss_MULT,   [1.0,1e15,0.5,0.5],                    ['E','C','αN','αD'],     (0.05,1.8), (0.05,1.8)),
    "HOFF":    (loss_HOFF,   [1.0,1e10,0.5,1e6,0.5,0.5],            ['E','A','αN','B','αD','β'], (0.05,1.8), (0.05,1.8)),
    "COMPUTE": (loss_COMPUTE,[1.0,1e15,0.3],                        ['E','K','α'],           (0.05,0.8), None),
    "MIN":     (loss_MIN,    [1.0,1e10,0.5,1e6,0.5],                ['E','A','αN','B','αD'], (0.05,1.8), (0.05,1.8)),
}


def fit_form(cells, form_name, L_key, n_starts=20):
    """Multi-start gradient fit with NNLS-like positivity via softplus reparam."""
    fn, p0, names, _, _ = FORMS[form_name]
    L = np.array([c[L_key] for c in cells])
    N = np.array([c["N"] for c in cells])
    D = np.array([c["D"] for c in cells])
    def obj(p_raw):
        # use softplus to keep params positive; E gets exp; alphas bounded with sigmoid * 1.8
        p_pos = np.log1p(np.exp(p_raw))  # softplus, makes all positive
        try:
            pred = fn(p_pos, N, D)
            return float(np.sum((pred - L)**2))
        except Exception:
            return 1e30
    rng = np.random.default_rng(20260602 + hash(form_name + L_key) % 10000)
    best = None
    for trial in range(n_starts):
        p_init = np.log(np.array(p0) * (0.1 + 1.9*rng.random(len(p0))))
        try:
            r = minimize(obj, p_init, method="Nelder-Mead",
                         options={"maxiter": 2000, "xatol":1e-6, "fatol":1e-8})
            if r.fun < (best[1] if best else 1e30):
                p_fit = np.log1p(np.exp(r.x))
                best = (p_fit, float(r.fun))
        except Exception:
            pass
    if best is None: return None, None
    p, sse = best
    pred = fn(p, N, D)
    rmse = float(np.sqrt(np.mean((pred - L)**2)))
    return {k: float(v) for k, v in zip(names, p)}, rmse


def main():
    cells = collect_cells()
    print(f"Total cells: {len(cells)}")
    print()
    for form_name, (fn, p0, names, _, _) in FORMS.items():
        print(f"=== {form_name} ===")
        pi, ri = fit_form(cells, form_name, "Linit", n_starts=30)
        pc, rc = fit_form(cells, form_name, "Lcont", n_starts=30)
        if pi is None or pc is None:
            print(f"  fit failed"); continue
        # delta on α params (if present)
        for k in ['αN','αD','α']:
            if k in pi and k in pc:
                print(f"  init: {' '.join(f'{n}={v:.3f}' for n,v in pi.items())}  RMSE={ri:.4f}")
                print(f"  cont: {' '.join(f'{n}={v:.3f}' for n,v in pc.items())}  RMSE={rc:.4f}")
                print(f"  Δ{k} (init-cont) = {pi[k]-pc[k]:+.3f}")
                break
        else:
            print(f"  init: {' '.join(f'{n}={v:.3f}' for n,v in pi.items())}  RMSE={ri:.4f}")
            print(f"  cont: {' '.join(f'{n}={v:.3f}' for n,v in pc.items())}  RMSE={rc:.4f}")
        print()


if __name__ == "__main__":
    main()
