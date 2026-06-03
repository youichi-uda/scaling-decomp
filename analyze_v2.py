#!/usr/bin/env python
"""v2 analysis: word-freq-matched position effect + floor-subtracted per-class scaling fits
+ held-out adjudication (M_shared vs M_perclass).

Usage: python analyze_v2.py v2_70m.json v2_160m.json ... [--holdout 6.9b,12b]
"""
import sys, json, argparse
import numpy as np

PARAMS = {"pythia-70m":70e6,"pythia-160m":160e6,"pythia-410m":410e6,"pythia-1b":1.0e9,
          "pythia-1.4b":1.4e9,"pythia-2.8b":2.8e9,"pythia-6.9b":6.9e9,"pythia-12b":12e9}


def key(n): return n.split("/")[-1]


def midwf(d, lo=4, hi=7, field="by_wordfreqdecile_position"):
    """weighted mean loss for init and cont over mid word-freq deciles [lo,hi]."""
    fp = d[field]; iw=cw=inn=cnn=0.0
    for b in range(lo, hi+1):
        a=fp[str(b)]["word_initial"]; c=fp[str(b)]["word_continuation"]
        if a["mean_loss"] is not None: iw+=a["mean_loss"]*a["n"]; inn+=a["n"]
        if c["mean_loss"] is not None: cw+=c["mean_loss"]*c["n"]; cnn+=c["n"]
    return (iw/inn if inn else np.nan, cw/cnn if cnn else np.nan)


def fit_shared_alpha(N, Ls, alphas=None):
    """Ls: dict class->loss array aligned with N. Fit single shared alpha, per-class (E,A) linear.
    Returns (alpha, {cls:(E,A)}, sse)."""
    N=np.asarray(N,float)
    if alphas is None: alphas=np.linspace(0.05,1.5,600)
    best=None
    for al in alphas:
        x=N**(-al); B=np.vstack([np.ones_like(x),x]).T
        sse=0.0; params={}
        for c,L in Ls.items():
            L=np.asarray(L,float); coef,*_=np.linalg.lstsq(B,L,rcond=None)
            E,A=coef; pred=B@coef; sse+=np.sum((pred-L)**2); params[c]=(E,A)
        if best is None or sse<best[2]: best=(al,params,sse)
    return best


def fit_perclass_alpha(N, Ls, alphas=None):
    N=np.asarray(N,float)
    if alphas is None: alphas=np.linspace(0.05,1.5,600)
    out={}
    for c,L in Ls.items():
        L=np.asarray(L,float); best=None
        for al in alphas:
            x=N**(-al); B=np.vstack([np.ones_like(x),x]).T
            coef,*_=np.linalg.lstsq(B,L,rcond=None); pred=B@coef
            sse=np.sum((pred-L)**2)
            if best is None or sse<best[2]: best=(al,tuple(coef),sse)
        out[c]=best
    return out


def predict(N, al, E, A): return E+A*np.asarray(N,float)**(-al)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("files",nargs="+")
    ap.add_argument("--holdout",default="6.9b,12b")
    args=ap.parse_args()
    data={}
    for f in args.files:
        d=json.load(open(f)); data[key(d["model"])]=d
    order=[k for k in PARAMS if k in data]
    Ns=[PARAMS[k] for k in order]
    print("sizes:",order,"\n")

    print("=== overall by-position ===")
    print(f"{'size':12}{'overall':>9}{'init':>8}{'cont':>8}{'other':>8}{'gap':>8}")
    for k in order:
        p=data[k]["by_position"]; i=p['word_initial']['mean_loss']; c=p['word_continuation']['mean_loss']; o=p['other']['mean_loss']
        print(f"{k:12}{data[k]['overall_mean_loss']:9.3f}{i:8.3f}{c:8.3f}{o:8.3f}{i-c:8.3f}")

    print("\n=== WORD-freq-matched init vs cont (mid deciles 4-7) — the control ===")
    print(f"{'size':12}{'init_wf':>9}{'cont_wf':>9}{'gap':>8}")
    iwf=[]; cwf=[]
    for k in order:
        i,c=midwf(data[k]); iwf.append(i); cwf.append(c)
        print(f"{k:12}{i:9.3f}{c:9.3f}{i-c:8.3f}")
    g0,g1=iwf[0]-cwf[0],iwf[-1]-cwf[-1]
    print(f"word-freq-matched gap change {order[0]}->{order[-1]}: {g0:.3f}->{g1:.3f} ({100*(g1-g0)/g0:+.1f}%)")

    print("\n=== seq-band mid check (init/cont, mid sequence positions) ===")
    for k in order:
        m=data[k]["by_seqband_position"]["mid"]
        print(f"{k:12} init {m['word_initial']['mean_loss']:.3f}  cont {m['word_continuation']['mean_loss']:.3f}")

    # ---- scaling fits on word-freq-matched classes ----
    classes={"init_wf":iwf,"cont_wf":cwf}
    holds=set("pythia-"+h for h in args.holdout.split(","))
    train=[k for k in order if k not in holds]
    test=[k for k in order if k in holds]
    if len(train)>=4 and test:
        Ntr=[PARAMS[k] for k in train]; Nte=[PARAMS[k] for k in test]
        Ltr={c:[v[order.index(k)] for k in train] for c,v in classes.items()}
        Lte={c:[v[order.index(k)] for k in test] for c,v in classes.items()}
        print(f"\n=== held-out adjudication (train {train} -> test {test}) ===")
        al_s,par_s,_=fit_shared_alpha(Ntr,Ltr)
        per=fit_perclass_alpha(Ntr,Ltr)
        print(f"M_shared alpha={al_s:.3f}; M_perclass alpha: "+", ".join(f"{c}={per[c][0]:.3f}" for c in classes))
        da=per['init_wf'][0]-per['cont_wf'][0]
        print(f"per-class alpha diff (init-cont) = {da:+.3f}")
        rmse_s=rmse_p=0.0; ns=0
        for c in classes:
            E,A=par_s[c]; ps=predict(Nte,al_s,E,A)
            alp,(Ep,Ap),_=per[c]; pp=predict(Nte,alp,Ep,Ap)
            for j,k in enumerate(test):
                rmse_s+=(ps[j]-Lte[c][j])**2; rmse_p+=(pp[j]-Lte[c][j])**2; ns+=1
                print(f"  {c:10} {k}: actual {Lte[c][j]:.3f}  shared-pred {ps[j]:.3f}  perclass-pred {pp[j]:.3f}")
        rmse_s=(rmse_s/ns)**.5; rmse_p=(rmse_p/ns)**.5
        print(f"held-out RMSE: M_shared={rmse_s:.4f}  M_perclass={rmse_p:.4f}  (perclass better by {rmse_s-rmse_p:+.4f})")
        print("VERDICT THRESHOLDS: position-alpha REAL if perclass beats shared by >0.02 nats; "
              "uniform-alpha if shared within 0.03 and perclass doesn't beat by >0.02")
    else:
        print(f"\n(need >=4 train sizes + holdout present; have train={train} test={test})")


if __name__=="__main__":
    main()
