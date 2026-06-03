#!/usr/bin/env python
"""Per-D N-axis fit with bootstrap CIs at each D. Addresses Codex critique
that §4.3 has no CIs and railed fits are mis-interpreted as Δα=0.
"""
import json, numpy as np, os

PARAMS = {"160m":160e6,"410m":410e6,"1b":1.0e9,"1.4b":1.4e9,
          "2.8b":2.8e9,"6.9b":6.9e9,"12b":12e9}
TOKS = 2_097_152
ALL_STEPS = [1000,2000,4000,7000,10000,15000,22000,25000,30000,42000,
             50000,70000,100000,130000,143000]

def midwf(d, lo=4, hi=7):
    fp = d.get("by_wordfreqdecile_position_per_word")
    if fp is None: return None, None
    iw=cw=inn=cnn=0.0
    for b in range(lo, hi+1):
        x = fp[str(b)]; a = x["word_initial"]; c = x["word_continuation"]
        if a["mean_loss"] is not None: iw += a["mean_loss"]*a["n_words"]; inn += a["n_words"]
        if c["mean_loss"] is not None: cw += c["mean_loss"]*c["n_words"]; cnn += c["n_words"]
    return (iw/inn if inn else None, cw/cnn if cnn else None)

def fit(N, L, grid=np.linspace(0.02, 1.8, 800), require_pos=True):
    N=np.asarray(N,float); L=np.asarray(L,float); best=None
    railed = False
    for al in grid:
        X=np.column_stack([np.ones_like(N), N**(-al)])
        coef,_,_,_=np.linalg.lstsq(X,L,rcond=None)
        if require_pos and (coef[1] <= 0 or coef[0] < 0): continue
        pred=X@coef; sse=float(np.sum((pred-L)**2))
        if best is None or sse<best[3]: best=(al,coef[0],coef[1],sse,pred)
    if best is None: return None, True
    # check railing at grid edge
    if best[0] == grid[0] or best[0] == grid[-1]:
        railed = True
    return best, railed

def boot(N, init, cont, B=1000, seed=20260603):
    fi,r_i=fit(N,init); fc,r_c=fit(N,cont)
    if fi is None or fc is None: return None,None,(np.nan,np.nan),True
    railed = r_i or r_c
    ri=np.array(init)-fi[4]; rc=np.array(cont)-fc[4]
    rng=np.random.default_rng(seed); diffs=[]
    fails=0
    for _ in range(B):
        Li=fi[4]+rng.choice(ri,size=len(N),replace=True)
        Lc=fc[4]+rng.choice(rc,size=len(N),replace=True)
        try:
            fbi,_=fit(N,Li); fbc,_=fit(N,Lc)
            if fbi is None or fbc is None: fails+=1; continue
            diffs.append(fbi[0]-fbc[0])
        except: fails+=1
    if len(diffs) < B*0.5: return fi,fc,(np.nan,np.nan),True
    return fi,fc, np.quantile(diffs,[0.025,0.975]), railed


print('=== Per-D N-axis fit with bootstrap CIs (per-word aggregation) ===')
print(f'{"step":>7}{"D(B)":>9}{"n_N":>4}{"α_init":>9}{"α_cont":>9}{"Δα_N":>9}{"95% CI":>22}{"railed?":>9}')
print('=' * 80)
results = []
for s in ALL_STEPS:
    Ns=[]; inits=[]; conts=[]
    for nk, N in PARAMS.items():
        f = f"v2mt_{nk}_step{s}.json" if s != 143000 else f"v2mt_{nk}.json"
        if not os.path.exists(f): continue
        d = json.load(open(f))
        li, lc = midwf(d)
        if li is None or lc is None: continue
        # also skip known-bad 12b step50000
        if nk == '12b' and s == 50000 and (li > 5 or lc > 5): continue
        Ns.append(N); inits.append(li); conts.append(lc)
    if len(Ns) < 4: continue
    fi,fc,(q025,q975),railed = boot(Ns, inits, conts)
    if fi is None:
        print(f'{s:>7}{s*TOKS/1e9:>9.1f}{len(Ns):>4}  FIT FAILED')
        continue
    diff = fi[0] - fc[0]
    railed_str = "YES" if railed else "no"
    ci_str = f'[{q025:+.3f},{q975:+.3f}]' if not np.isnan(q025) else '[unstable]'
    print(f'{s:>7}{s*TOKS/1e9:>9.1f}{len(Ns):>4}{fi[0]:>9.3f}{fc[0]:>9.3f}{diff:>+9.3f}  {ci_str:>20}  {railed_str:>7}')
    results.append({'step':s,'D':s*TOKS,'n_N':len(Ns),'alpha_init':float(fi[0]),'alpha_cont':float(fc[0]),
                    'diff':float(diff),'CI95':[float(q025),float(q975)] if not np.isnan(q025) else None,
                    'railed':bool(railed)})

print('=' * 80)
json.dump(results, open('final_perD_Nfit_bootstrap.json','w'), indent=2)
print('\nSaved -> final_perD_Nfit_bootstrap.json')
