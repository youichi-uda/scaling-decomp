#!/usr/bin/env python
"""Final consistent N-axis fit using per-word-matched aggregation throughout.
This replaces the older multi-token-strict numbers in §4.1.
"""
import json, numpy as np, os

PARAMS = {"70m":70e6,"160m":160e6,"410m":410e6,"1b":1.0e9,"1.4b":1.4e9,
          "2.8b":2.8e9,"6.9b":6.9e9,"12b":12e9}

def midwf(d, lo=4, hi=7):
    fp = d.get("by_wordfreqdecile_position_per_word")
    if fp is None: return None, None
    iw=cw=inn=cnn=0.0
    for b in range(lo, hi+1):
        x = fp[str(b)]; a = x["word_initial"]; c = x["word_continuation"]
        if a["mean_loss"] is not None: iw += a["mean_loss"]*a["n_words"]; inn += a["n_words"]
        if c["mean_loss"] is not None: cw += c["mean_loss"]*c["n_words"]; cnn += c["n_words"]
    return (iw/inn if inn else None, cw/cnn if cnn else None)

def fit(N, L, grid=np.linspace(0.02, 1.8, 800)):
    N=np.asarray(N,float); L=np.asarray(L,float); best=None
    for al in grid:
        X=np.column_stack([np.ones_like(N), N**(-al)])
        coef,_,_,_=np.linalg.lstsq(X,L,rcond=None)
        if coef[1] <= 0 or coef[0] < 0: continue
        pred=X@coef; sse=float(np.sum((pred-L)**2))
        if best is None or sse<best[3]: best=(al,coef[0],coef[1],sse,pred)
    return best

def boot(N, init, cont, B=2000, seed=20260603):
    fi=fit(N,init); fc=fit(N,cont)
    ri=np.array(init)-fi[4]; rc=np.array(cont)-fc[4]
    rng=np.random.default_rng(seed); diffs=[]
    for _ in range(B):
        Li=fi[4]+rng.choice(ri,size=len(N),replace=True)
        Lc=fc[4]+rng.choice(rc,size=len(N),replace=True)
        try:
            fbi=fit(N,Li); fbc=fit(N,Lc); diffs.append(fbi[0]-fbc[0])
        except: pass
    diffs=np.array(diffs)
    return fi, fc, np.quantile(diffs,[0.025,0.975]), diffs

# 7 non-anomalous sizes
sizes = ['160m','410m','1b','1.4b','2.8b','6.9b','12b']
Ns = [PARAMS[s] for s in sizes]
print('=== Per-word matched aggregation, 7 non-anomalous sizes, mid deciles 4-7 ===')
print(f'{"size":>5}{"L_init":>9}{"L_cont":>9}{"gap":>9}')
inits=[]; conts=[]
for s in sizes:
    d=json.load(open(f'v2mt_{s}.json'))
    i,c=midwf(d); inits.append(i); conts.append(c)
    print(f'{s:>5}{i:>9.3f}{c:>9.3f}{i-c:>+9.3f}')

fi,fc,(q025,q975),diffs = boot(Ns, inits, conts)
print()
print(f'α_init = {fi[0]:.3f}  E_init = {fi[1]:.3f}  A_init = {fi[2]:.2e}')
print(f'α_cont = {fc[0]:.3f}  E_cont = {fc[1]:.3f}  A_cont = {fc[2]:.2e}')
print(f'Δα = {fi[0]-fc[0]:+.3f}  95% bootstrap CI [{q025:+.3f}, {q975:+.3f}]')
print(f'P(Δα >= 0) = {(diffs>=0).mean():.4f}')

# Held-out adjudication: train on 160m-1.4b, hold out 2.8b/6.9b/12b
print()
print('=== Held-out adjudication (train: 160m-1.4b, holdout: 2.8b/6.9b/12b) ===')
tr=sizes[:4]; te=sizes[4:]
Ntr=[PARAMS[s] for s in tr]; Nte=[PARAMS[s] for s in te]
Litr=inits[:4]; Lite=inits[4:]; Lctr=conts[:4]; Lcte=conts[4:]

# Fit per-class on train
fi_tr=fit(Ntr,Litr); fc_tr=fit(Ntr,Lctr)
print(f'M_perclass train fit: α_init={fi_tr[0]:.3f}  α_cont={fc_tr[0]:.3f}  Δα_train={fi_tr[0]-fc_tr[0]:+.3f}')

# Shared-α fit on train
def fit_shared(N, Ls, grid=np.linspace(0.02,1.8,800)):
    N=np.asarray(N,float); best=None
    for al in grid:
        x=N**(-al); B=np.vstack([np.ones_like(x),x]).T
        sse=0; params={}
        all_pos=True
        for c,L in Ls.items():
            L=np.asarray(L,float); coef,_,_,_=np.linalg.lstsq(B,L,rcond=None)
            if coef[1]<=0 or coef[0]<0: all_pos=False; break
            pred=B@coef; sse+=float(np.sum((pred-L)**2)); params[c]=(float(coef[0]),float(coef[1]))
        if not all_pos: continue
        if best is None or sse<best[2]: best=(float(al),params,sse)
    return best

shared = fit_shared(Ntr, {'init':Litr,'cont':Lctr})
al_s, par_s, _ = shared
print(f'M_shared train fit: α={al_s:.3f}')

# Held-out RMSE
import math
sse_pc = sse_sh = 0.0; n=0
for cls,(fp,L_te) in [('init',(fi_tr,Lite)),('cont',(fc_tr,Lcte))]:
    for j,(N,L) in enumerate(zip(Nte,L_te)):
        pred_pc = fp[1] + fp[2]*N**(-fp[0])
        E_s,A_s = par_s[cls]
        pred_sh = E_s + A_s*N**(-al_s)
        sse_pc += (pred_pc-L)**2; sse_sh += (pred_sh-L)**2; n+=1
        print(f'  {cls:5} {[s for s in te][j]:>5} N={N:.1e}: actual={L:.3f}  M_perclass={pred_pc:.3f}  M_shared={pred_sh:.3f}')
rmse_pc = math.sqrt(sse_pc/n); rmse_sh = math.sqrt(sse_sh/n)
print()
print(f'Held-out RMSE M_perclass = {rmse_pc:.4f}')
print(f'Held-out RMSE M_shared   = {rmse_sh:.4f}')
print(f'M_perclass improvement = {rmse_sh-rmse_pc:+.4f} nats')
print(f'Pre-registered threshold: 0.02 → {"PASS" if rmse_sh-rmse_pc > 0.02 else "FAIL"}')

# Also Wikitext
print()
print('=== Wikitext-103 secondary corpus (train sizes 70m-1.4b) ===')
wsizes=['70m','160m','410m','1b','1.4b']
wNs=[PARAMS[s] for s in wsizes]
winits=[]; wconts=[]
for s in wsizes:
    d=json.load(open(f'v2_wikitext_{s}.json'))
    # wikitext data may use older multi-token strict aggregation only
    # check what aggregations are available
    if 'by_wordfreqdecile_position_per_word' in d:
        i,c=midwf(d)
    elif 'by_wordfreqdecile_position_multitokenonly' in d:
        fp=d['by_wordfreqdecile_position_multitokenonly']; iw=cw=inn=cnn=0.0
        for b in range(4,8):
            x=fp[str(b)]; a=x['word_initial']; cc=x['word_continuation']
            if a['mean_loss'] is not None: iw+=a['mean_loss']*a['n']; inn+=a['n']
            if cc['mean_loss'] is not None: cw+=cc['mean_loss']*cc['n']; cnn+=cc['n']
        i,c=iw/inn,cw/cnn
    else:
        # oldest format: by_wordfreqdecile_position (token-weighted, all words)
        fp=d['by_wordfreqdecile_position']; iw=cw=inn=cnn=0.0
        for b in range(4,8):
            x=fp[str(b)]; a=x['word_initial']; cc=x['word_continuation']
            if a['mean_loss'] is not None: iw+=a['mean_loss']*a['n']; inn+=a['n']
            if cc['mean_loss'] is not None: cw+=cc['mean_loss']*cc['n']; cnn+=cc['n']
        i,c=iw/inn,cw/cnn
    winits.append(i); wconts.append(c)
fi_w=fit(wNs,winits); fc_w=fit(wNs,wconts)
print(f'  α_init={fi_w[0]:.3f}  α_cont={fc_w[0]:.3f}  Δα={fi_w[0]-fc_w[0]:+.3f}')
print('  (Wikitext used multi-token-strict aggregation; not per-word matched. Note in paper.)')

# save final canonical numbers
json.dump({
    'method':'per-word matched aggregation, mid word-freq deciles 4-7',
    'sizes':sizes, 'n_N':len(sizes),
    'alpha_init':float(fi[0]),'alpha_cont':float(fc[0]),
    'E_init':float(fi[1]),'E_cont':float(fc[1]),
    'A_init':float(fi[2]),'A_cont':float(fc[2]),
    'diff':float(fi[0]-fc[0]),
    'CI95':[float(q025),float(q975)],
    'P_nonneg':float((diffs>=0).mean()),
    'held_out':{
        'train_sizes':tr,'test_sizes':te,
        'M_perclass_train':{'alpha_init':float(fi_tr[0]),'alpha_cont':float(fc_tr[0]),'diff':float(fi_tr[0]-fc_tr[0])},
        'M_shared_train':{'alpha':float(al_s)},
        'M_perclass_test_RMSE':float(rmse_pc),
        'M_shared_test_RMSE':float(rmse_sh),
        'improvement':float(rmse_sh-rmse_pc)},
    'wikitext':{'sizes':wsizes,'alpha_init':float(fi_w[0]),'alpha_cont':float(fc_w[0]),'diff':float(fi_w[0]-fc_w[0])},
}, open('final_Naxis_perword.json','w'), indent=2)
print('\nSaved -> final_Naxis_perword.json')
