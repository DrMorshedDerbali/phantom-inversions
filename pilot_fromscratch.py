"""
pilot.py - runs the three things the professor asked for:
  (A) Detector validation on KNOWN-ANSWER policies (must-flag / must-not-flag).
  (B) eta=1 symmetric control: inversions MUST vanish without value heterogeneity.
  (C) eta=8 (rho x mu) signal grid: does any real learned policy invert?
Saves results to pilot_results.json and arrays for the figure.
"""
import json, time, numpy as np
import invsim as S

EVAL_SEEDS = list(range(50000, 50120))     # 120 CRN-paired evaluation episodes
TRAIN_SEED = 7000
N_TRAIN = 4000
BETA = 0.04
P_IN = 0.4

def run_cell(eta, alpha_r, p_out, net_seed=0, n_train=N_TRAIN):
    sim = S.Sim(p_in=P_IN, p_out=p_out, beta=BETA, eta=eta, alpha_r=alpha_r,
                net_seed=net_seed)
    Q = sim.train(n_train, seed_base=TRAIN_SEED)
    opt_m = sim.evaluate(S.greedy_policy(Q), EVAL_SEEDS)
    bestJ, kl, kh, nai_m = S.tune_naive(sim, EVAL_SEEDS)
    md, ci_hi, sig = S.paired_advantage(opt_m['J'], nai_m['J'])
    det = S.detect_inversion(opt_m, nai_m, sig)
    return dict(sim=sim, Q=Q, opt=opt_m, nai=nai_m, naive_k=(kl, kh),
                advJ=md, advCIhi=ci_hi, det=det)

def summ(tag, r):
    o, n = r['opt'], r['nai']
    d = r['det']
    print(f"{tag:26s} | optJ {o['J'].mean():7.1f}  naiJ {n['J'].mean():7.1f}  "
          f"dJ {r['advJ']:+7.1f}(hi {r['advCIhi']:+6.1f}) | "
          f"infA o/n {o['inf_A'].mean():4.1f}/{n['inf_A'].mean():4.1f}  "
          f"infB o/n {o['inf_B'].mean():4.1f}/{n['inf_B'].mean():4.1f}  "
          f"cost o/n {o['cost'].mean():5.1f}/{n['cost'].mean():5.1f} | "
          f"P{int(d['P'])} X{int(d['X'])} T{int(d['T'])} "
          f"adv{int(d['advantage'])} => INV={int(d['inversion'])}")

results = {}

# ---------------------------------------------------------------- (A) detector validation
print("="*120)
print("(A) DETECTOR VALIDATION on known-answer policies "
      "(cell: eta=8, rho=2, weak bridge p_out=0.02)")
print("-"*120)
val_sim = S.Sim(p_in=P_IN, p_out=0.02, beta=BETA, eta=8.0, alpha_r=2.0, net_seed=0)
bestJ, kl, kh, val_nai = S.tune_naive(val_sim, EVAL_SEEDS)
# known-answer: definitionally inverted
inv_m = val_sim.evaluate(S.pol_inverted_burn, EVAL_SEEDS)
md_i, ci_i, sig_i = S.paired_advantage(inv_m['J'], val_nai['J'])
det_i = S.detect_inversion(inv_m, val_nai, sig_i)
# known-answer: definitionally monotone (should NOT flag)
mon_m = val_sim.evaluate(S.pol_monotone, EVAL_SEEDS)
md_m, ci_m, sig_m = S.paired_advantage(mon_m['J'], val_nai['J'])
det_m = S.detect_inversion(mon_m, val_nai, sig_m)
print(f"best-naive params (klow,khigh)=({kl},{kh}), naiveJ={bestJ:.1f}")
summ("KNOWN inverted-burn", dict(opt=inv_m, nai=val_nai, advJ=md_i, advCIhi=ci_i, det=det_i))
summ("KNOWN monotone", dict(opt=mon_m, nai=val_nai, advJ=md_m, advCIhi=ci_m, det=det_m))
print(f"\n  PASS check: inverted flagged? {det_i['inversion']}  (want True)   |   "
      f"monotone flagged? {det_m['inversion']}  (want False)")
results['validation'] = dict(
    inverted=dict(det=det_i, advJ=md_i, infA=float(inv_m['inf_A'].mean()),
                  infB=float(inv_m['inf_B'].mean())),
    monotone=dict(det=det_m, advJ=md_m),
    pass_inverted=bool(det_i['inversion']), pass_monotone=bool(not det_m['inversion']))

# ---------------------------------------------------------------- (B) eta=1 control
print("\n" + "="*120)
print("(B) eta=1 SYMMETRIC CONTROL  (no value heterogeneity -> inversions MUST vanish)")
print("-"*120)
ctrl = {}
for rho in [2.0, 5.0, 15.0]:
    r = run_cell(eta=1.0, alpha_r=rho, p_out=0.08)
    summ(f"eta=1 rho={rho:g} mu=0.20", r)
    ctrl[f"rho{rho:g}"] = r
any_inv_ctrl = any(v['det']['inversion'] for v in ctrl.values())
print(f"\n  CONTROL check: any inversion flagged at eta=1? {any_inv_ctrl}  (want False)")
results['control_eta1_any_inversion'] = bool(any_inv_ctrl)

# ---------------------------------------------------------------- (C) eta=8 signal grid
print("\n" + "="*120)
print("(C) eta=8 SIGNAL GRID  (rho x mu)   [the de-risking pilot]")
print("-"*120)
RHOS = [2.0, 5.0, 15.0]
POUTS = [0.02, 0.08, 0.16]                 # mu = p_out/p_in = 0.05, 0.20, 0.40
grid = {}
t0 = time.time()
for rho in RHOS:
    for pout in POUTS:
        r = run_cell(eta=8.0, alpha_r=rho, p_out=pout)
        mu = pout / P_IN
        summ(f"eta=8 rho={rho:<4g} mu={mu:.2f}", r)
        grid[(rho, pout)] = r
print(f"\n  grid wall-clock: {time.time()-t0:.1f}s")

# matrices for the figure
flag = np.zeros((len(RHOS), len(POUTS)))
adv = np.zeros_like(flag)
typ = np.empty((len(RHOS), len(POUTS)), dtype=object)
for i, rho in enumerate(RHOS):
    for j, pout in enumerate(POUTS):
        d = grid[(rho, pout)]['det']
        flag[i, j] = 1 if d['inversion'] else 0
        adv[i, j] = -grid[(rho, pout)]['advJ']         # positive = optimal better
        labs = [k for k in ('P', 'X', 'T') if d[k]]
        typ[i, j] = '+'.join(labs) if (d['inversion'] and labs) else ('~' if labs else '')

np.savez('/home/claude/pilot_grid.npz',
         flag=flag, adv=adv,
         rhos=np.array(RHOS), pouts=np.array(POUTS), p_in=P_IN,
         typ=np.array(typ, dtype=object))

n_inv = int(flag.sum())
print("\n" + "="*120)
print(f"PILOT SUMMARY: {n_inv}/{flag.size} eta=8 cells flagged as inversions.")
print(f"  detector validation: inverted-flagged={results['validation']['pass_inverted']}, "
      f"monotone-not-flagged={results['validation']['pass_monotone']}")
print(f"  eta=1 control clean (no inversions): {not results['control_eta1_any_inversion']}")
results['eta8_n_inversions'] = n_inv
results['eta8_grid_size'] = int(flag.size)
results['eta8_types'] = {f"rho{r}_pout{p}": grid[(r, p)]['det'] for r in RHOS for p in POUTS}

with open('/home/claude/pilot_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print("\nsaved pilot_results.json and pilot_grid.npz")
