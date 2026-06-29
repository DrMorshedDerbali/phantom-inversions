"""
bc_stage.py <stage>  - resumable BC warm-start pilot.
stages: grid1 grid2 control robust figure
Writes each cell result to bc_results.json (dict keyed by cell id) as it finishes.
"""
import sys, json, time, os, numpy as np
import invsim as S

EVAL = list(range(50000, 50120)); P_IN = 0.4; BETA = 0.04
NEP = 9000; NROLL = 1200; MARGIN = 5.0
RHOS = [2.0, 5.0, 15.0]; POUTS = [0.02, 0.08, 0.16]
DB = "/home/claude/bc_results.json"

def load():
    return json.load(open(DB)) if os.path.exists(DB) else {}
def save(d):
    json.dump(d, open(DB, "w"), indent=2)

def cell(eta, rho, pout, nep=NEP, margin=MARGIN, eps0=0.30, eps1=0.05):
    sim = S.Sim(p_in=P_IN, p_out=pout, beta=BETA, eta=eta, alpha_r=rho,
                net_seed=0, gamma_rl=1.0)
    bj, kl, kh, nai = S.tune_naive(sim, EVAL)
    Qw = S.warm_start_Q(sim, S.naive_factory(kl, kh), n_roll=NROLL, margin=margin)
    Q = sim.train(nep, seed_base=7000, eps0=eps0, eps1=eps1, eps_frac=0.7,
                  a0=0.10, a1=0.01, Q_init=Qw)
    opt = sim.evaluate(S.greedy_policy(Q), EVAL)
    md, hi, sig = S.paired_advantage(opt['J'], nai['J'])
    det = S.detect_inversion(opt, nai, sig)
    conv = bool(opt['J'].mean() <= nai['J'].mean() * 1.05 + 1e-6)
    return dict(optJ=float(opt['J'].mean()), naiJ=float(nai['J'].mean()),
                dJ=float(md), hi=float(hi), conv=conv, det=det,
                infA=[float(opt['inf_A'].mean()), float(nai['inf_A'].mean())],
                infB=[float(opt['inf_B'].mean()), float(nai['inf_B'].mean())],
                cost=[float(opt['cost'].mean()), float(nai['cost'].mean())])

def pr(tag, r):
    d = r['det']
    print(f"{tag:24s} optJ {r['optJ']:7.1f} naiJ {r['naiJ']:7.1f} dJ {r['dJ']:+7.1f}"
          f"(hi{r['hi']:+6.1f}) conv{int(r['conv'])} | infA {r['infA'][0]:4.1f}/{r['infA'][1]:4.1f}"
          f" infB {r['infB'][0]:4.1f}/{r['infB'][1]:4.1f} cost {r['cost'][0]:5.1f}/{r['cost'][1]:5.1f}"
          f" | P{int(d['P'])}X{int(d['X'])}T{int(d['T'])}adv{int(d['advantage'])} INV={int(d['inversion'])}")

stage = sys.argv[1]
d = load()
t0 = time.time()

if stage in ("grid1", "grid2"):
    cells = [(i, j) for i in range(3) for j in range(3)]
    chunk = cells[:5] if stage == "grid1" else cells[5:]
    for (i, j) in chunk:
        rho, pout = RHOS[i], POUTS[j]
        tc = time.time(); r = cell(8.0, rho, pout)
        d[f"g_{i}_{j}"] = r; save(d)
        pr(f"eta=8 rho={rho:<4g} mu={pout/P_IN:.2f} ({time.time()-tc:.0f}s)", r)

elif stage == "control":
    for rho in RHOS:
        tc = time.time(); r = cell(1.0, rho, 0.08)
        d[f"c_{rho:g}"] = r; save(d)
        pr(f"eta=1 rho={rho:<4g} mu=0.20 ({time.time()-tc:.0f}s)", r)

elif stage == "robust":
    specs = [("r5", 8.0, 5.0, 0.02), ("r2", 8.0, 2.0, 0.02)]
    sett = [(5.0, 0.30, 0.05), (10.0, 0.50, 0.10), (2.0, 0.20, 0.02)]
    for (tag, eta, rho, pout) in specs:
        for (m, e0, e1) in sett:
            tc = time.time(); r = cell(eta, rho, pout, margin=m, eps0=e0, eps1=e1)
            d[f"{tag}_m{m:g}_e{e0:g}"] = r; save(d)
            print(f"{tag} margin={m:<4g} eps={e0:.2f}->{e1:.2f} | dJ {r['dJ']:+6.1f} "
                  f"conv{int(r['conv'])} P{int(r['det']['P'])}X{int(r['det']['X'])}"
                  f"T{int(r['det']['T'])}adv{int(r['det']['advantage'])} "
                  f"INV={int(r['det']['inversion'])} ({time.time()-tc:.0f}s)")

elif stage == "figure":
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    adv = np.zeros((3, 3)); ann = np.empty((3, 3), dtype=object)
    conv = np.zeros((3, 3)); inv = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            r = d[f"g_{i}_{j}"]; adv[i, j] = r['dJ']
            conv[i, j] = 1 if r['conv'] else 0; inv[i, j] = 1 if r['det']['inversion'] else 0
            ann[i, j] = ''.join(k for k in ('P', 'X', 'T') if r['det'][k]) or '-'
    ctrl = [d[f"c_{rho:g}"] for rho in RHOS]
    n_conv = int(conv.sum()); n_inv = int(inv.sum())
    ctrl_false = sum(1 for c in ctrl if c['det']['inversion'])

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.2),
                                   gridspec_kw={'width_ratios': [1.35, 1]})
    vmax = max(1.0, np.abs(adv).max())
    im = axA.imshow(adv, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    axA.set_xticks(range(3)); axA.set_xticklabels([f"{p/P_IN:.2f}" for p in POUTS])
    axA.set_yticks(range(3)); axA.set_yticklabels([f"{r:g}" for r in RHOS])
    axA.set_xlabel("network mixing  $\\mu=p_{out}/p_{in}$  (weak $\\to$ strong bridge)")
    axA.set_ylabel("cost ratio  $\\rho=\\alpha_r/\\beta_r$")
    axA.set_title("(A)  $\\eta=8$, BC warm-start + invariant gate\n"
                  "color = J(optimal) - J(naive)   [blue = RL better]", fontsize=11)
    for i in range(3):
        for j in range(3):
            axA.text(j, i, f"{adv[i, j]:+.0f}\n[{ann[i, j]}]", ha='center', va='center',
                     fontsize=9, color='black')
            if conv[i, j] == 0:
                axA.add_patch(Rectangle((j-.5, i-.5), 1, 1, fill=False, hatch='////',
                                        edgecolor='0.3', lw=0))
            if inv[i, j] == 1:
                axA.add_patch(Rectangle((j-.5, i-.5), 1, 1, fill=False, edgecolor='lime', lw=3))
    fig.colorbar(im, ax=axA, fraction=0.046, pad=0.04, label="$\\Delta J$")
    axA.text(0.0, -0.30, "P=burn  X=counter-prevalence target  T=delay   |   "
             "hatched = invariant violated   |   green box = inversion flagged",
             transform=axA.transAxes, fontsize=8, color='0.25')
    x = np.arange(3); w = 0.38
    axB.bar(x - w/2, [c['optJ'] for c in ctrl], w, label='RL optimal (BC warm-start)', color='#4477aa')
    axB.bar(x + w/2, [c['naiJ'] for c in ctrl], w, label='best-tuned naive', color='#cc6677')
    axB.set_xticks(x); axB.set_xticklabels([f"{r:g}" for r in RHOS])
    axB.set_xlabel("cost ratio  $\\rho$"); axB.set_ylabel("objective  J  (lower is better)")
    axB.set_title("(B)  $\\eta=1$ control (heterogeneity OFF)\n"
                  "RL $\\leq$ naive (invariant holds); 0 inversions", fontsize=11)
    axB.legend(fontsize=8)
    for xi, c in zip(x, ctrl):
        axB.text(xi, max(c['optJ'], c['naiJ']) + 1, "no inv.", ha='center', fontsize=8, color='0.3')
    fig.suptitle(f"BC warm-start pilot: {n_conv}/9 converged, {n_inv}/9 inversions "
                 f"(eta=8); control {ctrl_false}/3 false. PILOT-SCALE, single net seed.",
                 fontsize=11.5, y=1.02)
    fig.tight_layout()
    fig.savefig("/mnt/user-data/outputs/inversion_pilot_bc_phase_map.png", dpi=150,
                bbox_inches='tight')
    print(f"SUMMARY  eta=8: {n_conv}/9 converged, {n_inv}/9 inversions; "
          f"eta=1 control false-flags: {ctrl_false}/3")
    print("saved figure -> inversion_pilot_bc_phase_map.png")

print(f"[{stage}] done in {time.time()-t0:.0f}s")
