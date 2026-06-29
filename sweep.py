#!/usr/bin/env python3
"""
sweep.py - full library-based phase-diagram sweep (inversion-survival study).

PRODUCTION PATH = NO RL. For each cell we:
  1. tune the naive comparator (best (klow,khigh) by mean J under CRN),
  2. screen the inverted-policy LIBRARY against that naive under common random
     numbers on a held-out evaluation seed split,
  3. apply the heterogeneity-specific two-gate detector (structural AND
     significant J advantage) to the BEST library policy,
  4. CHECKPOINT a json verdict + an npz of raw per-episode arrays to disk.

Parallel across CPU cores. Resumable (skips cells already on disk). Raw arrays
are stored so detection thresholds / significance margins can be changed POST-HOC
by re-running analyze.py WITHOUT recomputing the sweep.

Axes (edit GRID below or override via CLI):
  rho  = alpha_r / beta_r        cost ratio           (x of the main phase plane)
  mu   = p_out / p_in            network mixing        (y of the main phase plane)
  eta  = high-value multiplier   value heterogeneity   (mechanism slice; eta=1 = control)
  kappa= cost(shieldB)/cost(suppressA)  action-cost asymmetry  (cheap-protection axis)
  topo = {sbm, bridge, artic}    topology family       (does a real cut matter?)
  seed = network seed            replicate

Usage:
  python sweep.py --out results/ --cores 32
  python sweep.py --out results/ --cores 8 --quick          # tiny smoke grid
After it finishes:  python analyze.py results/
"""
import os, sys, json, time, argparse, itertools, numpy as np
import invsim as S

# --------------------------------------------------------------- default grid
GRID = dict(
    rho=[0.5, 1, 2, 5, 10, 20, 50],
    mu=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5],          # p_out = mu * p_in
    eta=[1, 2, 5, 8],                            # eta=1 = symmetric control
    kappa=[0.25, 0.5, 1.0],                      # cheap -> equal targeted protection
    topo=["sbm", "bridge", "artic"],
    seed=[0, 1, 2, 3, 4, 5, 6, 7],
)
# NB: the headline 15,120-cell grid uses eta=[1,2,5,8,16] (5 values).
# Set eta below to the 5-value list for the full run; default keeps 4 for safety.
GRID_FULL_ETA = [1, 2, 5, 8, 16]

QUICK = dict(rho=[2, 5], mu=[0.05, 0.4], eta=[1, 8], kappa=[0.25, 1.0],
             topo=["sbm", "bridge", "artic"], seed=[0])

P_IN = 0.4
EVAL_SEEDS = list(range(50000, 50120))           # 120 CRN-paired eval episodes
# held-out split: tune naive on TUNE seeds, report advantage on TEST seeds
TUNE_SEEDS = EVAL_SEEDS[:60]
TEST_SEEDS = EVAL_SEEDS[60:]

def cell_id(rho, mu, eta, kappa, topo, seed):
    return f"rho{rho:g}_mu{mu:g}_eta{eta:g}_kap{kappa:g}_{topo}_seed{seed}"

def run_one(args):
    (rho, mu, eta, kappa, topo, seed, outdir) = args
    cid = cell_id(rho, mu, eta, kappa, topo, seed)
    jpath = os.path.join(outdir, cid + ".json")
    if os.path.exists(jpath):
        return (cid, "skip")
    pout = mu * P_IN
    sim = S.Sim(p_in=P_IN, p_out=pout, beta=0.04, eta=eta, alpha_r=rho,
                net_seed=seed, kappa=kappa, topology=topo)

    # 1. tune naive on the TUNE split only (no peeking at the test seeds)
    _, kl, kh, _ = S.tune_naive(sim, TUNE_SEEDS)
    naive_fn = S.naive_factory(kl, kh)
    nai_test = sim.evaluate(naive_fn, TEST_SEEDS)            # naive floor on held-out

    # 1b. matched eta=1 TWIN (same rho, mu, kappa, topology, seed; heterogeneity OFF)
    #     -> used by the differential eta-gate so a credited inversion MUST vanish
    #     at eta=1. For an eta=1 cell the twin is itself, which forces het_driven
    #     to False and keeps the control provably clean.
    eta1_sim = S.Sim(p_in=P_IN, p_out=pout, beta=0.04, eta=1.0, alpha_r=rho,
                     net_seed=seed, kappa=kappa, topology=topo)
    _, kl1, kh1, _ = S.tune_naive(eta1_sim, TUNE_SEEDS)
    eta1_nai_test = eta1_sim.evaluate(S.naive_factory(kl1, kh1), TEST_SEEDS)

    # 2-3. screen the inverted-policy library on the held-out TEST split, gated
    best, recs = S.screen_library_gated(sim, nai_test, TEST_SEEDS,
                                        eta1_naive_m=eta1_nai_test, eta1_sim=eta1_sim)

    det = best['det']
    # safe-gate / convergence invariant: a verdict is trustworthy only if the
    # best library policy does not blow past the naive floor (it should be <=).
    converged = bool(best['J'] <= nai_test['J'].mean() * 1.05 + 1e-6)

    rec = dict(cid=cid, rho=rho, mu=mu, eta=eta, kappa=kappa, topo=topo, seed=seed,
               naive_k=[kl, kh], naiJ=float(nai_test['J'].mean()),
               bestlib=best['name'], bestlib_tag=best['tag'],
               optJ=float(best['J']), dJ=float(best['dJ']), ci_hi=float(best['ci_hi']),
               converged=converged, det=det,
               infA=[float(best['infA']), float(nai_test['inf_A'].mean())],
               infB=[float(best['infB']), float(nai_test['inf_B'].mean())],
               cost=[float(best['cost']), float(nai_test['cost'].mean())],
               # full per-policy table (J, dJ, det) for post-hoc inspection
               library={r['name']: dict(J=r['J'], dJ=r['dJ'], ci_hi=r['ci_hi'],
                                        tag=r['tag'], det=r['det']) for r in recs})
    json.dump(rec, open(jpath, "w"), indent=2)

    # 4. raw arrays (best library policy + naive) for post-hoc re-detection
    bm = best['_m']
    np.savez(os.path.join(outdir, cid + ".npz"),
             opt_J=bm['J'], opt_infA=bm['inf_A'], opt_infB=bm['inf_B'],
             opt_cost=bm['cost'], opt_first_agg=bm['first_agg'],
             opt_shieldB_offfire=bm['shieldB_offfire'], opt_active=bm['active_steps'],
             nai_J=nai_test['J'], nai_infA=nai_test['inf_A'], nai_infB=nai_test['inf_B'],
             nai_cost=nai_test['cost'], nai_first_agg=nai_test['first_agg'])
    return (cid, f"INV={int(det['inversion'])} conv={int(converged)} "
                 f"best={best['name']} dJ={best['dJ']:+.1f}")

def build_grid(g, full_eta):
    eta = GRID_FULL_ETA if (full_eta and 'eta' in g) else g['eta']
    return list(itertools.product(g['rho'], g['mu'], eta, g['kappa'],
                                  g['topo'], g['seed']))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--cores", type=int, default=max(1, os.cpu_count() - 1))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--full-eta", action="store_true",
                    help="use the 5-value eta grid [1,2,5,8,16] -> 15,120 cells")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    g = QUICK if a.quick else GRID
    combos = build_grid(g, a.full_eta and not a.quick)
    tasks = [(r, m, e, k, tp, s, a.out) for (r, m, e, k, tp, s) in combos]
    print(f"{len(tasks)} cells | {a.cores} cores | library-based (no RL) | out={a.out}")
    t0 = time.time(); done = 0; n_inv = 0
    from multiprocessing import Pool
    with Pool(a.cores) as pool:
        for (cid, msg) in pool.imap_unordered(run_one, tasks):
            done += 1
            if "INV=1" in msg:
                n_inv += 1
            if done % 50 == 0 or "INV=1" in msg:
                print(f"[{done}/{len(tasks)}] {cid}: {msg}  "
                      f"({time.time()-t0:.0f}s, inv so far {n_inv})")
    print(f"DONE {done} cells in {time.time()-t0:.0f}s; inversions flagged: {n_inv}. "
          f"Run: python analyze.py {a.out}")

if __name__ == "__main__":
    main()
