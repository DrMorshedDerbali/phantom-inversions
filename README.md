# Inversion-Survival Study — Library-Based Sweep Bundle (RunPod-ready)

A self-contained pipeline that tests whether the celebrated **non-monotone
optimal-control structures** of mean-field epidemic models (singular-arc
*burn-to-herd-immunity*, *delayed* onset, *counter-prevalence* bridge targeting)
**survive** the transition to **stochastic, finite, value-heterogeneous contact
networks** under sequential, cost-aware intervention — by mapping a phase diagram
over (cost-ratio rho x mixing mu x value-heterogeneity eta x action-cost
asymmetry kappa x topology family).

This bundle is the **final, library-based** pipeline used for the methodology
paper + survival case study. **The production verdict uses NO reinforcement
learning.** A library of structurally-inverted policies is evaluated directly
against a best-tuned naive comparator under common random numbers on a held-out
seed split, gated by a heterogeneity-specific, differential eta-gate detector.

---

## What the pipeline does, per cell

1. **Tune the naive** comparator (best (klow,khigh) by mean J) on the TUNE seed
   split only — never on the test seeds.
2. **Build a matched eta=1 twin** of the cell (same rho, mu, kappa, topology,
   seed; value heterogeneity OFF) and tune its own naive.
3. **Screen the inverted-policy library** against the naive on the held-out TEST
   split, under CRN-paired evaluation.
4. **Apply the differential eta-gate detector.** A library policy is credited
   with an inversion at the cell iff it is (a) **structurally** inverted
   (burn P / counter-prevalence X / delay T), (b) **significantly better** than
   the naive on the true undiscounted objective J at the cell's eta, AND
   (c) **does NOT** also beat the eta=1 twin's naive. Condition (c) certifies the
   advantage is *driven by value heterogeneity*, not a cost artifact — and makes
   the eta=1 control provably clean **by construction**.
5. **Checkpoint** a json verdict + an npz of raw per-episode arrays to disk
   immediately. Resumable; thresholds can be changed post-hoc via analyze.py
   without recomputing.

### Why the differential gate exists (a methodology lesson, baked in)
A naive two-gate detector ("structural AND beats naive") still false-positives
when targeted protection is cheap (small kappa) on a cut topology: a
"let-A-burn" policy can post a lower J than a mis-tuned naive **with no
heterogeneity mechanism at all**, and this fires even at eta=1. The differential
eta-gate is the principled fix. This confound, and its fix, is exactly the kind
of thing the methodology paper documents.

---

## Files

| File | Purpose |
|---|---|
| `invsim.py` | Engine: SEIR on N=40, three topology families (SBM, two-clique-plus-bridge, articulation-point), kappa-parameterized action costs, block-targeted actions, value heterogeneity eta. Inverted-policy **library**, tuned-naive comparator, CRN evaluation, two-gate detector, and the **differential eta-gate** production screen (`screen_library_gated`). Tabular Q-learning + BC warm-start retained ONLY for the methodology case study (the discount / representational-ceiling / RL-degradation narrative). |
| `sweep.py` | **Main RunPod driver.** Parallel, library-based phase sweep with rho x mu x eta x kappa x topology x seed axes; held-out TUNE/TEST seed split; per-cell eta=1 twin; per-cell JSON verdict + raw `.npz`; resumable. |
| `analyze.py` | Aggregates results -> `summary.csv` (per cell) **and `configs.csv`** (per plausible-box configuration: seed-pooled bootstrap + Wilcoxon, BH at q=0.05, and the four-part survival verdict). Renders one (rho x mu) phase-diagram PNG per (eta, kappa, topology) slice, audits the eta=1 control, and emits the PRIMARY OUTCOME line. Re-detects from raw arrays with new thresholds, no recompute. |
| `false_positives.py` | Read-only add-on over the JSON verdicts + `configs.csv`: emits `false_positives.csv`, the per-(topo,kappa,eta) phantom counter — how many naive-flagged wins the framework rejects and via which defense. Does not modify `analyze.py` or recompute the sweep. |
| `requirements.txt` | numpy, networkx, matplotlib, scipy. |

---

## Quickstart

```bash
pip install -r requirements.txt

# smoke test (48 cells, ~6 min on 4 cores):
python sweep.py --out results_quick/ --cores 4 --quick
python analyze.py results_quick/

# FULL 15,120-cell sweep (the headline grid):
python sweep.py --out results/ --cores 32 --full-eta
python analyze.py results/
python false_positives.py results/        # phantom counter (the §IV prevalence panel)
```

`--full-eta` uses eta in {1,2,5,8,16} (5 values) -> exactly **15,120 cells**.
Without it the default grid uses eta in {1,2,5,8} (4 values) -> 12,096 cells.

`sweep.py` is **resumable**: re-running skips cells already on disk. A crash at
hour N loses nothing; detection thresholds can be changed later via `analyze.py`.

### Wall-clock (measured ~30 core-seconds/cell, incl. the eta=1 twin)
| Cores | 15,120-cell full grid |
|---|---|
| 16 | ~7.8 h |
| 32 | ~3.9 h |
| 64 | ~1.9 h |

---

## Axes

```
rho   = alpha_r / beta_r            cost ratio          {0.5,1,2,5,10,20,50}
mu    = p_out / p_in                network mixing      {0.05,0.1,0.2,0.3,0.4,0.5}
eta   = high-value multiplier       heterogeneity       {1,2,5,8,16}  (eta=1 = control)
kappa = cost(shieldB)/cost(suppressA)  cost asymmetry   {0.25,0.5,1.0}
topo  = topology family             {sbm, bridge, artic}
seed  = network seed                {0..7}
```

`kappa < 1` makes targeted protection cheap — the regime in which burn-to-protect
/ counter-prevalence inversions can *arithmetically* win. `bridge` and `artic`
give the graph a genuine cut, which is where counter-prevalence "bridge target"
structures are most natural.

---

## Rigor checklist

- **True objective J** (minimized) = `alpha_r * (value-weighted ever-infected) + total cost`,
  undiscounted. No discounted-proxy mismatch.
- **Held-out seed split**: naive is tuned on TUNE seeds; advantage is reported on
  disjoint TEST seeds, CRN-paired for variance reduction.
- **Naive is a hard floor**: the convergence invariant `J(best-lib) <= 1.05 J(naive)`
  gates whether a cell's verdict is trustworthy. `analyze.py` only counts
  inversions in converged cells.
- **Inversion = structural AND significant AND heterogeneity-driven.** The
  differential eta-gate (does NOT beat the eta=1 twin) is mandatory.
- **eta=1 must stay clean.** With the differential gate this holds by
  construction; `analyze.py` still audits it and screams if any eta=1 cell flags.
- **Pre-register** the significance margin and the plausible parameter box before
  the full run; with thousands of cells, control FDR or pre-declare the region of
  interest so you are not implicitly testing thousands of hypotheses.

---

## Outcome interpretation

- **Inversions appear in some (kappa-cheap, bridge/artic) corner** -> "which
  mean-field structures survive, and exactly when" — a positive, mapped result.
- **No inversion survives anywhere once the invariant holds** -> a clean,
  strongly-supported contrarian null: the celebrated structures do not transfer
  to small value-heterogeneous stochastic cohorts; a simple tuned
  prevalence-follower suffices. (Network extension of Russell–Cunniffe 2025.)

Either way, the **methodology** — library-based detection, safe-gated naive
floor, held-out splits, and the differential heterogeneity-specific gate — is the
reusable contribution; the survival phase diagram is the case study proving it
earns its keep.
