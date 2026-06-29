#!/usr/bin/env python3
"""
analyze.py <results_dir>  - aggregate the library sweep and apply the
PRE-REGISTERED verdict procedure (see PREREGISTRATION.txt).

Outputs:
  summary.csv                    one row per cell (raw effects + per-cell gate)
  configs.csv                    one row per (rho,mu,eta,kappa,topo) configuration
                                 in the plausible box, with seed-pooled bootstrap
                                 + Wilcoxon p-values and the BH decision
  phase_<topo>_kap<k>_eta<e>.png (rho x mu) panel per slice; BH-significant +
                                 gated + converged configurations boxed in green
  stdout                         plausible-box BH tally, eta=1 control audit,
                                 and the PRIMARY OUTCOME (null vs survival)

The seed-pooled test, the 10% relative margin, the differential eta-gate, the
>=6/8 convergence rule, and BH at q=0.05 are all fixed by the pre-registration.
Raw per-episode arrays in the .npz files allow LABELED sensitivity reanalysis
(e.g. a different margin) without recomputing the sweep.
"""
import os, sys, json, glob, csv, numpy as np

# ---- pre-registered constants (must match PREREGISTRATION.txt) ----
MARGIN = 0.10                 # 10% relative improvement in J
Q_FDR = 0.05                  # Benjamini-Hochberg level
CONV_MIN_SEEDS = 6            # convergence invariant must hold in >=6 of 8 seeds
N_BOOT = 10000
BOX = dict(rho={1, 2, 5, 10, 20}, mu={0.1, 0.2, 0.3, 0.4},
           eta={2, 5, 8}, kappa={0.25, 0.5, 1.0}, topo={"sbm", "bridge"})

def in_box(r):
    return (r["rho"] in BOX["rho"] and r["mu"] in BOX["mu"] and r["eta"] in BOX["eta"]
            and r["kappa"] in BOX["kappa"] and r["topo"] in BOX["topo"])

def boot_wilcoxon(d, base_arr, margin=MARGIN, n_boot=N_BOOT, seed=777):
    """Seed-pooled paired test. d = pooled (J_lib - J_naive) per seed/episode;
    base_arr = paired J_naive. Returns (p_boot, p_wil, delta_hat, sig)."""
    d = np.asarray(d, float); base = np.asarray(base_arr, float)
    n = len(d)
    if n < 2 or base.mean() <= 0:
        return 1.0, 1.0, 0.0, False
    delta_hat = d.mean() / base.mean()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    bd = d[idx].mean(axis=1); bb = base[idx].mean(axis=1)
    bb = np.where(bb <= 0, np.nan, bb)
    bdelta = bd / bb
    # one-sided bootstrap p for H1: delta < -margin
    p_boot = float(np.nanmean(bdelta >= -margin))
    try:
        from scipy.stats import wilcoxon
        nz = d[d != 0.0]
        if nz.size >= 1:
            _, p_wil = wilcoxon(nz, alternative="less")
        else:
            p_wil = 1.0
    except Exception:
        p_wil = 0.0 if (np.median(d) < 0 and np.mean(d < 0) > 0.5) else 1.0
    med_ok = np.median(d) < -margin * np.median(base) if np.median(base) > 0 else False
    sig = bool(p_boot < Q_FDR and p_wil < 0.05 and med_ok and delta_hat < -margin)
    return p_boot, float(p_wil), float(delta_hat), sig

def bh(pvals, q=Q_FDR):
    """Benjamini-Hochberg rejections at level q, via scipy (validated against a
    hand-rolled step-up; identical). Returns boolean array of rejections."""
    p = np.asarray(pvals, float)
    if p.size == 0:
        return np.zeros(0, bool)
    try:
        from scipy.stats import false_discovery_control
        return false_discovery_control(p, method="bh") <= q
    except Exception:
        # fallback step-up
        m = len(p); order = np.argsort(p)
        passed = p[order] <= q * (np.arange(1, m + 1) / m)
        k = np.where(passed)[0]; out = np.zeros(m, bool)
        if k.size:
            out[order[:k.max() + 1]] = True
        return out

def main():
    rd = sys.argv[1]
    recs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(rd, "*.json")))]
    if not recs:
        print("no results found"); return
    by_cid = {r["cid"]: r for r in recs}

    # ---- per-cell CSV ----
    keys = ["cid", "rho", "mu", "eta", "kappa", "topo", "seed",
            "optJ", "naiJ", "dJ", "ci_hi", "converged", "bestlib", "bestlib_tag"]
    with open(os.path.join(rd, "summary.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(keys + ["P", "X", "T", "het_driven", "inversion"])
        for r in recs:
            d = r["det"]
            w.writerow([r.get(k, "") for k in keys] +
                       [int(d["P"]), int(d["X"]), int(d["T"]),
                        int(d.get("het_driven", False)), int(d["inversion"])])
    print(f"{len(recs)} cells -> summary.csv")

    # ---- configuration-level pooling within the plausible box ----
    # group by (rho,mu,eta,kappa,topo); pool the per-seed best-library episodes
    # vs naive from the stored npz arrays for a seed-pooled paired test.
    cfg = {}
    for r in recs:
        if not in_box(r):
            continue
        key = (r["rho"], r["mu"], r["eta"], r["kappa"], r["topo"])
        cfg.setdefault(key, []).append(r)

    rows = []
    for key, cells in sorted(cfg.items()):
        d_all, base_all = [], []
        conv_seeds = 0; het_seeds = 0; tags = {}
        for c in cells:
            npz = os.path.join(rd, c["cid"] + ".npz")
            if os.path.exists(npz):
                z = np.load(npz)
                d_all.append(z["opt_J"] - z["nai_J"]); base_all.append(z["nai_J"])
            if c["converged"]:
                conv_seeds += 1
            if c["det"].get("het_driven", False):
                het_seeds += 1
            tags[c["bestlib"]] = tags.get(c["bestlib"], 0) + 1
        if not d_all:
            continue
        d_all = np.concatenate(d_all); base_all = np.concatenate(base_all)
        p_boot, p_wil, delta, sig = boot_wilcoxon(d_all, base_all)
        # pre-registered Wilcoxon confirmatory carries a MEDIAN-effect requirement
        # (PREREGISTRATION.txt Section 3): the median paired improvement must also
        # exceed `margin` of the median naive J. Enforced in the final verdict below.
        base_med = float(np.median(base_all))
        med_ok = bool(base_med > 0 and np.median(d_all) < -MARGIN * base_med)
        modal_lib = max(tags, key=tags.get)
        rows.append(dict(key=key, n_seeds=len(cells), p_boot=p_boot, p_wil=p_wil,
                         delta=delta, sig_raw=sig, med_ok=med_ok, conv_seeds=conv_seeds,
                         het_seeds=het_seeds, modal_lib=modal_lib))

    # ---- Benjamini-Hochberg over the box configurations ----
    if rows:
        bh_pass = bh([row["p_boot"] for row in rows], q=Q_FDR)
        for row, bp in zip(rows, bh_pass):
            row["bh"] = bool(bp)
            # final pre-registered verdict (PREREGISTRATION.txt Sections 3 & 5):
            #   BH-significant bootstrap AND Wilcoxon at 0.05 AND the median-margin
            #   clause AND heterogeneity-driven in >= half the seeds AND convergence
            #   in >= 6/8 seeds.
            row["inversion"] = bool(bp and row["p_wil"] < 0.05 and row["med_ok"]
                                    and row["het_seeds"] >= max(1, row["n_seeds"] // 2)
                                    and row["conv_seeds"] >= CONV_MIN_SEEDS)
    with open(os.path.join(rd, "configs.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rho", "mu", "eta", "kappa", "topo", "n_seeds", "delta",
                    "p_boot", "p_wil", "bh", "med_ok", "het_seeds", "conv_seeds",
                    "modal_lib", "inversion"])
        for row in rows:
            k = row["key"]
            w.writerow([*k, row["n_seeds"], f"{row['delta']:.4f}",
                        f"{row['p_boot']:.4g}", f"{row['p_wil']:.4g}",
                        int(row["bh"]), int(row["med_ok"]), row["het_seeds"],
                        row["conv_seeds"], row["modal_lib"], int(row["inversion"])])
    print(f"{len(rows)} plausible-box configurations -> configs.csv")

    # ---- phase panels (per eta,kappa,topo slice over the FULL grid) ----
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        inv_keys = {row["key"] for row in rows if row.get("inversion")}
        etas = sorted({r["eta"] for r in recs}); kappas = sorted({r["kappa"] for r in recs})
        topos = sorted({r["topo"] for r in recs}); rhos = sorted({r["rho"] for r in recs})
        mus = sorted({r["mu"] for r in recs})
        for topo in topos:
            for kappa in kappas:
                for eta in etas:
                    M = np.full((len(rhos), len(mus)), np.nan)
                    for i, rho in enumerate(rhos):
                        for j, mu in enumerate(mus):
                            sub = [r for r in recs if r["eta"] == eta and r["kappa"] == kappa
                                   and r["topo"] == topo and r["rho"] == rho and r["mu"] == mu]
                            if sub:
                                M[i, j] = np.mean([r["dJ"] for r in sub])
                    if not np.isfinite(M).any():
                        continue
                    fig, ax = plt.subplots(figsize=(1.4*len(mus)+2, 1.0*len(rhos)+2))
                    vmax = np.nanmax(np.abs(M)) or 1.0
                    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
                    ax.set_xticks(range(len(mus))); ax.set_xticklabels([f"{m:g}" for m in mus])
                    ax.set_yticks(range(len(rhos))); ax.set_yticklabels([f"{r:g}" for r in rhos])
                    ax.set_xlabel("mixing  mu = p_out/p_in"); ax.set_ylabel("cost ratio  rho")
                    ax.set_title(f"topo={topo}, kappa={kappa:g}, eta={eta}: dJ=J(lib)-J(naive); "
                                 f"green box = pre-registered inversion")
                    for i in range(len(rhos)):
                        for j in range(len(mus)):
                            if np.isfinite(M[i, j]):
                                ax.text(j, i, f"{M[i,j]:+.0f}", ha="center", va="center", fontsize=8)
                            if (rhos[i], mus[j], eta, kappa, topo) in inv_keys:
                                ax.add_patch(Rectangle((j-.5, i-.5), 1, 1, fill=False,
                                                       edgecolor="lime", lw=3))
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="J(lib)-J(naive)")
                    fig.tight_layout()
                    out = os.path.join(rd, f"phase_{topo}_kap{kappa:g}_eta{eta:g}.png")
                    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
        print("  phase panels saved")
    except Exception as e:
        print("  (phase panels skipped:", e, ")")

    # ---- eta=1 control audit + PRIMARY OUTCOME ----
    eta1_false = [r["cid"] for r in recs
                  if r["eta"] == 1 and r["det"]["inversion"] and r["converged"]]
    if eta1_false:
        print(f"!! eta=1 CONTROL VIOLATED in {len(eta1_false)} cells: {eta1_false[:5]} ...")
    else:
        print("eta=1 control CLEAN: 0 false inversions (gate vanishes when "
              "heterogeneity is off, as required).")
    n_box_inv = sum(1 for row in rows if row.get("inversion"))
    print("=" * 70)
    if n_box_inv == 0:
        print("PRIMARY OUTCOME = NULL: no mean-field inversion survives in the "
              "plausible box under the pre-registered criterion.")
    else:
        print(f"PRIMARY OUTCOME = SURVIVAL: {n_box_inv} configuration(s) in the "
              f"plausible box satisfy the pre-registered survival criterion:")
        for row in rows:
            if row.get("inversion"):
                k = row["key"]
                print(f"   rho={k[0]} mu={k[1]} eta={k[2]} kappa={k[3]} {k[4]}  "
                      f"delta={row['delta']:+.1%}  via {row['modal_lib']}")
    print("=" * 70)

if __name__ == "__main__":
    main()
