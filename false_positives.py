#!/usr/bin/env python3
"""
false_positives.py <results_dir>  -  the phantom counter.

Quantifies how many apparent inversions a NAIVE evaluation pipeline would report,
and how many the falsification framework rejects, per (topo, kappa, eta) slice.
This is the "per-region counter" promised in the paper's contribution claims.

It is an ADD-ON: it reads only the stored per-cell JSON verdicts (and configs.csv
if present) written by sweep.py / analyze.py. It does NOT recompute the sweep and
does NOT touch the pre-registered analyze.py. Run it any time after the sweep:

    python false_positives.py results/

Definitions (all grounded in invsim.detect_inversion / screen_library_gated):
  naive_flag   : a cell a two-gate pipeline would publish = structural AND a
                 significant J advantage (det['structural'] and det['advantage']).
                 This is exactly the detector WITHOUT the differential eta-gate
                 and WITHOUT held-out BH/Wilcoxon control.
  het_gate_kill: naive_flag AND NOT det['het_driven'] -> the advantage did not
                 vanish at the eta=1 twin, so Defense 5 (differential eta-gate)
                 rejects it (the "loud phantom" / cost-arbitrage type).
  cell_inv     : det['inversion'] -> survives structure + significance + eta-gate
                 at the cell level.
  phantom      : naive_flag AND NOT cell_inv -> an apparent win the framework
                 rejects at the cell level.
Config-level certification (configs.csv 'inversion') additionally requires
seed-pooled BH + Wilcoxon + >=6/8 convergence; cells that are cell_inv=1 but whose
configuration is NOT certified are the "subtle phantom" (significance-floor) type,
caught by Defenses 3-4.
"""
import os, sys, json, glob, csv
from collections import defaultdict

# plausible box (mirror of analyze.BOX; used only to label rows, not to filter)
BOX = dict(rho={1, 2, 5, 10, 20}, mu={0.1, 0.2, 0.3, 0.4},
           eta={2, 5, 8}, kappa={0.25, 0.5, 1.0}, topo={"sbm", "bridge"})


def in_box(r):
    return (r["rho"] in BOX["rho"] and r["mu"] in BOX["mu"] and r["eta"] in BOX["eta"]
            and r["kappa"] in BOX["kappa"] and r["topo"] in BOX["topo"])


def load_certified_configs(rd):
    """Return set of (rho,mu,eta,kappa,topo) configs marked inversion=1 in
    configs.csv, or None if configs.csv is absent."""
    path = os.path.join(rd, "configs.csv")
    if not os.path.exists(path):
        return None
    certified = set()
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("inversion", "0") in ("1", "True", "true"):
                certified.add((_num(row["rho"]), _num(row["mu"]), _num(row["eta"]),
                               _num(row["kappa"]), row["topo"]))
    return certified


def _num(s):
    f = float(s)
    return int(f) if f.is_integer() else f


def main():
    if len(sys.argv) < 2:
        print("usage: python false_positives.py <results_dir>"); return
    rd = sys.argv[1]
    recs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(rd, "*.json")))]
    if not recs:
        print("no results found in", rd); return

    certified_cfgs = load_certified_configs(rd)

    # per (topo, kappa, eta) slice tallies
    slc = defaultdict(lambda: dict(n=0, naive=0, het_kill=0, cell_inv=0,
                                   box_cells=0, box_naive=0))
    g_naive = g_het_kill = g_cell_inv = g_total = 0
    ctrl_naive = 0          # naive flags on the eta=1 control (must net to 0 real)

    for r in recs:
        d = r["det"]
        structural = bool(d.get("structural", d.get("P") or d.get("X") or d.get("T")))
        advantage = bool(d.get("advantage", False))
        het_driven = bool(d.get("het_driven", False))
        cell_inv = bool(d.get("inversion", False))
        naive_flag = structural and advantage

        key = (r["topo"], r["kappa"], r["eta"])
        s = slc[key]
        s["n"] += 1
        g_total += 1
        if naive_flag:
            s["naive"] += 1; g_naive += 1
            if not het_driven:
                s["het_kill"] += 1; g_het_kill += 1
            if r["eta"] == 1:
                ctrl_naive += 1
        if cell_inv:
            s["cell_inv"] += 1; g_cell_inv += 1
        if in_box(r):
            s["box_cells"] += 1
            if naive_flag:
                s["box_naive"] += 1

    # ---- write false_positives.csv ----
    out = os.path.join(rd, "false_positives.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["topo", "kappa", "eta", "n_cells", "naive_flag",
                    "het_gate_kill", "cell_inversion", "phantoms_rejected",
                    "config_certified", "in_box"])
        for key in sorted(slc):
            topo, kappa, eta = key
            s = slc[key]
            if certified_cfgs is None:
                cfg_cert = ""
            else:
                cfg_cert = sum(1 for c in certified_cfgs
                               if c[4] == topo and c[3] == kappa and c[2] == eta)
            box_flag = int((topo in BOX["topo"]) and (kappa in BOX["kappa"])
                           and (eta in BOX["eta"]))
            w.writerow([topo, f"{kappa:g}", eta, s["n"], s["naive"], s["het_kill"],
                        s["cell_inv"], s["naive"] - s["cell_inv"], cfg_cert, box_flag])
    print(f"{len(slc)} (topo,kappa,eta) slices -> {out}")

    # ---- global tally (the headline number for the paper) ----
    bar = "=" * 70
    print(bar)
    print("PHANTOM TALLY (whole grid)")
    print(f"  cells evaluated ................. {g_total}")
    print(f"  naive-pipeline 'discoveries' .... {g_naive}   "
          f"(structural AND significant)")
    print(f"    rejected by differential eta-gate (D5): {g_het_kill}")
    print(f"  survive cell-level gates ........ {g_cell_inv}")
    print(f"  PHANTOMS REJECTED (cell-level) .. {g_naive - g_cell_inv}")
    if ctrl_naive:
        print(f"  of which on the eta=1 control ... {ctrl_naive}  "
              f"(apparent wins the gate voids by construction)")
    if certified_cfgs is not None:
        print(f"  CONFIG-LEVEL certified inversions (configs.csv): "
              f"{len(certified_cfgs)}")
        print(f"  -> phantoms additionally rejected by the held-out significance "
              f"floor (D3/D4) = cell_inv configs not certified")
    else:
        print("  (configs.csv not found: cell-level tally only; run analyze.py "
              "first for the config-level / significance-floor breakdown)")
    print(bar)


if __name__ == "__main__":
    main()
