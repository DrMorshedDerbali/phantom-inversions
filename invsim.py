"""
invsim.py - Inversion-survival engine (phase-diagram methodology paper).

This is the FINAL library-based engine. RL training is retained for the
methodology case study (it is part of the debugging narrative: discount
misalignment, representational ceiling, maximization-bias degradation), but the
production verdict is produced WITHOUT RL: a library of structurally-inverted
policies is evaluated directly against a best-tuned naive comparator under
common random numbers, gated by a heterogeneity-specific detector.

Enrichments that make an inversion POSSIBLE:
  (1) Structured value heterogeneity eta: infections in the high-value block (B)
      cost eta x more than in the low-value block (A) / crew.
  (2) Block-targeted actions: shield B or suppress A specifically.
  (3) Action-cost asymmetry kappa: cost(shield B) = kappa * cost(suppress A).
      kappa < 1 makes targeted protection CHEAP -> the regime in which the
      burn-to-protect / counter-prevalence inversions can arithmetically win.
  (4) Topology family: SBM, two-clique-plus-bridge, articulation-point. The
      counter-prevalence "bridge target" inversion is most natural when the
      graph has a genuine cut structure, not just block-mixing.

Without (1), the cost-optimal policy is provably monotone in prevalence and no
inversion can occur (this is exactly what the eta=1 control must demonstrate).

Dynamics: SEIR on an N=40, 3-block graph (crew=10, PassengersA=15, PassengersB=15).
Both E (presymptomatic, half infectivity) and I transmit, so "isolate
symptomatic" is IMPERFECT and population measures retain a role.

States: per-block infection signal + coarse B-exposure flag + time -> 144 states.
Objective J (to MINIMIZE) = alpha_r * weighted_ever_infected + total_cost.
"""

import numpy as np
import networkx as nx

# ---------------------------------------------------------------- topology
SIZES = [10, 15, 15]                     # crew, PassengersA, PassengersB
N = sum(SIZES)
BLOCK = np.array([0]*SIZES[0] + [1]*SIZES[1] + [2]*SIZES[2])
IS_CREW = BLOCK == 0
IS_A = BLOCK == 1                         # LOW value (can "burn" cheaply)
IS_B = BLOCK == 2                         # HIGH value (worth protecting)
P0 = SIZES[0]                             # patient zero = first node of block A

# ---------------------------------------------------------------- actions / cost
NACT = 5
# a0 none, a1 isolate symptomatic, a2 shield B, a3 suppress A, a4 full quarantine
# Base cost vector. cost(shield B)=COST[2] is overwritten per-cell via kappa:
#   COST[2] = kappa * COST[3].  kappa=1.0 reproduces the original pilot.
BASE_COST = np.array([0.0, 1.0, 2.0, 2.0, 5.0])

def cost_vector(kappa=1.0):
    c = BASE_COST.copy()
    c[2] = kappa * c[3]                   # shield-B cost = kappa * suppress-A cost
    return c

# ---------------------------------------------------------------- topology families
def _sbm_adj(p_in, p_out, seed):
    P = [[p_in if i == j else p_out for j in range(3)] for i in range(3)]
    G = nx.stochastic_block_model(SIZES, P, seed=int(seed))
    A = nx.to_numpy_array(G)
    np.fill_diagonal(A, 0.0)
    return A

def _two_clique_bridge_adj(p_in, p_out, seed):
    """Two dense passenger cliques (A, B) joined only through the crew block,
    which acts as the bridge. Within-block density ~ p_in; the ONLY A<->B paths
    go through crew (no direct A-B edges). Crew connects to both at p_out-scaled
    density. This gives a genuine cut: shield crew and A,B are separated.
    """
    rng = np.random.default_rng(int(seed) + 101)
    A = np.zeros((N, N))
    idx = {0: np.where(IS_CREW)[0], 1: np.where(IS_A)[0], 2: np.where(IS_B)[0]}
    for b in (0, 1, 2):                              # dense within-block edges
        nodes = idx[b]
        for u in range(len(nodes)):
            for v in range(u + 1, len(nodes)):
                if rng.random() < p_in:
                    A[nodes[u], nodes[v]] = A[nodes[v], nodes[u]] = 1.0
    bridge_p = max(p_out, 1.5 / max(1, len(idx[0])))  # keep bridge connected
    for b in (1, 2):                                  # crew<->A, crew<->B; NO A<->B
        for c in idx[0]:
            for v in idx[b]:
                if rng.random() < bridge_p:
                    A[c, v] = A[v, c] = 1.0
    np.fill_diagonal(A, 0.0)
    return A

def _articulation_adj(p_in, p_out, seed):
    """A single high-betweenness articulation node (last crew node) is the sole
    cut vertex between the A-side {crew\\{art} + A} and the B-side {B}. B reaches
    the rest of the network ONLY through art -> textbook defensible bridge.
    """
    rng = np.random.default_rng(int(seed) + 202)
    A = np.zeros((N, N))
    crew = np.where(IS_CREW)[0]
    art = crew[-1]
    a_side = np.concatenate([crew[:-1], np.where(IS_A)[0]])
    b_side = np.where(IS_B)[0]
    for u in range(len(a_side)):                      # dense A-side
        for v in range(u + 1, len(a_side)):
            if rng.random() < p_in:
                A[a_side[u], a_side[v]] = A[a_side[v], a_side[u]] = 1.0
    for u in range(len(b_side)):                      # dense B-side
        for v in range(u + 1, len(b_side)):
            if rng.random() < p_in:
                A[b_side[u], b_side[v]] = A[b_side[v], b_side[u]] = 1.0
    art_p = max(p_out, 1.5 / max(1, len(b_side)))
    for v in a_side:
        if rng.random() < max(p_out, art_p):
            A[art, v] = A[v, art] = 1.0
    for v in b_side:
        if rng.random() < art_p:
            A[art, v] = A[v, art] = 1.0
    A[art, a_side[0]] = A[a_side[0], art] = 1.0       # guarantee art touches both
    A[art, b_side[0]] = A[b_side[0], art] = 1.0
    np.fill_diagonal(A, 0.0)
    return A

TOPOLOGIES = ("sbm", "bridge", "artic")

def make_adj(p_in, p_out, seed, topology="sbm"):
    if topology == "sbm":
        return _sbm_adj(p_in, p_out, seed)
    if topology == "bridge":
        return _two_clique_bridge_adj(p_in, p_out, seed)
    if topology == "artic":
        return _articulation_adj(p_in, p_out, seed)
    raise ValueError(f"unknown topology {topology!r}; choose from {TOPOLOGIES}")

# ---------------------------------------------------------------- state binning
# crew_I(2) * A_I(4) * B_I(3) * B_E(2) * t(3) = 144
NSTATES = 2 * 4 * 3 * 2 * 3

def state_index(cnt, t):
    Ic, Ia, Ib, Eb = cnt['I_crew'], cnt['I_A'], cnt['I_B'], cnt['E_B']
    ci = 0 if Ic == 0 else 1
    if Ia == 0: ai = 0
    elif Ia <= 2: ai = 1
    elif Ia <= 5: ai = 2
    else: ai = 3
    if Ib == 0: bi = 0
    elif Ib <= 2: bi = 1
    else: bi = 2
    be = 0 if Eb == 0 else 1
    ti = min(t // 20, 2)
    return ((((ci * 4 + ai) * 3 + bi) * 2 + be) * 3 + ti)


class Sim:
    def __init__(self, p_in, p_out, beta, eta, alpha_r,
                 sigma=1/5, gamma=1/8, net_seed=42, beta_r=1.0, gamma_rl=0.95,
                 T=60, kappa=1.0, topology="sbm"):
        self.topology = topology
        self.A = make_adj(p_in, p_out, net_seed, topology=topology)
        self.beta = beta
        self.sigma = sigma
        self.gamma = gamma
        self.eta = eta
        self.alpha_r = alpha_r
        self.beta_r = beta_r
        self.gamma_rl = gamma_rl
        self.T = T
        self.kappa = kappa
        self.COST = cost_vector(kappa)                  # per-cell action cost vector
        self.wnode = np.array([1.0, 1.0, eta])[BLOCK]   # per-node infection weight
        self._B_idx = np.where(IS_B)[0]
        self._A_idx = np.where(IS_A)[0]

    def _modifier(self, action, Iidx):
        M = np.ones((N, N))
        if action == 1:                          # isolate symptomatic (imperfect)
            if Iidx.size:
                M[Iidx, :] = 0.4
                M[:, Iidx] = 0.4
        elif action == 2:                        # shield B
            M[self._B_idx, :] = 0.3
            M[:, self._B_idx] = 0.3
        elif action == 3:                        # suppress A
            M[self._A_idx, :] = 0.3
            M[:, self._A_idx] = 0.3
        elif action == 4:                        # full quarantine
            M[:, :] = 0.25
        return M

    def _p_infection(self, x, action):
        I = x == 2
        E = x == 1
        Iidx = np.where(I)[0]
        M = self._modifier(action, Iidx)
        srcfac = np.where(I, 1.0, np.where(E, 0.5, 0.0))         # infectivity of source j
        arg = self.beta * M * srcfac[None, :]                    # force j->i
        np.clip(arg, 0.0, 0.999, out=arg)
        L = self.A * np.log1p(-arg)
        return 1.0 - np.exp(L.sum(axis=1))

    @staticmethod
    def counts(x):
        I = x == 2; E = x == 1
        return dict(
            I_crew=int(I[IS_CREW].sum()), I_A=int(I[IS_A].sum()), I_B=int(I[IS_B].sum()),
            E_A=int(E[IS_A].sum()), E_B=int(E[IS_B].sum()),
            I_tot=int(I.sum()), E_tot=int(E.sum()))

    def train(self, n_ep, seed_base, eps0=1.0, eps1=0.05, eps_frac=0.7,
              a0=0.3, a1=0.01, Q_init=None):
        """Retained for the methodology case study only (NOT the production path)."""
        COST = self.COST
        Q = np.zeros((NSTATES, NACT)) if Q_init is None else Q_init.copy()
        for ep in range(n_ep):
            rng = np.random.default_rng(seed_base + ep)
            frac = ep / max(1, n_ep - 1)
            lr = a0 + (a1 - a0) * frac
            ef = min(1.0, ep / max(1, eps_frac * n_ep))
            eps = eps0 + (eps1 - eps0) * ef
            x = np.zeros(N, dtype=np.int8); x[P0] = 2
            cnt = self.counts(x); s = state_index(cnt, 0)
            for t in range(self.T):
                if rng.random() < eps:
                    a = rng.integers(NACT)
                else:
                    a = int(np.argmax(Q[s]))
                pinf = self._p_infection(x, a)
                xn = x.copy()
                S = x == 0; E = x == 1; I = x == 2
                se = S & (rng.random(N) < pinf)
                ei = E & (rng.random(N) < self.sigma)
                ir = I & (rng.random(N) < self.gamma)
                xn[se] = 1; xn[ei] = 2; xn[ir] = 3
                wexp = float(self.wnode[se].sum())
                r = -(self.alpha_r * wexp + self.beta_r * COST[a])
                cntn = self.counts(xn); sn = state_index(cntn, min(t + 1, self.T - 1))
                done = (cntn['E_tot'] == 0 and cntn['I_tot'] == 0)
                target = r if done else r + self.gamma_rl * float(np.max(Q[sn]))
                Q[s, a] += lr * (target - Q[s, a])
                x = xn; s = sn
                if done:
                    break
        return Q

    def evaluate(self, action_fn, seeds):
        """action_fn(cnt, t) -> action. Returns dict of per-episode arrays.
        CRN: episode e uses default_rng(seed) so the noise stream is shared
        across policies evaluated on the same seed list (paired comparison)."""
        COST = self.COST
        out = dict(J=[], cost=[], inf_tot=[], inf_A=[], inf_B=[], inf_crew=[],
                   peakI_A=[], peakI_B=[], first_agg=[], shieldB_offfire=[],
                   active_steps=[])
        for sd in seeds:
            rng = np.random.default_rng(sd)
            x = np.zeros(N, dtype=np.int8); x[P0] = 2
            ever = x != 0
            J = 0.0; cost = 0.0
            peakA = 0; peakB = 0; first_agg = self.T; sb_off = 0; active = 0
            for t in range(self.T):
                cnt = self.counts(x)
                a = int(action_fn(cnt, t))
                if cnt['I_tot'] + cnt['E_tot'] > 0:
                    active += 1
                    if a >= 2 and first_agg == self.T:
                        first_agg = t
                    if a == 2 and cnt['I_A'] > cnt['I_B']:   # shield B while fire is in A
                        sb_off += 1
                pinf = self._p_infection(x, a)
                xn = x.copy()
                S = x == 0; E = x == 1; I = x == 2
                se = S & (rng.random(N) < pinf)
                ei = E & (rng.random(N) < self.sigma)
                ir = I & (rng.random(N) < self.gamma)
                xn[se] = 1; xn[ei] = 2; xn[ir] = 3
                ever = ever | (xn != 0)
                wexp = float(self.wnode[se].sum())
                J += self.alpha_r * wexp + self.beta_r * COST[a]
                cost += COST[a]
                peakA = max(peakA, int((xn == 2)[IS_A].sum()))
                peakB = max(peakB, int((xn == 2)[IS_B].sum()))
                x = xn
                if cnt['I_tot'] == 0 and cnt['E_tot'] == 0 and t > 0:
                    break
            out['J'].append(J); out['cost'].append(cost)
            out['inf_tot'].append(int(ever.sum()))
            out['inf_A'].append(int(ever[IS_A].sum()))
            out['inf_B'].append(int(ever[IS_B].sum()))
            out['inf_crew'].append(int(ever[IS_CREW].sum()))
            out['peakI_A'].append(peakA); out['peakI_B'].append(peakB)
            out['first_agg'].append(first_agg); out['shieldB_offfire'].append(sb_off)
            out['active_steps'].append(active)
        return {k: np.array(v, dtype=float) for k, v in out.items()}


# ----------------------------------------------------------------- policies
def greedy_policy(Q):
    return lambda cnt, t: int(np.argmax(Q[state_index(cnt, min(t, 59))]))

def naive_factory(klow, khigh):
    """Tunable 'fight the biggest fire, immediately, harder when bigger' policy.
    Targets the highest-prevalence block; escalates to full quarantine when big."""
    def pol(cnt, t):
        It = cnt['I_tot']
        if It == 0:
            return 0
        if It >= khigh:
            return 4
        if It >= klow:
            hot = max(('crew', cnt['I_crew']), ('A', cnt['I_A']), ('B', cnt['I_B']),
                      key=lambda kv: kv[1])[0]
            return {'B': 2, 'A': 3, 'crew': 1}[hot]
        return 0
    return pol


# ----------------------------------------------------------------- INVERTED-POLICY LIBRARY
# Each library policy embodies one (or more) of the mean-field "inversion"
# structures. The detector confirms WHICH structures are present (P/X/T); the
# CRN-paired advantage test confirms whether the structure actually BEATS the
# tuned naive on the true objective J. A structure "survives" in a cell iff some
# library policy is both structurally inverted AND significantly better than naive.

def pol_inverted_burn(cnt, t):
    """Definitionally inverted: shield B whenever B not yet saturated, ignore A's
    fire entirely -> lets A burn (P) and shields B regardless of prevalence (X).
    Known-answer MUST-FLAG policy for detector validation."""
    if cnt['I_B'] < 8:
        return 2
    return 0

def pol_monotone(cnt, t):
    """Definitionally monotone (reasonable human heuristic). MUST NOT be flagged.
    Known-answer MUST-NOT-FLAG policy for detector validation."""
    It = cnt['I_tot']
    if It == 0:
        return 0
    if It >= 5:
        return 4
    hot = max(('crew', cnt['I_crew']), ('A', cnt['I_A']), ('B', cnt['I_B']),
              key=lambda kv: kv[1])[0]
    return {'B': 2, 'A': 3, 'crew': 1}[hot]

# --- the survival library (structural inversions to test against the naive) ---
def lib_protect_only(cnt, t):
    """Pure burn-to-protect: only ever shield B; never spend on A. Cheapest
    possible counter-prevalence policy -> pays only if kappa is small."""
    if cnt['I_B'] + cnt['E_B'] > 0 or cnt['I_A'] > 0:
        return 2 if cnt['I_B'] < 10 else 0
    return 0

def lib_protect_until_threat(cnt, t):
    """Shield B while the fire is in A (counter-prevalence target X); switch to
    suppress A only if A's fire grows large enough to threaten breakthrough."""
    if cnt['I_A'] >= cnt['I_B'] and cnt['I_A'] > 0:
        if cnt['I_A'] >= 6:
            return 3                      # concede and fight A only when severe
        return 2                          # otherwise shield B (counter-prevalence)
    if cnt['I_B'] > 0:
        return 2
    return 0

def delay_factory(onset):
    """Delayed onset (T): do nothing until t >= onset, then behave like a sensible
    prevalence-follower. Tests whether deferring the first aggressive action pays."""
    def pol(cnt, t):
        if t < onset:
            return 0
        It = cnt['I_tot']
        if It == 0:
            return 0
        if It >= 5:
            return 4
        hot = max(('crew', cnt['I_crew']), ('A', cnt['I_A']), ('B', cnt['I_B']),
                  key=lambda kv: kv[1])[0]
        return {'B': 2, 'A': 3, 'crew': 1}[hot]
    return pol

def lib_burn_then_protect(cnt, t):
    """Burn-to-herd-immunity flavour (P): let A burn entirely (never suppress A),
    spend only on shielding B once B is actually exposed."""
    if cnt['I_B'] + cnt['E_B'] > 0:
        return 2
    return 0

def library_policies():
    """Named library of structurally-inverted candidate policies.
    Each entry: (name, action_fn, primary_structure_tag)."""
    return [
        ("protect_only",         lib_protect_only,          "X"),
        ("protect_until_threat", lib_protect_until_threat,  "X"),
        ("burn_then_protect",    lib_burn_then_protect,     "P"),
        ("delay8",               delay_factory(8),          "T"),
        ("delay15",              delay_factory(15),         "T"),
    ]


# ----------------------------------------------------------------- comparator + detector
def tune_naive(sim, seeds, grid_low=(1, 2, 3), grid_high=(4, 6, 8)):
    best = None
    for kl in grid_low:
        for kh in grid_high:
            if kh <= kl:
                continue
            m = sim.evaluate(naive_factory(kl, kh), seeds)
            mj = m['J'].mean()
            if best is None or mj < best[0]:
                best = (mj, kl, kh, m)
    return best   # (meanJ, klow, khigh, metrics)

def paired_advantage(J_opt, J_naive, margin=0.10, n_boot=10000, seed=12345):
    """Pre-registered CRN-paired advantage test (see PREREGISTRATION.txt).

    Effect measure: relative improvement delta = mean(d)/mean(J_naive), where
    d = J_opt - J_naive over the held-out TEST seeds (paired). A candidate has a
    MEANINGFUL advantage iff it is significantly more than `margin` (=10%) better:

      PRIMARY  : one-sided bootstrap percentile interval on delta (n_boot
                 resamples of the paired differences); require the upper 95%
                 bound on delta < -margin.  Bootstrap (not normal-SE) because J
                 is right-skewed.
      CONFIRM  : one-sided Wilcoxon signed-rank on the raw paired differences
                 (require median improvement also beyond `margin` of median
                 naive J).  Distribution-free robustness check.

    Returns (mean_diff, ci_hi_relative, significant) where `significant` is True
    iff BOTH tests pass. `ci_hi_relative` is the upper 95% bootstrap bound on the
    RELATIVE effect delta (so the threshold is a clean < -margin)."""
    d = np.asarray(J_opt, float) - np.asarray(J_naive, float)
    n = len(d)
    md = float(d.mean())
    base = float(np.mean(J_naive))
    if n < 2 or base <= 0:
        return md, 0.0, False

    # ---- primary: one-sided bootstrap upper bound on relative effect ----
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_d = d[idx].mean(axis=1)
    boot_base = np.asarray(J_naive, float)[idx].mean(axis=1)
    boot_base = np.where(boot_base <= 0, np.nan, boot_base)
    boot_delta = boot_d / boot_base
    ci_hi_rel = float(np.nanpercentile(boot_delta, 95))     # upper 95% bound on delta
    primary = bool(ci_hi_rel < -margin)

    # ---- confirmatory: one-sided Wilcoxon + median effect size ----
    confirm = False
    try:
        from scipy.stats import wilcoxon
        nz = d[d != 0.0]
        if nz.size >= 1:
            stat, p = wilcoxon(nz, alternative="less")        # H1: differences < 0
            med_naive = float(np.median(J_naive))
            med_eff_ok = (np.median(d) < -margin * med_naive) if med_naive > 0 else False
            confirm = bool(p < 0.05 and med_eff_ok)
    except Exception:
        # SciPy absent: fall back to a sign-test style check on the median effect
        med_naive = float(np.median(J_naive))
        confirm = bool((np.median(d) < -margin * med_naive) and (np.mean(d < 0) > 0.5))

    return md, ci_hi_rel, bool(primary and confirm)

def detect_inversion(opt_m, naive_m, J_advantage_sig,
                     burn_margin=0.5, delay_margin=3.0, shield_frac=0.10):
    """Inversion = structural contradiction AND a significant J advantage.
    P (burn): candidate infects MORE of low-value block A than the naive.
    X (counter-prevalence target): candidate shields B while A is the hotter
      block for a non-trivial fraction of active steps.
    T (delay): candidate's first aggressive action is materially later."""
    P = (opt_m['inf_A'].mean() > naive_m['inf_A'].mean() + burn_margin)
    X = ((opt_m['shieldB_offfire'].sum() / max(1, opt_m['active_steps'].sum()))
         > shield_frac)
    T = (opt_m['first_agg'].mean() > naive_m['first_agg'].mean() + delay_margin)
    structural = bool(P or X or T)
    flagged = bool(structural and J_advantage_sig)
    return dict(P=bool(P), X=bool(X), T=bool(T),
                structural=structural, advantage=bool(J_advantage_sig),
                inversion=flagged)

def screen_library(sim, naive_m, seeds, policies=None):
    """Evaluate every library policy against the tuned naive under CRN.
    Returns (best_record, all_records). The CELL verdict is an inversion iff the
    BEST library policy (lowest J) is both structurally inverted AND beats naive.
    Heterogeneity-specific by construction: at eta=1 the value weights are flat,
    so shielding B buys nothing the naive cannot, and no library policy attains a
    significant advantage -> the detector vanishes, as required for the control."""
    if policies is None:
        policies = library_policies()
    recs = []
    best = None
    for (name, fn, tag) in policies:
        m = sim.evaluate(fn, seeds)
        md, hi, sig = paired_advantage(m['J'], naive_m['J'])
        det = detect_inversion(m, naive_m, sig)
        rec = dict(name=name, tag=tag, J=float(m['J'].mean()), dJ=float(md),
                   ci_hi=float(hi), det=det,
                   infA=float(m['inf_A'].mean()), infB=float(m['inf_B'].mean()),
                   cost=float(m['cost'].mean()),
                   first_agg=float(m['first_agg'].mean()),
                   shieldB_offfire=float(m['shieldB_offfire'].sum()),
                   active_steps=float(m['active_steps'].sum()),
                   _m=m)                       # raw arrays kept for npz dump
        recs.append(rec)
        if best is None or rec['J'] < best['J']:
            best = rec
    return best, recs


def screen_library_gated(sim, naive_m, seeds, eta1_naive_m=None, eta1_sim=None,
                         policies=None):
    """PRODUCTION screen with a DIFFERENTIAL eta-gate.

    The plain `screen_library` gate ('structural AND beats naive') is not enough:
    when targeted protection is cheap (small kappa) and the cohort is small, a
    'let-A-burn' policy can post a lower J than a naive that happens to be
    mis-tuned for a cut topology -- WITHOUT any value-heterogeneity mechanism.
    That false positive shows up even at eta=1 (see the methodology case study).

    The principled fix makes 'the effect must vanish at eta=1' a PER-CELL hard
    requirement, not just a global audit. A library policy is credited with an
    inversion at this cell iff:
        (i)  it significantly beats the naive at the cell's true eta, AND
        (ii) it does NOT significantly beat the matched eta=1 twin's naive
             (same rho, mu, kappa, topology, seed; heterogeneity OFF).
    Condition (ii) certifies the advantage is *driven by value heterogeneity*:
    if the same structural policy also wins with flat values, the win is a cost
    artifact, not a surviving mean-field inversion.

    Pass a prebuilt eta=1 twin (eta1_sim + its tuned naive metrics eta1_naive_m)
    to avoid recomputation; otherwise this builds one with eta=1.0.
    """
    if policies is None:
        policies = library_policies()
    if eta1_sim is None or eta1_naive_m is None:
        raise ValueError("screen_library_gated requires the matched eta=1 twin "
                         "(eta1_sim and eta1_naive_m); the driver supplies both.")

    recs = []
    best = None
    for (name, fn, tag) in policies:
        m = sim.evaluate(fn, seeds)
        md, hi, sig = paired_advantage(m['J'], naive_m['J'])
        det = detect_inversion(m, naive_m, sig)

        # differential eta-gate: does this SAME policy also beat the eta=1 twin?
        m1 = eta1_sim.evaluate(fn, seeds)
        _, _, sig1 = paired_advantage(m1['J'], eta1_naive_m['J'])
        het_driven = bool(sig and not sig1)         # wins at eta, not at eta=1

        det = dict(det)
        det['advantage_eta1'] = bool(sig1)
        det['het_driven'] = het_driven
        det['inversion'] = bool(det['structural'] and sig and het_driven)

        rec = dict(name=name, tag=tag, J=float(m['J'].mean()), dJ=float(md),
                   ci_hi=float(hi), det=det,
                   infA=float(m['inf_A'].mean()), infB=float(m['inf_B'].mean()),
                   cost=float(m['cost'].mean()),
                   first_agg=float(m['first_agg'].mean()),
                   shieldB_offfire=float(m['shieldB_offfire'].sum()),
                   active_steps=float(m['active_steps'].sum()),
                   _m=m)
        recs.append(rec)
        if best is None or rec['J'] < best['J']:
            best = rec
    return best, recs


# ----------------------------------------------------------------- BC warm-start
# (Retained ONLY for the methodology case study / the RL-degradation narrative.)
def rollout_record(sim, action_fn, seed):
    rng = np.random.default_rng(seed)
    x = np.zeros(N, dtype=np.int8); x[P0] = 2
    traj = []
    COST = sim.COST
    for t in range(sim.T):
        cnt = sim.counts(x)
        s = state_index(cnt, min(t, sim.T - 1))
        a = int(action_fn(cnt, t))
        pinf = sim._p_infection(x, a)
        xn = x.copy(); Sm = x == 0; Em = x == 1; Im = x == 2
        se = Sm & (rng.random(N) < pinf)
        ei = Em & (rng.random(N) < sim.sigma)
        ir = Im & (rng.random(N) < sim.gamma)
        xn[se] = 1; xn[ei] = 2; xn[ir] = 3
        wexp = float(sim.wnode[se].sum())
        r = -(sim.alpha_r * wexp + sim.beta_r * COST[a])
        traj.append((s, a, r))
        x = xn
        if cnt['I_tot'] == 0 and cnt['E_tot'] == 0 and t > 0:
            break
    return traj

def warm_start_Q(sim, action_fn, n_roll=1500, margin=5.0, seed0=900000):
    V_sum = np.zeros(NSTATES); V_cnt = np.zeros(NSTATES)
    act_cnt = np.zeros((NSTATES, NACT))
    for k in range(n_roll):
        traj = rollout_record(sim, action_fn, seed0 + k)
        G = 0.0
        for (s, a, r) in reversed(traj):
            G += r
            V_sum[s] += G; V_cnt[s] += 1; act_cnt[s, a] += 1
    visited = V_cnt > 0
    V = np.zeros(NSTATES)
    V[visited] = V_sum[visited] / V_cnt[visited]
    modal = np.argmax(act_cnt, axis=1)
    Q = np.zeros((NSTATES, NACT))
    for s in range(NSTATES):
        base = V[s] if visited[s] else 0.0
        Q[s, :] = base - margin
        Q[s, modal[s]] = base
    return Q
