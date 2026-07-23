"""
preflight_teachers_v2_check.py  —  GATE G3 for Approach 2 (M3).

Validates the 5-teacher panel and the oracle table BEFORE any dataset is
generated. A dataset built on a broken oracle table would silently poison M4 and
every result downstream, so this gate is mandatory (design spec S8).

REVISION HISTORY (kept because each round caught a real, distinct bug and the
next reader should not have to rediscover why the code looks like this):
  v1: 6-teacher panel including ETX-Dijkstra, backpressure with a strict-
      positive gate, majority-vote-style post-hoc diagnostics. Caught: 4/6
      teachers returned byte-identical PDR (graph was built with
      interference_mw=0, so packet_error_rate was always 0).
  v2: fixed the graph to reflect measured (load-dependent) channel state.
      First attempt used EXPECTED interference power -> link_quality collapsed
      to ~0.12 (Jensen's inequality: quality is convex in interference, mean
      power under-estimates mean quality). Second attempt used P(lethal
      interferer fires) instead -> physically sane (0.67 -> 0.50 across load).
  v3: full grid run exposed backpressure == GPSR in every cell (byte-identical).
      Root cause: the strict-positive gate on the queue differential almost
      never fired (measured occupancy is 0.02-0.14), so backpressure silently
      degenerated into greedy progress on ~100% of decisions, while a post-hoc
      diagnostic (sampled on a frozen graph after the episode ended) wrongly
      reported this as a 58% fallback rate. Fixed: gate removed entirely
      (argmax over ALL neighbours, matching Tassiulas-Ephremides), diagnostic
      rebuilt to count LIVE during the actual rollout.
  v4: ETX-Dijkstra lost to plain Dijkstra in every cell. Hypothesis was path-
      stretching from ETX's 1/(1-PER) blow-up; replaced with LQ-weighted
      Dijkstra (bounded via an eps floor) on that theory.
  v5 (THIS FIX): the v4 hypothesis was WRONG. A head-to-head including a
      THIRD candidate (arq_etx — bounded, physically exact for this
      simulator's actual ARQ mechanism) showed mean hop count is identical
      (1.89-1.95) across every candidate: nobody was stretching paths. Plain
      Dijkstra beat ALL THREE dynamic-metric variants in 9/12 cells. The real
      mechanism: link_quality is genuinely informative (corr -0.38 with ARQ
      retry count) but noisy and frame-stale; a noiseless objective (hop
      count) beats a weakly-correlated one when optimized globally over a
      multi-hop path, because estimation error compounds hop over hop. The
      SAME signal helps when used for a ONE-HOP greedy decision (SP-BP,
      DA-GPSR), where the estimate is freshest and never compounds. Panel
      reduced to 5 members; the negative result is reported directly (see
      simulator_v2.PANEL comment) rather than carrying a teacher that
      structurally never wins.
      ALSO in this pass: a re-run with the v3 diagnostic fix showed the fix
      was itself broken by the v4 rewrite (backpressure_next_hop's rewritten
      body no longer incremented the module-level _BP_STATS counters the
      harness reads), reporting a false 0.000 at every load. Restored, and
      cross-checked against the simulator's own live per-run counters so the
      two paths cannot silently diverge again.
      FURTHER, auditing the checks themselves (not just the numbers they
      print) found that "regime-dependent ranking" (checks 3/4) was passing
      almost entirely on cells with a winning margin under noise level at the
      seed counts then in use (2-3 seeds): every winner OTHER than the
      dominant teacher came from a cell with margin < 0.02. Checks 3 and 4 are
      now computed on the ROBUST-margin subset of cells as the primary
      criterion, with the full (unfiltered) picture still printed for
      transparency. Seed count raised to 30 (from this project's established
      Phase-1 rigor bar), and Phase A parallelized via ProcessPoolExecutor so
      the resulting 30-seed x 5-teacher x 4-scenario x 3-rate grid (4500 runs)
      completes in minutes rather than hours on a multi-core machine.

THE SIX CHECKS (docs/M3_TEACHERS_ORACLE_DESIGN.md S7, as revised above):
  1. All panel teachers beat a random-neighbour policy at every load.
  2. A backpressure-family teacher (backpressure or spbp) tops the ranking at
     HIGH load, restricted to cells where congestion is actually the
     bottleneck (>=50% of drops congestion-caused) and to statistically
     robust cells (winner margin > THIN_MARGIN).
  3. The WINNER changes across regimes, using ONLY robust-margin cells as
     evidence. The full (unfiltered) picture is also printed so the gap
     between "looks regime-dependent" and "is regime-dependent" is visible.
  4. Same, across scenario class specifically.
  5. Vote agreement is materially below 1.0 under load.
  6. Reproducible under a fixed seed.

Usage:
    python src\\preflight_teachers_v2_check.py
    python src\\preflight_teachers_v2_check.py --seeds 1 2 3 ... 30 --max_workers 16
"""

import os, sys, argparse
from collections import Counter
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulator_v2 import FANETSimulatorV2, PANEL
from teacher_panel import (build_oracle_table, scenario_class, load_bucket,
                           collect_votes, vote_agreement, welch_ttest)

BP_FAMILY = ('backpressure', 'spbp')

# Winner margins below this are inside the seed noise this project has
# repeatedly measured (~2-8% depending on n); such an entry must not count as
# evidence of regime-dependence. RETAINED as a secondary/display heuristic;
# the PRIMARY robustness criterion is now welch_ttest below, since n=30 gives
# enough samples to test significance properly rather than guess a threshold.
THIN_MARGIN = 0.02
ALPHA = 0.05   # significance level for the Welch's t-test robustness criterion

# Scenario grid spanning four genuinely distinct topology regimes (by expected
# node degree; G1 showed dense = congestion-limited, sparse = range/partition-
# limited, and these are physically different failure modes).
SCENARIOS = {
    'very_dense':  dict(num_drones=45, area_x=700,  area_y=700,  comm_range=250,
                        speed_min=5,  speed_max=15, pause_max=5.0),   # degree 18.0
    'dense_slow':  dict(num_drones=30, area_x=800,  area_y=800,  comm_range=250,
                        speed_min=5,  speed_max=15, pause_max=5.0),   # degree 9.2
    'medium_slow': dict(num_drones=30, area_x=1300, area_y=1300, comm_range=280,
                        speed_min=5,  speed_max=15, pause_max=5.0),   # degree 4.4
    'sparse_fast': dict(num_drones=20, area_x=1500, area_y=1500, comm_range=300,
                        speed_min=35, speed_max=50, pause_max=2.0),   # degree 2.5
}

DEFAULT_SEEDS = list(range(1, 31))   # 30 seeds, 1..30


def cell_significance(table, table_stats, key, alpha=ALPHA):
    """Welch's t-test between the #1 and #2 teacher in a cell. Returns
    (p_value, significant: bool). This is the PRIMARY robustness criterion
    (replaces a fixed 0.02 margin heuristic used when only 2-3 seeds were
    available); THIN_MARGIN is retained only as a quick-glance display value.
    """
    ranked = table[key]
    if len(ranked) < 2:
        return 1.0, False
    (t1, m1), (t2, m2) = ranked[0], ranked[1]
    stats = table_stats.get(key, {})
    s1, s2 = stats.get(t1), stats.get(t2)
    if s1 is None or s2 is None:
        return 1.0, False
    _, _, p = welch_ttest(s1['mean'], s1['std'], s1['n'],
                          s2['mean'], s2['std'], s2['n'])
    return p, (p < alpha)


def robust_cells(table, table_stats):
    """Cells whose #1-vs-#2 difference is statistically significant
    (Welch's t, alpha=0.05) — the subset trustworthy enough to use as evidence
    of regime-dependence at this seed count."""
    return {k: v for k, v in table.items()
            if cell_significance(table, table_stats, k)[1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS)
    ap.add_argument('--rates', type=float, nargs='+', default=[0.5, 2.0, 4.0])
    ap.add_argument('--duration', type=float, default=40.0)
    ap.add_argument('--drain_time', type=float, default=10.0)
    ap.add_argument('--max_workers', type=int, default=None,
                    help='ProcessPoolExecutor worker count; default = all cores')
    args = ap.parse_args()

    base = dict(z_min=50, z_max=150, duration=args.duration,
                interference_on=True, drain_time=args.drain_time)

    n_jobs = len(SCENARIOS) * len(args.rates) * len(args.seeds) * len(PANEL)
    print("\n" + "=" * 78)
    print("  GATE G3 — TEACHER PANEL + ORACLE TABLE VALID?")
    print("=" * 78)
    print(f"  panel: {', '.join(PANEL)}")
    print(f"  scenarios: {', '.join(SCENARIOS)}")
    print(f"  rates: {args.rates}   n_seeds: {len(args.seeds)}")
    print(f"  duration={args.duration:.0f}s  drain={args.drain_time:.0f}s "
          f"(PDR measured over pre-drain packets)")
    print(f"  total runs this grid: {n_jobs}")

    # ── Phase A: build the oracle table (parallel) ───────────────────────────
    print("\n" + "-" * 78)
    print("  PHASE A — TEACHER BENCHMARK GRID")
    print("-" * 78)
    table, raw, congestion, diagnostics, table_stats = build_oracle_table(
        SCENARIOS, args.rates, args.seeds, base_cfg=base, verbose=True,
        max_workers=args.max_workers)

    def margin_of(key):
        ranked = table[key]
        return ranked[0][1] - ranked[1][1] if len(ranked) > 1 else float('inf')

    # ── Per-cell ranking table ───────────────────────────────────────────────
    print("\n" + "-" * 78)
    print("  ORACLE TABLE  (mean network PDR over seeds, ranked)")
    print("-" * 78)
    for key in sorted(table):
        ranked = table[key]
        line = "  ".join(f"{t}={v:.3f}" for t, v in ranked)
        cg = congestion.get(key, 0.0)
        tag = "CONGESTION-limited" if cg >= 0.50 else "range/partition-limited"
        margin = margin_of(key)
        p_val, sig = cell_significance(table, table_stats, key)
        flag = "" if sig else "  <-- NOT SIGNIFICANT (p>=0.05), not robust evidence"
        print(f"    {key[0]:<12} {key[1]:<7} [{tag:<23} cong={cg:.2f}] "
              f"margin={margin:+.3f} p={p_val:.4f}{flag}")
        print(f"        {line}")

    # ── Random baseline for the sanity floor ─────────────────────────────────
    print("\n" + "-" * 78)
    print("  RANDOM BASELINE (sanity floor)")
    print("-" * 78)
    random_by_rate = {}
    worst_teacher_by_rate = {}
    for pr in args.rates:
        cfg = {**base, **SCENARIOS['medium_slow'], 'packet_rate': pr}
        rand_table, _, _, _, _ = build_oracle_table(
            {'medium_slow': SCENARIOS['medium_slow']}, [pr], args.seeds,
            panel=['random'], base_cfg=base, verbose=False,
            max_workers=args.max_workers)
        key0 = next(iter(rand_table))
        random_by_rate[pr] = rand_table[key0][0][1]
        key = (scenario_class({**base, **SCENARIOS['medium_slow'], 'packet_rate': pr}),
              load_bucket(pr))
        worst_teacher_by_rate[pr] = min(v for _, v in table[key])
        print(f"    rate={pr:<5.2f} random={random_by_rate[pr]:.3f}   "
              f"worst panel teacher={worst_teacher_by_rate[pr]:.3f}")

    # ── Diagnostics: override rate (ALL teachers) + bp zero-gradient rate ────
    # Read directly from the SAME runs Phase A already computed — no separate
    # re-simulation, so this cannot silently diverge from the PDR numbers above
    # (that disconnect was exactly the v3 bug).
    print("\n" + "-" * 78)
    print("  LOOP-OVERRIDE & BACKPRESSURE ZERO-GRADIENT DIAGNOSTICS (live, unified)")
    print("-" * 78)
    med_keys = {pr: (scenario_class({**base, **SCENARIOS['medium_slow'], 'packet_rate': pr}),
                     load_bucket(pr)) for pr in args.rates}
    bpzd_by_bucket = {}
    for pr in args.rates:
        key = med_keys[pr]
        d = diagnostics.get(key, {'override': {}, 'bp_zerodiff': 0.0})
        bucket = load_bucket(pr)
        bpzd_by_bucket[bucket] = d['bp_zerodiff']
        ov = "  ".join(f"{t}={d['override'].get(t, 0.0):.3f}" for t in PANEL)
        print(f"    rate={pr:<5.2f} bucket={bucket:<7} bp_zerodiff={d['bp_zerodiff']:.3f}"
              f"   override_rate: {ov}")

    # ── Vote agreement (lightweight, single-seed graph sample; supplementary) ─
    print("\n" + "-" * 78)
    print("  VOTE AGREEMENT (sampled on a mid-rollout graph, seed[0] only)")
    print("-" * 78)
    agree_by_bucket = {}
    for pr in args.rates:
        cfg = {**base, **SCENARIOS['medium_slow'], 'packet_rate': pr,
               'seed': args.seeds[0], 'actor': 'spbp'}
        sim = FANETSimulatorV2(cfg)
        sim.run()
        G = sim._build_graph()
        agrees = []
        nodes = [n for n in G.nodes if list(G.neighbors(n))]
        rng = np.random.default_rng(0)
        for _ in range(200):
            if len(nodes) < 2:
                break
            c, d = rng.choice(nodes, size=2, replace=False)
            votes, _ = collect_votes(G, int(c), int(d))
            if votes.get('spbp') is not None:
                agrees.append(vote_agreement(votes, votes['spbp']))
        bucket = load_bucket(pr)
        agree_by_bucket[bucket] = float(np.mean(agrees)) if agrees else 1.0
        print(f"    rate={pr:<5.2f} bucket={bucket:<7} vote_agreement={agree_by_bucket[bucket]:.3f}")

    # ── Reproducibility ──────────────────────────────────────────────────────
    cfg = {**base, **SCENARIOS['medium_slow'], 'packet_rate': args.rates[-1],
           'seed': args.seeds[0], 'actor': 'spbp'}
    r1 = FANETSimulatorV2(cfg).run()
    r2 = FANETSimulatorV2(cfg).run()
    c6 = abs(r1['pdr_predrain'] - r2['pdr_predrain']) < 1e-12

    # ── CHECKS ───────────────────────────────────────────────────────────────
    # 1. every teacher beats random at every load
    c1 = all(worst_teacher_by_rate[pr] > random_by_rate[pr] for pr in args.rates)

    # 2. Backpressure family tops HIGH load WHERE CONGESTION IS THE BOTTLENECK
    #    AND the cell is statistically robust. Sparse/range-limited regimes are
    #    excluded on principle (G1: sparse fails from partition/range, not
    #    congestion, so congestion-aware routing has nothing to exploit there).
    TIE_TOL = 0.01
    CONG_LIMITED = 0.50
    high_cells = [k for k in table if k[1] == 'high']
    cong_high = [k for k in high_cells if congestion.get(k, 0.0) >= CONG_LIMITED]
    cong_high_robust = [k for k in cong_high
                        if cell_significance(table, table_stats, k)[1]]

    def bp_ok(k):
        ranked = table[k]
        top_name, top_val = ranked[0]
        if top_name in BP_FAMILY:
            return True
        best_bp = max((v for t, v in ranked if t in BP_FAMILY), default=-1.0)
        return (top_val - best_bp) <= TIE_TOL

    c2 = bool(cong_high_robust) and all(bp_ok(k) for k in cong_high_robust)

    # 3 & 4: regime-dependence, computed on the ROBUST subset (the actual fix
    # this revision makes). The full unfiltered picture is reported alongside
    # so the gap between "looks regime-dependent" and "is" stays visible.
    cell_winners_all = [ranked[0][0] for ranked in table.values()]
    win_counts_all = Counter(cell_winners_all)
    top_teacher_all, top_wins_all = win_counts_all.most_common(1)[0]

    r_cells = robust_cells(table, table_stats)
    cell_winners_robust = [ranked[0][0] for ranked in r_cells.values()]
    all_winners_robust = set(cell_winners_robust)
    distinct_orders_robust = len({tuple(t for t, _ in r_cells[k]) for k in r_cells})
    c3 = len(r_cells) >= 3 and len(all_winners_robust) > 1

    winners_by_class_robust = {}
    for key, ranked in r_cells.items():
        winners_by_class_robust.setdefault(key[0], set()).add(ranked[0][0])
    c4 = len(r_cells) >= 3 and \
         len({frozenset(v) for v in winners_by_class_robust.values()}) > 1

    # 5. vote agreement materially below 1.0 under load
    loaded = [v for b, v in agree_by_bucket.items() if b in ('medium', 'high')]
    c5 = bool(loaded) and max(loaded) < 0.90

    # ── VERDICT ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("  VERDICT")
    print("=" * 78)
    checks = [
        ("1. All teachers beat random at every load", c1,
         "; ".join(f"r{pr}: {worst_teacher_by_rate[pr]:.3f}>{random_by_rate[pr]:.3f}"
                   for pr in args.rates)),
        ("2. Backpressure family tops congested HIGH load (robust cells)", c2,
         f"robust congestion-limited high cells: {cong_high_robust} "
         f"(of {len(cong_high)} congestion-limited total)"),
        ("3. Winner changes across regimes (ROBUST cells only)", c3,
         f"{len(r_cells)}/{len(table)} cells robust; "
         f"{len(all_winners_robust)} distinct winners among them, "
         f"{distinct_orders_robust} distinct orderings"),
        ("4. Winner changes across scenario class (ROBUST cells only)", c4,
         f"winners by class (robust only): "
         f"{ {c: sorted(s) for c, s in winners_by_class_robust.items()} }"),
        ("5. Teachers disagree under load", c5,
         f"max loaded agreement={max(loaded) if loaded else float('nan'):.3f} (<0.90)"),
        ("6. Reproducible (fixed seed)", c6,
         f"PDR {r1['pdr_predrain']:.6f} == {r2['pdr_predrain']:.6f}"),
    ]
    for name, ok, detail in checks:
        print(f"    [{'PASS' if ok else 'FAIL'}] {name:<52} {detail}")
    print(f"    [INFO] {'Oracle dominance, ALL cells (unfiltered)':<52} "
          f"{top_teacher_all} wins {top_wins_all}/{len(cell_winners_all)} cells "
          f"({100*top_wins_all/len(cell_winners_all):.0f}%)")
    print(f"    [INFO] {'BP zero-gradient rate by bucket':<52} "
          f"{ {b: round(v, 3) for b, v in bpzd_by_bucket.items()} }")

    passed = all(c for _, c, _ in checks)
    print()
    if passed:
        print("    G3 PASS — the panel spans genuinely different policies, the")
        print("    congestion model rewards congestion-awareness under load, and the")
        print("    oracle table's regime-dependence holds on statistically robust")
        print("    evidence, not just cells within seed noise.")
        print("    Dataset generation (Phase B) is now safe. PROCEED.")
    else:
        print("    G3 FAIL — do NOT generate the dataset yet.")
        print("    A dataset built on a broken oracle table would silently poison M4")
        print("    and everything downstream. Inspect the failing checks above.")
    print("=" * 78 + "\n")
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
