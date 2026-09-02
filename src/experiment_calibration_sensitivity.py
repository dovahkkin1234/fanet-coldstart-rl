"""
experiment_calibration_sensitivity.py  —  Do the conclusions survive the
calibration choices, or were they manufactured by them?

ADDRESSES REVIEWER FINDING M-5
==============================
Four calibration decisions made during M2 bring-up were chosen, explicitly,
BECAUSE they produced the dynamics the thesis needs:

  ACT_BETA = 0        interference driven only by real backlog. Adopted after
                      measuring that it concentrated interference at genuinely
                      congested nodes rather than diffusing it -- i.e. it
                      created the avoidable hot-spots congestion-aware routing
                      needs in order to look good.
  max_retx = 5        ARQ. Introduced specifically because without it PER was a
                      near-step function of SINR, interference behaved as
                      spatially-uniform random loss, and NO routing policy
                      could beat any other (measured actor spread ~0.008).
                      This is the largest discretionary intervention in the
                      whole environment.
  ACT_MAX = 0.5       ceiling on per-node transmission activity.
  IF_RANGE_MULT = 2.0 interference range as a multiple of comm range.

"We chose the calibration that produced our effect" is a clean line of attack,
and it is currently unanswered: there is no evidence the qualitative
conclusions survive a plausible range of these values. This experiment
supplies it.

WHAT IS TESTED (claims, not numbers)
------------------------------------
Absolute PDR will obviously move as the physics changes. That is expected and
uninteresting. What must survive is the QUALITATIVE structure the paper
depends on:

  C1  SP-BP still tops the teacher panel.
  C2  Headroom is still monotonically increasing in load
      (routing matters little at low load, a lot at high load).
  C3  Congestion-aware teachers (spbp, da_gpsr) still beat congestion-blind
      references (dijkstra, gpsr) at high load.
  C4  Mean link quality still degrades with load -- the G1 premise that the
      whole Approach-2 programme rests on.

If all four hold at every setting, the findings are properties of the
environment's structure, not of the constants. If any flips, that must be
reported as a scope limit on the claim.

DESIGN: ONE-AT-A-TIME, AND ITS LIMITATION STATED
-------------------------------------------------
A full factorial over 4 parameters x 3 levels is 81 configurations, which at
this grid size is prohibitive. This sweep is ONE-AT-A-TIME: each parameter is
varied across its range while the others hold at their locked defaults.

LIMITATION, stated rather than glossed: OAT explores the axes through the
baseline point and therefore CANNOT detect interaction effects -- a
combination of two off-baseline settings could behave differently from either
alone. OAT is the standard, defensible choice for "are the conclusions robust
to the knobs we chose", and it is honest to say it answers that and not
"we have mapped the parameter space". If a reviewer asks for interactions, the
follow-up is a fractional factorial on whichever axes prove most sensitive
here -- which is precisely what this experiment identifies.

Usage:
    python src\\experiment_calibration_sensitivity.py --max_workers 16
    python src\\experiment_calibration_sensitivity.py --quick
"""

import os, sys, json, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from teacher_panel import build_oracle_table

# Config now lives in config_v2.py -- see the note there on why eight
# independent copies of this block were a latent hazard.
from config_v2 import SCENARIOS, RATES, BASE, get_suite, provenance  # noqa: F401

# Core panel: two congestion-blind references and the three congestion-aware
# teachers that carry the C1/C3 claims. The full 8-teacher panel is not needed
# -- the claims turn on SP-BP vs its runner-up and vs the blind references.
PANEL = ['dijkstra', 'gpsr', 'backpressure', 'spbp', 'da_gpsr']

BLIND = ('dijkstra', 'gpsr')
AWARE = ('spbp', 'da_gpsr')

# locked defaults, then the alternatives probed one axis at a time
DEFAULTS = {'act_beta': 0.0, 'max_retx': 5, 'act_max': 0.5, 'if_range_mult': 2.0}
SWEEP = {
    'act_beta':      [0.02, 0.05],
    'max_retx':      [3, 8],
    'act_max':       [0.3, 0.7],
    'if_range_mult': [1.5, 2.5],
}


def evaluate_config(overrides, scen, rates, seeds, max_workers):
    """Run the panel under one calibration and extract the four claims."""
    base = {**BASE, **overrides}
    table, raw, cong, diag, stats = build_oracle_table(
        scen, rates, seeds, panel=PANEL, base_cfg=base,
        verbose=False, max_workers=max_workers)

    means = {t: float(np.mean([dict(table[k]).get(t, np.nan) for k in table]))
             for t in PANEL}
    ranked = sorted(means.items(), key=lambda kv: -kv[1])

    # C1: SP-BP tops the panel
    c1 = ranked[0][0] == 'spbp'
    margin = ranked[0][1] - ranked[1][1]

    # C3: congestion-aware beat blind at HIGH load
    hi = [k for k in table if k[1] == 'high']
    aware_hi, blind_hi = [], []
    for k in hi:
        bt = dict(table[k])
        aware_hi.append(max(bt.get(t, np.nan) for t in AWARE))
        blind_hi.append(max(bt.get(t, np.nan) for t in BLIND))
    c3_gap = float(np.mean(aware_hi) - np.mean(blind_hi)) if hi else float('nan')
    c3 = c3_gap > 0

    # C2/C4 need per-run detail: PDR by load for the best teacher, and link quality
    pdr_by_bucket = {}
    for k in table:
        pdr_by_bucket.setdefault(k[1], []).append(dict(table[k]).get('spbp', np.nan))
    pdr_lb = {b: float(np.nanmean(v)) for b, v in pdr_by_bucket.items()}

    return {'overrides': overrides, 'means': means, 'winner': ranked[0][0],
            'margin': float(margin), 'c1': bool(c1),
            'c3_gap': c3_gap, 'c3': bool(c3), 'pdr_by_bucket': pdr_lb}


def measure_c2_c4(overrides, scen, rates, seeds, max_workers):
    """Headroom monotonicity (C2) and link-quality degradation (C4).

    Run directly rather than via the oracle table, because both need
    per-episode diagnostics the table aggregates away."""
    from experiment_headroom import HeadroomSimulator, ROUTING_ADDRESSABLE
    from collections import Counter
    from concurrent.futures import ProcessPoolExecutor, as_completed

    jobs = [(sc, cfg, r, sd, overrides)
            for sc, cfg in scen.items() for r in rates for sd in seeds]
    out = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_headroom_job, j) for j in jobs]
        for f in as_completed(futs):
            out.append(f.result())

    by_rate = {}
    for o in out:
        by_rate.setdefault(o['rate'], []).append(o)
    headroom, linkq = {}, {}
    for r, rs in by_rate.items():
        tot = sum(x['n_routable'] for x in rs)
        agg = Counter()
        for x in rs:
            agg.update(x['drops_routable'])
        headroom[r] = sum(agg.get(c, 0) for c in ROUTING_ADDRESSABLE) / max(tot, 1)
        linkq[r] = float(np.mean([x['mean_link_quality'] for x in rs]))
    rates_sorted = sorted(headroom)
    hv = [headroom[r] for r in rates_sorted]
    lv = [linkq[r] for r in rates_sorted]
    c2 = all(hv[i] <= hv[i + 1] + 1e-9 for i in range(len(hv) - 1))
    c4 = all(lv[i] >= lv[i + 1] - 1e-9 for i in range(len(lv) - 1))
    return {'headroom_by_rate': {str(k): float(v) for k, v in headroom.items()},
            'linkq_by_rate': {str(k): float(v) for k, v in linkq.items()},
            'c2': bool(c2), 'c4': bool(c4)}


def _headroom_job(job):
    from experiment_headroom import HeadroomSimulator
    sc, cfg, rate, seed, overrides = job
    full = {**BASE, **overrides, **cfg, 'packet_rate': rate,
            'seed': seed, 'actor': 'spbp'}
    m = HeadroomSimulator(full).run()
    return {'rate': rate, 'n_routable': m['n_routable'],
            'drops_routable': m['drops_routable'],
            'mean_link_quality': m['mean_link_quality']}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=list(range(1, 16)))
    ap.add_argument('--rates', type=float, nargs='+', default=RATES)
    ap.add_argument('--max_workers', type=int, default=None)
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--out', default='results/calibration_sensitivity.json')
    args = ap.parse_args()

    scen = ({'medium_slow': SCENARIOS['medium_slow']} if args.quick
            else SCENARIOS)
    configs = [('BASELINE (locked)', {})]
    for param, values in SWEEP.items():
        for v in values:
            configs.append((f'{param}={v}', {param: v}))

    n_panel = len(configs) * len(scen) * len(args.rates) * len(args.seeds) * len(PANEL)
    n_head = len(configs) * len(scen) * len(args.rates) * len(args.seeds)
    print("\n" + "=" * 78)
    print("  CALIBRATION SENSITIVITY (reviewer finding M-5)")
    print("=" * 78)
    print(f"  {len(configs)} configurations, one-at-a-time about the locked defaults")
    print(f"  defaults: {DEFAULTS}")
    print(f"  runs: {n_panel} panel + {n_head} headroom = {n_panel + n_head}")
    print("  OAT cannot detect INTERACTIONS between parameters -- stated as a")
    print("  limitation, not glossed. It answers 'are the conclusions robust to")
    print("  the knobs we chose', which is the reviewer's actual question.")

    results = []
    for name, ov in configs:
        print(f"\n  --- {name} ---")
        r = evaluate_config(ov, scen, args.rates, args.seeds, args.max_workers)
        r2 = measure_c2_c4(ov, scen, args.rates, args.seeds, args.max_workers)
        r.update(r2)
        r['name'] = name
        results.append(r)
        print(f"      winner={r['winner']:<12} margin={r['margin']:+.4f}   "
              f"C1={'OK' if r['c1'] else 'FAIL'}  C2={'OK' if r['c2'] else 'FAIL'}  "
              f"C3={'OK' if r['c3'] else 'FAIL'} (gap {r['c3_gap']:+.4f})  "
              f"C4={'OK' if r['c4'] else 'FAIL'}")
        print(f"      headroom by rate: "
              f"{ {k: f'{100*v:.1f}%' for k, v in sorted(r['headroom_by_rate'].items())} }")

    # ---- summary ----
    print("\n" + "-" * 78)
    print("  CLAIM SURVIVAL ACROSS CALIBRATIONS")
    print("-" * 78)
    print(f"  {'configuration':<22}{'winner':<12}{'margin':>9}{'C1':>5}{'C2':>5}{'C3':>5}{'C4':>5}")
    for r in results:
        print(f"  {r['name']:<22}{r['winner']:<12}{r['margin']:>9.4f}"
              f"{'OK' if r['c1'] else 'FAIL':>5}{'OK' if r['c2'] else 'FAIL':>5}"
              f"{'OK' if r['c3'] else 'FAIL':>5}{'OK' if r['c4'] else 'FAIL':>5}")

    # which axis moves things most?
    base = results[0]
    print("\n  sensitivity of SP-BP's margin to each axis (vs baseline):")
    for r in results[1:]:
        print(f"    {r['name']:<22} margin {r['margin']:+.4f} "
              f"({r['margin'] - base['margin']:+.4f} vs baseline)")

    all_ok = all(r['c1'] and r['c2'] and r['c3'] and r['c4'] for r in results)
    print("\n" + "=" * 78)
    print("  VERDICT")
    print("=" * 78)
    if all_ok:
        print("  ALL FOUR CLAIMS SURVIVE EVERY CALIBRATION TESTED.")
        print("  The findings are properties of the environment's structure, not of")
        print("  the constants chosen during M2 bring-up. The 'you tuned to your")
        print("  result' objection is answered with evidence rather than assertion.")
        print("  Absolute PDR moves with the physics, as it must; the qualitative")
        print("  structure does not.")
    else:
        print("  AT LEAST ONE CLAIM FLIPS. This is important and must be reported,")
        print("  not buried: it bounds the scope of the affected claim to the")
        print("  calibration range where it holds. Failing configurations:")
        for r in results:
            bad = [c for c in ('c1', 'c2', 'c3', 'c4') if not r[c]]
            if bad:
                print(f"    {r['name']:<22} fails {bad}")
    print("=" * 78 + "\n")

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({'results': results, 'defaults': DEFAULTS, 'sweep': SWEEP,
                   'panel': PANEL, 'seeds': list(args.seeds),
                   'design': 'one-at-a-time; interactions NOT explored'},
                  f, indent=2)
    print(f"  saved to {args.out}\n")
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
