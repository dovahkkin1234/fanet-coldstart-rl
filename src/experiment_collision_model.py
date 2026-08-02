"""
experiment_collision_model.py  —  Does the Bianchi saturation assumption matter?

ADDRESSES REVIEWER FINDING M-4
==============================
Bianchi (2000) derives its collision fixed point under SATURATION -- every
station always has a packet queued. Our measured node activity is 0.02-0.09
and mean queue occupancy is under 0.14, so the network is firmly non-saturated
and the model is being applied outside its stated regime. The standard
corrections (Malone, Duffy & Leith, IEEE/ACM ToN 2007; Liaw et al.) add an
idle/empty-buffer state for exactly this reason. A networking-side TNSM
reviewer will catch this on sight.

A SECOND, WORSE DEFECT FOUND WHILE INVESTIGATING
------------------------------------------------
The call site computed `n_cont = 1 + int(round(sum of activities))` and fed
that integer to the saturated model. Since Bianchi returns exactly 0 for
n <= 1, any activity summing below 0.5 gave p_collision == 0.0 EXACTLY.
Measured at 15 carrier-sense neighbours:

    activity   OLD (rounded+saturated)   NEW (non-saturated)
      0.01            0.00000                  0.01610
      0.02            0.00000                  0.03146
      0.05            0.10462                  0.07354
      0.09            0.10462                  0.12138

p_collision was a STEP FUNCTION of load with a hard-zero plateau, and our
operating range sat mostly inside that plateau -- MAC contention was modelled
as exactly absent much of the time, then jumped discontinuously. That is a
quantisation defect, not a modelling approximation.

WHAT THE REPLACEMENT DOES
-------------------------
Keeps Bianchi's backoff chain for tau, but lets station k contend only when
its buffer is non-empty (probability q_k), so the tagged station sees

    p = 1 - prod_k (1 - q_k * tau)

solved jointly with tau. The product over individual q_k handles HETEROGENEOUS
activity, which matters because congested and idle nodes coexist by design.
Validated at import: it reduces EXACTLY to saturated Bianchi at q=1 (so it is
demonstrably the same chain), returns 0 at q=0, is monotone in both activity
and station count, and is smooth -- no plateau.

Honest scope: this is the standard effective-transmission-probability
approximation to the non-saturated case, NOT a reimplementation of the full
Malone-Duffy-Leith Markov chain. It corrects the saturation assumption and
removes the quantisation defect; it does not model post-backoff or finite
buffer states.

WHY THE DEFAULT IS NOT SIMPLY FLIPPED
-------------------------------------
Changing the physics changes G2's regression constant (PDR 0.310714), which
has caught silent behaviour changes repeatedly. So the new model is OPT-IN,
the difference is MEASURED here, and only then is the default flipped with a
new regression constant recorded. Silently changing physics and losing the
anchor would be the worse trade.

WHAT IS TESTED
--------------
The same four claims used for the M-5 calibration sweep, so the two are
directly comparable:

  C1  SP-BP still tops the teacher panel
  C2  headroom still increases monotonically with load
  C3  congestion-aware teachers still beat congestion-blind at high load
  C4  mean link quality still degrades with load

Plus the absolute shift in PDR and link quality, so the cost of the correction
is quantified rather than asserted.

Usage:
    python src\\experiment_collision_model.py --max_workers 16
    python src\\experiment_collision_model.py --quick
"""

import os, sys, json, argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from teacher_panel import build_oracle_table, paired_ttest

SCENARIOS = {
    'very_dense':  dict(num_drones=45, area_x=700,  area_y=700,  comm_range=250,
                        speed_min=5,  speed_max=15, pause_max=5.0),
    'dense_slow':  dict(num_drones=30, area_x=800,  area_y=800,  comm_range=250,
                        speed_min=5,  speed_max=15, pause_max=5.0),
    'medium_slow': dict(num_drones=30, area_x=1300, area_y=1300, comm_range=280,
                        speed_min=5,  speed_max=15, pause_max=5.0),
    'sparse_fast': dict(num_drones=20, area_x=1500, area_y=1500, comm_range=300,
                        speed_min=35, speed_max=50, pause_max=2.0),
}
RATES = [0.5, 2.0, 4.0]
BASE = dict(z_min=50, z_max=150, duration=40.0, drain_time=10.0,
            interference_on=True)
PANEL = ['dijkstra', 'gpsr', 'backpressure', 'spbp', 'da_gpsr']
BLIND, AWARE = ('dijkstra', 'gpsr'), ('spbp', 'da_gpsr')
MODELS = ['saturated', 'unsaturated']


def _headroom_job(job):
    from experiment_headroom import HeadroomSimulator
    sc, cfg, rate, seed, model = job
    full = {**BASE, **cfg, 'packet_rate': rate, 'seed': seed,
            'actor': 'spbp', 'collision_model': model}
    m = HeadroomSimulator(full).run()
    return {'rate': rate, 'n_routable': m['n_routable'],
            'drops_routable': m['drops_routable'],
            'mean_link_quality': m['mean_link_quality'],
            'pdr': m['pdr_predrain']}


def evaluate(model, scen, rates, seeds, max_workers):
    from experiment_headroom import ROUTING_ADDRESSABLE
    base = {**BASE, 'collision_model': model}
    table, raw, cong, diag, stats = build_oracle_table(
        scen, rates, seeds, panel=PANEL, base_cfg=base,
        verbose=False, max_workers=max_workers)

    means = {t: float(np.mean([dict(table[k]).get(t, np.nan) for k in table]))
             for t in PANEL}
    ranked = sorted(means.items(), key=lambda kv: -kv[1])
    c1 = ranked[0][0] == 'spbp'
    margin = ranked[0][1] - ranked[1][1]

    hi = [k for k in table if k[1] == 'high']
    aware = [max(dict(table[k]).get(t, np.nan) for t in AWARE) for k in hi]
    blind = [max(dict(table[k]).get(t, np.nan) for t in BLIND) for k in hi]
    c3_gap = float(np.mean(aware) - np.mean(blind)) if hi else float('nan')

    jobs = [(sc, cfg, r, sd, model)
            for sc, cfg in scen.items() for r in rates for sd in seeds]
    out = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_headroom_job, j) for j in jobs]
        for f in as_completed(futs):
            out.append(f.result())

    by_rate = {}
    for o in out:
        by_rate.setdefault(o['rate'], []).append(o)
    headroom, linkq, pdr = {}, {}, {}
    for r, rs in by_rate.items():
        tot = sum(x['n_routable'] for x in rs)
        agg = Counter()
        for x in rs:
            agg.update(x['drops_routable'])
        headroom[r] = sum(agg.get(c, 0) for c in ROUTING_ADDRESSABLE) / max(tot, 1)
        linkq[r] = float(np.mean([x['mean_link_quality'] for x in rs]))
        pdr[r] = float(np.mean([x['pdr'] for x in rs]))
    rs_sorted = sorted(headroom)
    hv = [headroom[r] for r in rs_sorted]
    lv = [linkq[r] for r in rs_sorted]

    return {'model': model, 'means': means, 'winner': ranked[0][0],
            'margin': float(margin), 'c1': bool(c1),
            'c3_gap': c3_gap, 'c3': bool(c3_gap > 0),
            'c2': bool(all(hv[i] <= hv[i+1] + 1e-9 for i in range(len(hv)-1))),
            'c4': bool(all(lv[i] >= lv[i+1] - 1e-9 for i in range(len(lv)-1))),
            'headroom': {str(k): float(v) for k, v in headroom.items()},
            'linkq': {str(k): float(v) for k, v in linkq.items()},
            'pdr': {str(k): float(v) for k, v in pdr.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=list(range(1, 16)))
    ap.add_argument('--rates', type=float, nargs='+', default=RATES)
    ap.add_argument('--max_workers', type=int, default=None)
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--out', default='results/collision_model.json')
    args = ap.parse_args()

    scen = {'medium_slow': SCENARIOS['medium_slow']} if args.quick else SCENARIOS

    print("\n" + "=" * 78)
    print("  COLLISION MODEL: SATURATED vs NON-SATURATED (reviewer finding M-4)")
    print("=" * 78)
    print("  Bianchi (2000) assumes saturation; measured activity here is 0.02-0.09.")
    print("  The old call site also ROUNDED the summed activity to an integer")
    print("  station count, so p_collision was exactly 0 below activity ~0.5/n --")
    print("  a hard-zero plateau across much of our operating range.")

    results = []
    for model in MODELS:
        print(f"\n  --- {model} ---")
        r = evaluate(model, scen, args.rates, args.seeds, args.max_workers)
        results.append(r)
        print(f"      winner={r['winner']:<10} margin={r['margin']:+.4f}   "
              f"C1={'OK' if r['c1'] else 'FAIL'} C2={'OK' if r['c2'] else 'FAIL'} "
              f"C3={'OK' if r['c3'] else 'FAIL'} C4={'OK' if r['c4'] else 'FAIL'}")
        print(f"      PDR by rate:      "
              f"{ {k: round(v,4) for k,v in sorted(r['pdr'].items())} }")
        print(f"      link quality:     "
              f"{ {k: round(v,4) for k,v in sorted(r['linkq'].items())} }")
        print(f"      headroom by rate: "
              f"{ {k: f'{100*v:.1f}%' for k,v in sorted(r['headroom'].items())} }")

    sat, uns = results[0], results[1]
    print("\n" + "-" * 78)
    print("  IMPACT OF THE CORRECTION")
    print("-" * 78)
    print(f"  {'quantity':<22}{'saturated':>12}{'unsaturated':>14}{'delta':>10}")
    print(f"  {'SP-BP margin':<22}{sat['margin']:>12.4f}{uns['margin']:>14.4f}"
          f"{uns['margin']-sat['margin']:>+10.4f}")
    for r in sorted(sat['pdr']):
        print(f"  {'PDR @ rate '+r:<22}{sat['pdr'][r]:>12.4f}{uns['pdr'][r]:>14.4f}"
              f"{uns['pdr'][r]-sat['pdr'][r]:>+10.4f}")
    for r in sorted(sat['linkq']):
        print(f"  {'link quality @ '+r:<22}{sat['linkq'][r]:>12.4f}"
              f"{uns['linkq'][r]:>14.4f}{uns['linkq'][r]-sat['linkq'][r]:>+10.4f}")

    print("\n" + "=" * 78)
    print("  VERDICT")
    print("=" * 78)
    claims_hold = all(uns[c] for c in ('c1', 'c2', 'c3', 'c4'))
    if claims_hold:
        print("  ALL FOUR CLAIMS SURVIVE the corrected collision model.")
        print("  RECOMMENDATION: adopt 'unsaturated' as the default. It is the")
        print("  literature-appropriate model for a non-saturated network AND it")
        print("  removes a genuine quantisation defect (p_collision exactly 0 across")
        print("  much of the operating range). Record the new G2 regression constant")
        print("  when the default flips -- do not lose the anchor.")
        print("  For the paper: report that the saturation assumption was tested")
        print("  rather than assumed, and that conclusions are unchanged under the")
        print("  corrected model. That converts a likely referee objection into")
        print("  evidence of care.")
    else:
        bad = [c for c in ('c1', 'c2', 'c3', 'c4') if not uns[c]]
        print(f"  CLAIMS {bad} DO NOT SURVIVE the corrected model.")
        print("  This is important and must be reported, not worked around: the")
        print("  affected conclusions were partly artifacts of applying a saturated")
        print("  model outside its regime. The corrected model is the defensible")
        print("  one; the claims must be rescoped to match it.")
    print("=" * 78 + "\n")

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({'results': results, 'seeds': list(args.seeds),
                   'note': 'effective-transmission-probability approximation; '
                           'not a full Malone-Duffy-Leith chain'}, f, indent=2)
    print(f"  saved to {args.out}\n")
    return 0


if __name__ == '__main__':
    sys.exit(main())
