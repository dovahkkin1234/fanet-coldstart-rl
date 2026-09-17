"""map_operating_regimes.py -- where, if anywhere, does congestion dominate?

WHY THIS EXISTS

scan_congestion_window.py reported "the queue term is worth ~0.3pp, effect is
genuinely small". That verdict was WRONG, because the scan printed only two loss
channels -- q_ovf and link_err -- and hid the one that actually dominated:

    dense_slow, dur 200 s, alt 100-300, spbp
      rate 20:  energy_depleted  9038 (34.0%)   q_ovf   72   DEAD NODES 7/30
      rate 40:  energy_depleted 39034 (73.4%)   q_ovf  199   DEAD NODES 7/30

Seven of thirty nodes were flat dead at EVERY rate tested. Queue overflow peaked
at 415 packets against 39,034 energy deaths. The entire rate sweep ran inside an
ENERGY-LIMITED regime, so it measured battery exhaustion, not congestion.

THE STRUCTURAL PROBLEM, ANALYTICALLY

  INITIAL_ENERGY = 100 per node, TX_COST = 0.02, RX_COST = 0.01
  -> each hop costs 0.03 across the two nodes involved
  -> a node dies after 5000 transmit attempts

  Network energy budget = N x 100. Cost per packet = hops x 0.03.
  Max packets an episode can carry, WITH PERFECT LOAD SPREADING:

    dense_slow   N=30  ->  45,455 packets
      dur  200 s  ->  max sustainable per-flow rate ~ 34
      dur 1000 s  ->  max sustainable per-flow rate ~  6.6

  Real traffic is skewed -- central relays carry far more than average -- so the
  true ceiling is well below these. And congestion (queue overflow dominating)
  requires per-flow rates around 40.

  At 1000 s -- THE DATASET'S TARGET DURATION -- the energy ceiling (~6.6) sits
  far BELOW the congestion onset (~40). If that holds empirically, the target
  duration and the congestion regime are MUTUALLY EXCLUSIVE under this energy
  model, and no rate grid can satisfy both.

WHAT THIS SCRIPT DOES

  PHASE 1 -- 2D regime map over duration x rate. For every cell it reports the
  FULL drop-reason breakdown, dead-node count, and classifies the dominant loss
  channel into a regime: clean / congestion / interference / energy / partition.
  A cell only counts as usable for the thesis if QUEUE OVERFLOW dominates.

  PHASE 2 -- only in cells classified 'congestion', measures the matched queue
  ablation (spbp_ab_full vs spbp_ab_noqueue). Refuses to report a queue value
  anywhere else, because that number is meaningless when another channel
  dominates -- which is exactly the error this script exists to correct.

USAGE
    python src\\map_operating_regimes.py --max_workers 16
    python src\\map_operating_regimes.py --max_workers 16 \\
        --durations 40 100 200 --rates 5 10 20 40 80 --seeds 3
"""
import argparse, json, math, os, sys, time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import experiment_spbp_mechanism      # noqa: F401 -- registers spbp_ab_*
from simulator_v2 import (FANETSimulatorV2, INITIAL_ENERGY, EnergyModel,
                          SLOTS_PER_FRAME, FRAME_DT)
from config_v2 import BASE, SCENARIOS, provenance

# a cell is only usable for the thesis if queue overflow is the dominant
# ROUTING-relevant loss AND non-routing channels are not swamping it
ENERGY_VETO = 0.10      # >10% of generated lost to energy -> regime is 'energy'
DOMINANCE = 1.0         # q_ovf must at least match the next channel to 'win'


def _run(job):
    sc, cfg, dur, rate, seed, actor, alt = job
    base = {**BASE, 'duration': dur, 'z_min': alt[0], 'z_max': alt[1]}
    s = FANETSimulatorV2({**base, **cfg, 'packet_rate': rate,
                          'seed': seed, 'actor': actor})
    m = s.run()
    ph = m.get('n_phantom_slots')
    if ph is None:
        raise RuntimeError("n_phantom_slots missing -- v12 not applied.")
    if ph != 0:
        raise RuntimeError(f"{ph} phantom slot(s) at {sc} dur={dur} rate={rate}")
    dead = sum(1 for e in s.energy if e <= 0.01)
    gen = max(m['n_generated'], 1)
    d = m['drop_reasons']
    return {'scenario': sc, 'duration': dur, 'rate': rate, 'seed': seed,
            'actor': actor, 'pdr': m['network_pdr'],
            'n_generated': m['n_generated'], 'n_delivered': m['n_delivered'],
            'dead_nodes': dead, 'n_nodes': s.N,
            'drops': {k: int(v) for k, v in d.items()},
            'share': {k: v / gen for k, v in d.items()}}


def classify(share, dead_frac):
    """Which loss channel dominates? Only 'congestion' cells are usable."""
    e = share.get('energy_depleted', 0.0)
    q = share.get('queue_overflow', 0.0)
    l = share.get('link_error', 0.0)
    n = share.get('no_route', 0.0)
    total = e + q + l + n
    if total < 0.02:
        return 'clean'                    # almost nothing is being lost
    if e >= ENERGY_VETO or dead_frac >= 0.10:
        return 'energy'                   # battery death swamps everything
    if q >= max(l, n) * DOMINANCE and q > 0.02:
        return 'congestion'               # THE regime the thesis needs
    if n >= max(l, q):
        return 'partition'
    return 'interference'


def paired(diffs):
    n = len(diffs)
    if n < 2:
        return (diffs[0] if diffs else float('nan')), float('nan'), float('nan')
    mean = sum(diffs) / n
    var = sum((x - mean) ** 2 for x in diffs) / (n - 1)
    se = math.sqrt(var / n)
    t = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447,
         8: 2.365, 9: 2.306, 10: 2.262, 15: 2.145}.get(n, 1.96)
    return mean, mean - t * se, mean + t * se


def run_batch(jobs, workers, label):
    t0, rows, done = time.time(), [], 0
    print(f"  {label}: {len(jobs)} episodes")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_run, j) for j in jobs]
        for f in as_completed(futs):
            rows.append(f.result()); done += 1
            if done % max(1, len(jobs) // 8) == 0:
                el = time.time() - t0
                print(f"    {done}/{len(jobs)}  ({el:.0f}s, "
                      f"~{el/done*(len(jobs)-done):.0f}s left)")
    return rows


def energy_ceiling(cfg, dur, hops=2.2):
    """Max sustainable per-flow rate under PERFECT load spreading."""
    N = cfg['num_drones']; flows = max(1, N // 4)
    budget = N * INITIAL_ENERGY
    per_pkt = hops * (EnergyModel.TX_COST + EnergyModel.RX_COST)
    return (budget / per_pkt) / (max(dur - 10, 1) * flows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', default='dense_slow')
    ap.add_argument('--durations', type=float, nargs='+', default=[40., 100., 200.])
    ap.add_argument('--rates', type=float, nargs='+', default=[5., 10., 20., 40., 80.])
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--z_min', type=float, default=100)
    ap.add_argument('--z_max', type=float, default=300)
    ap.add_argument('--max_workers', type=int, default=None)
    ap.add_argument('--out', default='results/operating_regimes.json')
    args = ap.parse_args()

    seeds = list(range(1, args.seeds + 1))
    cfg = SCENARIOS[args.scenario]
    alt = (args.z_min, args.z_max)
    nflows = max(1, cfg['num_drones'] // 4)
    cap = SLOTS_PER_FRAME / FRAME_DT

    print('=' * 100)
    print('  OPERATING-REGIME MAP -- does a congestion regime exist at all?')
    print('=' * 100)
    print(f"  scenario {args.scenario}  N={cfg['num_drones']}  flows={nflows}  "
          f"alt {alt[0]:.0f}-{alt[1]:.0f}m  seeds {args.seeds}")
    print(f"  per-node service capacity = {cap:.0f} pkt/s")
    print(f"  node dies after {INITIAL_ENERGY/EnergyModel.TX_COST:.0f} transmit attempts")
    print('\n  ANALYTICAL ENERGY CEILING (max per-flow rate, PERFECT spreading):')
    for dur in list(args.durations) + [1000.0]:
        ec = energy_ceiling(cfg, dur)
        tag = '   <- DATASET TARGET DURATION' if dur == 1000.0 else ''
        print(f"     dur {dur:>6.0f}s  ->  per-flow rate <= {ec:6.2f}{tag}")
    print('  Real traffic is skewed, so true ceilings are LOWER than these.')

    # ---------------- PHASE 1 : regime map ------------------------------
    print('\n' + '-' * 100)
    print('  PHASE 1 -- regime map (spbp), FULL loss accounting')
    print('-' * 100)
    jobs = [(args.scenario, cfg, d, r, s, 'spbp', alt)
            for d in args.durations for r in args.rates for s in seeds]
    rows = run_batch(jobs, args.max_workers, 'phase 1')

    cells = defaultdict(list)
    for r in rows:
        cells[(r['duration'], r['rate'])].append(r)

    print(f"\n  {'dur':>6}{'rate':>6}{'offer':>7}{'pdr':>7}"
          f"{'energy':>8}{'q_ovf':>7}{'link':>7}{'noRt':>6}{'dead':>7}  regime")
    print('  ' + '-' * 78)
    regime_map = {}
    for key in sorted(cells):
        dur, rate = key
        g = cells[key]
        def sh(k): return sum(x['share'].get(k, 0.) for x in g) / len(g)
        pdr = sum(x['pdr'] for x in g) / len(g)
        dead = sum(x['dead_nodes'] for x in g) / len(g)
        dfrac = dead / g[0]['n_nodes']
        share = {k: sh(k) for k in
                 ('energy_depleted', 'queue_overflow', 'link_error', 'no_route')}
        reg = classify(share, dfrac)
        regime_map[f'{dur}|{rate}'] = dict(duration=dur, rate=rate, pdr=pdr,
                                           dead_nodes=dead, regime=reg, **share)
        mark = '  <== USABLE' if reg == 'congestion' else ''
        print(f"  {dur:>6.0f}{rate:>6.0f}{nflows*rate:>7.0f}{pdr:>7.3f}"
              f"{share['energy_depleted']:>8.3f}{share['queue_overflow']:>7.3f}"
              f"{share['link_error']:>7.3f}{share['no_route']:>6.3f}"
              f"{dead:>6.1f}  {reg}{mark}")

    usable = [v for v in regime_map.values() if v['regime'] == 'congestion']
    print(f"\n  cells where QUEUE OVERFLOW dominates: {len(usable)}/{len(regime_map)}")
    if not usable:
        print("  *** NO CONGESTION REGIME FOUND anywhere in this map. ***")
        print("  The network goes clean -> interference/energy without passing")
        print("  through a congestion-limited regime. If this holds across")
        print("  scenarios, the thesis premise -- that congestion-aware routing")
        print("  matters -- has no operating point in this simulator as configured,")
        print("  and the ENERGY MODEL (not the rate grid) is what must change.")
    else:
        for v in usable:
            print(f"     dur {v['duration']:>5.0f}  rate {v['rate']:>4.0f}  "
                  f"q_ovf {v['queue_overflow']:.3f}  pdr {v['pdr']:.3f}")

    # ---------------- PHASE 2 : queue value, ONLY where valid -----------
    phase2 = []
    if usable:
        print('\n' + '-' * 100)
        print('  PHASE 2 -- matched queue ablation, ONLY in congestion cells')
        print('-' * 100)
        jobs2 = [(args.scenario, cfg, v['duration'], v['rate'], s, a, alt)
                 for v in usable for s in seeds
                 for a in ('spbp_ab_full', 'spbp_ab_noqueue')]
        rows2 = run_batch(jobs2, args.max_workers, 'phase 2')
        p = defaultdict(dict)
        for r in rows2:
            p[(r['duration'], r['rate'], r['actor'])][r['seed']] = r['pdr']
        print(f"\n  {'dur':>6}{'rate':>6}{'ab_full':>9}{'ab_noq':>9}"
              f"{'QUEUE VAL':>11}{'95% CI':>20}{'cons':>6}")
        print('  ' + '-' * 62)
        for v in usable:
            f = p[(v['duration'], v['rate'], 'spbp_ab_full')]
            n = p[(v['duration'], v['rate'], 'spbp_ab_noqueue')]
            ss = sorted(set(f) & set(n))
            d = [f[s] - n[s] for s in ss]
            qv, lo, hi = paired(d)
            cons = sum(1 for x in d if x > 0) / len(d)
            ci = f"[{100*lo:6.2f},{100*hi:6.2f}]" if lo == lo else "[   n/a       ]"
            phase2.append(dict(duration=v['duration'], rate=v['rate'],
                               queue_value=qv, ci_lo=lo, ci_hi=hi, consistency=cons))
            print(f"  {v['duration']:>6.0f}{v['rate']:>6.0f}"
                  f"{sum(f[s] for s in ss)/len(ss):>9.4f}"
                  f"{sum(n[s] for s in ss)/len(ss):>9.4f}"
                  f"{100*qv:>10.2f}p{ci:>20}{cons:>6.1f}")
    else:
        print('\n  PHASE 2 SKIPPED -- no congestion cell to measure in.')
        print('  Reporting a queue value from a non-congestion cell is exactly the')
        print('  error that produced the previous "effect is genuinely small" verdict.')

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({'schema': 'operating_regimes_v1', 'scenario': args.scenario,
                   'seeds': seeds, 'alt': list(alt),
                   'provenance': {**provenance(),
                                  'energy': {'initial': INITIAL_ENERGY,
                                             'tx': EnergyModel.TX_COST,
                                             'rx': EnergyModel.RX_COST}},
                   'regime_map': regime_map, 'phase2': phase2,
                   'rows': rows}, f, indent=2)
    print(f"\n  saved to {args.out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
