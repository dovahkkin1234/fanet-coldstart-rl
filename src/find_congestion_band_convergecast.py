"""find_congestion_band_convergecast.py -- usable band for SINK scenarios.

FRESH, SELF-CONTAINED script -- not a patch. find_usable_band.py (Suite A's
version) was never committed to the tracked repo and this session's sandbox
copy of it is gone, so there is nothing to verify an anchor against. Rather
than guess at an anchor for a file I cannot see, this reimplements the same
proven logic (elasticity-based band detection, paired matched-queue ablation,
tiered Holm correction, the same self-tests) fresh, targeting
`get_suite('convergecast')` directly -- Suite A's version only ever looked up
`config_v2.SCENARIOS`, which does not contain sink_N scenarios at all.

WHY THIS IS NEEDED BEFORE THE ORACLE PANEL

The original convergecast oracle panel (panel_convergecast_oracle.py) used the
Suite A rate grid [0.02, 0.05, 0.10, 0.25, 0.40] -- the SAME stale grid class
G2's sweep just needed fixing for. Every rate showed "nearly lossless" PDR
(0.98-0.99) at every teacher, in every cell -- the panel could not have
detected an oracle difference because nothing was congested enough to create
one. That panel's "keep spbp" verdict is a null result on an under-stressed
grid, not a confirmed finding.

Convergecast's bottleneck is structurally different from Suite A's: ALL flows
funnel into ONE sink node, so the SINK's own per-node capacity
(SLOTS_PER_FRAME/FRAME_DT, ~100 pkt/s), not aggregate network capacity, is
what should saturate first. For sink_50 (flows = N//4 = 12), the sink's
incoming rate is 12 x per-flow-rate, so naive estimate: capacity binds around
rate ~= 100/12 ~= 8.3 -- an order of magnitude below anything tested before
and far below Suite A's usable rates (35-70). This is an ESTIMATE to bound the
first probe, not a result -- the whole point of this script is to measure it.

RUNS AT A SHORT DURATION FIRST, same strategy as Suite A: locate the band
cheaply (default 40s, matching the current G1/G2 parity reference), leaving
duration-transfer to 1000s/INITIAL_ENERGY=8000 as a follow-up once a band is
found, not assumed.

USAGE
    python find_congestion_band_convergecast.py --max_workers 16
    python find_congestion_band_convergecast.py --max_workers 16 --scenario sink_100
"""
import argparse, json, math, os, sys, time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import experiment_spbp_mechanism          # noqa: F401 -- registers spbp_ab_*
from simulator_v2 import FANETSimulatorV2
from config_v2 import BASE, get_suite, provenance

VARIANTS = ['spbp_ab_noqueue', 'spbp_ab_full',
            'spbp_ab_candqueue', 'spbp_ab_additive']
REFERENCE = 'spbp_ab_noqueue'
ALPHA = 0.05

ELASTIC_LO, ELASTIC_HI = 0.05, 0.85
MIN_QOVF = 0.02
MAX_ENERGY = 0.05
NO_COLLAPSE_TOL = 0.10
TIER1 = ['spbp_ab_full']
TIER2 = ['spbp_ab_candqueue', 'spbp_ab_additive']


def episode(scenario_cfg, dur, rate, seed, actor, energy=None):
    cfg = {**BASE, **scenario_cfg, 'duration': dur,
           'packet_rate': rate, 'seed': seed, 'actor': actor}
    if energy is not None:
        cfg['initial_energy'] = energy
    s = FANETSimulatorV2(cfg)
    m = s.run()
    ph = m.get('n_phantom_slots')
    if ph is None:
        raise RuntimeError('n_phantom_slots missing -- v12 not applied')
    if ph != 0:
        raise RuntimeError(f'{ph} phantom slot(s): leak present, results void')
    g = max(m['n_generated'], 1)
    d = m['drop_reasons']
    # FIX 1: sink-specific energy, not just the whole-network dead count.
    # Convergecast concentrates ALL flows onto one node -- at the top of this
    # script's default probe range, receive-side cost alone (RX_COST x
    # incoming attempts) can consume ~96% of the sink's default battery in
    # just 40s (measured: 12 flows x 20/s x 40s x RX_COST=0.01 = 96 units).
    # That is a DIFFERENT mechanism from Suite A's long-duration accumulation,
    # and the generic 'dead' count alone cannot distinguish "the sink
    # specifically is dying" (expected here) from "something else is".
    sink_id = scenario_cfg.get('sink_node', 0)
    # self.initial_energy is set by v14 -- pre-v14 it does not exist at all,
    # and s.initial_energy would raise a bare, unclear AttributeError instead
    # of a clean guard. Same style as the existing v12 phantom-slot check.
    if not hasattr(s, 'initial_energy'):
        raise RuntimeError('s.initial_energy missing -- apply_energy_config_v14.py '
                           'not applied. Needed for the sink-energy diagnostic.')
    return {'actor': actor, 'seed': seed, 'rate': rate, 'duration': dur,
            'pdr': m['network_pdr'], 'n_generated': m['n_generated'],
            'n_delivered': m['n_delivered'],
            'dead_nodes': sum(1 for e in s.energy if e <= 0.01),
            'sink_energy_frac': s.energy[sink_id] / max(s.initial_energy, 1e-9),
            'q_ovf': d.get('queue_overflow', 0) / g,
            'energy': d.get('energy_depleted', 0) / g,
            'link': d.get('link_error', 0) / g,
            'mean_hops': m.get('mean_hops')}


def _job(a):
    return episode(*a)


def paired_t(x, y):
    d = [xi - yi for xi, yi in zip(x, y)]
    n = len(d)
    if n < 2:
        return (d[0] if d else float('nan')), float('nan'), float('nan'), float('nan'), float('nan')
    mean = sum(d) / n
    var = sum((v - mean) ** 2 for v in d) / (n - 1)
    se = math.sqrt(var / n) if var > 0 else 0.0
    cons = sum(1 for v in d if v > 0) / n
    if se == 0:
        return mean, mean, mean, (0.0 if mean != 0 else 1.0), cons
    t = mean / se
    try:
        from scipy import stats
        return (mean, mean - stats.t.ppf(1 - ALPHA / 2, n - 1) * se,
                mean + stats.t.ppf(1 - ALPHA / 2, n - 1) * se,
                float(2 * stats.t.sf(abs(t), n - 1)), cons)
    except Exception:
        c = {5: 2.776, 10: 2.262, 15: 2.145, 20: 2.093, 30: 2.045}.get(n, 1.96)
        return mean, mean - c * se, mean + c * se, float('nan'), cons


def holm(p):
    idx = sorted(range(len(p)), key=lambda i: p[i])
    m, adj, run = len(p), [0.0] * len(p), 0.0
    for k, i in enumerate(idx):
        run = max(run, (m - k) * p[i])
        adj[i] = min(1.0, run)
    return adj


def holm_dict(pmap):
    keys = list(pmap)
    return dict(zip(keys, holm([pmap[k] for k in keys])))


def run_batch(jobs, workers, label):
    t0, rows, done = time.time(), [], 0
    print(f'  {label}: {len(jobs)} episodes')
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_job, j) for j in jobs]
        for f in as_completed(futs):
            rows.append(f.result()); done += 1
            if done % max(1, len(jobs) // 8) == 0:
                el = time.time() - t0
                print(f'    {done}/{len(jobs)}  ({el:.0f}s, '
                      f'~{el/done*(len(jobs)-done):.0f}s left)')
    return rows


def self_tests(scenario_cfg, dur, rate, energy):
    print('\n' + '=' * 92)
    print('  SELF-TESTS')
    print('=' * 92)
    # FIX: this used to run at args.rates[-1] (the sweep's highest, most
    # stressed rate, ~11s/episode here) across 6 of its 8 sequential episodes,
    # with ZERO progress printing -- a normal ~60-90s wait looked identical to
    # a hang. Confirmed by direct timing (sink_timing_probe.py, this machine):
    # 4.9s at rate=4 scaling smoothly to 11.0s at rate=20, no pathological
    # cliff anywhere -- it was never stuck, just silent. Two changes: print
    # BEFORE every episode, not just after; and the caller now passes a
    # cheaper rate (sweep midpoint) instead of the maximum, since self-tests
    # only need to ESTABLISH validity, not stress-test the worst case.
    fails = []
    print('  running S1 (v12 present) ...', flush=True)
    r1 = episode(scenario_cfg, dur, rate, 1, REFERENCE, energy)
    print('  S1 v12 present (phantom==0)     : OK')
    print('  running S2 (determinism) ...', flush=True)
    r2 = episode(scenario_cfg, dur, rate, 1, REFERENCE, energy)
    ok = r1['pdr'] == r2['pdr'] and r1['n_delivered'] == r2['n_delivered']
    print(f"  S2 determinism                  : {'OK' if ok else '*** FAIL ***'}")
    if not ok:
        fails.append('S2 determinism')
    print('  running S3 (paired-seed identity, 4 variants) ...', flush=True)
    gens = {}
    for a in VARIANTS:
        print(f'    ... {a}', flush=True)
        gens[a] = episode(scenario_cfg, dur, rate, 1, a, energy)['n_generated']
    ok = len(set(gens.values())) == 1
    print(f"  S3 paired-seed traffic identity : {'OK' if ok else '*** DIFFER: ' + str(gens) + ' ***'}")
    if not ok:
        fails.append('S3 paired-seed integrity')
    print('  running S4 (no collapse, half-rate vs rate) ...', flush=True)
    d_lo = episode(scenario_cfg, dur, rate * 0.5, 1, REFERENCE, energy)['n_delivered']
    d_hi = episode(scenario_cfg, dur, rate, 1, REFERENCE, energy)['n_delivered']
    drop = (d_lo - d_hi) / max(d_lo, 1)
    ok = drop <= NO_COLLAPSE_TOL
    print(f"  S4 no delivery collapse         : {d_lo} -> {d_hi}  "
          f"({100*drop:+.1f}%, tolerance {100*NO_COLLAPSE_TOL:.0f}%)  "
          f"{'OK' if ok else '*** COLLAPSE ***'}")
    if not ok:
        fails.append(f'S4 delivery collapsed by {100*drop:.1f}%')
    print('\n  ' + ('SELF-TESTS PASSED' if not fails else f'FAILED: {fails}'))
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', default='sink_50')
    ap.add_argument('--rates', type=float, nargs='+',
                    default=[2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0])
    ap.add_argument('--map-seeds', type=int, default=3)
    ap.add_argument('--measure-seeds', type=int, default=30)
    ap.add_argument('--duration', type=float, default=40.0)
    ap.add_argument('--initial_energy', type=float, default=None)
    ap.add_argument('--max_workers', type=int, default=None)
    ap.add_argument('--skip-self-test', action='store_true')
    ap.add_argument('--out', default='results/congestion_band_convergecast.json')
    args = ap.parse_args()

    try:
        suite = get_suite('convergecast')
    except RuntimeError as e:
        print(f"  ERROR: {e}"); return 1
    if args.scenario not in suite:
        print(f"  ERROR: {args.scenario} not in convergecast suite: {sorted(suite)}")
        return 1
    cfg = suite[args.scenario]
    nflows = max(1, cfg['num_drones'] // 4)
    mseeds = list(range(1, args.map_seeds + 1))

    print('=' * 92)
    print('  CONVERGECAST CONGESTION-BAND SEARCH')
    print('=' * 92)
    print(f"  scenario {args.scenario}  N={cfg['num_drones']}  flows={nflows}  "
          f"(all funnel to ONE sink)  dur {args.duration}s")
    print(f"  initial_energy: {args.initial_energy if args.initial_energy is not None else 'DEFAULT (100.0)'}")
    print(f"  estimate: sink saturates near rate ~= 100/{nflows} = {100/nflows:.1f} "
          f"(per-node capacity / flow count) -- UNVERIFIED, this run measures it")

    if len(args.rates) < 2:
        print("\n  ERROR: need >=2 rates for elasticity."); return 1

    if not args.skip_self_test:
        # FIX: was args.rates[-1] (the sweep's highest/most stressed rate) --
        # self-tests only need to establish validity, not run at the worst
        # case. Midpoint is enough and roughly halves the wait.
        if self_tests(cfg, args.duration, args.rates[len(args.rates) // 2], args.initial_energy):
            print('\n  ABORTING -- self-tests failed.'); return 1

    print('\n' + '-' * 92)
    print('  PHASE 1 -- delivery elasticity sweep')
    print('-' * 92)
    jobs = [(cfg, args.duration, r, s, REFERENCE, args.initial_energy)
            for r in args.rates for s in mseeds]
    rows = run_batch(jobs, args.max_workers, 'phase 1')

    agg = defaultdict(list)
    for r in rows:
        agg[r['rate']].append(r)
    curve = []
    for rate in args.rates:
        g = agg[rate]
        curve.append(dict(
            rate=rate, offered=nflows * rate,
            delivered=sum(x['n_delivered'] for x in g) / len(g),
            generated=sum(x['n_generated'] for x in g) / len(g),
            pdr=sum(x['pdr'] for x in g) / len(g),
            q_ovf=sum(x['q_ovf'] for x in g) / len(g),
            energy=sum(x['energy'] for x in g) / len(g),
            link=sum(x['link'] for x in g) / len(g),
            dead=sum(x['dead_nodes'] for x in g) / len(g),
            sink_energy_frac=sum(x['sink_energy_frac'] for x in g) / len(g)))

    for i, c in enumerate(curve):
        if i == 0:
            c['elasticity'] = float('nan')
        else:
            p = curve[i - 1]
            dd = (c['delivered'] - p['delivered']) / max(p['delivered'], 1)
            dr = (c['rate'] - p['rate']) / p['rate']
            c['elasticity'] = dd / dr if dr else float('nan')
        e = c['elasticity']
        c['usable'] = bool(e == e and ELASTIC_LO <= e <= ELASTIC_HI
                           and c['q_ovf'] >= MIN_QOVF and c['energy'] <= MAX_ENERGY
                           and c['dead'] == 0)

    print(f"\n  {'rate':>6}{'offer':>7}{'deliv':>8}{'pdr':>7}{'elastic':>9}"
          f"{'q_ovf':>8}{'energy':>8}{'dead':>6}{'sinkE%':>8}  verdict")
    print('  ' + '-' * 82)
    for c in curve:
        e = f"{c['elasticity']:>9.3f}" if c['elasticity'] == c['elasticity'] else f"{'--':>9}"
        v = 'USABLE' if c['usable'] else (
            'saturated' if (c['elasticity'] == c['elasticity'] and c['elasticity'] < ELASTIC_LO)
            else ('energy' if c['energy'] > MAX_ENERGY else
                  ('clean' if c['q_ovf'] < MIN_QOVF else '-')))
        # FIX 1 continued: flag explicitly when the SINK itself is the one
        # draining, distinct from generic dead-node counting.
        if c['sink_energy_frac'] < 0.20 and c['dead'] == 0:
            v += '  [sink draining]'
        print(f"  {c['rate']:>6.1f}{c['offered']:>7.1f}{c['delivered']:>8.0f}"
              f"{c['pdr']:>7.3f}{e}{c['q_ovf']:>8.3f}{c['energy']:>8.3f}"
              f"{c['dead']:>6.1f}{100*c['sink_energy_frac']:>7.1f}%  {v}")

    usable = [c for c in curve if c['usable']]
    print(f"\n  usable cells: {len(usable)}/{len(curve)}")
    results = {}
    if not usable:
        print('  *** NO USABLE BAND in this rate range. Extend --rates and re-run')
        print('      before concluding none exists -- the 100/flows estimate above')
        print('      is a rough guide, not a measurement. ***')
    else:
        print('  ' + ', '.join(f"rate {c['rate']:.1f}" for c in usable))

        seeds = list(range(1, args.measure_seeds + 1))
        print('\n' + '-' * 92)
        print('  PHASE 2 -- matched queue ablation, ONLY inside the usable band')
        print('-' * 92)
        jobs2 = [(cfg, args.duration, c['rate'], s, a, args.initial_energy)
                 for c in usable for s in seeds for a in VARIANTS]
        rows2 = run_batch(jobs2, args.max_workers, 'phase 2')
        by = defaultdict(dict)
        for r in rows2:
            by[(r['rate'], r['actor'])][r['seed']] = r
        praw, tags = [], []
        for c in usable:
            rate = c['rate']
            ref = by[(rate, REFERENCE)]
            print(f"\n  rate {rate:.1f}  |  elasticity {c['elasticity']:.3f}  "
                  f"q_ovf {c['q_ovf']:.3f}")
            print(f"  {'variant':<20}{'pdr':>9}{'deliv':>8}{'vs noqueue':>12}"
                  f"{'95% CI':>20}{'cons':>6}{'hops':>7}")
            print('  ' + '-' * 82)
            cell = {}
            for a in VARIANTS:
                cur = by[(rate, a)]
                ss = sorted(set(cur) & set(ref))
                x = [cur[s]['pdr'] for s in ss]
                y = [ref[s]['pdr'] for s in ss]
                mean, lo, hi, pv, cons = paired_t(x, y)
                # nan-safe mean_hops: a seed with zero deliveries for this
                # (rate, actor) cell reports mean_hops=nan (np.mean of an
                # empty list). A plain sum() is poisoned to nan by a SINGLE
                # such seed -- exactly the mechanism found once already in
                # Suite A (medium_slow: all four variants read nan despite
                # ~1200 average deliveries). Filter to finite values instead.
                hops_vals = [cur[s]['mean_hops'] for s in ss
                             if cur[s]['mean_hops'] == cur[s]['mean_hops']]
                cell[a] = dict(pdr=sum(x) / len(x), delta=mean, ci_lo=lo, ci_hi=hi,
                               p_raw=pv, consistency=cons,
                               delivered=sum(cur[s]['n_delivered'] for s in ss) / len(ss),
                               mean_hops=(sum(hops_vals) / len(hops_vals)
                                          if hops_vals else float('nan')),
                               n_zero_delivery_seeds=len(ss) - len(hops_vals))
                if a != REFERENCE and pv == pv:
                    praw.append(pv); tags.append((rate, a))
                ci = f"[{100*lo:6.2f},{100*hi:6.2f}]" if lo == lo else f"{'[  n/a  ]':>20}"
                print(f"  {a:<20}{cell[a]['pdr']:>9.4f}{cell[a]['delivered']:>8.0f}"
                      f"{100*mean:>11.2f}p{ci:>20}{cons:>6.2f}"
                      f"{cell[a]['mean_hops']:>7.2f}")
            results[str(rate)] = cell

        flat_adj = holm_dict(dict(zip(tags, praw))) if praw else {}
        def holm_over(names):
            fam = {(r, a): p for (r, a), p in zip(tags, praw) if a in names}
            return holm_dict(fam) if fam else {}
        tier1_adj, tier2_adj = holm_over(TIER1), holm_over(TIER2)
        for c in usable:
            for a in VARIANTS:
                if a == REFERENCE:
                    continue
                results[str(c['rate'])][a].setdefault('p_holm_flat', float('nan'))
                results[str(c['rate'])][a].setdefault('significant_flat', False)
                results[str(c['rate'])][a].setdefault('p_holm_tier1', float('nan'))
                results[str(c['rate'])][a].setdefault('significant_tier1', False)
                results[str(c['rate'])][a].setdefault('p_holm_tier2', float('nan'))
                results[str(c['rate'])][a].setdefault('significant_tier2', False)
                results[str(c['rate'])][a].setdefault('significant', False)
        for (rate, a), ph in zip(tags, praw):
            key = (rate, a)
            results[str(rate)][a]['p_holm_flat'] = flat_adj.get(key, float('nan'))
            results[str(rate)][a]['significant_flat'] = bool(flat_adj.get(key, 1.0) < ALPHA)
            if a in TIER1:
                results[str(rate)][a]['p_holm_tier1'] = tier1_adj.get(key, float('nan'))
                results[str(rate)][a]['significant_tier1'] = bool(tier1_adj.get(key, 1.0) < ALPHA)
            if a in TIER2:
                results[str(rate)][a]['p_holm_tier2'] = tier2_adj.get(key, float('nan'))
                results[str(rate)][a]['significant_tier2'] = bool(tier2_adj.get(key, 1.0) < ALPHA)
            results[str(rate)][a]['significant'] = (
                results[str(rate)][a].get('significant_tier1', False) if a in TIER1
                else results[str(rate)][a].get('significant_tier2', False))

        print('\n' + '=' * 92)
        print('  VERDICT -- is SP-BP still right for this convergecast cell?')
        print('=' * 92)
        for c in usable:
            rate = c['rate']; cell = results[str(rate)]
            print(f"\n  rate {rate:.1f}:")
            for a in VARIANTS:
                if a == REFERENCE:
                    continue
                cc = cell[a]
                tag = (f"tier1 p_holm={cc.get('p_holm_tier1', float('nan')):.4g}" if a in TIER1
                       else f"tier2 p_holm={cc.get('p_holm_tier2', float('nan')):.4g}")
                print(f"    {a:<20}{100*cc['delta']:+7.2f}pp   {tag}   "
                      f"{'SIG' if cc['significant'] else 'ns'}")
            beats = [a for a in TIER1 if cell[a]['delta'] > 0 and cell[a]['significant']]
            hurts = [a for a in TIER1 if cell[a]['delta'] < 0 and cell[a]['significant']]
            if beats:
                print(f"    spbp's queue term HELPS here.")
            elif hurts:
                print(f"    spbp's queue term HURTS here -- oracle needs reconsidering")
                print(f"    for this cell, same as three Suite A scenarios already found.")
            else:
                print(f"    not significant at this rate individually.")

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({'schema': 'congestion_band_convergecast_v1',
                   'scenario': args.scenario, 'duration': args.duration,
                   'initial_energy': args.initial_energy,
                   'criteria': {'elastic_lo': ELASTIC_LO, 'elastic_hi': ELASTIC_HI,
                                'min_qovf': MIN_QOVF, 'max_energy': MAX_ENERGY},
                   'provenance': provenance(), 'curve': curve,
                   'results': results}, f, indent=2)
    print(f'\n  saved to {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
