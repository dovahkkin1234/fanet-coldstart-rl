"""panel_oracle_generalized_v1.py -- the 8-teacher (+candqueue) panel, at the
TRUE final operating point, using per-scenario rate grids measured directly
(not assumed) by the Phase-1 band searches this script depends on.

FRESH, SELF-CONTAINED. panel_convergecast_oracle.py was never committed to the
tracked repo and this session has no verified copy of it to patch against --
same situation find_congestion_band_convergecast.py was written fresh to
avoid. This reuses that script's PROVEN comparison logic (the v16-fixed
delta = means[t] - means[incumbent], not the sign-flip bug that shipped
before v16) rather than reinventing it.

WHY A NINTH CANDIDATE

spbp_ab_candqueue is not one of the original 8 named teachers -- it is an
ablation variant of spbp built for mechanistic investigation. It has now been
measured, at the true 1000s/battery=8000 operating point, across all 5
scenarios this project studies:

    very_dense    candqueue best, every rate, consistency 1.00
    dense_slow    candqueue HELPS at 60/80, neutral at 100 (full only hurts-or-ns)
    convergecast  candqueue best of three, significant
    sparse_fast   candqueue least-negative, not yet significant
    medium_slow   candqueue still hurts, but LESS than full -- the one exception

5/5: never worse than full, best in 4/5. That is strong enough to test
directly as a ninth oracle candidate here, rather than as a separate pass
after this panel returns.

RATE GRIDS -- MEASURED, NOT ASSUMED

Per-scenario usable rates below are exactly what the Phase-1 band searches
found at 1000s/battery=8000 (Suite A scenarios) and 1000s/battery=8000
(convergecast). sink_50's grid is a single rate (30.0) because every higher
rate tested showed negative elasticity -- the single-sink bottleneck
saturates over a long-enough duration, a distinct mechanism from energy
depletion (sinkE% stayed >70% throughout; this is a queueing, not a battery,
ceiling).

DECISION RULE (same as the original convergecast panel, post-v16 fix)

spbp is the INCUMBENT. A challenger replaces it in a given cell only if it
beats spbp by >=1pp AND is significant after Holm correction. Argmax alone is
never used -- ties or near-ties would otherwise read as wins.

USAGE
    python panel_oracle_generalized_v1.py --max_workers 8
    python panel_oracle_generalized_v1.py --max_workers 8 --seeds 10
"""
import argparse, json, math, os, sys, time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import experiment_spbp_mechanism          # noqa: F401 -- registers spbp_ab_*
from simulator_v2 import FANETSimulatorV2
from config_v2 import BASE, SCENARIOS, get_suite, provenance

TEACHERS = ['dijkstra', 'gpsr', 'backpressure', 'spbp', 'da_gpsr', 'dpp',
            'car', 'spbp_lookahead', 'spbp_ab_candqueue']
INCUMBENT = 'spbp'
ALPHA = 0.05
REPLACE_MARGIN = 0.01   # 1pp

# scenario -> (suite, cfg-lookup, rates)  -- rates are exactly what Phase 1 found usable
CELLS = {
    'dense_slow':  ('suiteA', 'dense_slow',  [60.0, 80.0, 100.0]),
    'very_dense':  ('suiteA', 'very_dense',  [60.0, 80.0]),
    'medium_slow': ('suiteA', 'medium_slow', [40.0, 60.0, 80.0]),
    # sparse_fast is INCLUDED but flagged: its two "usable" cells have
    # elasticity 0.80/0.85, both pressed against the ELASTIC_HI=0.85 ceiling,
    # while every other scenario's usable cells sit at 0.05-0.47. Its
    # link_error is ~0.007 (near zero) against a PDR of ~0.30, so its dominant
    # loss is neither link nor queue -- most likely route-unavailability,
    # matching this scenario's original connectivity-limited characterisation.
    # Its results are reported SEPARATELY in the verdict and are NOT used to
    # set a per-cell oracle, because "which queue policy wins" is not a
    # well-posed question in a cell whose losses are not queue-driven.
    'sparse_fast': ('suiteA', 'sparse_fast', [80.0, 100.0]),
    'sink_50':     ('convergecast', 'sink_50', [30.0]),
}
# Scenarios whose cells are reported but NOT used to set a per-cell oracle.
QUALIFIED_SCENARIOS = {'sparse_fast'}

DURATION = 1000.0
INITIAL_ENERGY = 8000.0


def scenario_cfg(suite, name):
    if suite == 'suiteA':
        return SCENARIOS[name]
    return get_suite('convergecast')[name]


def episode(scenario, rate, seed, actor):
    suite, cfg_name, _ = CELLS[scenario]
    cfg = {**BASE, **scenario_cfg(suite, cfg_name), 'duration': DURATION,
           'initial_energy': INITIAL_ENERGY, 'packet_rate': rate,
           'seed': seed, 'actor': actor}
    s = FANETSimulatorV2(cfg)
    m = s.run()
    ph = m.get('n_phantom_slots')
    if ph is None:
        raise RuntimeError('n_phantom_slots missing -- v12 not applied')
    if ph != 0:
        raise RuntimeError(f'{ph} phantom slot(s): leak present, results void')
    return {'scenario': scenario, 'rate': rate, 'seed': seed, 'actor': actor,
            'pdr': m['network_pdr'], 'n_generated': m['n_generated'],
            'n_delivered': m['n_delivered']}


def _job(a):
    return episode(*a)


def paired_ttest(x, y):
    d = [xi - yi for xi, yi in zip(x, y)]
    n = len(d)
    if n < 2:
        return float('nan')
    mean = sum(d) / n
    var = sum((v - mean) ** 2 for v in d) / (n - 1)
    if var == 0:
        return 0.0 if mean != 0 else 1.0
    se = math.sqrt(var / n)
    t = mean / se
    try:
        from scipy import stats
        return float(2 * stats.t.sf(abs(t), n - 1))
    except Exception:
        return float('nan')


def holm_dict(pmap):
    keys = list(pmap)
    vals = [pmap[k] for k in keys]
    idx = sorted(range(len(vals)), key=lambda i: vals[i])
    m, adj, run = len(vals), [0.0] * len(vals), 0.0
    for k, i in enumerate(idx):
        run = max(run, (m - k) * vals[i])
        adj[i] = min(1.0, run)
    return dict(zip(keys, adj))


def run_batch(jobs, workers, label):
    t0, rows, done = time.time(), [], 0
    print(f'  {label}: {len(jobs)} episodes', flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_job, j) for j in jobs]
        for f in as_completed(futs):
            rows.append(f.result()); done += 1
            if done % max(1, len(jobs) // 20) == 0:
                el = time.time() - t0
                print(f'    {done}/{len(jobs)}  ({el:.0f}s, '
                      f'~{el/done*(len(jobs)-done):.0f}s left)', flush=True)
    return rows


def self_tests():
    print('\n' + '=' * 96)
    print('  SELF-TESTS')
    print('=' * 96)
    fails = []
    print('  running S1/S2 (v12 present, determinism) ...', flush=True)
    r1 = episode('dense_slow', 60.0, 1, INCUMBENT)
    r2 = episode('dense_slow', 60.0, 1, INCUMBENT)
    ok = r1['pdr'] == r2['pdr']
    print(f"  S1/S2 v12 present + determinism : {'OK' if ok else '*** FAIL ***'}")
    if not ok:
        fails.append('determinism')
    print('  running S3 (all 9 teachers see identical traffic) ...', flush=True)
    gens = {a: episode('dense_slow', 60.0, 1, a)['n_generated'] for a in TEACHERS}
    ok = len(set(gens.values())) == 1
    print(f"  S3 paired-seed traffic identity : {'OK' if ok else '*** DIFFER: ' + str(gens) + ' ***'}")
    if not ok:
        fails.append('traffic identity')
    print('\n  ' + ('SELF-TESTS PASSED' if not fails else f'FAILED: {fails}'))
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--max_workers', type=int, default=None)
    ap.add_argument('--skip-self-test', action='store_true')
    ap.add_argument('--out', default='results/panel_oracle_generalized.json')
    args = ap.parse_args()

    print('=' * 96)
    print('  GENERALIZED ORACLE PANEL -- Suite A + convergecast, true operating point')
    print('=' * 96)
    print(f'  duration={DURATION}s  initial_energy={INITIAL_ENERGY}  seeds={args.seeds}')
    print(f'  teachers ({len(TEACHERS)}): {TEACHERS}')
    total_cells = sum(len(rates) for _, _, rates in CELLS.values())
    print(f'  scenarios: {list(CELLS)}  ({total_cells} cells total)')
    for name, (_, _, rates) in CELLS.items():
        print(f'    {name:<14} rates {rates}')

    if not args.skip_self_test:
        if self_tests():
            print('\n  ABORTING -- self-tests failed.')
            return 1

    jobs = [(scenario, rate, seed, actor)
            for scenario, (_, _, rates) in CELLS.items()
            for rate in rates
            for seed in range(1, args.seeds + 1)
            for actor in TEACHERS]
    print(f'\n  total episodes: {len(jobs)}')
    rows = run_batch(jobs, args.max_workers, 'panel')

    by_cell = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        by_cell[(r['scenario'], r['rate'])][r['actor']][r['seed']] = r['pdr']

    cell_results, praw, tags = {}, [], []
    qualified_notes = {}
    for key, per in by_cell.items():
        means = {t: sum(v.values()) / len(v) for t, v in per.items() if v}
        winner = max(means, key=means.get)
        comps = {}
        for t in TEACHERS:
            if t == INCUMBENT or INCUMBENT not in per or t not in per:
                continue
            seeds = sorted(set(per[INCUMBENT]) & set(per[t]))
            if len(seeds) < 2:
                continue
            x = [per[t][s] for s in seeds]
            y = [per[INCUMBENT][s] for s in seeds]
            pv = paired_ttest(x, y)
            delta = means[t] - means[INCUMBENT]   # v16-correct: challenger minus INCUMBENT
            comps[t] = {'delta': delta, 'p_raw': pv, 'n_pairs': len(seeds)}
            if pv == pv:
                praw.append(pv); tags.append((key, t))
        cell_results[key] = {'means': means, 'winner': winner, 'comparisons': comps}

    holm_adj = holm_dict(dict(zip(tags, praw))) if praw else {}
    for (key, t), ph in zip(tags, praw):
        c = cell_results[key]['comparisons'][t]
        c['p_holm'] = holm_adj.get((key, t), float('nan'))
        c['significant'] = bool(c['p_holm'] < ALPHA)

    print('\n' + '=' * 96)
    print('  PER-CELL RESULTS')
    print('=' * 96)
    replacements = {}
    for scenario, (_, _, rates) in CELLS.items():
        for rate in rates:
            key = (scenario, rate)
            if key not in cell_results:
                continue
            cr = cell_results[key]
            print(f'\n  {scenario} @ rate {rate}')
            print(f"    {'teacher':<20}{'mean PDR':>10}{'vs spbp':>10}{'p_holm':>10}   ")
            for t in TEACHERS:
                if t not in cr['means']:
                    continue
                m = cr['means'][t]
                if t == INCUMBENT:
                    print(f"    {t:<20}{m:>10.4f}{'  (ref)':>10}")
                    continue
                c = cr['comparisons'].get(t, {})
                d = c.get('delta', float('nan'))
                ph = c.get('p_holm', float('nan'))
                sig = c.get('significant', False)
                flag = ' <-- BEATS spbp' if (sig and d >= REPLACE_MARGIN) else ''
                print(f"    {t:<20}{m:>10.4f}{100*d:>+9.2f}p{ph:>10.4g}{flag}")
            beats = [t for t, c in cr['comparisons'].items()
                     if c.get('significant') and c.get('delta', 0) >= REPLACE_MARGIN]
            if beats:
                best = max(beats, key=lambda t: cr['comparisons'][t]['delta'])
                if scenario in QUALIFIED_SCENARIOS:
                    print(f'    [{scenario} is flagged -- reported, not used to set an oracle]')
                    qualified_notes[key] = best
                else:
                    replacements[key] = best

    print('\n' + '=' * 96)
    print('  VERDICT -- per-cell oracle recommendation')
    print('=' * 96)
    if not replacements:
        print('  No cell shows a challenger beating spbp by >=1pp significantly.')
        print('  -> KEEP spbp as the incumbent oracle everywhere tested.')
    else:
        print('  spbp should be REPLACED in the following cells:')
        for (scenario, rate), teacher in replacements.items():
            print(f'    {scenario:<14} rate {rate:<6} -> {teacher}')
        untouched = [f'{s} rate {r}' for s, (_, _, rates) in CELLS.items()
                     for r in rates if (s, r) not in replacements
                     and s not in QUALIFIED_SCENARIOS]
        print(f'  spbp stands in every other cell: {untouched}')
    if qualified_notes:
        print('\n  FLAGGED cells (reported only, no oracle set -- see CELLS comment):')
        for (scenario, rate), teacher in qualified_notes.items():
            print(f'    {scenario:<14} rate {rate:<6} best was {teacher}')

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    serializable = {f'{s}|{r}': v for (s, r), v in cell_results.items()}
    with open(args.out, 'w') as f:
        json.dump({'schema': 'panel_oracle_generalized_v1', 'duration': DURATION,
                   'initial_energy': INITIAL_ENERGY, 'teachers': TEACHERS,
                   'cells': {k: v[2] for k, v in CELLS.items()},
                   'provenance': provenance(), 'results': serializable,
                   'replacements': {f'{s}|{r}': t for (s, r), t in replacements.items()}},
                  f, indent=2)
    print(f'\n  saved to {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
