"""panel_contenders_v2.py -- settle the oracle question, with per-seed storage.

WHY THIS EXISTS

The generalized panel (v1) displaced spbp in 7 of 11 cells but left two gaps:

  1. spbp_ab_noqueue was NOT in the teacher set. Given the panel's own finding
     -- that the dominant loss is interference, that PDR rank is inverse hop
     rank, and that the LEAST queue-aware teachers win -- a pure no-queue
     variant of spbp is precisely the thing most likely to be competitive, and
     it never got a run.

  2. v1 saved only per-cell MEANS, not per-seed PDRs. That makes the result
     un-extendable: adding one teacher later cannot be paired against the
     stored data, forcing a re-run of everything it should have been comparable
     to. This script stores every per-seed PDR so a future teacher can be added
     for the cost of that teacher alone.

TEACHER SET (default 5, configurable)

  spbp_ab_noqueue  -- the gap being closed
  spbp             -- the displaced incumbent, kept as a reference point
  dijkstra         -- won all 3 medium_slow cells in v1
  gpsr             -- won dense_slow@100 in v1
  da_gpsr          -- won 3 cells in v1

backpressure, dpp, car and spbp_lookahead are excluded: all four lost in every
cell in v1, several by 10-28pp. Re-running them would spend hours confirming a
settled result. They remain in v1's record.

DECISION RULE -- changed from v1, deliberately

v1 treated spbp as the incumbent and asked "does anyone beat it?". spbp is no
longer the presumptive oracle, so that framing no longer fits. This script
instead:

  1. ranks teachers by mean PDR within each cell;
  2. tests the top-ranked teacher against every other, paired by seed, Holm
     corrected across the whole panel;
  3. reports the LEADING GROUP -- the top teacher plus everyone statistically
     indistinguishable from it;
  4. recommends the SIMPLEST member of the leading group, on parsimony.

Step 4 matters: if noqueue and gpsr are statistically tied, picking either on
argmax alone would be reading noise. Preferring the simpler policy is a stated
tiebreak, not a silent one.

sparse_fast is measured and reported but does NOT set an oracle -- its usable
cells sit against the elasticity ceiling (0.80/0.85 vs 0.05-0.47 elsewhere) and
its losses are neither link nor queue driven. v1 had this filter but the run
that produced results/panel_oracle_generalized.json did not apply it.

USAGE
    python src\\panel_contenders_v2.py --max_workers 8
    python src\\panel_contenders_v2.py --max_workers 8 --seeds 10
"""
import argparse, json, math, os, sys, time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import experiment_spbp_mechanism          # noqa: F401 -- registers spbp_ab_*
from simulator_v2 import FANETSimulatorV2
from config_v2 import BASE, SCENARIOS, get_suite, provenance

# ordered simplest -> most complex, used for the parsimony tiebreak
TEACHERS = ['dijkstra', 'gpsr', 'da_gpsr', 'spbp_ab_noqueue', 'spbp']
COMPLEXITY = {t: i for i, t in enumerate(TEACHERS)}

ALPHA = 0.05
# A teacher leaves the leading group only if it is BOTH significantly worse
# AND worse by a practically meaningful margin. v1 carried a 1pp margin; an
# early draft of v2 dropped it, which a synthetic test caught: two teachers
# separated by a tiny but perfectly consistent gap produce p=0 (zero paired
# variance) and would set the oracle on a ~0.1pp difference. Significance
# alone is not sufficient grounds to prefer one teacher over another.
MIN_PRACTICAL_DELTA = 0.01   # 1pp
QUALIFIED_SCENARIOS = {'sparse_fast'}   # reported, never oracle-setting

CELLS = {
    'dense_slow':  ('suiteA', 'dense_slow',  [60.0, 80.0, 100.0]),
    'very_dense':  ('suiteA', 'very_dense',  [60.0, 80.0]),
    'medium_slow': ('suiteA', 'medium_slow', [40.0, 60.0, 80.0]),
    'sparse_fast': ('suiteA', 'sparse_fast', [80.0, 100.0]),
    'sink_50':     ('convergecast', 'sink_50', [30.0]),
}
DURATION = 1000.0
INITIAL_ENERGY = 8000.0


def scenario_cfg(suite, name):
    return SCENARIOS[name] if suite == 'suiteA' else get_suite('convergecast')[name]


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
            'n_delivered': m['n_delivered'], 'mean_hops': m.get('mean_hops')}


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
    t = mean / math.sqrt(var / n)
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


def self_tests():
    print('\n' + '=' * 96)
    print('  SELF-TESTS')
    print('=' * 96)
    fails = []
    print('  running S1/S2 (v12 present, determinism) ...', flush=True)
    a = episode('dense_slow', 60.0, 1, 'spbp')
    b = episode('dense_slow', 60.0, 1, 'spbp')
    ok = a['pdr'] == b['pdr']
    print(f"  S1/S2 v12 + determinism         : {'OK' if ok else '*** FAIL ***'}")
    if not ok:
        fails.append('determinism')
    print(f'  running S3 (all {len(TEACHERS)} teachers see identical traffic) ...', flush=True)
    gens = {t: episode('dense_slow', 60.0, 1, t)['n_generated'] for t in TEACHERS}
    ok = len(set(gens.values())) == 1
    print(f"  S3 paired-seed traffic identity : {'OK' if ok else '*** DIFFER: ' + str(gens) + ' ***'}")
    if not ok:
        fails.append('traffic identity')
    # S4: noqueue must actually be registered and distinct from spbp
    nq = episode('dense_slow', 60.0, 1, 'spbp_ab_noqueue')['pdr']
    sp = episode('dense_slow', 60.0, 1, 'spbp')['pdr']
    ok = nq != sp
    print(f"  S4 noqueue distinct from spbp   : {nq:.4f} vs {sp:.4f}  "
          f"{'OK' if ok else '*** IDENTICAL -- ablation not registered ***'}")
    if not ok:
        fails.append('noqueue not distinct')
    print('\n  ' + ('SELF-TESTS PASSED' if not fails else f'FAILED: {fails}'))
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--max_workers', type=int, default=8)
    ap.add_argument('--teachers', nargs='+', default=None)
    ap.add_argument('--skip-self-test', action='store_true')
    ap.add_argument('--out', default='results/panel_contenders_v2.json')
    args = ap.parse_args()

    global TEACHERS
    if args.teachers:
        TEACHERS = args.teachers

    n_cells = sum(len(r) for _, _, r in CELLS.values())
    n_eps = len(TEACHERS) * n_cells * args.seeds
    print('=' * 96)
    print('  CONTENDER PANEL v2 -- closing the noqueue gap, per-seed storage')
    print('=' * 96)
    print(f'  duration={DURATION}s  initial_energy={INITIAL_ENERGY}  seeds={args.seeds}')
    print(f'  teachers ({len(TEACHERS)}): {TEACHERS}')
    print(f'  cells: {n_cells}   episodes: {n_eps}   est. ~{n_eps*74/3600:.1f}h at 8 workers')
    print(f'  leading group = top + all within {100*MIN_PRACTICAL_DELTA:.0f}pp or not significantly worse')
    print(f'  oracle-setting cells: '
          f'{sum(len(r) for s,(_,_,r) in CELLS.items() if s not in QUALIFIED_SCENARIOS)}'
          f'   flagged (reported only): '
          f'{sum(len(r) for s,(_,_,r) in CELLS.items() if s in QUALIFIED_SCENARIOS)}')

    if not args.skip_self_test and self_tests():
        print('\n  ABORTING -- self-tests failed.')
        return 1

    jobs = [(s, r, seed, t) for s, (_, _, rates) in CELLS.items()
            for r in rates for seed in range(1, args.seeds + 1) for t in TEACHERS]
    t0, rows, done = time.time(), [], 0
    print(f'\n  panel: {len(jobs)} episodes', flush=True)
    with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
        futs = [ex.submit(_job, j) for j in jobs]
        for f in as_completed(futs):
            rows.append(f.result()); done += 1
            if done % max(1, len(jobs) // 20) == 0:
                el = time.time() - t0
                print(f'    {done}/{len(jobs)}  ({el:.0f}s, '
                      f'~{el/done*(len(jobs)-done):.0f}s left)', flush=True)

    # ---- per-seed storage (the v1 flaw this fixes) ----
    per_seed = defaultdict(lambda: defaultdict(dict))
    hops = defaultdict(lambda: defaultdict(list))
    for r in rows:
        per_seed[f"{r['scenario']}|{r['rate']}"][r['actor']][str(r['seed'])] = r['pdr']
        if r['mean_hops'] == r['mean_hops']:
            hops[f"{r['scenario']}|{r['rate']}"][r['actor']].append(r['mean_hops'])

    # ---- rank, then test top vs all others ----
    cell_out, praw, tags = {}, [], []
    for key, per in per_seed.items():
        means = {t: sum(v.values()) / len(v) for t, v in per.items() if v}
        ranked = sorted(means, key=means.get, reverse=True)
        top = ranked[0]
        comps = {}
        for t in ranked[1:]:
            seeds = sorted(set(per[top]) & set(per[t]))
            if len(seeds) < 2:
                continue
            pv = paired_ttest([per[top][s] for s in seeds], [per[t][s] for s in seeds])
            comps[t] = {'delta_vs_top': means[t] - means[top], 'p_raw': pv,
                        'n_pairs': len(seeds)}
            if pv == pv:
                praw.append(pv); tags.append((key, t))
        cell_out[key] = {'means': means, 'ranked': ranked, 'top': top,
                         'comparisons': comps,
                         'mean_hops': {t: (sum(v) / len(v) if v else None)
                                       for t, v in hops[key].items()}}

    adj = holm_dict(dict(zip(tags, praw))) if praw else {}
    for (key, t), _ in zip(tags, praw):
        c = cell_out[key]['comparisons'][t]
        c['p_holm'] = adj.get((key, t), float('nan'))
        c['significant'] = bool(c['p_holm'] < ALPHA)
        c['practically_worse'] = bool(c['delta_vs_top'] <= -MIN_PRACTICAL_DELTA)
        # v25: membership by EFFECT SIZE, not by failure to reject.
        # The previous rule (significant AND practically_worse) let a teacher
        # 4.40pp worse stay in the leading group whenever the paired test was
        # underpowered at n=10 -- parsimony then picked it for being simplest.
        # "Not significantly worse" is not "equally good". Significance is
        # still computed and printed, but as confirmation, not as the gate.
        c['sig_worse_than_top'] = bool(c['practically_worse'])

    # ---- leading group + parsimony recommendation ----
    for key, cr in cell_out.items():
        # v25: leading group = every teacher whose POINT ESTIMATE sits within
        # MIN_PRACTICAL_DELTA of the top scorer.
        lead = [cr['top']] + [t for t, c in cr['comparisons'].items()
                              if c.get('delta_vs_top', -1) > -MIN_PRACTICAL_DELTA]
        cr['leading_group'] = lead
        cr['recommended'] = min(lead, key=lambda t: COMPLEXITY.get(t, 99))

    # ---- report ----
    print('\n' + '=' * 96)
    print('  PER-CELL RESULTS  (ranked by mean PDR; tested against the top)')
    print('=' * 96)
    for scenario, (_, _, rates) in CELLS.items():
        for rate in rates:
            key = f'{scenario}|{rate}'
            if key not in cell_out:
                continue
            cr = cell_out[key]
            flag = '   [FLAGGED -- reported, no oracle set]' if scenario in QUALIFIED_SCENARIOS else ''
            print(f'\n  {scenario} @ rate {rate}{flag}')
            print(f"    {'teacher':<20}{'mean PDR':>10}{'vs top':>10}{'p_holm':>11}{'hops':>8}   ")
            for t in cr['ranked']:
                m = cr['means'][t]
                h = cr['mean_hops'].get(t)
                hs = f'{h:.2f}' if h else '--'
                if t == cr['top']:
                    print(f"    {t:<20}{m:>10.4f}{'  (top)':>10}{'':>11}{hs:>8}")
                    continue
                c = cr['comparisons'].get(t, {})
                if c.get('sig_worse_than_top'):
                    mark = ''
                elif c.get('significant'):
                    mark = '  ~tied (sig, but <1pp)'
                else:
                    mark = '  ~tied'
                print(f"    {t:<20}{m:>10.4f}{100*c.get('delta_vs_top',0):>+9.2f}p"
                      f"{c.get('p_holm',float('nan')):>11.4g}{hs:>8}{mark}")
            print(f"    leading group: {cr['leading_group']}")
            print(f"    -> recommended (parsimony): {cr['recommended']}")

    print('\n' + '=' * 96)
    print('  ORACLE ASSIGNMENT')
    print('=' * 96)
    assign = {}
    for scenario, (_, _, rates) in CELLS.items():
        for rate in rates:
            key = f'{scenario}|{rate}'
            if key not in cell_out:
                continue
            if scenario in QUALIFIED_SCENARIOS:
                print(f'  {key:<24} FLAGGED -- best was {cell_out[key]["top"]}, no oracle set')
                continue
            assign[key] = cell_out[key]['recommended']
            print(f'  {key:<24} -> {assign[key]}')

    nq_wins = [k for k, v in assign.items() if v == 'spbp_ab_noqueue']
    print(f"\n  cells where spbp_ab_noqueue is recommended: {len(nq_wins)}/{len(assign)}")
    if nq_wins:
        print(f'    {nq_wins}')
        print('  -> H1 supported: removing the queue term rescues the SP-BP family.')
    else:
        print('  -> H2 supported: even without its queue term, the SP-BP family')
        print('     loses to simpler teachers. The family is not the right oracle.')

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({'schema': 'panel_contenders_v2', 'duration': DURATION,
                   'initial_energy': INITIAL_ENERGY, 'teachers': TEACHERS,
                   'seeds': args.seeds, 'alpha': ALPHA,
                   'qualified_scenarios': sorted(QUALIFIED_SCENARIOS),
                   'run_params': {'duration': DURATION, 'initial_energy': INITIAL_ENERGY,
                                  'seeds': args.seeds,
                                  'note': 'actual operating point of THIS run; the '
                                          'provenance block reports config_v2.BASE, '
                                          'the parity reference'},
                   'provenance': provenance(),
                   'per_seed_pdr': {k: dict(v) for k, v in per_seed.items()},
                   'results': cell_out, 'oracle_assignment': assign}, f, indent=2)
    print(f'\n  saved to {args.out}')
    print('  (per_seed_pdr is stored -- a future teacher can be added for the')
    print('   cost of that teacher alone, without re-running this panel)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
