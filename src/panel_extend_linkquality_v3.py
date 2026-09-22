"""panel_extend_linkquality_v3.py -- add the two untested link-metric teachers.

WHAT THIS RUNS AND WHY

`lq_dijkstra` and `etx_dijkstra` sit in simulator_v2.TEACHERS and have never
been in a panel. lq_dijkstra's own docstring says it "REPLACES ETX-Dijkstra in
the panel" -- it was written specifically to be tested, and then never added to
the teacher list. That is the gap this closes.

It matters because link error dominates loss in every measured cell (1.3x to
6x queue overflow), and because the current best teacher (da_gpsr) carries a
+w_quality*link_quality term. Whether link-quality awareness or queue awareness
is da_gpsr's active ingredient is currently unknown -- lq_dijkstra isolates it:
shortest path weighted purely by link quality, with NO queue term at all.

    lq_dijkstra   w(v,u) = 1 / max(lq, eps)**alpha      bounded (eps floor)
    etx_dijkstra  w(v,u) = 1 / max(1-per, eps)          unbounded blow-up

Running both is a controlled pair: they differ essentially in whether the
metric is bounded. The lq_dijkstra docstring reports etx losing to plain
Dijkstra in every oracle cell (0.484 vs 0.532, 0.272 vs 0.305, 0.148 vs 0.179)
and attributes lq's advantage to the FLOOR rather than to inverting a different
probability -- but those numbers predate the corrected simulator, the corrected
operating point and the corrected rate grids, so they are not evidence about
this panel. Re-measuring both is what makes the claim testable here.

MERGE, NOT RE-RUN

panel_contenders_v2 stores per_seed_pdr, so only the two NEW teachers are
simulated (220 episodes, ~4.5h) and their results are merged into the existing
five. That merge is only sound if this run reproduces the stored simulator
behaviour exactly, so an EQUIVALENCE CONTROL runs first: one already-measured
teacher is re-simulated at one cell and must reproduce its stored per-seed PDRs
bit-for-bit. If it does not, the merge is refused rather than silently
producing a blended result from two different simulator states.

DECISION RULE -- the v25-corrected one

leading group = every teacher whose point estimate is within MIN_PRACTICAL_DELTA
of the top scorer; parsimony breaks ties inside that group. The superseded rule
(exclude only if significant AND >=1pp worse) let underpowered losers into the
group and picked teachers up to 4.40pp worse.

USAGE
    python src\\panel_extend_linkquality_v3.py --max_workers 8
"""
import argparse, json, math, os, sys, time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import experiment_spbp_mechanism          # noqa: F401
from simulator_v2 import FANETSimulatorV2
from config_v2 import BASE, SCENARIOS, get_suite, provenance

NEW_TEACHERS = ['lq_dijkstra', 'etx_dijkstra']
# simplest -> most complex, for the parsimony tiebreak. The two link-metric
# teachers sit just after plain dijkstra: both are single-metric shortest path,
# more complex than hop count, simpler than any multi-term score function.
COMPLEXITY = {'dijkstra': 0, 'lq_dijkstra': 1, 'etx_dijkstra': 2, 'gpsr': 3,
              'da_gpsr': 4, 'spbp_ab_noqueue': 5, 'spbp': 6}
ALPHA = 0.05
MIN_PRACTICAL_DELTA = 0.01
QUALIFIED_SCENARIOS = {'sparse_fast'}

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
            'pdr': m['network_pdr'], 'mean_hops': m.get('mean_hops')}


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


def equivalence_control(prior):
    """Re-simulate one ALREADY-MEASURED cell/teacher and require bit-identical
    reproduction of the stored per-seed PDRs. Without this, merging new results
    into old ones could silently blend two different simulator states."""
    print('\n' + '=' * 96)
    print('  EQUIVALENCE CONTROL -- is the stored panel reproducible right now?')
    print('=' * 96)
    key = 'dense_slow|60.0'
    ref_actor = 'dijkstra'
    stored = prior.get('per_seed_pdr', {}).get(key, {}).get(ref_actor)
    if not stored:
        print(f'  ERROR: no stored per-seed data for {key}/{ref_actor}')
        return False
    seeds = sorted(stored, key=int)[:3]          # 3 seeds is enough to catch drift
    print(f'  re-simulating {ref_actor} @ {key}, seeds {seeds} ...', flush=True)
    ok = True
    for s in seeds:
        got = episode('dense_slow', 60.0, int(s), ref_actor)['pdr']
        exp = stored[s]
        match = abs(got - exp) < 1e-12
        print(f"    seed {s}: stored={exp:.10f}  now={got:.10f}  "
              f"{'OK' if match else '*** DRIFT ***'}")
        ok &= match
    print(f"\n  {'MERGE IS SOUND' if ok else 'MERGE REFUSED -- simulator state differs from the stored run'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prior', default='results/panel_contenders_v2.json')
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--max_workers', type=int, default=8)
    ap.add_argument('--skip-equivalence', action='store_true')
    ap.add_argument('--out', default='results/panel_extended_v3.json')
    args = ap.parse_args()

    if not os.path.exists(args.prior):
        print(f'  ERROR: prior panel not found: {args.prior}'); return 1
    prior = json.load(open(args.prior))
    if 'per_seed_pdr' not in prior:
        print('  ERROR: prior has no per_seed_pdr -- cannot merge, re-run the full panel.')
        return 1

    n_cells = sum(len(r) for _, _, r in CELLS.values())
    n_eps = len(NEW_TEACHERS) * n_cells * args.seeds
    print('=' * 96)
    print('  PANEL EXTENSION v3 -- lq_dijkstra + etx_dijkstra')
    print('=' * 96)
    print(f'  prior: {args.prior}  ({len(prior["teachers"])} teachers already measured)')
    print(f'  new teachers: {NEW_TEACHERS}')
    print(f'  cells: {n_cells}   NEW episodes: {n_eps}   est. ~{n_eps*74/3600:.1f}h')
    print(f'  (merging instead of re-running saves '
          f'{len(prior["teachers"])*n_cells*args.seeds} episodes)')

    if not args.skip_equivalence and not equivalence_control(prior):
        print('\n  ABORTING -- the stored panel does not reproduce. Investigate before merging.')
        return 1

    jobs = [(s, r, seed, t) for s, (_, _, rates) in CELLS.items()
            for r in rates for seed in range(1, args.seeds + 1) for t in NEW_TEACHERS]
    t0, rows, done = time.time(), [], 0
    print(f'\n  running {len(jobs)} new episodes', flush=True)
    with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
        futs = [ex.submit(_job, j) for j in jobs]
        for f in as_completed(futs):
            rows.append(f.result()); done += 1
            if done % max(1, len(jobs) // 15) == 0:
                el = time.time() - t0
                print(f'    {done}/{len(jobs)}  ({el:.0f}s, '
                      f'~{el/done*(len(jobs)-done):.0f}s left)', flush=True)

    # ---- merge ----
    per_seed = {k: dict(v) for k, v in prior['per_seed_pdr'].items()}
    hops = defaultdict(dict)
    for k, cell in prior.get('results', {}).items():
        for t, h in (cell.get('mean_hops') or {}).items():
            hops[k][t] = h
    for r in rows:
        key = f"{r['scenario']}|{r['rate']}"
        per_seed.setdefault(key, {}).setdefault(r['actor'], {})[str(r['seed'])] = r['pdr']
        hops[key].setdefault(r['actor'], [])
    newhops = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r['mean_hops'] == r['mean_hops']:
            newhops[f"{r['scenario']}|{r['rate']}"][r['actor']].append(r['mean_hops'])
    for k, d2 in newhops.items():
        for t, v in d2.items():
            hops[k][t] = sum(v) / len(v)

    # ---- rank + test, v25-corrected rule ----
    cells_out, praw, tags = {}, [], []
    for key, per in per_seed.items():
        means = {t: sum(v.values()) / len(v) for t, v in per.items() if v}
        top = max(means, key=means.get)
        comps = {}
        for t in means:
            if t == top:
                continue
            seeds = sorted(set(per[top]) & set(per[t]))
            if len(seeds) < 2:
                continue
            pv = paired_ttest([per[t][s] for s in seeds], [per[top][s] for s in seeds])
            comps[t] = {'delta_vs_top': means[t] - means[top], 'p_raw': pv}
            if pv == pv:
                praw.append(pv); tags.append((key, t))
        cells_out[key] = {'means': means, 'top': top, 'comparisons': comps,
                          'mean_hops': dict(hops.get(key, {}))}

    adj = holm_dict(dict(zip(tags, praw))) if praw else {}
    for (key, t), _ in zip(tags, praw):
        cells_out[key]['comparisons'][t]['p_holm'] = adj.get((key, t), float('nan'))

    print('\n' + '=' * 96)
    print('  PER-CELL RESULTS  (all teachers, v25-corrected leading group)')
    print('=' * 96)
    assignment, newwins = {}, 0
    for scenario, (_, _, rates) in CELLS.items():
        for rate in rates:
            key = f'{scenario}|{rate}'
            if key not in cells_out:
                continue
            cr = cells_out[key]
            means, top = cr['means'], cr['top']
            lead = [t for t, m in means.items() if m >= means[top] - MIN_PRACTICAL_DELTA]
            rec = min(lead, key=lambda t: COMPLEXITY.get(t, 99))
            cr['leading_group'], cr['recommended'] = lead, rec
            flag = '   [FLAGGED]' if scenario in QUALIFIED_SCENARIOS else ''
            print(f'\n  {scenario} @ {rate}{flag}')
            print(f"    {'teacher':<20}{'mean PDR':>10}{'vs top':>10}{'p_holm':>11}{'hops':>8}")
            for t in sorted(means, key=means.get, reverse=True):
                h = cr['mean_hops'].get(t)
                hs = f'{h:.2f}' if isinstance(h, float) else '--'
                star = ' *NEW*' if t in NEW_TEACHERS else ''
                if t == top:
                    print(f"    {t:<20}{means[t]:>10.4f}{'  (top)':>10}{'':>11}{hs:>8}{star}")
                    continue
                c = cr['comparisons'].get(t, {})
                print(f"    {t:<20}{means[t]:>10.4f}"
                      f"{100*c.get('delta_vs_top',0):>+9.2f}p"
                      f"{c.get('p_holm',float('nan')):>11.4g}{hs:>8}{star}")
            print(f'    leading group: {lead}')
            print(f'    -> recommended: {rec}')
            if scenario not in QUALIFIED_SCENARIOS:
                assignment[key] = rec
                if rec in NEW_TEACHERS:
                    newwins += 1

    print('\n' + '=' * 96)
    print('  ORACLE ASSIGNMENT (extended panel)')
    print('=' * 96)
    for k in sorted(assignment):
        old = prior.get('oracle_assignment', {}).get(k, '?')
        ch = '   CHANGED' if assignment[k] != old else ''
        print(f'  {k:<24}{old:<18}-> {assignment[k]}{ch}')
    print(f'\n  cells won by a NEW teacher: {newwins}/{len(assignment)}')
    if newwins:
        print('  -> link-quality metric is competitive; da_gpsr\'s advantage is at')
        print('     least partly its link-quality term, not its queue term.')
    else:
        print('  -> neither link-metric teacher wins a cell. da_gpsr\'s advantage is')
        print('     NOT reducible to link quality alone -- the combination matters.')

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({'schema': 'panel_extended_v3',
                   'teachers': sorted(set(prior['teachers']) | set(NEW_TEACHERS)),
                   'new_teachers': NEW_TEACHERS, 'merged_from': args.prior,
                   'duration': DURATION, 'initial_energy': INITIAL_ENERGY,
                   'seeds': args.seeds,
                   'decision_rule': 'v25: leading group within 1pp of top, parsimony tiebreak',
                   'run_params': {'duration': DURATION, 'initial_energy': INITIAL_ENERGY,
                                  'seeds': args.seeds,
                                  'note': 'actual operating point of THIS run'},
                   'provenance': provenance(),
                   'per_seed_pdr': per_seed, 'results': cells_out,
                   'oracle_assignment': assignment}, f, indent=2)
    print(f'\n  saved to {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
