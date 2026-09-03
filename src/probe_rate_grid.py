"""probe_rate_grid.py -- choose the packet-rate grid from measurement.

Implements docs/PROBE_PREREGISTRATION.md exactly. Read that first; the thresholds
and the decision rule live there and are frozen. This file only executes them.

WHAT IT DOES
    Runs 4 scenarios x 6 candidate rates x N seeds x 6 actors at the NEW operating
    point (1000 s, altitude 100-300 m) and reports, per (scenario, rate):

      QUEUE VALUE     paired PDR_routable(spbp_ab_full) - PDR_routable(spbp_ab_noqueue)
                      with a paired 95% CI. PRIMARY metric -- a matched control,
                      identical code path, only the queue term neutralised.
      CARRIED LOAD    delivered pkt/s. Locates the collapse knee, bounds the grid top.
      ADDRESSABLE     the existing headroom metric. Continuity only. Never selects.

    then applies the frozen rule (validity gate -> knee -> anchor -> peak -> 5 rates)
    and emits the grid, or refuses to and says why.

WHY PAIRED SEEDS. Every actor runs the SAME seed list, so actor differences are paired
differences. Unpaired, between-seed variance would swamp a 3 pp effect at n=5.

WHY spbp_ab_noqueue AND NOT dijkstra. `spbp - dijkstra` conflates queue value with
link-quality value and the two move oppositely in load; measured at 40 s it falls with
load while the matched ablation does not. See the pre-registration §2.

COST. 4 x 6 x 5 x 6 = 720 episodes at 1000 s, roughly 1.4 h at 16 workers.

USAGE
    # smoke test at the OLD operating point (fast, ~5 min) -- verifies plumbing
    python src\\probe_rate_grid.py --smoke --max_workers 16

    # the real probe
    python src\\probe_rate_grid.py --max_workers 16 --out results\\probe_rate_grid.json
"""
import argparse, json, math, os, sys, time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import experiment_spbp_mechanism            # noqa: F401 -- registers spbp_ab_* actors
from experiment_headroom import HeadroomSimulator, ROUTING_ADDRESSABLE
from config_v2 import BASE, SCENARIOS, provenance

# ── frozen constants, from the pre-registration ────────────────────────────
PROBE_RATES = [0.02, 0.05, 0.10, 0.15, 0.25, 0.40]
ACTORS = ['spbp', 'spbp_ab_full', 'spbp_ab_noqueue', 'dijkstra', 'gpsr', 'random']
PRIMARY_HI, PRIMARY_LO = 'spbp_ab_full', 'spbp_ab_noqueue'

MIN_ROUTABLE = 200        # §4.1 validity
MIN_VS_RANDOM = 0.02      # §4.1 validity, 2 pp
BAND_STRONG = 0.08        # §4.2
BAND_USABLE = 0.03        # §4.2
GRID_SIZE = 5             # §4.3
PEAK_STEP_DISAGREE = 2    # §4.5


def _run(job):
    """One episode. Module-level for pickling under Windows spawn."""
    sc, cfg, rate, seed, actor, base = job
    full = {**base, **cfg, 'packet_rate': rate, 'seed': seed, 'actor': actor}
    m = HeadroomSimulator(full).run()
    addressable = sum(m['drops_routable'].get(c, 0) for c in ROUTING_ADDRESSABLE)
    return {
        'scenario': sc, 'rate': rate, 'seed': seed, 'actor': actor,
        'pdr_routable': m['pdr_routable'], 'pdr_raw': m['pdr_predrain'],
        'n_routable': m['n_routable'], 'n_delivered': m['n_delivered'],
        'n_generated': m['n_generated'],
        'addressable': addressable,
        'drops_routable': m['drops_routable'],
        'duration': base['duration'],
    }


def paired_ci(diffs):
    """Mean and 95% CI of a paired difference. t-approx; n is small by design."""
    n = len(diffs)
    if n == 0:
        return float('nan'), float('nan'), float('nan')
    mean = sum(diffs) / n
    if n == 1:
        return mean, float('nan'), float('nan')
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    se = math.sqrt(var / n)
    tcrit = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
             7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}.get(n, 1.96)
    return mean, mean - tcrit * se, mean + tcrit * se


MIN_CONSISTENCY = 0.8     # §4.2 -- paired sign consistency


def band(point, consistency):
    """Band from the POINT ESTIMATE plus paired sign consistency (pre-reg §4.2).

    Banding on the CI lower bound was tried and rejected: measured paired-diff SD
    is ~13.5 pp at 40 s, so at n=5 a 13 pp effect has a CI half-width of 16.7 pp
    and every cell reads 'floor' -- 6/6 smoke cells did. Sign consistency is
    robust at small n; the CI is still computed and reported.
    """
    if point != point:                       # nan
        return '?'
    if consistency < MIN_CONSISTENCY:
        return 'floor'                       # demoted: direction not consistent
    if point >= BAND_STRONG:
        return 'strong'
    if point >= BAND_USABLE:
        return 'usable'
    return 'floor'


def analyse(rows, rates):
    """Apply the frozen decision rule. Returns per-scenario analysis dicts."""
    by = defaultdict(dict)                   # (sc, rate) -> actor -> [rows by seed]
    for r in rows:
        by[(r['scenario'], r['rate'])].setdefault(r['actor'], []).append(r)

    out = {}
    for sc in sorted({k[0] for k in by}):
        cells = []
        for rate in rates:
            a = by.get((sc, rate))
            if not a:
                continue
            seeds = sorted({x['seed'] for x in a[PRIMARY_HI]})
            hi = {x['seed']: x for x in a[PRIMARY_HI]}
            lo = {x['seed']: x for x in a[PRIMARY_LO]}
            diffs = [hi[s]['pdr_routable'] - lo[s]['pdr_routable']
                     for s in seeds if s in hi and s in lo]
            qv, ci_lo, ci_hi = paired_ci(diffs)
            consistency = (sum(1 for d in diffs if d > 0) / len(diffs)
                           if diffs else float('nan'))

            def mean_of(actor, field):
                v = [x[field] for x in a.get(actor, [])]
                return sum(v) / len(v) if v else float('nan')

            n_routable = sum(x['n_routable'] for x in a[PRIMARY_HI])
            dur = a[PRIMARY_HI][0]['duration']
            carried = mean_of('spbp', 'n_delivered') / dur
            offered = mean_of('spbp', 'n_generated') / dur
            addr_tot = sum(x['addressable'] for x in a['spbp'])
            rt_tot = sum(x['n_routable'] for x in a['spbp'])
            addressable = addr_tot / max(rt_tot, 1)
            vs_random = mean_of('spbp', 'pdr_routable') - mean_of('random', 'pdr_routable')

            # v3 regression check (P1)
            drift = max((abs(hi[s]['pdr_routable'] - x['pdr_routable'])
                         for s in seeds
                         for x in a.get('spbp', []) if x['seed'] == s), default=0.0)

            reasons = []
            if n_routable < MIN_ROUTABLE:
                reasons.append(f'n_routable={n_routable}<{MIN_ROUTABLE}')
            if vs_random < MIN_VS_RANDOM:
                reasons.append(f'spbp-random={100*vs_random:.2f}pp<{100*MIN_VS_RANDOM:.0f}pp')

            cells.append(dict(
                rate=rate, queue_value=qv, ci_lo=ci_lo, ci_hi=ci_hi,
                consistency=consistency, band=band(qv, consistency),
                carried=carried, offered=offered,
                addressable=addressable, n_routable=n_routable,
                vs_random=vs_random, spbp_pdr=mean_of('spbp', 'pdr_routable'),
                rel_spread=(qv / mean_of(PRIMARY_HI, 'pdr_routable')
                            if mean_of(PRIMARY_HI, 'pdr_routable') > 1e-9 else float('nan')),
                r_rel=(qv / addressable if addressable >= 0.01 else float('nan')),
                spbp_vs_abfull_drift=drift,
                valid=not reasons, invalid_reason='; '.join(reasons),
            ))

        valid = [c for c in cells if c['valid']]
        res = dict(cells=cells, n_valid=len(valid))
        if not valid:
            res.update(decision='NO VALID CELLS', grid=None)
            out[sc] = res
            continue

        # §4.3.1 as AMENDED (A2): last MATERIAL rise, not argmax. argmax returned
        # 0.40 where the 0.25->0.40 carried-load increment is +0.4%.
        knee = valid[0]['rate']
        for prev, cur in zip(valid, valid[1:]):
            if prev['carried'] > 1e-12 and (cur['carried'] - prev['carried']) / prev['carried'] >= 0.02:
                knee = cur['rate']
        anchor = min(c['rate'] for c in valid)                       # §4.3.2
        # §4.3.3 as AMENDED (A1): a peak may not come from a floor-band cell.
        non_floor = [c for c in valid if c['band'] != 'floor']
        if not non_floor:
            res.update(decision='NO MEASURABLE EFFECT -- every valid cell is floor band',
                       grid=None, knee=knee, anchor=anchor, peak=None,
                       note='no cell clears the floor band; sign consistency never '
                            'reaches 0.8. Scenario needs its own decision (pre-reg §6.4).')
            out[sc] = res
            continue
        peak = max(non_floor, key=lambda c: c['queue_value'])['rate']

        # §4.4 boundary rule, as AMENDED (A3): a boundary argmax only means the
        # RANGE is wrong if there is a genuine gradient. If relative queue value is
        # flat, the MODEL is wrong -- there is no peak to bracket.
        probed = [c['rate'] for c in cells]
        boundary = peak in (min(probed), max(probed))
        rels = [c['rel_spread'] for c in valid
                if c['rel_spread'] == c['rel_spread'] and c['rel_spread'] > 0]
        flat = bool(rels) and (max(rels) / min(rels) < 2.0)
        res.update(knee=knee, anchor=anchor, peak=peak, boundary_peak=boundary,
                   flat_response=flat,
                   rel_spread_ratio=(max(rels) / min(rels)) if rels else float('nan'))
        if boundary and flat:
            res.update(decision='NO PEAK -- FLAT RESPONSE; select by coverage',
                       grid=None,
                       note=f'relative queue value varies by only '
                            f'{max(rels)/min(rels):.2f}x across valid cells; the '
                            f'inverted-U model is refuted, do NOT extend the range '
                            f'(pre-reg §4.4 A3, §6.1)')
            out[sc] = res
            continue
        if boundary:
            direction = 'LOWER' if peak == min(probed) else 'HIGHER'
            res.update(decision=f'EXTEND RANGE {direction} AND RE-PROBE',
                       grid=None,
                       note=f'argmax queue value at probed endpoint {peak}; '
                            f'true peak lies outside the range (pre-reg §4.4)')
            out[sc] = res
            continue

        span = [c['rate'] for c in valid if anchor <= c['rate'] <= knee]
        if peak not in span:
            span = sorted(set(span) | {peak})
        if len(span) <= GRID_SIZE:
            grid = span
        else:
            idx = [round(i * (len(span) - 1) / (GRID_SIZE - 1)) for i in range(GRID_SIZE)]
            grid = sorted(set([span[i] for i in idx] + [peak]))[:GRID_SIZE]
        res.update(decision='GRID SELECTED', grid=grid)
        out[sc] = res
    return out


def report(analysis):
    for sc, a in analysis.items():
        print('\n' + '=' * 100)
        print(f'  {sc}')
        print('=' * 100)
        print(f"  {'rate':>6}{'queue value':>26}{'cons':>6}{'band':>8}{'carried':>10}"
              f"{'offered':>9}{'addr':>8}{'spbp':>7}{'vs rnd':>8}{'n_rt':>7}  valid")
        print('  ' + '-' * 104)
        for c in a['cells']:
            ci = (f"{100*c['queue_value']:6.2f}pp "
                  f"[{100*c['ci_lo']:6.2f},{100*c['ci_hi']:6.2f}]"
                  if c['ci_lo'] == c['ci_lo'] else f"{100*c['queue_value']:6.2f}pp  [   n/a      ]")
            flag = 'yes' if c['valid'] else 'NO  ' + c['invalid_reason']
            print(f"  {c['rate']:>6}{ci:>26}{c['consistency']:>6.1f}"
                  f"{c['band']:>8}{c['carried']:>10.2f}"
                  f"{c['offered']:>9.2f}{c['addressable']:>8.3f}{c['spbp_pdr']:>7.3f}"
                  f"{100*c['vs_random']:>7.1f}p{c['n_routable']:>7}  {flag}")
        drift = max((c['spbp_vs_abfull_drift'] for c in a['cells']), default=0.0)
        print(f"\n  P1 regression check  spbp vs spbp_ab_full  max drift = {drift:.2e}"
              f"  {'OK' if drift < 1e-9 else '*** FAIL -- v3 REGRESSION ***'}")
        if a.get('grid'):
            print(f"  anchor={a['anchor']}  peak={a['peak']}  knee={a['knee']}")
            print(f"  DECISION: {a['decision']} -> {a['grid']}")
            if len(a['grid']) < GRID_SIZE:
                print(f"  *** UNDERSIZED: {len(a['grid'])} rates, wanted {GRID_SIZE}."
                      f" Probe range too narrow or too coarse (pre-reg 4.3)."
                      f" Add probe points before selecting.")
        else:
            print(f"  DECISION: {a['decision']}")
            if a.get('note'):
                print(f"            {a['note']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenarios', nargs='+', default=sorted(SCENARIOS))
    ap.add_argument('--rates', type=float, nargs='+', default=PROBE_RATES)
    ap.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    ap.add_argument('--actors', nargs='+', default=ACTORS)
    ap.add_argument('--duration', type=float, default=1000.0)
    ap.add_argument('--drain_time', type=float, default=BASE['drain_time'])
    ap.add_argument('--z_min', type=float, default=100)
    ap.add_argument('--z_max', type=float, default=300)
    ap.add_argument('--max_workers', type=int, default=None)
    ap.add_argument('--out', default='results/probe_rate_grid.json')
    ap.add_argument('--smoke', action='store_true',
                    help='fast plumbing check at the OLD 40s operating point, '
                         '2 scenarios x 3 rates x 2 seeds')
    ap.add_argument('--reanalyse', metavar='JSON', default=None,
                    help='re-run the decision rule on a saved probe JSON without '
                         'any re-simulation. Use after amending the rules.')
    args = ap.parse_args()

    if args.reanalyse:
        with open(args.reanalyse) as f:
            prev = json.load(f)
        print('=' * 100)
        print(f'  RE-ANALYSIS of {args.reanalyse} under the CURRENT decision rule')
        print('  (no re-simulation; rows are replayed exactly as recorded)')
        print('=' * 100)
        print(f"  rows: {len(prev['rows_per_seed'])}  seeds: {prev['seeds']}")
        print(f"  operating point: {prev['provenance']['resolved_base']}")
        again = analyse(prev['rows_per_seed'], prev['rates'])
        report(again)
        outp = args.reanalyse.replace('.json', '_reanalysed.json')
        with open(outp, 'w') as f:
            json.dump({**prev, 'schema': 'probe_rate_grid_v1_reanalysed',
                       'analysis': again}, f, indent=2)
        print(f'\n  saved to {outp}')
        return 0

    if args.smoke:
        args.duration, args.z_min, args.z_max = BASE['duration'], BASE['z_min'], BASE['z_max']
        args.rates = [0.5, 2.0, 4.0]
        args.seeds = [1, 2]
        args.scenarios = ['dense_slow', 'medium_slow']
        args.out = args.out.replace('.json', '_smoke.json')

    base = {**BASE, 'duration': args.duration, 'drain_time': args.drain_time,
            'z_min': args.z_min, 'z_max': args.z_max}

    jobs = [(sc, SCENARIOS[sc], r, sd, ac, base)
            for sc in args.scenarios for r in args.rates
            for sd in args.seeds for ac in args.actors]

    print('=' * 100)
    print('  RATE-GRID PROBE -- executes docs/PROBE_PREREGISTRATION.md (frozen)')
    print('=' * 100)
    print(f"  operating point : duration={base['duration']}s drain={base['drain_time']}s "
          f"alt={base['z_min']}-{base['z_max']}m")
    print(f"  scenarios       : {args.scenarios}")
    print(f"  rates           : {args.rates}")
    print(f"  seeds (PAIRED)  : {args.seeds}")
    print(f"  actors          : {args.actors}")
    print(f"  primary metric  : {PRIMARY_HI} - {PRIMARY_LO}  (matched control)")
    print(f"  episodes        : {len(jobs)}")
    if args.smoke:
        print('  *** SMOKE MODE -- old operating point, plumbing check only.')
        print('  *** Its numbers must NOT be used to pick a grid.')

    t0 = time.time()
    rows, done = [], 0
    with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
        futs = [ex.submit(_run, j) for j in jobs]
        for f in as_completed(futs):
            rows.append(f.result())
            done += 1
            if done % max(1, len(jobs) // 20) == 0:
                el = time.time() - t0
                print(f'    {done}/{len(jobs)}  ({el:.0f}s elapsed, '
                      f'~{el/done*(len(jobs)-done):.0f}s left)')

    analysis = analyse(rows, args.rates)
    report(analysis)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({'schema': 'probe_rate_grid_v1',
                   'preregistration': 'docs/PROBE_PREREGISTRATION.md',
                   'smoke': args.smoke,
                   'provenance': {**provenance(), 'resolved_base': base},
                   'actors': args.actors, 'seeds': args.seeds,
                   'rates': args.rates,
                   'primary': [PRIMARY_HI, PRIMARY_LO],
                   'thresholds': {'min_routable': MIN_ROUTABLE,
                                  'min_vs_random': MIN_VS_RANDOM,
                                  'band_strong': BAND_STRONG,
                                  'band_usable': BAND_USABLE},
                   'analysis': analysis, 'rows_per_seed': rows}, f, indent=2)
    print(f'\n  saved to {args.out}')

    grids = {sc: a.get('grid') for sc, a in analysis.items()}
    if all(g for g in grids.values()):
        peaks = {sc: a['peak'] for sc, a in analysis.items()}
        idx = {sc: args.rates.index(p) for sc, p in peaks.items()}
        spread = max(idx.values()) - min(idx.values())
        print(f"\n  per-scenario peaks: {peaks}")
        print(f"  peak disagreement: {spread} grid step(s) "
              f"(>= {PEAK_STEP_DISAGREE} triggers per-scenario grids, pre-reg §4.5)")
        if spread >= PEAK_STEP_DISAGREE:
            print('  -> PER-SCENARIO GRIDS INDICATED. Note the cost: load_bucket()')
            print('     becomes scenario-aware, touching teacher_panel and the dataset')
            print('     schema, and every "12-cell panel" reference needs REWORDING.')
        else:
            union = sorted(set().union(*[set(g) for g in grids.values()]))
            print(f'  -> ONE GLOBAL GRID indicated: {union}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
