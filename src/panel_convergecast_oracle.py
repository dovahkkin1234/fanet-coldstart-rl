"""panel_convergecast_oracle.py -- which teacher is the ORACLE for Suite C?

WHAT THIS DECIDES

`generate_dataset_v2.py` hardcodes `ORACLE_TEACHER = 'spbp'` on the strength of
G3, which measured the 8-teacher panel across the twelve **Suite A** cells and
found SP-BP won all of them. Suite C (convergecast, many-to-one to a pinned
ground sink) was never in that comparison.

The Dataset V2 plan puts 30% convergecast episodes into the training mix, so the
warmstart policy will imitate whatever teacher labels those episodes. Labelling
them with SP-BP on Suite A evidence is an assumption, not a measurement. This
script is the G3-equivalent that turns it into a measurement.

WHY IT MIGHT COME OUT DIFFERENT. The convergecast bottleneck is the sink's
immediate neighbourhood: at N=200 the sink has roughly 35 contending neighbours,
and Bianchi throughput with 35 stations is a fraction of nominal. That is a
MAC-layer bottleneck, which SP-BP's queue differential only partly captures. A
link-quality-weighted teacher (ETX-style) or a lookahead variant could plausibly
win there. If one does, `oracle_teacher_for()` -- which already exists in
teacher_panel and is currently bypassed by the hardcoded constant -- becomes the
right mechanism.

OPERATING POINT. Runs at the FINAL point (1000 s, altitude 100-300) via explicit
CLI overrides, NOT by touching config_v2.BASE. The oracle must be chosen where
the dataset will actually be generated -- a teacher ranking measured at 40 s /
50-150 m would not transfer, since the whole reason for the rate recalibration
was that 40 s and 1000 s behave differently. config_v2.BASE stays at 40 s so the
SP-BP parity gate keeps its fixed reference (FILE2 §3).

METHOD (mirrors G3)
  * every teacher runs the SAME seed list per cell, so all differences are PAIRED
  * per cell: mean routable PDR per teacher, winner, and paired t-tests of the
    winner against every other teacher
  * Holm-Bonferroni correction ACROSS ALL comparisons, via teacher_panel's own
    holm_bonferroni -- not a reimplementation
  * verdict: does one teacher win every cell, and are the wins significant?

COST WARNING. Convergecast runs at N = 50-200 against Suite A's 20-45, and cost
scales roughly with N^2 per slot. The full grid is expensive. FILE2 §7 carries a
standing rule about exactly this -- the last extrapolation of this kind was 13x
wrong -- so START WITH --quick, read the reported per-episode timing, and scale
from measurement.

USAGE
    # timing probe + smoke, N=50 only (do this FIRST)
    python src\\panel_convergecast_oracle.py --quick --max_workers 16

    # staged: the two cheaper node counts
    python src\\panel_convergecast_oracle.py --scenarios sink_50 sink_100 \\
        --seeds 1 2 3 4 5 --max_workers 16 --out results\\panel_cc_small.json

    # full grid (expensive -- only after the timing probe justifies it)
    python src\\panel_convergecast_oracle.py --max_workers 16 \\
        --out results\\panel_convergecast_oracle.json
"""
import argparse, json, os, sys, time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulator_v2 import FANETSimulatorV2, PANEL
from teacher_panel import paired_ttest, holm_bonferroni
from config_v2 import BASE, get_suite, provenance

FROZEN_RATES = [0.02, 0.05, 0.10, 0.25, 0.40]   # docs/PROBE_PREREGISTRATION.md §6.3
ALPHA = 0.05
MIN_DELTA = 0.01   # 1 pp -- a challenger must beat SP-BP by at least this
                   # AND significantly. Guards against tie-broken-by-argmax.


def _run(job):
    """One episode. Module-level for pickling under Windows spawn."""
    sc, cfg, rate, seed, teacher, base = job
    t0 = time.time()
    full = {**base, **cfg, 'packet_rate': rate, 'seed': seed, 'actor': teacher}
    m = FANETSimulatorV2(full).run()
    gen = max(m['n_generated'], 1)
    return {
        'scenario': sc, 'rate': rate, 'seed': seed, 'teacher': teacher,
        'pdr': m['n_delivered'] / gen,
        'n_generated': m['n_generated'], 'n_delivered': m['n_delivered'],
        'drop_reasons': dict(m['drop_reasons']),
        'mean_delay_ms': m.get('mean_delay_ms'),
        'secs': time.time() - t0, 'num_drones': cfg['num_drones'],
    }


def analyse(rows, teachers):
    """Per cell: winner + paired tests of winner vs all others. Holm across all."""
    cells = defaultdict(lambda: defaultdict(dict))     # (sc,rate) -> teacher -> seed -> pdr
    for r in rows:
        cells[(r['scenario'], r['rate'])][r['teacher']][r['seed']] = r['pdr']

    results, raw_p, tags = {}, [], []
    for key in sorted(cells):
        per = cells[key]
        means = {t: (sum(v.values()) / len(v)) if v else float('nan')
                 for t, v in per.items()}
        present = [t for t in teachers if t in means and means[t] == means[t]]
        if not present:
            continue
        # Winner by argmax is reported for information only. The DECISION uses
        # the challenger framing below: SP-BP is the incumbent oracle and is
        # replaced only if something beats it by a real margin, significantly.
        # A first version of this script used argmax alone and, in a cell where
        # all eight teachers scored EXACTLY 0.5952, declared a tie-broken-by-
        # list-order "winner" that beat SP-BP by +0.0000 -- and recommended
        # abandoning the oracle on that basis.
        winner = max(present, key=lambda t: means[t])
        tied = [t for t in present if abs(means[t] - means[winner]) < 1e-12]
        # every comparison is CHALLENGER vs SP-BP, paired on seeds
        comps = {}
        incumbent = 'spbp'
        for t in present:
            if t == incumbent or incumbent not in per:
                continue
            seeds = sorted(set(per[incumbent]) & set(per[t]))
            if len(seeds) < 2:
                continue
            x = [per[t][s] for s in seeds]          # challenger
            y = [per[incumbent][s] for s in seeds]  # incumbent
            try:
                _t, _df, p = paired_ttest(x, y)   # returns (t, df, p)
            except Exception:
                p = float('nan')
            # v16 FIX: this was 'means[winner] - means[t]' -- a leftover
            # from an earlier argmax-winner design. The loop below already
            # correctly pairs seeds and runs the t-test against `incumbent`;
            # only this delta line still referenced the wrong comparator.
            # Confirmed by exact arithmetic match on a real run: every stored
            # delta equalled means[winner]-means[t], not means[t]-means[incumbent].
            comps[t] = {'delta': means[t] - means[incumbent], 'p_raw': float(p),
                        'n_pairs': len(seeds)}
            if p == p:
                raw_p.append(float(p)); tags.append((key, t))
        results[f'{key[0]}|{key[1]}'] = {
            'scenario': key[0], 'rate': key[1], 'means': means,
            'winner': winner, 'tied_with_winner': tied,
            'all_tied': len(tied) == len(present), 'comparisons': comps,
        }

    if raw_p:
        p_holm, rejected = holm_bonferroni(raw_p, ALPHA)
        for (key, t), ph, rej in zip(tags, p_holm, rejected):
            c = results[f'{key[0]}|{key[1]}']['comparisons'][t]
            c['p_holm'], c['significant'] = float(ph), bool(rej)
    return results


def report(results, teachers):
    print('\n' + '=' * 108)
    print('  PER-CELL TEACHER COMPARISON  (mean routable-free PDR = delivered / generated)')
    print('=' * 108)
    hdr = f"  {'scenario':<11}{'rate':>6}  " + ''.join(f'{t[:11]:>12}' for t in teachers)
    print(hdr); print('  ' + '-' * (len(hdr) - 2))
    for k in sorted(results):
        r = results[k]
        row = f"  {r['scenario']:<11}{r['rate']:>6}  "
        for t in teachers:
            v = r['means'].get(t, float('nan'))
            mark = '*' if t == r['winner'] else ' '
            row += f"{v:>11.4f}{mark}" if v == v else f"{'--':>12}"
        print(row)
    print("\n  * = cell winner")

    print('\n' + '=' * 108)
    print('  VERDICT -- is SP-BP still the right oracle for Suite C?')
    print('=' * 108)
    print(f"  Decision rule: SP-BP is the INCUMBENT. It is replaced only where a")
    print(f"  challenger beats it by >= {100*MIN_DELTA:.0f} pp AND significantly after Holm.")
    print(f"  Argmax alone is NOT used -- exact ties would otherwise read as wins.")

    n_tied = sum(1 for r in results.values() if r.get('all_tied'))
    if n_tied:
        print(f"\n  NOTE: {n_tied}/{len(results)} cell(s) had ALL teachers exactly tied.")
        print("        Routing choice is irrelevant there; they carry no oracle signal.")

    beaten = {}
    for k, r in results.items():
        hits = {t: c for t, c in r['comparisons'].items()
                if c.get('significant') and c['delta'] >= MIN_DELTA}
        if hits:
            beaten[k] = hits

    n_sig = sum(1 for r in results.values()
                for c in r['comparisons'].values() if c.get('significant'))
    n_tot = sum(len(r['comparisons']) for r in results.values())
    print(f"\n  cells: {len(results)}   challenger comparisons: {n_tot}   "
          f"significant after Holm: {n_sig}")

    if not beaten:
        print(f"\n  NO challenger beats SP-BP by >= {100*MIN_DELTA:.0f} pp significantly "
              f"in ANY cell.")
        print("  -> KEEP ORACLE_TEACHER = 'spbp' for Suite C.")
        print("     The Suite A assumption is now a MEASUREMENT, not an assumption.")
    else:
        print(f"\n  SP-BP is significantly beaten in {len(beaten)}/{len(results)} cell(s):")
        for k in sorted(beaten):
            r = results[k]
            for t, c in sorted(beaten[k].items(), key=lambda kv: -kv[1]['delta']):
                print(f"       {r['scenario']:<11} rate {r['rate']:<6} {t:<16} "
                      f"beats spbp by {c['delta']:+.4f}  "
                      f"p_holm={c.get('p_holm', float('nan')):.4g}")
        print("\n  -> DO NOT pin 'spbp' for Suite C. Replace the hardcoded")
        print("     ORACLE_TEACHER with the per-cell oracle_teacher_for() lookup")
        print("     that already exists in teacher_panel.py.")

    # is the frozen Suite A rate grid even sensible for convergecast?
    print('\n  RATE-GRID SANITY (is the Suite A grid appropriate here?)')
    for k in sorted(results):
        r = results[k]
        best = r['means'][r['winner']]
        flag = ''
        if best < 0.05:
            flag = '  <-- near-total collapse, cell carries little signal'
        elif best > 0.95:
            flag = '  <-- nearly lossless, cell carries little signal'
        print(f"    {r['scenario']:<11} rate {r['rate']:<6} best PDR {best:.4f}{flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenarios', nargs='+', default=None,
                    help='default: all convergecast scenarios')
    ap.add_argument('--rates', type=float, nargs='+', default=FROZEN_RATES)
    ap.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    ap.add_argument('--teachers', nargs='+', default=list(PANEL))
    ap.add_argument('--duration', type=float, default=1000.0)
    ap.add_argument('--drain_time', type=float, default=BASE['drain_time'])
    ap.add_argument('--z_min', type=float, default=100)
    ap.add_argument('--z_max', type=float, default=300)
    ap.add_argument('--max_workers', type=int, default=None)
    ap.add_argument('--quick', action='store_true',
                    help='TIMING PROBE: sink_50 only, 3 rates, 3 seeds')
    ap.add_argument('--out', default='results/panel_convergecast_oracle.json')
    args = ap.parse_args()

    try:
        suite = get_suite('convergecast')
    except RuntimeError as e:
        print(f"  ERROR: {e}")
        print("  Apply v8a first: python apply_v8a_suites_v1.py --root .")
        return 1

    if args.quick:
        args.scenarios = ['sink_50']
        args.rates = [0.05, 0.10, 0.25]
        args.seeds = [1, 2, 3]
        args.out = args.out.replace('.json', '_quick.json')

    scen = {k: suite[k] for k in (args.scenarios or sorted(suite))}
    missing = [k for k in (args.scenarios or []) if k not in suite]
    if missing:
        print(f"  ERROR: unknown convergecast scenarios {missing}; "
              f"available: {sorted(suite)}"); return 1

    base = {**BASE, 'duration': args.duration, 'drain_time': args.drain_time,
            'z_min': args.z_min, 'z_max': args.z_max}

    jobs = [(sc, cfg, r, sd, t, base)
            for sc, cfg in scen.items() for r in args.rates
            for sd in args.seeds for t in args.teachers]

    print('=' * 108)
    print('  CONVERGECAST ORACLE PANEL -- G3-equivalent for Suite C')
    print('=' * 108)
    print(f"  operating point : duration={base['duration']}s "
          f"alt={base['z_min']}-{base['z_max']}m   (config_v2.BASE UNCHANGED at "
          f"{BASE['duration']}s -- parity reference preserved)")
    print(f"  scenarios       : {sorted(scen)}  "
          f"(N = {[c['num_drones'] for c in scen.values()]})")
    print(f"  rates           : {args.rates}")
    print(f"  seeds (PAIRED)  : {args.seeds}")
    print(f"  teachers        : {args.teachers}")
    print(f"  episodes        : {len(jobs)}")
    if args.quick:
        print('  *** QUICK MODE -- timing probe. Read the per-episode timings below')
        print('  *** and scale the full run from MEASUREMENT, not extrapolation.')

    t0, rows, done = time.time(), [], 0
    with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
        futs = [ex.submit(_run, j) for j in jobs]
        for f in as_completed(futs):
            rows.append(f.result()); done += 1
            if done % max(1, len(jobs) // 20) == 0:
                el = time.time() - t0
                print(f'    {done}/{len(jobs)}  ({el:.0f}s elapsed, '
                      f'~{el/done*(len(jobs)-done):.0f}s left)')

    per_n = defaultdict(list)
    for r in rows:
        per_n[r['num_drones']].append(r['secs'])
    print('\n  MEASURED per-episode cost (use this to scale, do not extrapolate):')
    for n in sorted(per_n):
        v = per_n[n]
        print(f"    N={n:<4} mean {sum(v)/len(v):7.1f}s   max {max(v):7.1f}s   "
              f"episodes {len(v)}")

    results = analyse(rows, args.teachers)
    report(results, args.teachers)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({'schema': 'panel_convergecast_oracle_v1',
                   'quick': args.quick, 'alpha': ALPHA,
                   'provenance': {**provenance(), 'resolved_base': base},
                   'teachers': args.teachers, 'rates': args.rates,
                   'seeds': args.seeds, 'scenarios': sorted(scen),
                   'analysis': results, 'rows_per_episode': rows}, f, indent=2)
    print(f'\n  saved to {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
