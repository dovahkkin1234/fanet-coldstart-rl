"""verify_oracle_congestion.py -- is SP-BP's queue term actually harmful?

FRESH STANDALONE SCRIPT. Not a patch. Imports the validated simulator but
implements its own measurement, statistics and guards, so a defect in the
existing experiment scripts cannot propagate into this answer.

WHAT IT IS RE-TESTING

A 5-seed measurement at duration 20, rates 160-260 (a genuine congestion regime:
51-70% queue overflow, ZERO energy death) found SP-BP's queue term made PDR
WORSE: -2.66pp, -2.13pp, -1.64pp at consistency 0.2. That is a thesis-level
claim -- the hand-designed oracle being beaten by its own ablation -- and 5
seeds with no significance test is nowhere near enough to support it.

This script re-tests it properly and then asks the narrower question: if SP-BP's
queue FORMULATION is wrong rather than queue-awareness itself, a different
formulation should win. Two already exist and have never been tested here:

  spbp_ab_full       lq multiplicative, queue DIFFERENTIAL   (the real SP-BP)
  spbp_ab_noqueue    hop term only                           (no queue at all)
  spbp_ab_candqueue  CANDIDATE queue only, not differential  (DA-GPSR style)
  spbp_ab_additive   lq ADDITIVE instead of multiplicative

If candqueue or additive beats noqueue where full does not, the finding is
"SP-BP's queue term is formulated wrong" -- narrow and recoverable -- rather
than "congestion-awareness does not work".

SELF-TESTS RUN FIRST (--skip-self-test to bypass, not recommended)

  S1  v12 applied: n_phantom_slots present and zero
  S2  determinism: same seed twice -> identical metrics
  S3  PAIRED-SEED INTEGRITY: different actors on the same seed must see the
      SAME traffic (identical n_generated). If they do not, every paired
      difference in this script is comparing two different workloads and the
      whole measurement is void. This is the hidden-bug check that matters most.
  S4  P1 regression: spbp and spbp_ab_full must agree to < 1e-12
  S5  REGIME GUARD: the chosen cell really is congestion-dominated with
      negligible energy death -- refuses to measure otherwise
  S6  statistics sanity: the paired t-test agrees with scipy

USAGE
    python src\\verify_oracle_congestion.py --max_workers 16
    python src\\verify_oracle_congestion.py --max_workers 16 --seeds 20 --rate 200
"""
import argparse, json, math, os, sys, time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import experiment_spbp_mechanism           # noqa: F401 -- registers spbp_ab_*
from simulator_v2 import FANETSimulatorV2
from config_v2 import BASE, SCENARIOS, provenance

VARIANTS = ['spbp_ab_noqueue', 'spbp_ab_full',
            'spbp_ab_candqueue', 'spbp_ab_additive']
REFERENCE = 'spbp_ab_noqueue'     # the no-queue baseline everything is judged against
ALPHA = 0.05
MAX_ENERGY_SHARE = 0.05           # regime guard: energy loss must be negligible
MIN_QOVF_SHARE = 0.20             # regime guard: congestion must actually dominate


def episode(scenario, dur, rate, seed, actor, alt):
    """One episode. Module-level for pickling under Windows spawn."""
    cfg = {**BASE, **SCENARIOS[scenario], 'duration': dur,
           'z_min': alt[0], 'z_max': alt[1],
           'packet_rate': rate, 'seed': seed, 'actor': actor}
    s = FANETSimulatorV2(cfg)
    m = s.run()
    ph = m.get('n_phantom_slots')
    if ph is None:
        raise RuntimeError('n_phantom_slots missing -- v12 not applied')
    if ph != 0:
        raise RuntimeError(f'{ph} phantom slot(s): leak present, results void')
    g = max(m['n_generated'], 1)
    d = m['drop_reasons']
    return {'actor': actor, 'seed': seed, 'rate': rate, 'duration': dur,
            'pdr': m['network_pdr'], 'n_generated': m['n_generated'],
            'n_delivered': m['n_delivered'],
            'dead_nodes': sum(1 for e in s.energy if e <= 0.01),
            'q_ovf_share': d.get('queue_overflow', 0) / g,
            'energy_share': d.get('energy_depleted', 0) / g,
            'link_share': d.get('link_error', 0) / g,
            'mean_hops': m.get('mean_hops'), 'mean_delay_ms': m.get('mean_delay_ms')}


def _job(a):
    return episode(*a)


def paired_t(x, y):
    """Paired t-test. Returns (mean_diff, ci_lo, ci_hi, p, consistency)."""
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
        p = float(2 * stats.t.sf(abs(t), n - 1))
        crit = float(stats.t.ppf(1 - ALPHA / 2, n - 1))
    except Exception:
        p = float('nan')
        crit = {5: 2.776, 10: 2.262, 15: 2.145, 20: 2.093}.get(n, 1.96)
    return mean, mean - crit * se, mean + crit * se, p, cons


def holm(pvals):
    idx = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals)
    adj = [0.0] * m
    running = 0.0
    for k, i in enumerate(idx):
        val = (m - k) * pvals[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return adj


# ───────────────────────────── self tests ────────────────────────────────
def self_tests(scenario, dur, rate, alt, verbose=True):
    fails = []
    p = (lambda *a: print(*a)) if verbose else (lambda *a: None)
    p('\n' + '=' * 92)
    p('  SELF-TESTS')
    p('=' * 92)

    # S1 -- v12 applied
    r = episode(scenario, dur, rate, 1, 'spbp', alt)
    p(f"  S1 v12 fix present             : n_phantom_slots readable and zero  OK")

    # S2 -- determinism
    r2 = episode(scenario, dur, rate, 1, 'spbp', alt)
    same = (r['pdr'] == r2['pdr'] and r['n_generated'] == r2['n_generated'])
    p(f"  S2 determinism                 : {'identical  OK' if same else '*** NON-DETERMINISTIC ***'}")
    if not same:
        fails.append('S2 determinism')

    # S3 -- PAIRED-SEED INTEGRITY (the important hidden-bug check)
    gens = {}
    for a in VARIANTS:
        gens[a] = episode(scenario, dur, rate, 1, a, alt)['n_generated']
    uniq = set(gens.values())
    ok3 = len(uniq) == 1
    p(f"  S3 paired-seed traffic identity: n_generated across variants = {gens}")
    p(f"     {'all identical  OK' if ok3 else '*** DIFFER -- paired comparison would be INVALID ***'}")
    if not ok3:
        fails.append('S3 paired-seed integrity: variants see different traffic')

    # S4 -- P1 regression
    a1 = episode(scenario, dur, rate, 1, 'spbp', alt)['pdr']
    a2 = episode(scenario, dur, rate, 1, 'spbp_ab_full', alt)['pdr']
    ok4 = abs(a1 - a2) < 1e-12
    p(f"  S4 spbp == spbp_ab_full        : drift {abs(a1-a2):.2e}  "
      f"{'OK' if ok4 else '*** v3 REGRESSION ***'}")
    if not ok4:
        fails.append('S4 spbp != spbp_ab_full')

    # S5 -- regime guard
    p(f"  S5 regime guard                : q_ovf={r['q_ovf_share']:.3f} "
      f"energy={r['energy_share']:.3f} dead={r['dead_nodes']}")
    if r['q_ovf_share'] < MIN_QOVF_SHARE:
        fails.append(f"S5 not congestion-dominated (q_ovf {r['q_ovf_share']:.3f} "
                     f"< {MIN_QOVF_SHARE})")
    if r['energy_share'] > MAX_ENERGY_SHARE:
        fails.append(f"S5 energy death not negligible ({r['energy_share']:.3f} "
                     f"> {MAX_ENERGY_SHARE}) -- measuring batteries, not routing")
    if not [f for f in fails if f.startswith('S5')]:
        p('     congestion-dominated with negligible energy death  OK')

    # S6 -- statistics sanity
    try:
        from scipy import stats as sp
        x = [0.50, 0.55, 0.48, 0.52, 0.61]
        y = [0.45, 0.49, 0.47, 0.44, 0.55]
        mean, lo, hi, pv, cons = paired_t(x, y)
        st, sp_p = sp.ttest_rel(x, y)
        ok6 = abs(pv - sp_p) < 1e-9
        p(f"  S6 paired t-test vs scipy      : p={pv:.6g} vs {sp_p:.6g}  "
          f"{'OK' if ok6 else '*** MISMATCH ***'}")
        if not ok6:
            fails.append('S6 paired t-test disagrees with scipy')
    except ImportError:
        p('  S6 scipy unavailable -- skipped')

    p()
    if fails:
        p(f'  SELF-TESTS FAILED ({len(fails)}):')
        for f in fails:
            p('    - ' + f)
    else:
        p('  SELF-TESTS PASSED')
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', default='dense_slow')
    ap.add_argument('--duration', type=float, default=20.0)
    ap.add_argument('--rate', type=float, default=200.0)
    ap.add_argument('--rates', type=float, nargs='+', default=None,
                    help='override: test several rates')
    ap.add_argument('--seeds', type=int, default=20)
    ap.add_argument('--z_min', type=float, default=100)
    ap.add_argument('--z_max', type=float, default=300)
    ap.add_argument('--max_workers', type=int, default=None)
    ap.add_argument('--skip-self-test', action='store_true')
    ap.add_argument('--out', default='results/oracle_congestion.json')
    args = ap.parse_args()

    alt = (args.z_min, args.z_max)
    rates = args.rates or [args.rate]
    seeds = list(range(1, args.seeds + 1))

    print('=' * 92)
    print('  ORACLE RE-TEST IN A GENUINE CONGESTION REGIME')
    print('=' * 92)
    print(f"  scenario {args.scenario}  duration {args.duration}s  "
          f"alt {alt[0]:.0f}-{alt[1]:.0f}m  seeds {args.seeds} (paired)")
    print(f"  rates    {rates}")
    print(f"  variants {VARIANTS}")
    print(f"  reference for all comparisons: {REFERENCE}")

    if not args.skip_self_test:
        f = self_tests(args.scenario, args.duration, rates[0], alt)
        if f:
            print('\n  ABORTING -- self-tests failed. Measuring now would produce')
            print('  a number that cannot be trusted.')
            return 1

    jobs = [(args.scenario, args.duration, r, s, a, alt)
            for r in rates for s in seeds for a in VARIANTS]
    print('\n' + '-' * 92)
    print(f'  MEASURING -- {len(jobs)} episodes')
    print('-' * 92)
    t0, rows, done = time.time(), [], 0
    with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
        futs = [ex.submit(_job, j) for j in jobs]
        for fu in as_completed(futs):
            rows.append(fu.result()); done += 1
            if done % max(1, len(jobs) // 10) == 0:
                el = time.time() - t0
                print(f'    {done}/{len(jobs)}  ({el:.0f}s, '
                      f'~{el/done*(len(jobs)-done):.0f}s left)')

    by = defaultdict(dict)
    for r in rows:
        by[(r['rate'], r['actor'])][r['seed']] = r

    results, praw, tags = {}, [], []
    for rate in rates:
        ref = by[(rate, REFERENCE)]
        g0 = ref[seeds[0]]
        print(f"\n  rate {rate:.0f}  |  q_ovf {g0['q_ovf_share']:.3f}  "
              f"energy {g0['energy_share']:.3f}  dead {g0['dead_nodes']}")
        print(f"  {'variant':<20}{'pdr':>9}{'vs noqueue':>12}{'95% CI':>20}"
              f"{'cons':>6}{'hops':>7}")
        print('  ' + '-' * 74)
        cell = {}
        for a in VARIANTS:
            cur = by[(rate, a)]
            ss = sorted(set(cur) & set(ref))
            x = [cur[s]['pdr'] for s in ss]
            y = [ref[s]['pdr'] for s in ss]
            mean, lo, hi, pv, cons = paired_t(x, y)
            mp = sum(x) / len(x)
            hops = sum(cur[s]['mean_hops'] for s in ss) / len(ss)
            cell[a] = dict(pdr=mp, delta=mean, ci_lo=lo, ci_hi=hi, p_raw=pv,
                           consistency=cons, mean_hops=hops, n=len(ss))
            if a != REFERENCE and pv == pv:
                praw.append(pv); tags.append((rate, a))
            ci = f"[{100*lo:6.2f},{100*hi:6.2f}]" if lo == lo else "[   n/a       ]"
            tag = '  (reference)' if a == REFERENCE else ''
            print(f"  {a:<20}{mp:>9.4f}{100*mean:>11.2f}p{ci:>20}"
                  f"{cons:>6.2f}{hops:>7.2f}{tag}")
        results[str(rate)] = cell

    if praw:
        adj = holm(praw)
        for (rate, a), ph in zip(tags, adj):
            results[str(rate)][a]['p_holm'] = ph
            results[str(rate)][a]['significant'] = bool(ph < ALPHA)

    # ─────────────────────────── verdict ────────────────────────────────
    print('\n' + '=' * 92)
    print('  VERDICT')
    print('=' * 92)
    for rate in rates:
        cell = results[str(rate)]
        print(f"\n  rate {rate:.0f}:")
        full = cell['spbp_ab_full']
        sig = full.get('significant', False)
        if full['delta'] < 0 and sig:
            print(f"    SP-BP's queue term HURTS: {100*full['delta']:+.2f}pp, "
                  f"p_holm={full.get('p_holm', float('nan')):.4g} -- SIGNIFICANT")
        elif full['delta'] < 0:
            print(f"    SP-BP's queue term trends negative "
                  f"({100*full['delta']:+.2f}pp) but NOT significant after Holm")
        else:
            print(f"    SP-BP's queue term helps: {100*full['delta']:+.2f}pp"
                  f"{' (significant)' if sig else ' (not significant)'}")

        alts = {a: cell[a] for a in ('spbp_ab_candqueue', 'spbp_ab_additive')}
        winners = {a: c for a, c in alts.items()
                   if c['delta'] > 0 and c.get('significant')}
        if winners:
            best = max(winners, key=lambda a: winners[a]['delta'])
            print(f"    ALTERNATIVE FORMULATION WINS: {best} "
                  f"{100*winners[best]['delta']:+.2f}pp "
                  f"(p_holm={winners[best].get('p_holm'):.4g})")
            print(f"    -> the finding is 'SP-BP's queue term is FORMULATED WRONG',")
            print(f"       not 'congestion-awareness does not work'. Narrow and")
            print(f"       recoverable: re-pin the oracle's queue formulation.")
        else:
            print(f"    No alternative formulation beats no-queue significantly.")
            print(f"    -> queue-awareness as implemented gives no benefit in this")
            print(f"       regime, in ANY of the three formulations tested.")

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({'schema': 'oracle_congestion_v1',
                   'scenario': args.scenario, 'duration': args.duration,
                   'rates': rates, 'seeds': seeds, 'alpha': ALPHA,
                   'reference': REFERENCE,
                   'provenance': {**provenance(), 'alt': list(alt)},
                   'results': results, 'rows': rows}, f, indent=2)
    print(f'\n  saved to {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
