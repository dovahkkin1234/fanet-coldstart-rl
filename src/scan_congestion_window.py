"""scan_congestion_window.py -- does a usable congestion window exist post-v12?

WHY THIS EXISTS

The v12 leak fix collapsed the measured value of congestion-awareness. Paired
4-seed measurement, dense_slow, alt 100-300:

    rate  offered   ab_full  ab_noqueue   QUEUE VALUE  consistency   q_ovf
      40   280 p/s   0.8196      0.8031       +1.64pp         0.75      42
      60   420 p/s   0.4440      0.4438       +0.02pp         0.75    1299
      80   560 p/s   0.2583      0.2657       -0.74pp         0.75    4414

Roughly 1-2 pp at best, against 6-13 pp before the fix. The probe's band
thresholds (strong >=8pp / usable 3-8pp / floor <3pp) were calibrated on the
pre-fix figures -- the queue term at +6.45pp and the spbp-da_gpsr gap at 3.58pp
-- BOTH of which are now known to be leak-contaminated. So the thresholds are
calibrated to phantom effect sizes, and the full probe would report NO
MEASURABLE EFFECT everywhere by construction rather than by evidence.

This scan settles which of three explanations holds, before the 2-hour run:

  H1  the effect is genuinely ~1-2pp, and the thesis premise needs restating
  H2  the congestion window is being stepped over -- q_ovf jumps 42 -> 1299 ->
      4414 across three rates, very abrupt for a window that should have a
      usable middle
  H3  the queue term is MIS-SCALED for this operating point. The qscale sweep
      that found it saturated (flat from 0.5 to 10, best gain +0.001) was itself
      run on leaked data, so its conclusion does not carry over.

PHASE 1 -- fine rate scan across the region where q_ovf first appears, paired
seeds, reporting queue value, sign consistency and a paired CI. Distinguishes
H1 from H2: if a rate exists with a clean positive effect, H2 held and the
earlier scan simply stepped over it.

PHASE 2 -- qscale sweep at the best rate from Phase 1. Directly tests H3 using
the spbp_qX variants that already exist. If a qscale other than 1.0 restores a
usable effect, the queue term is mis-weighted, not worthless -- which would be
good news for the thesis and would change the oracle definition rather than
the premise.

Runs at the REAL operating point (alt 100-300) so the answer transfers to the
probe. config_v2.BASE is never mutated.

USAGE
    python src\\scan_congestion_window.py --max_workers 16
    python src\\scan_congestion_window.py --max_workers 16 --duration 200 --seeds 10
"""
import argparse, json, math, os, sys, time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import experiment_spbp_mechanism      # noqa: F401 -- registers spbp_ab_*
import experiment_queue_weight        # noqa: F401 -- registers spbp_qX
from simulator_v2 import FANETSimulatorV2
from config_v2 import BASE, SCENARIOS, provenance

PHASE1_RATES = [20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0]
QSCALES = ['spbp_q0.1', 'spbp_q0.3', 'spbp_q1', 'spbp_q3', 'spbp_q10']


def _run(job):
    sc, cfg, rate, seed, actor, base = job
    full = {**base, **cfg, 'packet_rate': rate, 'seed': seed, 'actor': actor}
    m = FANETSimulatorV2(full).run()
    ph = m.get('n_phantom_slots')
    if ph is None:
        raise RuntimeError("n_phantom_slots missing -- v12 not applied. "
                           "This scan would re-measure phantom congestion.")
    if ph != 0:
        raise RuntimeError(f"{ph} phantom slot(s) at {sc} rate={rate} seed={seed}")
    return {'scenario': sc, 'rate': rate, 'seed': seed, 'actor': actor,
            'pdr': m['network_pdr'],
            'q_ovf': m['drop_reasons'].get('queue_overflow', 0),
            'link_err': m['drop_reasons'].get('link_error', 0),
            'n_generated': m['n_generated'], 'secs': 0.0}


def paired(diffs):
    n = len(diffs)
    if n < 2:
        return (diffs[0] if diffs else float('nan')), float('nan'), float('nan')
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    se = math.sqrt(var / n)
    t = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447,
         8: 2.365, 9: 2.306, 10: 2.262, 12: 2.201, 15: 2.145}.get(n, 1.96)
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
                print(f"    {done}/{len(jobs)}  ({el:.0f}s, ~{el/done*(len(jobs)-done):.0f}s left)")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', default='dense_slow')
    ap.add_argument('--rates', type=float, nargs='+', default=PHASE1_RATES)
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--duration', type=float, default=200.0)
    ap.add_argument('--z_min', type=float, default=100)
    ap.add_argument('--z_max', type=float, default=300)
    ap.add_argument('--max_workers', type=int, default=None)
    ap.add_argument('--skip-qscale', action='store_true')
    ap.add_argument('--out', default='results/scan_congestion_window.json')
    args = ap.parse_args()

    seeds = list(range(1, args.seeds + 1))
    base = {**BASE, 'duration': args.duration,
            'z_min': args.z_min, 'z_max': args.z_max}
    cfg = SCENARIOS[args.scenario]
    nflows = max(1, cfg['num_drones'] // 4)

    print('=' * 96)
    print('  CONGESTION-WINDOW SCAN -- is there a usable effect post-v12?')
    print('=' * 96)
    print(f"  scenario   : {args.scenario}  (N={cfg['num_drones']}, flows={nflows})")
    print(f"  operating  : duration={args.duration}s alt={args.z_min}-{args.z_max}m "
          f"(config_v2.BASE untouched at {BASE['duration']}s)")
    print(f"  seeds      : {args.seeds} (paired)")

    # ---------------- PHASE 1 : fine rate scan --------------------------
    print('\n' + '-' * 96)
    print('  PHASE 1 -- fine rate scan  (H1: effect is small  vs  H2: window stepped over)')
    print('-' * 96)
    jobs = [(args.scenario, cfg, r, s, a, base)
            for r in args.rates for s in seeds
            for a in ('spbp_ab_full', 'spbp_ab_noqueue')]
    rows = run_batch(jobs, args.max_workers, 'phase 1')

    per = defaultdict(dict)
    for r in rows:
        per[(r['rate'], r['actor'])][r['seed']] = r
    print(f"\n  {'rate':>6}{'offered':>9}{'ab_full':>9}{'ab_noq':>9}"
          f"{'QUEUE VAL':>11}{'95% CI':>20}{'cons':>6}{'q_ovf':>8}{'link_err':>9}")
    print('  ' + '-' * 88)
    phase1 = []
    for rate in args.rates:
        f = per[(rate, 'spbp_ab_full')]; n = per[(rate, 'spbp_ab_noqueue')]
        ss = sorted(set(f) & set(n))
        d = [f[s]['pdr'] - n[s]['pdr'] for s in ss]
        qv, lo, hi = paired(d)
        cons = sum(1 for x in d if x > 0) / len(d)
        mf = sum(f[s]['pdr'] for s in ss) / len(ss)
        mn = sum(n[s]['pdr'] for s in ss) / len(ss)
        qo = sum(f[s]['q_ovf'] for s in ss) / len(ss)
        le = sum(f[s]['link_err'] for s in ss) / len(ss)
        phase1.append(dict(rate=rate, queue_value=qv, ci_lo=lo, ci_hi=hi,
                           consistency=cons, ab_full=mf, ab_noqueue=mn,
                           q_ovf=qo, link_err=le, offered=nflows * rate))
        ci = f"[{100*lo:6.2f},{100*hi:6.2f}]" if lo == lo else "[    n/a       ]"
        print(f"  {rate:>6.0f}{nflows*rate:>9.0f}{mf:>9.4f}{mn:>9.4f}"
              f"{100*qv:>10.2f}p{ci:>20}{cons:>6.1f}{qo:>8.0f}{le:>9.0f}")

    best = max(phase1, key=lambda c: c['queue_value'])
    clean = [c for c in phase1 if c['consistency'] >= 0.8 and c['ci_lo'] > 0]
    print(f"\n  best rate = {best['rate']:.0f} "
          f"(queue value {100*best['queue_value']:+.2f}pp, cons {best['consistency']:.1f})")
    if clean:
        print(f"  H2 SUPPORTED: {len(clean)} rate(s) with consistency>=0.8 AND CI excluding 0:")
        for c in clean:
            print(f"     rate {c['rate']:>5.0f}  {100*c['queue_value']:+.2f}pp  "
                  f"CI [{100*c['ci_lo']:+.2f},{100*c['ci_hi']:+.2f}]")
        print("  -> a usable window EXISTS; the earlier coarse scan stepped over it.")
    else:
        print("  H1 SUPPORTED: no rate has consistency>=0.8 with a CI excluding zero.")
        print("  -> the effect is genuinely small at qscale=1. Phase 2 tests whether")
        print("     that is because the queue term is mis-weighted (H3).")

    # ---------------- PHASE 2 : qscale sweep ----------------------------
    phase2 = []
    if not args.skip_qscale:
        print('\n' + '-' * 96)
        print(f"  PHASE 2 -- qscale sweep at rate {best['rate']:.0f}  (H3: queue term mis-scaled?)")
        print('  The sweep that found qscale saturated (flat 0.5-10, best gain +0.001)')
        print('  was run on LEAKED data, so its conclusion does not carry over.')
        print('-' * 96)
        jobs2 = [(args.scenario, cfg, best['rate'], s, a, base)
                 for s in seeds for a in QSCALES + ['spbp_ab_noqueue']]
        rows2 = run_batch(jobs2, args.max_workers, 'phase 2')
        p2 = defaultdict(dict)
        for r in rows2:
            p2[r['actor']][r['seed']] = r['pdr']
        baseq = p2['spbp_ab_noqueue']
        print(f"\n  {'qscale':>12}{'pdr':>9}{'vs noqueue':>12}{'95% CI':>20}{'cons':>6}")
        print('  ' + '-' * 60)
        for a in QSCALES:
            ss = sorted(set(p2[a]) & set(baseq))
            d = [p2[a][s] - baseq[s] for s in ss]
            qv, lo, hi = paired(d)
            cons = sum(1 for x in d if x > 0) / len(d)
            mp = sum(p2[a][s] for s in ss) / len(ss)
            phase2.append(dict(actor=a, pdr=mp, delta=qv, ci_lo=lo, ci_hi=hi,
                               consistency=cons))
            ci = f"[{100*lo:6.2f},{100*hi:6.2f}]" if lo == lo else "[    n/a       ]"
            print(f"  {a:>12}{mp:>9.4f}{100*qv:>11.2f}p{ci:>20}{cons:>6.1f}")
        bq = max(phase2, key=lambda c: c['delta'])
        print(f"\n  best qscale = {bq['actor']} ({100*bq['delta']:+.2f}pp, "
              f"cons {bq['consistency']:.1f})")
        cur = [c for c in phase2 if c['actor'] == 'spbp_q1']
        if cur and bq['delta'] > cur[0]['delta'] + 0.01 and bq['consistency'] >= 0.8:
            print("  H3 SUPPORTED: a different qscale restores a usable effect.")
            print("  -> the queue term is MIS-WEIGHTED, not worthless. This changes the")
            print("     ORACLE DEFINITION, not the thesis premise. Re-run the qscale")
            print("     sweep properly and re-pin SP-BP's weight before the grid probe.")
        else:
            print("  H3 NOT supported: no qscale restores a usable effect.")
            print("  -> combined with Phase 1, the effect is genuinely small post-v12.")
            print("     The band thresholds MUST be re-derived from these numbers")
            print("     (Amendment 2) before the full probe is worth running.")

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({'schema': 'scan_congestion_window_v1',
                   'scenario': args.scenario, 'seeds': seeds,
                   'provenance': {**provenance(), 'resolved_base': base},
                   'phase1': phase1, 'phase2': phase2,
                   'rows': rows}, f, indent=2)
    print(f"\n  saved to {args.out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
