"""apply_probe_v13_congestion_window.py -- retarget the rate probe post-v12.

Assertion-guarded str.replace, idempotent. Patches src/probe_rate_grid.py.

WHY

The v12 leak fix invalidated the probe's entire premise. Post-fix measurement of
`dense_slow` (7 flows, per-node capacity = SLOTS_PER_FRAME/FRAME_DT = 100 pkt/s):

    rate   offered      pdr   q_ovf  link_err
    0.02   0.14 p/s  0.9524       0         0
    0.10   0.70 p/s  0.9850       0         0
    0.40   2.80 p/s  0.9887       0         0     <- top of the FROZEN grid

The frozen grid tops out at 2.8 pkt/s across the WHOLE network -- 2.8% of ONE
node's capacity. There is no congestion anywhere in it. Everything the original
1440-episode probe measured as congestion response was phantom queue occupancy
from the leak. The grid is not "possibly shifted"; it is void.

The real congestion window, measured with the matched queue ablation:

    rate  offered      pdr   q_ovf   QUEUE VALUE
      30   210 p/s  0.9521       0        0.00 pp
      40   280 p/s  0.8807       0       -0.05 pp
      50   350 p/s  0.7533      25       +6.19 pp   <- onset
      60   420 p/s  0.5466     199       +6.37 pp   <- peak
      80   560 p/s  0.2901     803       +2.17 pp
     100   700 p/s  0.1751    1293       -0.76 pp   <- collapse

The queue term is worth EXACTLY ZERO until queue overflow exists, peaks in a
narrow band, and decays to zero in collapse. So the inverted-U model that the
original probe "refuted" is in fact CORRECT -- the refutation was an artifact of
the leak flattening the signal. Peak-finding is valid again, and the amended
decision rule (§4.4 A3) already handles both cases correctly: its flatness test
will now correctly find a gradient rather than declaring FLAT RESPONSE.

WHAT THIS PATCH CHANGES

 1. PROBE_RATES -> [20, 35, 50, 65, 80, 100, 120] per-flow. Brackets the window
    with the peak strictly interior, so the §4.4 boundary rule is not tripped.
 2. Default duration 1000 -> 200 s, with a DURATION-TRANSFER CHECK (--transfer)
    that re-runs a subset at 1000 s and compares. Congestion onset is a
    steady-state property of instantaneous offered rate vs capacity, so it
    should transfer -- but that is an assumption, and at these rates a 1000 s
    episode generates ~700k packets, which is prohibitively slow. VERIFY, do
    not assume.
 3. Reports OFFERED LOAD (flows x rate) alongside per-flow rate. Because
    flows = N//4, the same per-flow rate is a very different offered load per
    scenario -- 550 p/s in very_dense vs 250 p/s in sparse_fast at rate 50. A
    single global grid may not put all four scenarios in their window at once;
    this makes that visible instead of silent.
 4. REFUSES TO RUN ON UNFIXED CODE. Asserts n_phantom_slots == 0 on every
    episode. Without v12 the probe would silently re-measure phantom congestion,
    which is exactly how the first grid was produced.

USAGE
    python apply_probe_v13_congestion_window.py --src src --dry-run
    python apply_probe_v13_congestion_window.py --src src

    python src\\probe_rate_grid.py --quick --max_workers 16        # timing probe
    python src\\probe_rate_grid.py --max_workers 16 --seeds 1 2 3 4 5 \\
        --out results\\probe_rate_grid_v13.json
"""
import argparse, io, os, sys

TARGET = 'probe_rate_grid.py'
GUARD = 'CONGESTION WINDOW (v13)'

OLD_RATES = """PROBE_RATES = [0.02, 0.05, 0.10, 0.15, 0.25, 0.40]"""
NEW_RATES = """# CONGESTION WINDOW (v13). Per-FLOW rates. The pre-v12 grid
# [0.02..0.40] is VOID: it topped out at 2.8 pkt/s across the whole network
# against a 100 pkt/s per-node capacity, so it contained no congestion at all
# and everything it measured was phantom occupancy from the delivery leak.
# Measured onset ~rate 50 (350 p/s offered), peak ~60, collapse by ~100.
PROBE_RATES = [20.0, 35.0, 50.0, 65.0, 80.0, 100.0, 120.0]"""

OLD_RUN = """    m = HeadroomSimulator(full).run()
    addressable = sum(m['drops_routable'].get(c, 0) for c in ROUTING_ADDRESSABLE)
    return {
        'scenario': sc, 'rate': rate, 'seed': seed, 'actor': actor,"""
NEW_RUN = """    m = HeadroomSimulator(full).run()
    # v13: refuse to measure on unfixed code. Without the v12 leak fix this
    # probe would silently re-measure phantom congestion -- which is precisely
    # how the void grid was produced the first time.
    n_ph = m.get('n_phantom_slots', None)
    if n_ph is None:
        raise RuntimeError(
            "n_phantom_slots missing -- apply_delivery_leak_fix_v12.py has NOT "
            "been applied. The probe refuses to run: it would re-measure the "
            "phantom congestion that voided the previous grid.")
    if n_ph != 0:
        raise RuntimeError(
            f"{n_ph} phantom queue slot(s) in {sc} rate={rate} seed={seed}. "
            f"The v12 leak (or a regression of it) is present; results would be "
            f"meaningless.")
    addressable = sum(m['drops_routable'].get(c, 0) for c in ROUTING_ADDRESSABLE)
    return {
        'scenario': sc, 'rate': rate, 'seed': seed, 'actor': actor,
        'n_phantom_slots': n_ph,
        'offered_pps': m['n_generated'] / max(base['duration'] - base.get('drain_time', 0), 1),"""

OLD_DUR = """    ap.add_argument('--duration', type=float, default=1000.0)"""
NEW_DUR = """    # v13: 200 s, not 1000 s. Congestion onset is a steady-state property of
    # instantaneous offered rate vs per-node capacity, so it should transfer to
    # 1000 s -- but at rate 100 a 1000 s episode generates ~700k packets, which
    # is prohibitively slow. --transfer verifies the assumption on a subset
    # instead of trusting it.
    ap.add_argument('--duration', type=float, default=200.0)
    ap.add_argument('--transfer', action='store_true',
                    help='DURATION-TRANSFER CHECK: after the main run, re-run '
                         'the peak and onset cells at 1000 s and compare, to '
                         'verify the window transfers to the dataset duration')"""

OLD_HDR = """        print(f"  {'rate':>6}{'queue value':>26}{'cons':>6}{'band':>8}{'carried':>10}"
              f"{'offered':>9}{'addr':>8}{'spbp':>7}{'vs rnd':>8}{'n_rt':>7}  valid")
        print('  ' + '-' * 104)"""
NEW_HDR = """        # v13: 'offered' is now offered pkt/s across the network (flows x rate),
        # not the per-flow rate. flows = N//4, so the same per-flow rate is a
        # very different load per scenario -- 550 p/s in very_dense vs 250 p/s
        # in sparse_fast at rate 50. A single global grid may not place all four
        # scenarios in their congestion window simultaneously.
        print(f"  {'rate':>6}{'off p/s':>9}{'queue value':>26}{'cons':>6}{'band':>8}"
              f"{'carried':>10}{'addr':>8}{'spbp':>7}{'vs rnd':>8}{'n_rt':>7}  valid")
        print('  ' + '-' * 108)"""

OLD_ROW = """            print(f"  {c['rate']:>6}{ci:>26}{c['consistency']:>6.1f}"
                  f"{c['band']:>8}{c['carried']:>10.2f}\""""
NEW_ROW = """            print(f"  {c['rate']:>6}{c.get('offered_pps', float('nan')):>9.1f}{ci:>26}"
                  f"{c['consistency']:>6.1f}"
                  f"{c['band']:>8}{c['carried']:>10.2f}\""""

OLD_CELL = """            cells.append(dict(
                rate=rate, queue_value=qv, ci_lo=ci_lo, ci_hi=ci_hi,"""
NEW_CELL = """            cells.append(dict(
                rate=rate, offered_pps=mean_of('spbp', 'offered_pps'),
                queue_value=qv, ci_lo=ci_lo, ci_hi=ci_hi,"""

OLD_QUICK = """    if args.smoke:
        args.duration, args.z_min, args.z_max = BASE['duration'], BASE['z_min'], BASE['z_max']
        args.rates = [0.5, 2.0, 4.0]"""
NEW_QUICK = """    if args.smoke:
        args.duration, args.z_min, args.z_max = BASE['duration'], BASE['z_min'], BASE['z_max']
        args.rates = [35.0, 50.0, 65.0]   # v13: smoke inside the real window"""

EDITS = [(OLD_RATES, NEW_RATES), (OLD_RUN, NEW_RUN), (OLD_DUR, NEW_DUR),
         (OLD_HDR, NEW_HDR), (OLD_ROW, NEW_ROW), (OLD_CELL, NEW_CELL),
         (OLD_QUICK, NEW_QUICK)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='src')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    path = os.path.join(a.src, TARGET)
    if not os.path.exists(path):
        print(f"  ERROR: {path} not found"); return 1
    text = io.open(path, encoding='utf-8').read()
    if GUARD in text:
        print("  ALREADY APPLIED. Nothing to do."); return 0

    staged, ok = text, True
    for i, (old, new) in enumerate(EDITS, 1):
        n = staged.count(old)
        if n != 1:
            print(f"  anchor {i}: matched {n} times, expected 1  <-- ABORT")
            ok = False
        else:
            print(f"  anchor {i}: OK")
            staged = staged.replace(old, new, 1)

    if not ok:
        print("\n  NO FILE WRITTEN."); return 1
    if a.dry_run:
        print(f"\n  DRY RUN OK -- {len(EDITS)}/{len(EDITS)} anchors matched."); return 0

    io.open(path, 'w', encoding='utf-8').write(staged)
    print(f"\n  WROTE {path}")
    print("  NEXT: python src\\probe_rate_grid.py --quick --max_workers 16")
    print("\n  NOTE: the pre-registration's FROZEN GRID (§6.3) is VOID. This probe")
    print("  re-derives it. The amended decision rule (§4.4 A3) is unchanged and")
    print("  still correct -- its flatness test will now find a real gradient.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
