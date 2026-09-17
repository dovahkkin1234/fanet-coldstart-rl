"""apply_g2_sweep_rebaseline_v21.py -- fix G2's stale SWEEP range (checks 1-5).

Assertion-guarded str.replace, idempotent, one anchor.

WHY

v20 fixed check 6 (the regression anchor) by moving it to a rate with real,
measured losses. Checks 1-5 use a SEPARATE default rate list --
`[0.25, 0.5, 1.0, 2.0, 4.0]` -- which is the exact same stale grid, just never
updated. Confirmed directly (g2_exact_check.py, g2_sweep_probe.py, both run on
this machine, G2's own exact scenario): every one of those five rates produces
zero drops of any kind. Checks 1-5 test queue occupancy, PDR-vs-load,
taxonomy shift, activity, and interference ON/OFF -- all five need SOME load
range where congestion actually occurs, and this range has none.

MEASURED REPLACEMENT

    rate    pdr     dominant drop cause
      10   0.9993   link_error only (100%)
      20   0.9820   link_error only (100%)
      40   0.8125   link_error 51% / queue_overflow 18% / energy 30%
      60   0.4771   energy 53% / queue_overflow 24% / link_error 23%
      70   0.3745   energy 61% / queue_overflow 22% / link_error 18%

Monotonic (0.9993 > 0.9820 > 0.8125 > 0.4771 > 0.3745) -- check 2. Visible
taxonomy shift from pure interference at low rate to energy/queue-dominant at
high rate -- check 3. rate=60 matches v20's REGRESSION_RATE exactly, so the
sweep and the anchor now agree on what "heavy load" means at this scenario.

`mid = rates[len(rates)//2]` becomes 40.0 with this list -- the rate used by
check 5 (interference ON/OFF) and check 6's reproducibility half. The ON case
at rate=40 is measured (link_error=1080); the OFF case is not pre-measured,
but removing interference can only remove link_error and improve PDR, so
check 5's condition (le_off=0, le_on>0, off_pdr>on_pdr) is a safe bet, not a
verified one. Running G2 after this patch IS that verification -- if check 5
still fails, that is the next thing to fix, same loop as everything else here.

OLD VALUES KEPT IN A COMMENT, per this project's convention.

USAGE
    python apply_g2_sweep_rebaseline_v21.py --src src --dry-run
    python apply_g2_sweep_rebaseline_v21.py --src src
"""
import argparse, io, os, sys

TARGET = 'preflight_simulator_v2_check.py'
GUARD = 'v21: sweep re-baselined'

OLD = """    ap.add_argument('--rates', type=float, nargs='+',
                    default=[0.25, 0.5, 1.0, 2.0, 4.0])"""

NEW = """    # v21: sweep re-baselined. Old default [0.25, 0.5, 1.0, 2.0, 4.0] produced
    # ZERO drops of any kind at every one of its five rates under the current
    # simulator (measured directly on this machine, G2's own exact scenario) --
    # the same staleness class v20 fixed for the regression anchor, here
    # affecting checks 1-5 (queue occupancy, PDR-vs-load, taxonomy, activity,
    # interference ON/OFF), all of which need SOME congestion to test anything.
    # New values are measured, not estimated: monotonic PDR 0.9993->0.3745,
    # with a visible shift from pure link_error (low rate) to energy/queue-
    # dominant (high rate). rate=60 matches REGRESSION_RATE (v20) exactly.
    ap.add_argument('--rates', type=float, nargs='+',
                    default=[10.0, 20.0, 40.0, 60.0, 70.0])"""


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

    n = text.count(OLD)
    if n != 1:
        print(f"  anchor matched {n} times, expected 1  <-- ABORT")
        return 1
    print("  anchor 1: OK")
    if a.dry_run:
        print("\n  DRY RUN OK. Nothing written."); return 0

    io.open(path, 'w', encoding='utf-8').write(text.replace(OLD, NEW, 1))
    print(f"\n  WROTE {path}")
    print("  NEXT: python src\\preflight_simulator_v2_check.py")
    print("  If check 5 still fails (the OFF-case at rate=40 was not")
    print("  pre-measured), send the output -- that is the next fix, not a")
    print("  sign anything else is wrong.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
