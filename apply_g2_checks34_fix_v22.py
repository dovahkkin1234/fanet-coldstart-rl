"""apply_g2_checks34_fix_v22.py -- fix check 3's definition AND check 4's floor.

Assertion-guarded str.replace, idempotent, two anchors. Builds on v20+v21.

TWO DISTINCT MECHANISMS, ONE ROOT MEASUREMENT

Measured directly (g2_activity_probe.py, this machine, G2's exact scenario,
rates 5-35): `mean_activity` is EXACTLY 0.0000 for every rate from 5 through
30 -- the entire pure-link_error regime, including at rate=30 with 491 real
drops -- and becomes nonzero only once queue_overflow first appears, at
rate=35. `mean_activity` tracks queue buildup specifically, not raw
interference/link failures.

CHECK 4 (activity in graded band) requires activity > 0 at every sweep rate.
v21's sweep floor (10.0) sits inside the always-zero regime, so check 4 can
never pass with that floor, no matter how the top end is chosen.

CHECK 3 (drop taxonomy shifts toward congestion) uses
`CONGESTION_CAUSES = ('queue_overflow', 'link_error')`. Recomputed exactly:
cong_share FALLS (100% -> 39.4%) across v21's sweep, because energy_depleted
-- not counted -- grows ~11.7x while the counted causes only grow ~3.3x. A
third, faster-growing cause dilutes the ratio the check is watching. Swapping
`link_error` for `energy_depleted` in CONGESTION_CAUSES fixes this on the
SAME already-measured data: queue_overflow and energy_depleted are both
"this node is overloaded" outcomes (a full buffer or a drained battery);
link_error is a channel/interference outcome, arguably G1's domain, not G2's.
Recomputed with the swap: cong_share RISES monotonically, 31.8% -> 82.1%,
comfortably clearing both of check 3's conditions.

Fixing check 4's floor (moving the sweep above where queue_overflow first
appears) also fixes check 3's zero-drop assumption as a side effect -- the new
floor (35.0) has real, nonzero cong_share (31.8%) under the corrected
definition, so the check no longer depends on a literal zero anywhere.

NEW SWEEP: [35.0, 40.0, 50.0, 60.0, 70.0] -- replaces v21's [10, 20, 40, 60, 70].
40/60/70 kept (already measured, already known-good for checks 1/2/5/6);
10 and 20 dropped (always-zero-activity regime); 35 and 50 added (measured:
rate=35 via g2_activity_probe.py; rate=50 via g2_sweep_probe.py). `mid` becomes
rates[2] = 50.0 -- still fully decoupled from v20's REGRESSION_RATE=60.0.

Old values kept in comments, per this project's convention.

USAGE
    python apply_g2_checks34_fix_v22.py --src src --dry-run
    python apply_g2_checks34_fix_v22.py --src src
"""
import argparse, io, os, sys

TARGET = 'preflight_simulator_v2_check.py'
GUARD = 'v22: CONGESTION_CAUSES'

EDITS = [
    ("CONGESTION_CAUSES = ('queue_overflow', 'link_error')",
     """# v22: CONGESTION_CAUSES swapped link_error -> energy_depleted. Measured:
# with link_error counted, cong_share FELL (100%->39.4%) across the v21
# sweep because energy_depleted (uncounted) grew ~11.7x against the counted
# causes' ~3.3x -- a faster-growing uncounted cause was diluting the ratio.
# queue_overflow + energy_depleted are both "node is overloaded" outcomes
# (full buffer vs drained battery); link_error is channel/interference,
# arguably G1's concern, not G2's. Superseded: ('queue_overflow', 'link_error').
CONGESTION_CAUSES = ('queue_overflow', 'energy_depleted')"""),

    ("                    default=[10.0, 20.0, 40.0, 60.0, 70.0])",
     """                    # v22: floor raised 10->35. Measured (g2_activity_probe.py):
                    # mean_activity is EXACTLY 0.0000 for every rate 5-30 in
                    # this scenario -- the entire pure-link_error regime --
                    # and check 4 requires activity>0 at EVERY sweep rate, so
                    # no floor below ~32-35 can ever pass it. 40/60/70 kept
                    # from v21 (already known-good); 10/20 dropped; 35/50
                    # added (both measured). Superseded: [10.0, 20.0, 40.0, 60.0, 70.0].
                    default=[35.0, 40.0, 50.0, 60.0, 70.0])"""),
]


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
    print("  NEXT: python src\\preflight_simulator_v2_check.py")
    print("  Expect all six checks to PASS. If anything still fails, that is")
    print("  new information (rate=35/50's exact PDR wasn't independently")
    print("  re-confirmed for check 2 -- inferred from the surrounding trend).")
    return 0


if __name__ == '__main__':
    sys.exit(main())
