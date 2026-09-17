"""apply_g2_anchor_rebaseline_v20.py -- fix G2's stale AND uninformative anchor.

Assertion-guarded str.replace, idempotent, four anchors.

WHAT WAS WRONG -- TWO SEPARATE THINGS

  (1) STALE VALUES. The anchor (dijkstra 87/280, spbp 112/280) does not
      reproduce under the current, corrected simulator. Verified directly on
      the real hardware this runs on: at G2's own exact scenario (1000x1000m,
      comm_range=280, dur=40s) and its own anchor rate (1.0), BOTH actors now
      show 280/280 (100% PDR, zero drops of any kind). v10/v12/v17/v18 all
      post-date whenever 87/280 was originally measured.

  (2) THE ANCHOR RATE ITSELF WAS A POOR CHOICE, independent of staleness. At
      rate=1.0 the network carries essentially no load (280 packets total
      over 40s). A regression anchor there is UNINFORMATIVE: a build with
      EVERY drop mechanism silently disabled would ALSO read 100%/100% at
      this rate, so the anchor cannot distinguish a healthy simulator from a
      completely broken one -- which is the exact ambiguity that triggered a
      multi-turn investigation before this fix existed.

THE FIX

Move the anchor to rate=60 (offered load 420 pkt/s against this scenario's
30 nodes), where BOTH actors show real, MIXED-CAUSE losses -- queue_overflow,
link_error, AND energy_depleted all present simultaneously. Measured directly
on the machine this check runs on, not recomputed in a different sandbox
where small accumulated-patch differences could produce a non-bit-identical
number:

    dijkstra   8015/16800   drops={queue_overflow:2100, link_error:2000,
                                   energy_depleted:4685}
    spbp       7161/16800   drops={queue_overflow:1417, link_error:2513,
                                   no_route:178, energy_depleted:5531}

A NEW, DEDICATED constant (REGRESSION_RATE) carries this, rather than
repurposing `mid` (the sweep-table midpoint, ALSO used by check 5's
interference ON/OFF test and check 6's reproducibility test). Changing `mid`
would have risked altering what those two OTHER checks exercise; this keeps
the anchor rate fully decoupled from everything else in the file.

Old anchor values are kept in a comment, per this project's convention of
recording superseded numbers rather than deleting them.

WORTH NOTING, NOT ACTED ON HERE: spbp (42.63%) delivers LESS than dijkstra
(47.71%) at this operating point -- a third, independent scenario geometry
showing the same queue-term pattern found in very_dense/dense_slow/
medium_slow. Corroborating evidence for the oracle per-cell re-selection work
already planned separately; not something this anchor-fix patch addresses.

USAGE
    python apply_g2_anchor_rebaseline_v20.py --src src --dry-run
    python apply_g2_anchor_rebaseline_v20.py --src src
"""
import argparse, io, os, sys

TARGET = 'preflight_simulator_v2_check.py'
GUARD = 'REGRESSION_RATE'

EDITS = [
    ("REGRESSION_ACTOR = 'dijkstra'\nREGRESSION_DELIVERED = 87\nREGRESSION_GENERATED = 280",
     """# v20: anchor re-baselined. Old values (87/280, ~31%% PDR) do not reproduce
# under the corrected simulator (v10/v12/v17/v18 all post-date whenever they
# were measured) AND rate=1.0 was itself an uninformative choice -- a build
# with every drop mechanism disabled also reads 100%% there. Moved to
# REGRESSION_RATE (60.0, not the sweep midpoint `mid`, to avoid changing what
# check 5 / check 6 test) where both actors show real mixed-cause loss.
# Superseded: REGRESSION_DELIVERED=87, REGRESSION_GENERATED=280.
REGRESSION_RATE = 60.0
REGRESSION_ACTOR = 'dijkstra'
REGRESSION_DELIVERED = 8015
REGRESSION_GENERATED = 16800"""),

    ("REGRESSION_LQ_ACTOR = 'spbp'\nREGRESSION_LQ_DELIVERED = 112\nREGRESSION_LQ_GENERATED = 280",
     """# v20: same re-baseline, same reasoning as REGRESSION_ above.
# Superseded: REGRESSION_LQ_DELIVERED=112, REGRESSION_LQ_GENERATED=280.
REGRESSION_LQ_ACTOR = 'spbp'
REGRESSION_LQ_DELIVERED = 7161
REGRESSION_LQ_GENERATED = 16800"""),

    ("""    ra = (r1 if args.actor == REGRESSION_ACTOR else
          run({**base, 'packet_rate': mid, 'interference_on': True,
               'actor': REGRESSION_ACTOR}))""",
     """    # v20: REGRESSION_RATE, not mid -- r1 was built at `mid` for checks 5/6,
    # so it can no longer be reused here even when actors match.
    ra = run({**base, 'packet_rate': REGRESSION_RATE, 'interference_on': True,
              'actor': REGRESSION_ACTOR})"""),

    ("""    rb = run({**base, 'packet_rate': mid, 'interference_on': True,
              'actor': REGRESSION_LQ_ACTOR})""",
     """    rb = run({**base, 'packet_rate': REGRESSION_RATE, 'interference_on': True,
              'actor': REGRESSION_LQ_ACTOR})"""),
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
    print("  Anchor now at rate=60 (real mixed-cause losses), values from")
    print("  ground truth measured on this machine, not recomputed elsewhere.")
    print("  NEXT: python src\\preflight_simulator_v2_check.py   (should now PASS)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
