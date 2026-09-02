"""
apply_sink_fix_v9.py — fixes the one real code bug found in the v7/v8 audit.

Applies ON TOP OF v8.

════════════════════════════════════════════════════════════════════════════
THE BUG: the Suite-C sink was documented as a ground station but implemented
as an ordinary flying drone.
════════════════════════════════════════════════════════════════════════════
v8's docstring claimed `sink_node=0` designates a node "pinned stationary at
the area centre at the lowest altitude -- a ground station does not fly."

MEASURED: node 0 moved 156.4 m in 10 s and started at (517, 424, 215) --
mid-air, off-centre, and mobile. The pinning was never implemented.

WHY IT MATTERS RATHER THAN BEING COSMETIC. Suite C exists for ONE purpose:
to produce a number directly comparable with HCPMR / CQMR / IQMR. All three
route to a STATIONARY GROUND station -- HCPMR's GCC, CQMR/IQMR's TBS, the
latter explicitly "at the center of the base of the cylindrical region." A
flying, wandering sink is a different experiment, and any comparison drawn
from it would be invalid in exactly the dimension Suite C is meant to control.

  F1  A node named by `sink_node` is placed at the area centre at `z_min` and
      held there for the whole episode: velocity forced to zero every step,
      so it cannot drift, and no waypoint is ever pursued.

  F2  An assertion at construction confirms the sink really is where it should
      be. The bug slipped through because the docstring was believed instead
      of tested; this makes the claim checkable.

════════════════════════════════════════════════════════════════════════════
NOT FIXED HERE -- two issues that are decisions, not defects
════════════════════════════════════════════════════════════════════════════
(a) DATASET SIZE. At 1000 s the frame count per episode goes 80 -> 2000. Suite
    A at 50 seeds becomes 1,200,000 frames, projecting to ~3.3 GB for
    frames.npz against the current 136 MB, plus ~21 h of generation. The
    independent audit loads frames.npz fully into memory, so this is a real
    operational limit, not just disk. It needs a seed-count decision, and
    reduced seeds must keep at least one seed in EACH of the train / val /
    test ranges (101-135 / 136-142 / 143-150) or the v5 split assertion will
    correctly abort generation.

(b) LOAD SWEEP. RATES is still (0.5, 2.0, 4.0), calibrated for 40 s episodes.
    At 1000 s, dense_slow at rate 2.0 collapses from PDR 0.3286 to 0.0250.
    Rate 0.05 at 1000 s reproduces roughly what rate 0.5 gave at 40 s.
    Recalibration is a design decision and is deliberately left to the user.

    Note that recalibration also resolves a Suite-C bottleneck: at N=200 the
    sink receives 50 flows, and at rate 2.0 that is 100 pkt/s against a
    per-node service capacity of exactly 100 pkt/s (SERVICE_RATE=1 per 0.01 s
    slot) -- the sink saturates by construction, independent of routing. At
    rate 0.15 it is 7.5 pkt/s and the bottleneck disappears.

USAGE
    python apply_sink_fix_v9.py --src src [--dry-run]
"""

import argparse
import os
import sys

VERSION = "v9"


class PatchError(RuntimeError):
    pass


def sub(text, old, new, label, path):
    n = text.count(old)
    if n != 1:
        raise PatchError(
            f"[{path}] anchor for '{label}' matched {n} times, expected 1.\n"
            f"  anchor starts: {old[:110]!r}")
    return text.replace(old, new, 1)


A_OLD = """        # SUITE C: when set, every flow terminates at this node, which is also
        # pinned stationary at the area centre (see _make_flows and __init__).
        self.sink_node = config.get('sink_node', None)"""

A_NEW = '''        # SUITE C: when set, every flow terminates at this node, and the node
        # is pinned stationary at the area centre at z_min -- a ground station
        # does not fly. See _pin_sink() below.
        self.sink_node = config.get('sink_node', None)'''

# The pin must happen AFTER the drones are constructed. Anchor on the flow
# construction line, which follows drone creation.
B_OLD = """        self.num_flows = max(1, self.N // 4)
        self.flows = self._make_flows()"""

B_NEW = '''        self.num_flows = max(1, self.N // 4)
        self._pin_sink()
        self.flows = self._make_flows()'''

C_OLD = """    def _make_flows(self):"""

C_NEW = '''    def _pin_sink(self):
        """Place the Suite-C sink at the area centre, at z_min, and hold it.

        HCPMR's GCC and CQMR/IQMR's TBS are stationary GROUND stations --
        CQMR/IQMR place theirs explicitly "at the center of the base of the
        cylindrical region". Suite C exists to be comparable with those, so a
        wandering airborne sink would invalidate the one thing the suite
        controls for.

        The v8 version documented this and never implemented it: the sink
        moved 156 m in 10 s and started at (517, 424, 215). The assertion at
        the end makes the claim checkable rather than believed.
        """
        if self.sink_node is None:
            return
        i = int(self.sink_node)
        d = self.drones[i]
        cx, cy = self.area_x / 2.0, self.area_y / 2.0
        d.x, d.y, d.z = cx, cy, float(self.z_min)
        d.dest_x, d.dest_y, d.dest_z = cx, cy, float(self.z_min)
        d.current_speed = 0.0
        d.pause_max = 0.0
        d.speed_min = d.speed_max = 0.0

        # Freeze it: step() becomes a no-op for this drone, so nothing in the
        # mobility model can nudge it off the ground station's position.
        def _frozen(dt, _d=d, _cx=cx, _cy=cy, _z=float(self.z_min)):
            _d.x, _d.y, _d.z = _cx, _cy, _z
            _d.vx = _d.vy = _d.vz = 0.0
        d.step = _frozen

        d.step(1.0)
        assert abs(d.x - cx) < 1e-9 and abs(d.y - cy) < 1e-9 \\
            and abs(d.z - self.z_min) < 1e-9, \\
            f'sink pinning failed: node {i} at ({d.x},{d.y},{d.z}), ' \\
            f'expected ({cx},{cy},{self.z_min})'

    def _make_flows(self):'''


PATCHES = {
    'simulator_v2.py': [
        (A_OLD, A_NEW, 'F1 comment now matches behaviour'),
        (B_OLD, B_NEW, 'F1 pin the sink before flows are built'),
        (C_OLD, C_NEW, 'F1+F2 _pin_sink with a self-check'),
    ],
}

MARKER = '_pin_sink'
GUARD = 'simulator_v2.py'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='src')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    g = os.path.join(args.src, GUARD)
    if not os.path.isfile(g):
        print(f"ERROR: {g} not found. Run from the repo root or pass --src.")
        return 2
    txt = open(g, encoding='utf-8').read()
    if MARKER in txt:
        print(f"ERROR: {GUARD} already patched by v9. Restore from git first.")
        return 2
    if 'sink_node' not in txt:
        print(f"ERROR: {GUARD} has no sink_node — apply v8 first.")
        return 2

    staged = {}
    print(f"\n{'=' * 78}\n  SINK PINNING FIX {VERSION} — assertion-guarded\n{'=' * 78}")
    for fname, edits in PATCHES.items():
        path = os.path.join(args.src, fname)
        text = open(path, encoding='utf-8').read()
        for old, new, label in edits:
            text = sub(text, old, new, label, fname)
            print(f"    [ok] {fname:<20} {label}")
        staged[path] = text

    if args.dry_run:
        print(f"\n  DRY RUN — {len(staged)} file(s) would change, nothing written.")
        return 0
    for path, text in staged.items():
        open(path, 'w', encoding='utf-8').write(text)
    print(f"\n  WROTE {len(staged)} file(s).")
    print("\n  STILL OPEN — decisions, not defects (see module docstring):")
    print("    (a) 1000 s -> 1.2 M frames, ~3.3 GB, ~21 h for Suite A at 50 seeds")
    print("    (b) RATES (0.5, 2.0, 4.0) saturate at 1000 s; rate 2.0 gives PDR 0.025")
    print("=" * 78 + "\n")
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except PatchError as e:
        print(f"\nPATCH FAILED — nothing was written.\n{e}\n")
        sys.exit(1)
