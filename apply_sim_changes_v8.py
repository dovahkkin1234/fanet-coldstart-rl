"""
apply_sim_changes_v8.py — PART 1 simulator changes, all at once.

Applies ON TOP OF v7. Requires regeneration afterwards.

════════════════════════════════════════════════════════════════════════════
S1  EPISODE DURATION  40 s -> 1000 s
════════════════════════════════════════════════════════════════════════════
Matches HCPMR / CQMR / IQMR exactly (all run 1000 s). Unblocks two features
that are currently dead BY CONSTRUCTION, not by defect:

    energy                    mean 0.9899, std 0.0205  -- 40 s is too short
                              for battery drain to bind
    estimated_link_lifetime   60.4% of values at the ceiling, because
                              LIFETIME_REF = 60 s EXCEEDS a 40 s episode, so
                              the cap is literally unreachable

LIFETIME_REF IS DELIBERATELY NOT CHANGED. The saturation was an episode-length
artifact, and fixing both at once would make the improvement unattributable.
Fix the cause; leave the constant alone.

drain_time stays at 10 s. Its job is to let in-flight packets land, and at
2-3 hops x ~6 ms/hop a packet completes in well under a second -- 10 s was
already generous and does not need to scale with duration.

════════════════════════════════════════════════════════════════════════════
S2  ALTITUDE BAND  50-150 m (span 100)  ->  100-300 m (span 200)
════════════════════════════════════════════════════════════════════════════
Matches all three competitors, which all place nodes at 100-300 m.

MEASURED COST: about 8% of density.

    scenario       span 100   span 200   retained
    very_dense       17.16      15.75      91.8%
    dense_slow        8.66       7.95      91.8%
    medium_slow       4.14       3.87      93.5%
    sparse_fast       2.34       2.21      94.3%

WHY SO CHEAP. In a thin slab (h << 2R) the communication sphere is effectively
a cylinder, so neighbour volume and total volume BOTH scale linearly with h
and the ratio is independent of altitude. Only once h approaches 2R does the
sphere saturate and density fall as 1/h. With 2R = 500-600 m, span 200 is
still well inside the slab regime.

HONEST CAVEAT FOR THE PAPER: span 200 against 2R = 500-600 is still slab-like
(ratio 0.33-0.40). It matches the literature; it does not make connectivity
genuinely 3-D. That needs span ~ 2R -- which is what S3 probes.

════════════════════════════════════════════════════════════════════════════
S3  ALTITUDE SCOPE-PROBE CELLS, span 500 -- REPORTED SEPARATELY
════════════════════════════════════════════════════════════════════════════
Adds `*_tall` variants at 100-600 m (span 500), where span ~ 2R and
connectivity is genuinely three-dimensional. No competing paper evaluates
this.

    scenario           span 200   span 500   retained
    very_dense_tall      15.75       9.55      55.7%
    dense_slow_tall       7.95       4.82      55.7%
    medium_slow_tall      3.87       2.49      60.3%
    sparse_fast_tall      2.21       1.48      63.2%

THEY ARE EXCLUDED FROM THE MAIN GRID AND FROM ANY MEAN. Density falls ~40%,
so PDR falls with it -- folding these into an average would drag the headline
number down for a reason that has nothing to do with routing quality.
`SCENARIOS` keeps only the four standard cells; `SCENARIOS_TALL` is a separate
dict, and `SCENARIOS_ALL` exists only for scripts that explicitly want both.

════════════════════════════════════════════════════════════════════════════
S4  SUITE B -- CLEAN DENSITY SWEEP
════════════════════════════════════════════════════════════════════════════
FIXED: area 1000x1000, altitude 100-300, comm range 250, speed 10-30, pause 5
VARIED: node count only -- N in {50, 100, 150, 200}

Removes the four-factor confound in Suite A, where node count, area, comm
range AND speed all move together, so no result there is attributable to
density alone. Every parameter here matches HCPMR except the traffic pattern.

    density_50 -> 8.59   density_100 -> 17.37
    density_150 -> 26.14  density_200 -> 34.91   expected one-hop neighbours

════════════════════════════════════════════════════════════════════════════
S5  SUITE C -- CONVERGECAST COMPARABILITY
════════════════════════════════════════════════════════════════════════════
Identical to Suite B plus a single fixed ground sink: every flow terminates at
one stationary node.

WHY SUITE B IS NOT ENOUGH ON ITS OWN. Matching density does NOT make our
numbers comparable to HCPMR's, because the traffic pattern still differs --
random UAV pairs with no guaranteed path versus many-to-one convergecast to a
fixed station. Suite C matches density AND traffic AND altitude AND range AND
speed AND duration, and is therefore the only configuration that produces a
genuinely comparable number.

IMPLEMENTATION. `sink_node` in the config makes _make_flows draw sources
uniformly and fix every destination to node 0, which is pinned stationary at
the area centre at the lowest altitude (a ground station does not fly). This
is a flow-generation and mobility-pinning change, not new physics.

USAGE
    python apply_sim_changes_v8.py --src src [--dry-run]
"""

import argparse
import os
import sys

VERSION = "v8"


class PatchError(RuntimeError):
    pass


def sub(text, old, new, label, path):
    n = text.count(old)
    if n != 1:
        raise PatchError(
            f"[{path}] anchor for '{label}' matched {n} times, expected 1.\n"
            f"  anchor starts: {old[:110]!r}")
    return text.replace(old, new, 1)


# ── S1 + S2: BASE ────────────────────────────────────────────────────────────
A_OLD = """BASE = dict(z_min=50, z_max=150, duration=40.0, drain_time=10.0,
            interference_on=True)"""

A_NEW = '''# S1: duration 40 -> 1000 s, matching HCPMR/CQMR/IQMR. Unblocks `energy`
#     (std was 0.0205 -- 40 s is too short for drain to bind) and
#     `estimated_link_lifetime` (60.4% at ceiling because LIFETIME_REF = 60 s
#     exceeded the whole episode). LIFETIME_REF is deliberately NOT changed:
#     the saturation was an episode-length artifact, and changing both would
#     make the fix unattributable.
# S2: altitude 50-150 (span 100) -> 100-300 (span 200), matching all three
#     competitors. Costs ~8% of density; see the module docstring for why so
#     little (thin-slab regime, density independent of span until h ~ 2R).
# drain_time stays 10 s -- a 2-3 hop packet completes in well under a second,
# so it never needed to scale with duration.
BASE = dict(z_min=100, z_max=300, duration=1000.0, drain_time=10.0,
            interference_on=True)'''


# ── S3 + S4 + S5: scenario dicts ────────────────────────────────────────────
B_OLD = """SCENARIOS = {
    'very_dense':  dict(num_drones=45, area_x=700,  area_y=700,  comm_range=250,
                        speed_min=5,  speed_max=15, pause_max=5.0),
    'dense_slow':  dict(num_drones=30, area_x=800,  area_y=800,  comm_range=250,
                        speed_min=5,  speed_max=15, pause_max=5.0),
    'medium_slow': dict(num_drones=30, area_x=1300, area_y=1300, comm_range=280,
                        speed_min=5,  speed_max=15, pause_max=5.0),
    'sparse_fast': dict(num_drones=20, area_x=1500, area_y=1500, comm_range=300,
                        speed_min=35, speed_max=50, pause_max=2.0),
}"""

B_NEW = '''# ─────────────────────────────────────────────────────────────────────────────
# SUITE A -- the difficulty case. Unchanged geometry; altitude now comes from
# BASE (100-300 m). This is where partition stress and graceful degradation
# live: reachability 100% / 99.2% / 54.7% / 21.4%.
#
# KNOWN LIMITATION, stated rather than hidden: these four cells vary node
# count, area, comm range AND speed together, so a difference between them is
# not attributable to any one factor. SUITE B exists to fix exactly that.
# ─────────────────────────────────────────────────────────────────────────────
SCENARIOS = {
    'very_dense':  dict(num_drones=45, area_x=700,  area_y=700,  comm_range=250,
                        speed_min=5,  speed_max=15, pause_max=5.0),
    'dense_slow':  dict(num_drones=30, area_x=800,  area_y=800,  comm_range=250,
                        speed_min=5,  speed_max=15, pause_max=5.0),
    'medium_slow': dict(num_drones=30, area_x=1300, area_y=1300, comm_range=280,
                        speed_min=5,  speed_max=15, pause_max=5.0),
    'sparse_fast': dict(num_drones=20, area_x=1500, area_y=1500, comm_range=300,
                        speed_min=35, speed_max=50, pause_max=2.0),
}

# ─────────────────────────────────────────────────────────────────────────────
# S3 -- ALTITUDE SCOPE-PROBE, span 500 m (100-600). REPORT SEPARATELY.
#
# Here span ~ 2R, so connectivity is genuinely three-dimensional rather than
# slab-like. No competing paper evaluates this.
#
# *** NEVER FOLD THESE INTO A MEAN WITH SUITE A. *** Density drops ~40%
# (very_dense 15.75 -> 9.55, sparse_fast 2.21 -> 1.48), so PDR drops with it
# for reasons that have nothing to do with routing quality. Averaging them in
# would drag the headline number down and misrepresent both.
# ─────────────────────────────────────────────────────────────────────────────
TALL_BAND = dict(z_min=100, z_max=600)          # span 500

SCENARIOS_TALL = {
    f'{k}_tall': dict(v, **TALL_BAND) for k, v in SCENARIOS.items()
}

# ─────────────────────────────────────────────────────────────────────────────
# SUITE B -- CLEAN DENSITY SWEEP. Node count is the ONLY thing that varies.
# Area, altitude, comm range and speed are all fixed at HCPMR's values, so a
# density result here IS attributable to density.
# Expected one-hop neighbours: 8.59 / 17.37 / 26.14 / 34.91.
# ─────────────────────────────────────────────────────────────────────────────
_DENSITY_FIXED = dict(area_x=1000, area_y=1000, comm_range=250,
                      speed_min=10, speed_max=30, pause_max=5.0)

SCENARIOS_DENSITY = {
    f'density_{n}': dict(_DENSITY_FIXED, num_drones=n)
    for n in (50, 100, 150, 200)
}

# ─────────────────────────────────────────────────────────────────────────────
# SUITE C -- CONVERGECAST COMPARABILITY. Suite B plus one fixed ground sink.
#
# Suite B alone does NOT make our numbers comparable to HCPMR's: matching
# density leaves the TRAFFIC PATTERN different (random UAV pairs vs many-to-one
# to a fixed station). Suite C matches density, traffic, altitude, range, speed
# and duration, and is the only configuration that yields a directly comparable
# number.
#
# `sink_node=0` makes every flow terminate at node 0, which is pinned
# stationary at the area centre at the lowest altitude -- a ground station does
# not fly.
# ─────────────────────────────────────────────────────────────────────────────
SCENARIOS_CONVERGECAST = {
    f'sink_{n}': dict(_DENSITY_FIXED, num_drones=n, sink_node=0)
    for n in (50, 100, 150, 200)
}

# Convenience union for scripts that explicitly want everything. The default
# grid remains SCENARIOS alone -- nothing is silently averaged across suites.
SCENARIOS_ALL = {**SCENARIOS, **SCENARIOS_TALL,
                 **SCENARIOS_DENSITY, **SCENARIOS_CONVERGECAST}'''


# ── S5: sink support in the simulator ───────────────────────────────────────
C_OLD = """        self.num_flows = max(1, self.N // 4)"""
C_NEW = """        # SUITE C: when set, every flow terminates at this node, which is also
        # pinned stationary at the area centre (see _make_flows and __init__).
        self.sink_node = config.get('sink_node', None)
        self.num_flows = max(1, self.N // 4)"""

D_OLD = """        flows = []
        for fid in range(self.num_flows):
            src, dst = self.rng.choice(self.N, size=2, replace=False)"""

D_NEW = """        flows = []
        for fid in range(self.num_flows):
            if self.sink_node is not None:
                # Convergecast: destination fixed, source drawn from the rest.
                dst = int(self.sink_node)
                choices = [i for i in range(self.N) if i != dst]
                src = int(self.rng.choice(choices))
            else:
                src, dst = self.rng.choice(self.N, size=2, replace=False)"""


PATCHES = {
    'generate_dataset_v2.py': [
        (A_OLD, A_NEW, 'S1+S2 duration 1000 s, altitude 100-300'),
        (B_OLD, B_NEW, 'S3+S4+S5 tall probe, density sweep, convergecast'),
    ],
    'simulator_v2.py': [
        (C_OLD, C_NEW, 'S5 sink_node config'),
        (D_OLD, D_NEW, 'S5 convergecast flow generation'),
    ],
}

MARKER = 'SCENARIOS_CONVERGECAST'
GUARD = 'generate_dataset_v2.py'


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
        print(f"ERROR: {GUARD} already patched by v8. Restore from git first.")
        return 2

    staged = {}
    print(f"\n{'=' * 78}\n  SIMULATOR CHANGES {VERSION} — assertion-guarded\n{'=' * 78}")
    for fname, edits in PATCHES.items():
        path = os.path.join(args.src, fname)
        if not os.path.isfile(path):
            raise PatchError(f'missing file: {path}')
        text = open(path, encoding='utf-8').read()
        for old, new, label in edits:
            text = sub(text, old, new, label, fname)
            print(f"    [ok] {fname:<26} {label}")
        staged[path] = text

    if args.dry_run:
        print(f"\n  DRY RUN — {len(staged)} file(s) would change, nothing written.")
        return 0
    for path, text in staged.items():
        open(path, 'w', encoding='utf-8').write(text)
    print(f"\n  WROTE {len(staged)} file(s).\n")
    print("  Episodes are now 25x longer. RUN THE TIMING PROBE FIRST:")
    print("    python verify_sim_changes_v8.py --src src")
    print("  It measures cost at N = 45 / 100 / 200 so the suite schedule is")
    print("  a plan rather than an extrapolation.")
    print("=" * 78 + "\n")
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except PatchError as e:
        print(f"\nPATCH FAILED — nothing was written.\n{e}\n")
        sys.exit(1)
