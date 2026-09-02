"""apply_config_module_v11.py -- point every hardcoded config site at config_v2.

Assertion-guarded str.replace. Anchors must match EXACTLY ONCE per file; edits
are staged in memory and NOTHING is written unless every anchor in every file
matches. Idempotent.

Replaces the module-level SCENARIOS / RATES / BASE definitions in each consumer
with an import from config_v2, so there is one place to change and one place for
v8 to patch. Values are byte-identical today -- verified by comparison before
and after -- so this patch must not move any number.

Run apply_config_module_v11.py AFTER copying config_v2.py into src\\.
"""
import argparse, io, os, sys

# (filename, anchor) -- each anchor is the file's own config block
TARGETS = {}

SCEN_BLOCK = """SCENARIOS = {
    'very_dense':  dict(num_drones=45, area_x=700,  area_y=700,  comm_range=250,
                        speed_min=5,  speed_max=15, pause_max=5.0),
    'dense_slow':  dict(num_drones=30, area_x=800,  area_y=800,  comm_range=250,
                        speed_min=5,  speed_max=15, pause_max=5.0),
    'medium_slow': dict(num_drones=30, area_x=1300, area_y=1300, comm_range=280,
                        speed_min=5,  speed_max=15, pause_max=5.0),
    'sparse_fast': dict(num_drones=20, area_x=1500, area_y=1500, comm_range=300,
                        speed_min=35, speed_max=50, pause_max=2.0),
}
RATES = [0.5, 2.0, 4.0]
BASE = dict(z_min=50, z_max=150, duration=40.0, drain_time=10.0,
            interference_on=True)"""

SCEN_NEW = """# Config now lives in config_v2.py -- see the note there on why eight
# independent copies of this block were a latent hazard.
from config_v2 import SCENARIOS, RATES, BASE, get_suite, provenance  # noqa: F401"""

# generate_dataset_v2.py interleaves SEEDS between RATES and BASE, so it needs
# its own anchor. SEEDS stays local: it is disjoint from G3's 1-30 on purpose
# and is a property of the dataset, not of the scenario grid.
GD_BLOCK = SCEN_BLOCK.replace(
    "RATES = [0.5, 2.0, 4.0]\nBASE =",
    "RATES = [0.5, 2.0, 4.0]\nSEEDS = list(range(101, 151))          # 50 seeds, disjoint from G3\nBASE =")
GD_NEW = SCEN_NEW + "\nSEEDS = list(range(101, 151))          # 50 seeds, disjoint from G3"

FILES = ['experiment_headroom.py', 'experiment_queue_weight.py',
         'experiment_spbp_mechanism.py', 'experiment_collision_model.py',
         'experiment_calibration_sensitivity.py', 'generate_dataset_v2.py',
         'experiment_locality_cost.py', 'preflight_teachers_v2_check.py']

ANCHORS = {'generate_dataset_v2.py': (GD_BLOCK, GD_NEW)}

# -- gap fix: two files missed by the first cut of this patch --------------
# experiment_locality_cost.py and preflight_teachers_v2_check.py both define
# their OWN SCENARIOS (the second under different formatting -- per-scenario
# degree comments -- so it needs its own anchor). Worse: both also hardcode
# z_min=50, z_max=150 and a duration=40.0 default AS LITERALS, independent of
# BASE, in their own argparse setup. Value-identical today, disconnected by
# construction -- the same defect class as the SCENARIOS duplication, just one
# layer deeper. v8 touches exactly these two knobs.

LOC_OLD = """SCENARIOS = {
    'very_dense':  dict(num_drones=45, area_x=700,  area_y=700,  comm_range=250,
                        speed_min=5,  speed_max=15, pause_max=5.0),
    'dense_slow':  dict(num_drones=30, area_x=800,  area_y=800,  comm_range=250,
                        speed_min=5,  speed_max=15, pause_max=5.0),
    'medium_slow': dict(num_drones=30, area_x=1300, area_y=1300, comm_range=280,
                        speed_min=5,  speed_max=15, pause_max=5.0),
    'sparse_fast': dict(num_drones=20, area_x=1500, area_y=1500, comm_range=300,
                        speed_min=35, speed_max=50, pause_max=2.0),
}
RATES = [0.5, 2.0, 4.0]
HORIZONS = ['spbp_k1', 'spbp_k2', 'spbp_k3', 'spbp_k4', 'spbp_kinf']"""
LOC_NEW = """# Config now lives in config_v2.py.
from config_v2 import SCENARIOS, RATES, BASE  # noqa: F401
HORIZONS = ['spbp_k1', 'spbp_k2', 'spbp_k3', 'spbp_k4', 'spbp_kinf']"""

LOC_ARGS_OLD = """    ap.add_argument('--duration', type=float, default=40.0)
    ap.add_argument('--drain_time', type=float, default=10.0)"""
LOC_ARGS_NEW = """    ap.add_argument('--duration', type=float, default=BASE['duration'])
    ap.add_argument('--drain_time', type=float, default=BASE['drain_time'])"""

LOC_BASE_OLD = """    base = dict(z_min=50, z_max=150, duration=args.duration,
                interference_on=True, drain_time=args.drain_time)"""
LOC_BASE_NEW = """    base = dict(z_min=BASE['z_min'], z_max=BASE['z_max'], duration=args.duration,
                interference_on=BASE['interference_on'], drain_time=args.drain_time)"""

PF_OLD = """SCENARIOS = {
    'very_dense':  dict(num_drones=45, area_x=700,  area_y=700,  comm_range=250,
                        speed_min=5,  speed_max=15, pause_max=5.0),   # degree 18.0
    'dense_slow':  dict(num_drones=30, area_x=800,  area_y=800,  comm_range=250,
                        speed_min=5,  speed_max=15, pause_max=5.0),   # degree 9.2
    'medium_slow': dict(num_drones=30, area_x=1300, area_y=1300, comm_range=280,
                        speed_min=5,  speed_max=15, pause_max=5.0),   # degree 4.4
    'sparse_fast': dict(num_drones=20, area_x=1500, area_y=1500, comm_range=300,
                        speed_min=35, speed_max=50, pause_max=2.0),   # degree 2.5
}"""
PF_NEW = """# Config now lives in config_v2.py. Per-scenario node degree (measured in the
# M3.5 audit, kept here for readability): very_dense 18.0, dense_slow 9.2,
# medium_slow 4.4, sparse_fast 2.5.
from config_v2 import SCENARIOS, BASE  # noqa: F401"""

ANCHORS['experiment_locality_cost.py'] = (LOC_OLD, LOC_NEW)
ANCHORS['preflight_teachers_v2_check.py'] = (PF_OLD, PF_NEW)

# second-pass edits applied to the SAME files after the primary anchor, keyed
# by filename -> list of (old, new). Only literal z_min/z_max/duration fixes.
PF_ARGS_OLD = """    ap.add_argument('--rates', type=float, nargs='+', default=[0.5, 2.0, 4.0])
    ap.add_argument('--duration', type=float, default=40.0)
    ap.add_argument('--drain_time', type=float, default=10.0)"""
PF_ARGS_NEW = """    from config_v2 import RATES  # local import: only main() needs it
    ap.add_argument('--rates', type=float, nargs='+', default=RATES)
    ap.add_argument('--duration', type=float, default=BASE['duration'])
    ap.add_argument('--drain_time', type=float, default=BASE['drain_time'])"""

PF_BASE_OLD = """    base = dict(z_min=50, z_max=150, duration=args.duration,
                interference_on=True, drain_time=args.drain_time)"""
PF_BASE_NEW = """    base = dict(z_min=BASE['z_min'], z_max=BASE['z_max'], duration=args.duration,
                interference_on=BASE['interference_on'], drain_time=args.drain_time)"""

SECOND_PASS = {
    'experiment_locality_cost.py': [(LOC_ARGS_OLD, LOC_ARGS_NEW),
                                    (LOC_BASE_OLD, LOC_BASE_NEW)],
    'preflight_teachers_v2_check.py': [(PF_ARGS_OLD, PF_ARGS_NEW),
                                       (PF_BASE_OLD, PF_BASE_NEW)],
}

GUARD = 'from config_v2 import'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='src')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    if not os.path.exists(os.path.join(a.src, 'config_v2.py')):
        print(f"  ERROR: {a.src}/config_v2.py not found -- copy it in first")
        return 1

    staged, allok, skipped = {}, True, []
    for fn in FILES:
        path = os.path.join(a.src, fn)
        if not os.path.exists(path):
            print(f"  {fn:<42} NOT FOUND -- abort"); allok = False; continue
        text = io.open(path, encoding='utf-8').read()
        if GUARD in text:
            print(f"  {fn:<42} already applied"); skipped.append(fn); continue
        old_blk, new_blk = ANCHORS.get(fn, (SCEN_BLOCK, SCEN_NEW))
        n = text.count(old_blk)
        if n != 1:
            print(f"  {fn:<42} anchor matched {n} times, expected 1  <-- ABORT")
            allok = False
            continue
        print(f"  {fn:<42} anchor OK")
        text = text.replace(old_blk, new_blk, 1)

        ok2 = True
        for j, (o2, n2) in enumerate(SECOND_PASS.get(fn, []), 1):
            m = text.count(o2)
            if m != 1:
                print(f"  {fn:<42} 2nd-pass anchor {j} matched {m} times, "
                      f"expected 1  <-- ABORT")
                ok2 = False
            else:
                print(f"  {fn:<42} 2nd-pass anchor {j} OK")
                text = text.replace(o2, n2, 1)
        if not ok2:
            allok = False
            continue
        staged[path] = text

    if not allok:
        print("\n  NO FILES WRITTEN. Anchors must match exactly once in every file.")
        return 1
    if not staged:
        print("\n  Nothing to do."); return 0
    if a.dry_run:
        print(f"\n  DRY RUN OK -- {len(staged)} file(s) would change, "
              f"{len(skipped)} already done. Nothing written.")
        return 0
    for path, s in staged.items():
        io.open(path, 'w', encoding='utf-8').write(s)
    print(f"\n  WROTE {len(staged)} file(s)")
    print("  NEXT: python verify_config_module_v11.py --src " + a.src)
    return 0


if __name__ == '__main__':
    sys.exit(main())
