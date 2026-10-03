"""apply_v8b_operating_point_STAGED.py -- duration + altitude + energy, together.

*** DO NOT RUN THIS YET ***

This patch is staged, not scheduled. It exists now so that when the SP-BP
parity gate at the current 40 s / 50-150 m reference finally passes, the
operating-point change is one atomic, tested step -- not something reassembled
from memory weeks from now, at which point it would be easy to move the
reference point (duration, altitude) while forgetting the third piece that a
separate investigation thread discovered independently (energy).

WHY THIS EXISTS AS ITS OWN FILE

`apply_sim_changes_v8.py`'s S1+S2 block already contains the correct duration/
altitude change (BASE: 40s/50-150m -> 1000s/100-300m). It cannot be applied
as-is: its anchor targets `generate_dataset_v2.py`, but v11 moved BASE into
`config_v2.py`, so the anchor matches zero times there. v8a already split out
S3-S5 (the suite dicts + sink support) and applied those separately, correctly
targeting config_v2.py, while deliberately leaving S1+S2 (BASE) untouched.

This file is "v8b" -- S1+S2, retargeted to config_v2.py, PLUS the energy
finding, which did not exist when v8 was originally written:

    the smallest INITIAL_ENERGY that opens a congestion window at 1000 s is
    8000 (80x default) -- measured directly, battery=4000 still shows dead
    nodes and nonzero energy_share at every rate above 40; battery=8000 shows
    energy_share=0.000 and dead=0 at rates 60/80/120/160.

Bundling all three into one patch is the point: duration, altitude, and
energy must move together or the operating point is self-contradictory (1000 s
episodes at the default 100-energy battery reopen the exact energy-death
regime the whole investigation was about).

WHAT THIS PATCH DOES, PRECISELY

  1. BASE -- duration 1000s, altitude 100-300m (v8's original S1+S2 text,
     verbatim, retargeted to config_v2.py), PLUS initial_energy=8000 added as
     a new key.
  2. provenance() -- adds 'initial_energy': BASE['initial_energy'] to its
     returned dict, so every future results JSON that calls provenance()
     self-describes the battery without a separate 'resolved_base' patch-up.

WHAT THIS PATCH DELIBERATELY DOES NOT DO

  - Does not touch RATES. The per-scenario grid redesign (LOAD_DENSITY_
    DESIGN.md) is a separate change, applied after this one, once the parity
    gate has run at the NEW operating point this patch establishes.
  - Does not run G1/G2 for you. Re-run them immediately after applying, same
    discipline as v14/v17: neither gate references energy, so passing them
    confirms nothing else regressed, it does not validate the energy change
    itself -- the energy sweep and duration-transfer checks already did that.
  - Does not apply v18's max_queue/ttl/lifetime_ref pattern to anything new.
    Unrelated to this patch.

WHEN TO ACTUALLY RUN THIS

Only after the FANETEnvV2 / SP-BP parity gate has passed at the CURRENT 40 s /
50-150 m reference (FILE2 execution order, step 7). Running it earlier moves
the parity reference out from under a gate that hasn't run yet.

USAGE (when the time comes)
    python apply_v8b_operating_point_STAGED.py --src src --dry-run
    python apply_v8b_operating_point_STAGED.py --src src
"""
import argparse, io, os, sys

TARGET = 'config_v2.py'
GUARD = "'initial_energy': BASE['initial_energy']"

# ── S1+S2, verbatim from apply_sim_changes_v8.py's A_OLD/A_NEW, retargeted
# to config_v2.py (v8's own anchor targets generate_dataset_v2.py and cannot
# match here), PLUS initial_energy folded into the same BASE literal so the
# duration/altitude/energy move lands as one edit, not three.
OLD_BASE = """BASE = dict(z_min=50, z_max=150, duration=40.0, drain_time=10.0,
            interference_on=True)"""

NEW_BASE = '''# v8b: duration + altitude (verbatim from apply_sim_changes_v8.py S1+S2,
# retargeted here -- v8's own anchor cannot match config_v2.py, since v11
# moved BASE here after v8 was written) PLUS initial_energy, bundled into one
# change because 1000 s episodes and a congestion regime are mutually
# exclusive at the default battery -- moving duration without energy would
# silently reopen the energy-death regime this investigation started from.
#
# S1: duration 40 -> 1000 s, matching HCPMR/CQMR/IQMR. Unblocks `energy`
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
# ENERGY: INITIAL_ENERGY 100 -> 8000 (80x). Measured directly (not estimated):
#     battery=4000 still shows dead nodes and nonzero energy_share at every
#     rate above 40; battery=8000 shows energy_share=0.000, dead=0 at rates
#     60/80/120/160. 8000 is the smallest battery tested that clears it.
BASE = dict(z_min=100, z_max=300, duration=1000.0, drain_time=10.0,
            interference_on=True, initial_energy=8000.0)'''

OLD_PROV = """    return {'duration': BASE['duration'], 'drain_time': BASE['drain_time'],
            'z_min': BASE['z_min'], 'z_max': BASE['z_max'],
            'interference_on': BASE['interference_on'],
            'rates': list(RATES), 'scenarios': sorted(SCENARIOS),
            'config_module': 'config_v2'}"""

NEW_PROV = """    return {'duration': BASE['duration'], 'drain_time': BASE['drain_time'],
            'z_min': BASE['z_min'], 'z_max': BASE['z_max'],
            'interference_on': BASE['interference_on'],
            'initial_energy': BASE['initial_energy'],
            'rates': list(RATES), 'scenarios': sorted(SCENARIOS),
            'config_module': 'config_v2'}"""

EDITS = [(OLD_BASE, NEW_BASE), (OLD_PROV, NEW_PROV)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='src')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--i-confirm-the-parity-gate-passed', action='store_true',
                    help='required in addition to --dry-run absence; refuses '
                         'to write without it, since this moves the parity '
                         'reference point')
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
        print(f"\n  DRY RUN OK -- {len(EDITS)}/{len(EDITS)} anchors matched.")
        print("  Still staged. Not applying without --i-confirm-the-parity-gate-passed.")
        return 0
    if not a.i_confirm_the_parity_gate_passed:
        print("\n  REFUSING TO WRITE.")
        print("  This patch moves the 40s/50-150m parity reference that the")
        print("  FANETEnvV2 / SP-BP parity gate (FILE2 execution order step 7)")
        print("  needs to run against. Re-run with")
        print("  --i-confirm-the-parity-gate-passed once that gate has actually passed.")
        return 1

    io.open(path, 'w', encoding='utf-8').write(staged)
    print(f"\n  WROTE {path}")
    print("  BASE is now 1000s / 100-300m / initial_energy=8000.")
    print("  NEXT: re-run G1/G2 (won't validate energy, but confirms no other")
    print("  regression), then re-derive RATES per LOAD_DENSITY_DESIGN.md.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
