"""apply_energy_norm_fix_v17.py -- stop the energy feature from being mis-normalised.

Assertion-guarded str.replace, idempotent, one anchor.

THE BUG

features_v2.py carries its OWN copy of the energy constant:

    INITIAL_ENERGY_REF = 100.0                      # features_v2.py:151
    'initial_energy': INITIAL_ENERGY_REF,           # norm_constants()
    d.get('energy', 0.0) / nc['initial_energy']     # the energy node feature

This is entirely independent of simulator_v2.INITIAL_ENERGY. Every OTHER value
in the same norm_constants() dict is read from `cfg` -- area_y, z_min, z_max,
speed_max, comm_range -- and only initial_energy is hardcoded from a module
constant.

WHY IT MATTERS NOW

The operating point is moving to INITIAL_ENERGY = 8000 (80x), because that is
the smallest battery that opens a congestion window at 1000 s. With this bug in
place, regeneration would produce nodes starting at energy 8000, normalised by
100 -- so the energy node feature would read ~80.0 instead of ~1.0, for every
node in every frame, silently.

NOTHING WOULD CATCH IT. G1 has zero energy references; G2's six checks (queues,
PDR-vs-load, drop taxonomy, activity band, interference on/off, reproducibility)
have zero energy references. The dataset audit checks feature distributions but
against no absolute reference for this feature. The mis-scaled feature would
flow straight into the GNN.

This is the same config-duplication class v11 eliminated for SCENARIOS/RATES/
BASE. Energy simply was not in scope then.

THE FIX

    'initial_energy': INITIAL_ENERGY_REF,
->  'initial_energy': float(cfg.get('initial_energy', INITIAL_ENERGY_REF)),

Matching how every other entry in that dict already works. The default is
unchanged, so any config that does NOT set 'initial_energy' produces a
bit-identical dataset to before -- verified by equivalence control.

USAGE
    python apply_energy_norm_fix_v17.py --src src --dry-run
    python apply_energy_norm_fix_v17.py --src src
"""
import argparse, io, os, sys

TARGET = 'features_v2.py'
GUARD = "cfg.get('initial_energy'"

OLD = "        'initial_energy': INITIAL_ENERGY_REF,"
NEW = ("        # v17: follow the CONFIGURED battery, not a hardcoded twin of\n"
       "        # simulator_v2.INITIAL_ENERGY. Every other entry in this dict\n"
       "        # already reads from cfg; this one did not, so raising the\n"
       "        # simulator's battery to 8000 would have normalised energy by\n"
       "        # 100 and emitted a feature of ~80.0 instead of ~1.0, silently,\n"
       "        # with no gate able to detect it.\n"
       "        'initial_energy': float(cfg.get('initial_energy', INITIAL_ENERGY_REF)),")


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
    print("  Default unchanged -- configs without 'initial_energy' are bit-identical.")
    print("  NEXT: python verify_energy_norm_v17.py --src " + a.src)
    return 0


if __name__ == '__main__':
    sys.exit(main())
