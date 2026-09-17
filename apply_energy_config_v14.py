"""apply_energy_config_v14.py -- make INITIAL_ENERGY config-overridable.

Assertion-guarded str.replace, idempotent. One anchor.

WHY

The energy budget is the single remaining blocker on dataset regeneration.
Measured: at 1000 s the network can sustain only ~6.6 per-flow rate before
nodes start dying, while congestion requires ~40-80. A factor of ~10 apart, so
NO rate grid satisfies both -- 1000 s episodes and a congestion regime are
currently mutually exclusive.

    INITIAL_ENERGY = 100.0, TX_COST = 0.02  ->  node dies after 5,000 transmits

That is a calibration choice, not physics. A real UAV radio draws milliwatts
against a watt-hour battery; 5,000 packets is low by orders of magnitude. So
raising it is MORE defensible than the current value, not less.

This patch does NOT change the default. It makes the constant overridable per
config so the sweep can find the right value empirically before anything is
pinned. `INITIAL_ENERGY` stays 100.0 unless a config explicitly overrides it,
so every existing result and gate is bit-identical after this patch -- verified
by an equivalence control.

WHAT CHANGES

    self.energy = [INITIAL_ENERGY] * self.N
->  self.initial_energy = float(config.get('initial_energy', INITIAL_ENERGY))
    self.energy = [self.initial_energy] * self.N

USAGE
    python apply_energy_config_v14.py --src src --dry-run
    python apply_energy_config_v14.py --src src
"""
import argparse, io, os, sys

TARGET = 'simulator_v2.py'
GUARD = 'self.initial_energy'

OLD = """        self.energy = [INITIAL_ENERGY] * self.N"""

NEW = """        # v14: overridable per config. Default is UNCHANGED (INITIAL_ENERGY),
        # so every pre-v14 result is bit-identical unless a config explicitly
        # sets 'initial_energy'. Exists because the energy ceiling -- not the
        # rate grid -- is what makes 1000 s episodes and a congestion regime
        # mutually exclusive, and that needs to be measured before being pinned.
        self.initial_energy = float(config.get('initial_energy', INITIAL_ENERGY))
        self.energy = [self.initial_energy] * self.N"""


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
    print("  Default unchanged (100.0). Override with config 'initial_energy'.")
    print("  NEXT: python src\\sweep_energy_ceiling.py --max_workers 4")
    return 0


if __name__ == '__main__':
    sys.exit(main())
