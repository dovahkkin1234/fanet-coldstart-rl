"""apply_v8a_suites_v1.py -- v8 SPLIT: suites and sink support, WITHOUT the
operating-point change.

Assertion-guarded str.replace. Anchors must match EXACTLY ONCE; edits staged in
memory; nothing written unless every anchor in every file matches. Idempotent.

WHY v8 IS BEING SPLIT

`apply_sim_changes_v8.py` bundles two independent changes:

    S1 + S2   BASE: duration 40 -> 1000 s, altitude 50-150 -> 100-300
    S3 + S4 + S5   scenario dicts (TALL / DENSITY / CONVERGECAST) + sink support

Only S1+S2 moves the operating point, and the SP-BP parity gate for the new RL
environment needs the CURRENT 40 s / 50-150 m point as its fixed reference
(FILE2 §3). But the convergecast oracle panel -- which must run BEFORE any
convergecast training data is generated -- needs S3-S5.

So this patch applies S3-S5 only. S1+S2 ship separately as v8b, after the
parity gate, immediately before regeneration.

v8 also cannot be applied as written any more: its BASE anchor targets
`generate_dataset_v2.py`, but v11 moved BASE/SCENARIOS/RATES into
`src/config_v2.py`. v8a targets config_v2; v8b must be rewritten the same way.

SAFETY -- WHY THIS IS A NO-OP FOR SUITE A

The sink edit adds `self.sink_node = config.get('sink_node', None)` and branches
in `_make_flows`. When `sink_node` is unset the else-branch is **byte-identical**
to the current code, including the same `self.rng.choice(self.N, size=2,
replace=False)` call -- so RNG consumption, and therefore every Suite A result,
is unchanged. Verified by equivalence control after applying.

The three suite dicts are ADDITIVE: `config_v2.SCENARIOS` is untouched, and
`get_suite('default')` keeps returning the same four scenarios. What changes is
that `get_suite('density'|'convergecast'|'tall')` stops raising.

USAGE
    python apply_v8a_suites_v1.py --root . --dry-run
    python apply_v8a_suites_v1.py --root .
"""
import argparse, io, os, sys

CFG = os.path.join('src', 'config_v2.py')
SIM = os.path.join('src', 'simulator_v2.py')
# Guards must be strings that exist ONLY AFTER patching. The first attempt
# used 'SCENARIOS_CONVERGECAST = {' which also matches the pre-existing
# EMPTY declaration 'SCENARIOS_CONVERGECAST = {}', so config_v2 was
# wrongly reported as already applied and silently skipped.
GUARDS = {CFG: '_DENSITY_FIXED', SIM: 'self.sink_node'}

# ── config_v2.py: populate the three declared-but-empty suite dicts ─────────
CFG_OLD = """# ── v8 suites: declared, deliberately empty until v8 is applied ────────────
SCENARIOS_TALL = {}
SCENARIOS_DENSITY = {}
SCENARIOS_CONVERGECAST = {}"""

CFG_NEW = '''# ── v8a suites ─────────────────────────────────────────────────────────────
# Populated by apply_v8a_suites_v1.py. SCENARIOS (Suite A) is deliberately NOT
# touched, and BASE is NOT touched -- the operating-point change is v8b, which
# ships after the SP-BP parity gate.

# S3 -- ALTITUDE SCOPE-PROBE, span 500 m (100-600). REPORT SEPARATELY: this is
# a scope probe, not a Suite A cell, and must never be pooled with one.
SCENARIOS_TALL = {
    'tall_probe': dict(num_drones=30, area_x=1300, area_y=1300, comm_range=280,
                       speed_min=5, speed_max=15, pause_max=5.0,
                       z_min=100, z_max=600),
}

# S4 -- SUITE B, CLEAN DENSITY SWEEP. Fixed geometry; vary node count ONLY, so
# any density effect is attributable to density and nothing else. This is the
# control Suite A lacks (Suite A varies node count, area, range AND speed
# together -- FILE1 §4.3's recorded methodological weakness).
_DENSITY_FIXED = dict(area_x=1000, area_y=1000, comm_range=250,
                      speed_min=10, speed_max=30, pause_max=5.0)

SCENARIOS_DENSITY = {
    f'density_{n}': dict(_DENSITY_FIXED, num_drones=n)
    for n in (50, 100, 150, 200)
}

# S5 -- SUITE C, CONVERGECAST COMPARABILITY. Suite B plus one fixed ground
# sink. Suite B alone does NOT make our numbers comparable to HCPMR's: matching
# density leaves the TRAFFIC PATTERN different (random UAV pairs vs many-to-one
# to a fixed station). `sink_node=0` makes every flow terminate at node 0,
# which v9 pins stationary at the area centre at z_min -- a ground station does
# not fly.
SCENARIOS_CONVERGECAST = {
    f'sink_{n}': dict(_DENSITY_FIXED, num_drones=n, sink_node=0)
    for n in (50, 100, 150, 200)
}

# Convenience union for scripts that explicitly want everything. The default
# grid remains SCENARIOS alone -- nothing is silently averaged across suites.
SCENARIOS_ALL = {**SCENARIOS, **SCENARIOS_TALL,
                 **SCENARIOS_DENSITY, **SCENARIOS_CONVERGECAST}'''

# ── simulator_v2.py: sink support (no-op unless sink_node is set) ───────────
SIM_OLD_A = """        self.num_flows = max(1, self.N // 4)"""
# NOTE: this comment text is VERBATIM from v8. apply_sink_fix_v9.py anchors on
# it exactly; rewording it (as a first draft of this patch did) makes v9 fail
# with 0 matches. Do not "improve" the wording.
SIM_NEW_A = """        # SUITE C: when set, every flow terminates at this node, which is also
        # pinned stationary at the area centre (see _make_flows and __init__).
        self.sink_node = config.get('sink_node', None)
        self.num_flows = max(1, self.N // 4)"""

SIM_OLD_B = """        flows = []
        for fid in range(self.num_flows):
            src, dst = self.rng.choice(self.N, size=2, replace=False)"""
SIM_NEW_B = """        flows = []
        for fid in range(self.num_flows):
            if self.sink_node is not None:
                # Convergecast: destination fixed, source drawn from the rest.
                dst = int(self.sink_node)
                choices = [i for i in range(self.N) if i != dst]
                src = int(self.rng.choice(choices))
            else:
                src, dst = self.rng.choice(self.N, size=2, replace=False)"""

EDITS = {
    CFG: [(CFG_OLD, CFG_NEW)],
    SIM: [(SIM_OLD_A, SIM_NEW_A), (SIM_OLD_B, SIM_NEW_B)],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    staged, allok = {}, True
    for rel, edits in EDITS.items():
        path = os.path.join(a.root, rel)
        if not os.path.exists(path):
            print(f"  ERROR: {path} not found"); return 1
        text = io.open(path, encoding='utf-8').read()
        if GUARDS[rel] in text:
            print(f"  {rel}: ALREADY APPLIED, skipping"); continue
        for i, (old, new) in enumerate(edits, 1):
            n = text.count(old)
            if n != 1:
                print(f"  [{rel}] anchor {i}: matched {n} times, expected 1  <-- ABORT")
                allok = False
            else:
                print(f"  [{rel}] anchor {i}: OK")
                text = text.replace(old, new, 1)
        staged[path] = text

    if not allok:
        print("\n  NO FILES WRITTEN."); return 1
    if not staged:
        print("\n  Nothing to do."); return 0
    if a.dry_run:
        print(f"\n  DRY RUN OK -- {len(staged)} file(s) would change. Nothing written.")
        return 0
    for path, s in staged.items():
        io.open(path, 'w', encoding='utf-8').write(s)
        print(f"  WROTE {path}")
    print("\n  NEXT: python apply_sink_fix_v9.py --src src   (pins the sink)")
    print("        then the convergecast oracle panel.")
    print("  NOTE: BASE is UNCHANGED -- still 40 s / 50-150 m. That is deliberate;")
    print("        the operating-point change is v8b, after the parity gate.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
