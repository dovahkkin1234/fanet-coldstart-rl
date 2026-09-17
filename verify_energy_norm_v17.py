"""verify_energy_norm_v17.py -- prove the energy feature normalises correctly.

CHECKS
  1. DEFAULT UNCHANGED -- a config without 'initial_energy' produces the exact
     same norm constant as before, so pre-v17 datasets stay comparable.
  2. OVERRIDE FOLLOWS CONFIG -- setting initial_energy=8000 makes the norm
     constant 8000, not 100.
  3. FEATURE IN RANGE -- with the operating point's battery, a freshly started
     node's energy feature is ~1.0, not ~80.0.
  4. NEGATIVE CONTROL -- recomputes what the feature WOULD have been under the
     old hardcoded constant, proving the bug was real and this check can fail.
  5. NO OTHER HARDCODED NORM -- scans norm_constants() for any remaining entry
     that ignores cfg, since this bug class is duplication and one instance
     usually implies others.

USAGE
    python verify_energy_norm_v17.py --src src
"""
import argparse, os, re, sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='src')
    a = ap.parse_args()
    sys.path.insert(0, os.path.abspath(a.src))

    import features_v2 as F
    from config_v2 import BASE, SCENARIOS

    fails = []
    print('=' * 88)
    print('  VERIFY v17 -- energy feature normalisation')
    print('=' * 88)

    base_cfg = {**BASE, **SCENARIOS['dense_slow']}

    # 1 -- default unchanged
    nc_default = F.norm_constants(base_cfg)
    ok = abs(nc_default['initial_energy'] - F.INITIAL_ENERGY_REF) < 1e-9
    print(f"\n  1. default (no override)      : initial_energy="
          f"{nc_default['initial_energy']:.1f}  expected {F.INITIAL_ENERGY_REF:.1f}  "
          f"{'OK' if ok else '*** CHANGED ***'}")
    if not ok:
        fails.append('default norm constant changed')

    # 2 -- override followed
    nc_over = F.norm_constants({**base_cfg, 'initial_energy': 8000.0})
    ok = abs(nc_over['initial_energy'] - 8000.0) < 1e-9
    print(f"  2. override initial_energy=8000: initial_energy="
          f"{nc_over['initial_energy']:.1f}  "
          f"{'OK' if ok else '*** IGNORED -- v17 not applied ***'}")
    if not ok:
        fails.append('override ignored (v17 not applied)')

    # 3 -- feature in range at the operating point
    feat = 8000.0 / nc_over['initial_energy']
    ok = 0.5 <= feat <= 1.5
    print(f"  3. fresh node energy feature   : {feat:.4f}  (expect ~1.0)  "
          f"{'OK' if ok else '*** OUT OF RANGE ***'}")
    if not ok:
        fails.append(f'energy feature {feat:.2f} out of range')

    # 4 -- NEGATIVE CONTROL: what the bug would have produced
    buggy = 8000.0 / F.INITIAL_ENERGY_REF
    print(f"\n  4. NEGATIVE CONTROL")
    print(f"     under the OLD hardcoded constant the feature would be {buggy:.1f}")
    if buggy <= 1.5:
        fails.append('negative control did not reproduce the bug -- check is vacuous')
        print('     *** control failed to show a problem -- this check proves nothing ***')
    else:
        print(f"     {buggy:.1f} vs correct {feat:.4f} -- bug was real, fix is load-bearing  OK")

    # 5 -- scan for other hardcoded norm entries
    print(f"\n  5. other hardcoded norm_constants entries")
    src = open(os.path.join(a.src, 'features_v2.py')).read()
    m = re.search(r'def norm_constants\(.*?\n(.*?)\n\s*\}', src, re.S)
    suspicious = []
    if m:
        for line in m.group(1).splitlines():
            line = line.strip()
            if not line.startswith("'") or ':' not in line or line.startswith('#'):
                continue
            key = line.split(':')[0].strip("' ")
            if 'cfg' not in line and '_REF' in line:
                suspicious.append(key)
    if suspicious:
        print(f"     still hardcoded (ignore cfg): {suspicious}")
        print(f"     NOT failures by themselves -- these are genuine fixed references")
        print(f"     (MAX_QUEUE, TTL, etc). Flagged so a future operating-point change")
        print(f"     to any of them gets the same treatment as energy did.")
    else:
        print("     none found")

    print()
    if fails:
        print(f"  FAIL -- {len(fails)}:")
        for f in fails:
            print('    - ' + f)
        return 1
    print("  PASS -- energy normalisation follows the configured battery, and the")
    print("  negative control confirms the pre-v17 behaviour was genuinely wrong.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
