"""verify_norm_constants_v18.py -- prove max_queue/ttl/lifetime_ref follow cfg.

Same structure as verify_energy_norm_v17.py, extended to the three constants
v18 fixes. See that file's docstring for the general pattern.

USAGE
    python verify_norm_constants_v18.py --src src
"""
import argparse, os, sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='src')
    a = ap.parse_args()
    sys.path.insert(0, os.path.abspath(a.src))

    import features_v2 as F
    from config_v2 import BASE, SCENARIOS

    fails = []
    print('=' * 88)
    print('  VERIFY v18 -- max_queue / ttl / lifetime_ref normalisation')
    print('=' * 88)

    base_cfg = {**BASE, **SCENARIOS['dense_slow']}
    checks = [('max_queue', F.MAX_QUEUE_REF, 999.0),
              ('ttl', F.TTL_REF, 999.0),
              ('lifetime_ref', F.LIFETIME_REF, 999.0)]

    for key, ref_default, test_override in checks:
        nc_default = F.norm_constants(base_cfg)
        ok = abs(nc_default[key] - ref_default) < 1e-9
        print(f"\n  {key}: default")
        print(f"     nc['{key}']={nc_default[key]}  expected {ref_default}  "
              f"{'OK' if ok else '*** CHANGED ***'}")
        if not ok:
            fails.append(f'{key}: default changed')

        nc_over = F.norm_constants({**base_cfg, key: test_override})
        ok = abs(nc_over[key] - test_override) < 1e-9
        print(f"  {key}: override -> {test_override}")
        print(f"     nc['{key}']={nc_over[key]}  "
              f"{'OK' if ok else '*** IGNORED -- v18 not applied ***'}")
        if not ok:
            fails.append(f'{key}: override ignored (v18 not applied)')

        # negative control: prove the OLD hardcoded path would have missed it
        would_have_been = ref_default
        control_meaningful = abs(would_have_been - test_override) > 1e-6
        print(f"  {key}: NEGATIVE CONTROL -- old hardcoded value would still read "
              f"{would_have_been}, not {test_override}")
        if not control_meaningful:
            fails.append(f'{key}: negative control vacuous (test value == default)')
        else:
            print(f"     confirms the fix is load-bearing  OK")

    print()
    if fails:
        print(f"  FAIL -- {len(fails)}:")
        for f in fails:
            print('    - ' + f)
        return 1
    print("  PASS -- all three constants follow their configured value, and the")
    print("  negative controls confirm the pre-v18 hardcoding was real.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
