"""verify_config_module_v11.py -- verify the shared config module BY EXECUTION.

Written after the fact: apply_config_module_v11.py printed "NEXT: run this"
while this file did not exist. Same defect class as a docstring promising
output fields the code never wrote -- an instruction pointing at nothing.

CHECKS
  1. no patched file still defines its own SCENARIOS / RATES / BASE
  2. no patched file still carries a hardcoded z_min/z_max/duration literal
  3. all modules resolve SCENARIOS, RATES and BASE to the SAME VALUES
  4. POSITIVE CONTROL: mutate config_v2.BASE in memory and confirm every
     module observes the change. If they do not, the import is cosmetic and
     the files are still independent -- which is the whole failure this patch
     exists to prevent. This is the check that can actually fail.
  5. get_suite() raises loudly for a declared-but-empty suite (pre-v8) and
     returns the four-scenario grid for 'default'
  6. an episode still runs end to end through the patched config path
"""
import argparse, importlib, io, os, re, sys

PATCHED = ['experiment_headroom.py', 'experiment_queue_weight.py',
           'experiment_spbp_mechanism.py', 'experiment_collision_model.py',
           'experiment_calibration_sensitivity.py', 'generate_dataset_v2.py',
           'experiment_locality_cost.py', 'preflight_teachers_v2_check.py']

# literals v8 will change; none may survive outside config_v2.py
LITERALS = [r"z_min\s*=\s*50\b", r"z_max\s*=\s*150\b", r"duration\s*=\s*40\.0"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='src')
    a = ap.parse_args()
    src = os.path.abspath(a.src)
    sys.path.insert(0, src)

    print("=" * 84)
    print("  VERIFY v11 -- shared config module")
    print("=" * 84)
    fails = []

    # 1 + 2 -- source-level checks
    print(f"\n  {'file':<42}{'own defs':>10}{'literals':>10}")
    print("  " + "-" * 62)
    for fn in PATCHED:
        p = os.path.join(src, fn)
        if not os.path.exists(p):
            fails.append(f"{fn}: missing"); continue
        t = io.open(p, encoding='utf-8').read()
        own = [k for k in ('SCENARIOS = {', 'RATES = [', 'BASE = dict(') if k in t]
        lits = [L for L in LITERALS if re.search(L, t)]
        print(f"  {fn:<42}{(','.join(own) or 'none'):>10}{(str(len(lits)) if lits else 'none'):>10}")
        if own:
            fails.append(f"{fn}: still defines {own} locally")
        if lits:
            fails.append(f"{fn}: still has hardcoded literal(s) {lits}")

    # 3 -- value agreement
    import config_v2
    mods = {}
    for fn in PATCHED:
        name = fn[:-3]
        try:
            mods[name] = importlib.import_module(name)
        except Exception as e:
            fails.append(f"{name}: import failed: {e}")
    print(f"\n  {'module':<42}{'SCENARIOS':>11}{'RATES':>8}{'BASE':>8}")
    print("  " + "-" * 69)
    for name, m in mods.items():
        s = getattr(m, 'SCENARIOS', None)
        r = getattr(m, 'RATES', None)
        b = getattr(m, 'BASE', None)
        ok_s = s is None or s == config_v2.SCENARIOS
        ok_r = r is None or r == config_v2.RATES
        ok_b = b is None or b == config_v2.BASE
        print(f"  {name:<42}{('ok' if ok_s else 'DIFFERS'):>11}"
              f"{('ok' if ok_r else 'DIFFERS'):>8}{('ok' if ok_b else 'DIFFERS'):>8}")
        for lbl, ok in (('SCENARIOS', ok_s), ('RATES', ok_r), ('BASE', ok_b)):
            if not ok:
                fails.append(f"{name}.{lbl} disagrees with config_v2")

    # 4 -- POSITIVE CONTROL: the check that can actually fail
    print("\n  POSITIVE CONTROL -- mutate config_v2.BASE, do all modules see it?")
    original = config_v2.BASE['duration']
    sentinel = 1234.5
    config_v2.BASE['duration'] = sentinel
    try:
        blind = [n for n, m in mods.items()
                 if getattr(m, 'BASE', None) is not None
                 and m.BASE.get('duration') != sentinel]
        seeing = [n for n, m in mods.items()
                  if getattr(m, 'BASE', None) is not None
                  and m.BASE.get('duration') == sentinel]
        print(f"    modules observing the change : {len(seeing)}")
        print(f"    modules blind to it          : {len(blind)}  {blind if blind else ''}")
        if blind:
            fails.append(f"import is COSMETIC for {blind} -- they hold a separate "
                         f"copy, so v8 would still desynchronise them")
    finally:
        config_v2.BASE['duration'] = original
    assert config_v2.BASE['duration'] == original, "failed to restore BASE"
    print(f"    restored duration -> {config_v2.BASE['duration']}")

    # 5 -- suite selector
    print("\n  SUITE SELECTOR")
    try:
        d = config_v2.get_suite('default')
        print(f"    get_suite('default') -> {len(d)} scenarios: {sorted(d)}")
        if len(d) != 4:
            fails.append("get_suite('default') did not return 4 scenarios")
    except Exception as e:
        fails.append(f"get_suite('default') raised: {e}")
    # v11.1: the three v8 suites are populated by v8a. Before v8a they were
    # declared-but-empty and this check asserted they RAISE. Now it checks the
    # real property: populated suites return the expected grid, empty ones still
    # raise. The negative control below proves the guard, rather than assuming it.
    EXPECTED = {
        'density':      {f'density_{n}' for n in (50, 100, 150, 200)},
        'convergecast': {f'sink_{n}'    for n in (50, 100, 150, 200)},
        'tall':         {'tall_probe'},
    }
    for name, expect in EXPECTED.items():
        try:
            grid = config_v2.get_suite(name)
        except RuntimeError:
            print(f"    get_suite({name!r}) raises -- still EMPTY (v8a not applied)")
            continue
        except Exception as e:
            fails.append(f"get_suite({name!r}) raised {type(e).__name__}: {e}")
            continue
        got = set(grid)
        if got != expect:
            fails.append(f"get_suite({name!r}) returned {sorted(got)}, "
                         f"expected {sorted(expect)}")
        else:
            print(f"    get_suite({name!r}) -> {len(grid)} scenarios: {sorted(got)}")

    # sink scenarios must actually carry the sink_node key, or convergecast
    # silently degrades to ordinary random-pair traffic.
    try:
        cg = config_v2.get_suite('convergecast')
        missing = [k for k, v in cg.items() if 'sink_node' not in v]
        if missing:
            fails.append(f"convergecast scenarios missing sink_node: {missing}")
        else:
            print(f"    convergecast scenarios all carry sink_node")
    except RuntimeError:
        pass

    # NEGATIVE CONTROL -- empty suite must still raise. Temporarily empty one,
    # call get_suite, restore. Proves the guard works on today's code.
    print("\n  NEGATIVE CONTROL -- does an EMPTY suite still raise?")
    _saved = config_v2.SUITES.get('density')
    config_v2.SUITES['density'] = {}
    try:
        config_v2.get_suite('density')
        fails.append("empty suite did NOT raise -- an empty grid would run "
                     "silently; the guard is dead code")
        print("    did not raise  *** GUARD IS DEAD ***")
    except RuntimeError as e:
        print(f"    raises RuntimeError as required: {str(e)[:60]}...")
    except Exception as e:
        fails.append(f"empty suite raised {type(e).__name__}, expected RuntimeError")
    finally:
        if _saved is not None:
            config_v2.SUITES['density'] = _saved
    assert config_v2.SUITES.get('density') == _saved, "failed to restore SUITES"

    # unknown suite name must raise KeyError, not return anything
    try:
        config_v2.get_suite('no_such_suite')
        fails.append("unknown suite name did not raise")
    except KeyError:
        print("    unknown suite name raises KeyError as required")
    except Exception as e:
        fails.append(f"unknown suite raised {type(e).__name__}, expected KeyError")

    # 6 -- end to end
    print("\n  END-TO-END")
    try:
        from experiment_headroom import HeadroomSimulator
        cfg = {**config_v2.BASE, **config_v2.SCENARIOS['dense_slow'],
               'packet_rate': 2.0, 'seed': 1, 'actor': 'spbp'}
        m = HeadroomSimulator(cfg).run()
        print(f"    dense_slow seed 1 rate 2.0 -> pdr={m['pdr_predrain']:.4f} "
              f"stale_hits={m['cache_stale_hits']}")
        if m['cache_stale_hits'] != 0:
            fails.append("v10 cache regression: stale hits under the fixed key")
    except Exception as e:
        fails.append(f"end-to-end run failed: {type(e).__name__}: {e}")

    print()
    if fails:
        print(f"  FAIL -- {len(fails)} problem(s):")
        for f in fails:
            print("    - " + f)
        return 1
    print("  PASS -- 6/6. The shared import is real: every module observes a "
          "change made in config_v2.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
