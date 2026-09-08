"""apply_verifier_v11_1.py -- make the v11 suite-selector check v8a-aware.

Assertion-guarded str.replace, one anchor, idempotent.

THE PROBLEM

verify_config_module_v11.py asserts that get_suite('density'|'convergecast'|
'tall') RAISES. That was correct while those dicts were declared-but-empty. v8a
populates them, so they now return real grids and the verifier reports:

    FAIL -- 3 problem(s):
      - get_suite('density') returned instead of raising ...

This is a STALE CHECK, not a defect. The suites returning grids is exactly what
v8a was for.

WHY NOT JUST DELETE THE CHECK

The original property is still worth guarding: selecting a suite that has no
scenarios must fail loudly rather than silently running an empty grid. Deleting
the check would lose that. So the amended check does both:

  * populated suite  -> must return a non-empty grid with the EXPECTED keys
  * empty suite      -> must still raise RuntimeError. Verified by temporarily
                        emptying a suite in memory, calling get_suite, and
                        restoring it. This is a NEGATIVE CONTROL -- it proves the
                        guard still works rather than assuming it does, and it
                        keeps working no matter which suites are populated.
  * unknown suite    -> must raise KeyError

USAGE
    python apply_verifier_v11_1.py --root . --dry-run
    python apply_verifier_v11_1.py --root .
"""
import argparse, io, os, sys

TARGET = os.path.join('verify_config_module_v11.py')
GUARD = 'NEGATIVE CONTROL -- empty suite'

OLD = """    for empty in ('density', 'convergecast', 'tall'):
        try:
            config_v2.get_suite(empty)
            fails.append(f"get_suite({empty!r}) returned instead of raising "
                         f"-- an empty grid would run silently")
        except RuntimeError:
            print(f"    get_suite({empty!r}) raises RuntimeError as expected (pre-v8)")
        except Exception as e:
            fails.append(f"get_suite({empty!r}) raised {type(e).__name__}, "
                         f"expected RuntimeError")"""

NEW = """    # v11.1: the three v8 suites are populated by v8a. Before v8a they were
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
    print("\\n  NEGATIVE CONTROL -- does an EMPTY suite still raise?")
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
        fails.append(f"unknown suite raised {type(e).__name__}, expected KeyError")"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    path = os.path.join(a.root, TARGET)
    if not os.path.exists(path):
        print(f"  ERROR: {path} not found"); return 1
    text = io.open(path, encoding='utf-8').read()
    if GUARD in text:
        print("  ALREADY APPLIED. Nothing to do."); return 0

    n = text.count(OLD)
    if n != 1:
        print(f"  anchor matched {n} times, expected 1  <-- ABORT")
        print("  NO FILE WRITTEN."); return 1
    print("  anchor 1: OK")
    if a.dry_run:
        print("\n  DRY RUN OK. Nothing written."); return 0

    io.open(path, 'w', encoding='utf-8').write(text.replace(OLD, NEW, 1))
    print(f"\n  WROTE {path}")
    print("  NEXT: python verify_config_module_v11.py --src src")
    return 0


if __name__ == '__main__':
    sys.exit(main())
