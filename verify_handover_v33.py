"""verify_handover_v33.py -- verify v33 (docs only) by EXECUTION.

    python verify_handover_v33.py --root .

Checks that v33 changed no code, that the docs record what was decided on 2026-10-11 and no
longer say otherwise, that every earlier doc guard still holds (by running
verify_decisions_v32.py), and that the recorded warmstart still matches its sha256 when the
checkpoint is on this machine. Touches nothing in the repo. Under a minute.
"""
import argparse
import hashlib
import os
import subprocess
import sys
import traceback

RESULTS = []
SIGNATURE = '92e0c265044dd292'     # code signature of data\v3m
V32_COMMIT = '8e1198d'             # v32: the last commit that changed code


def check(name):
    def deco(fn):
        def run():
            print(f'\n--- {name}', flush=True)
            try:
                RESULTS.append((name, True, fn() or ''))
            except Exception as e:                      # noqa: BLE001
                traceback.print_exc()
                RESULTS.append((name, False, f'{type(e).__name__}: {e}'))
        return run
    return deco


def run(cmd, cwd, timeout=900):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, encoding='utf-8',
                           errors='replace', timeout=timeout)
    except subprocess.TimeoutExpired:
        return -9, f'TIMEOUT after {timeout}s: {cmd}'
    return r.returncode, r.stdout + r.stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.')
    a = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(errors='replace')
    root = os.path.abspath(a.root)
    src = os.path.join(root, 'src')
    sys.path.insert(0, src)
    py = sys.executable
    os.environ['PYTHONIOENCODING'] = 'utf-8'

    def text(*p):
        with open(os.path.join(root, *p), encoding='utf-8') as f:
            return f.read()

    def repo_state():
        _, st = run(['git', '-C', root, 'status', '--porcelain', '-uall'], root)
        _, df = run(['git', '-C', root, 'diff', 'HEAD', '--binary'], root)
        return hashlib.sha256((st + '\0' + df).encode('utf-8', 'replace')).hexdigest(), st

    state0 = repo_state()

    @check('1 no code changed: src/ is identical to v32 and the code signature is unchanged')
    def c1():
        import generate_dataset_v3 as G
        assert G.CODE_SIGNATURE == SIGNATURE, (G.CODE_SIGNATURE, SIGNATURE)
        rc, txt = run(['git', '-C', root, 'diff', '--stat', V32_COMMIT, '--', 'src'], root)
        assert rc == 0 and not txt.strip(), f'src/ differs from {V32_COMMIT}:\n{txt}'
        rc, txt = run(['git', '-C', root, 'status', '--porcelain', '-uall', '--', 'src'], root)
        assert rc == 0 and not txt.strip(), f'uncommitted files in src/:\n{txt}'
        return f'src/ identical to {V32_COMMIT}; code signature still {SIGNATURE}'
    c1()

    @check('2 the docs record the decisions of 2026-10-11 and no longer say otherwise')
    def c2():
        spec, cl = text('docs', 'DATASET_V3_SPEC.md'), text('docs', 'CLAUDE.md')
        for s in ('v33 (2026-10-11): **no regeneration is planned**',
                  '**Accepted for v3m (2026-10-11, v33): no regeneration for it.**',
                  'an SP-BP baseline in M5 breaks ties in node-id order',
                  '**Accepted for v3m (2026-10-11, v33):** that rule is enough.',
                  '**Accepted (2026-10-11, v33):** the anchors stay',
                  '9. SP-BP tie order (§0 #17): **accepted for v3m (2026-10-11), no regeneration**',
                  '10. Held-out seed coupling (§0 #18): **accepted for v3m (2026-10-11)**',
                  '11. Anchors (§0 #19): **accepted (2026-10-11)**',
                  '16. **Handover to Phase 0 / M5**', "selected_checkpoint('results/g4_v3m')",
                  'never deduplicate RL transitions', 'run at least one M5 seed from student_2000',
                  'measure the warmstart\'s starting point on fresh seeds', 'design exploration in',
                  '17. **Untested alternatives**'):
            assert s in spec, s
        for s in ('Fix at the next regeneration, with a pin that includes ties',
                  'Salting the mobility seed by scenario changes the simulator — next regeneration',
                  'or sweep a lower rate and regenerate'):
            assert s not in spec, f'the spec still says: {s}'
        for s in ('No regeneration is planned (v33, 2026-10-11)', 'is the dated record of 2026-09-30',
                  'Not the warmstart source: the M5 warmstart is'):
            assert s in cl, s
        for s in ('(the current record and plan)', 'a save step is still needed before Phase 4 can load one'):
            assert s not in cl, f'CLAUDE.md still says: {s}'
        return ('spec: status, §0 #17-#19 accepted with rules, §12 items 9-11, 16 (handover) and 17; '
                'CLAUDE.md: state pointer, warmstart source, status; the superseded wording is gone')
    c2()

    @check('3 every earlier doc guard still holds (verify_decisions_v32.py, plus the v31 report guards)')
    def c3():
        cl, rp = text('docs', 'CLAUDE.md'), text('docs', 'FANET_Full_Project_Report.md')
        assert 'Never edit a module in generate_dataset_v3.SIGNATURE_MODULES' in cl
        for s in ('> **Dated record (2026-09-30)', '**54 ledger entries', '| 54 | Checks claiming more than they test'):
            assert s in rp, s
        rc, txt = run([py, os.path.join(root, 'verify_decisions_v32.py'), '--root', root], root)
        assert rc == 0 and 'ALL PASS (7/7)' in txt, txt[-2000:]
        return 'verify_decisions_v32.py ALL PASS (7/7): the v30, v31 and v32 doc guards and the record script'
    c3()

    @check('4 the recorded warmstart still matches its sha256 (when the checkpoint is on this machine)')
    def c4():
        ck = os.path.join(root, 'results', 'g4_v3m', 'student_2001.pt')
        if not os.path.isfile(ck):
            return 'skipped: results/g4_v3m/student_2001.pt is not on this machine (not in git)'
        rc, txt = run([py, os.path.join(src, 'record_warmstart_v3.py'), '--gate',
                       os.path.join(root, 'results', 'g4_v3m'), '--check'], root)
        assert rc == 0 and 'OK' in txt, txt[-800:]
        return txt.strip().splitlines()[-1].strip()
    c4()

    @check('5 this verification changed nothing in the repo')
    def c5():
        state1 = repo_state()
        assert state1[0] == state0[0], f'git status / diff changed:\n{state0[1]}\n->\n{state1[1]}'
        return 'git status and git diff identical before and after'
    c5()

    print('\n' + '=' * 78)
    print('  v33 VERIFICATION')
    print('=' * 78)
    for name, ok, msg in RESULTS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}\n         {msg}")
    passed = all(ok for _, ok, _ in RESULTS)
    print(f"\n  {'ALL PASS' if passed else 'FAILURES ABOVE'} ({sum(ok for _, ok, _ in RESULTS)}/{len(RESULTS)})")
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
