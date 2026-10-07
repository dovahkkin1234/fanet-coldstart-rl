"""verify_grid_followup_v29.py -- verify the v29 patch by EXECUTION.

    python verify_grid_followup_v29.py --root .

Six checks, each guard also shown to fire on the input it exists to stop. Needs the
re-measured band files (results/band_*_v28.json). Runs in a temporary folder; touches
nothing in the repo. About a minute.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import traceback

RESULTS = []
PRE_V29_GRID = {                     # DATASET_GRID before v29 (bands measured on the trapping simulator)
    'dense_slow':  {'anchor': [40.0], 'band': [60.0, 80.0, 100.0]},
    'very_dense':  {'anchor': [40.0], 'band': [60.0, 80.0]},
    'medium_slow': {'anchor': [30.0], 'band': [40.0, 60.0, 80.0]},
    'sparse_fast': {'anchor': [40.0], 'band': [80.0, 100.0]},
    'sink_50':     {'anchor': [20.0], 'band': [30.0]},
}
V28_SIGNATURE = '73aff0fb2bb836a2'   # generator code signature of the v28 tree (verify_mobility_fix_v28)


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


def run(cmd, cwd):
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, encoding='utf-8', errors='replace')
    return r.returncode, r.stdout + r.stderr


def rule_of(curve):
    """The 2026-10-02 rule, restated: band = usable rates, anchor = lowest swept rate below it."""
    swept = sorted(float(c['rate']) for c in curve)
    use = sorted(float(c['rate']) for c in curve if c.get('usable'))
    return {'anchor': [swept[0]] if use and swept[0] < use[0] else [], 'band': use}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.')
    ap.add_argument('--keep', action='store_true', help='keep the temporary folder')
    a = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(errors='replace')
    root = os.path.abspath(a.root)
    src = os.path.join(root, 'src')
    res = os.path.join(root, 'results')
    sys.path.insert(0, src)
    tmp = tempfile.mkdtemp(prefix='v29_verify_')
    py = sys.executable
    os.environ['PYTHONIOENCODING'] = 'utf-8'

    import config_v2 as C
    import verify_dataset_grid_v3 as V
    from mobility import MOBILITY_VERSION

    def grid_of(sc):
        g = C.DATASET_GRID[sc]
        return {'anchor': [float(x) for x in g['anchor']], 'band': [float(x) for x in g['band']]}

    @check('1 DATASET_GRID follows the bands measured on the fixed simulator')
    def c1():
        curves = {}
        for sc, names in V.BAND_FILES.items():
            p = os.path.join(res, names[0])
            assert os.path.isfile(p), f'{names[0]} missing -- the v28 band re-measure must come first'
            b = json.load(open(p))
            assert (b.get('run_params') or {}).get('mobility') == MOBILITY_VERSION, f'{names[0]}: not v28'
            curves[sc] = b['curve']
        bad_now = [sc for sc in curves if rule_of(curves[sc]) != grid_of(sc)]
        assert not bad_now, {sc: (rule_of(curves[sc]), grid_of(sc)) for sc in bad_now}
        bad_old = sorted(sc for sc in curves if rule_of(curves[sc]) != PRE_V29_GRID[sc])
        assert bad_old == ['sink_50', 'sparse_fast'], f'negative control: pre-v29 grid differs in {bad_old}'
        n = len(C.dataset_cells())
        assert n == 18, n
        return (f'all 5 scenarios match the rule on their v28 band files ({n} cells); the pre-v29 grid '
                f'fails the same comparison in exactly sparse_fast and sink_50')
    c1()

    @check('2 the grid verifier passes all 18 cells and reports no band as moved')
    def c2():
        gv = os.path.join(tmp, 'gv.json')
        rc, txt = run([py, os.path.join(src, 'verify_dataset_grid_v3.py'), '--results', res, '--out', gv], root)
        g = json.load(open(gv))
        n_pass = sum(v['status'] == 'PASS' for v in g['cells'].values())
        assert n_pass == len(g['cells']) == 18, txt[-2000:]
        assert g.get('band_moved') == {} and 'BAND MOVED' not in txt, txt[-1500:]
        assert g.get('mobility') == MOBILITY_VERSION
        return '18/18 PASS, no BAND MOVED line, mobility recorded'
    c2()

    @check('3 generator: 18 cells, buckets, grid gate, new code signature')
    def c3():
        import generate_dataset_v3 as G
        exp = {('sparse_fast', 40): 'low', ('sparse_fast', 60): 'medium', ('sparse_fast', 80): 'medium',
               ('sparse_fast', 100): 'high', ('sink_50', 20): 'low', ('sink_50', 30): 'medium',
               ('sink_50', 40): 'high', ('dense_slow', 40): 'low', ('dense_slow', 100): 'high',
               ('very_dense', 80): 'high', ('medium_slow', 30): 'low', ('medium_slow', 80): 'high'}
        for (sc, r), want in exp.items():
            assert C.load_bucket_rel(sc, r) == want, (sc, r, C.load_bucket_rel(sc, r), want)
        cells = C.dataset_cells()
        g = json.load(open(os.path.join(tmp, 'gv.json')))
        assert G.unverified_cells(g, cells) == [], G.unverified_cells(g, cells)
        old = dict(g, cells={k: v for k, v in g['cells'].items() if k not in ('sparse_fast@60', 'sink_50@40')})
        miss = G.unverified_cells(old, cells)
        assert miss == ['sparse_fast@60=missing', 'sink_50@40=missing'], miss
        assert G.CODE_SIGNATURE != V28_SIGNATURE, 'config_v2 changed but the code signature did not'
        smoke = sorted((s, max(C.DATASET_GRID[s]['band'])) for s in C.DATASET_GRID)
        return (f'buckets as expected (sink_50@30 is now medium); a fresh 18-cell grid file verifies every '
                f'cell, a 16-cell one leaves exactly the two new cells unverified (re-run the grid verifier '
                f'after v29); code signature {V28_SIGNATURE} -> {G.CODE_SIGNATURE}; smoke cells {smoke}')
    c3()

    @check('4 gate: held-out cells reported by default, never gated')
    def c4():
        gate = [py, os.path.join(src, 'rollout_gate_v3.py'), '--stage', 'analyze']
        held = [f'{s}@{r:g}' for s, r in C.dataset_cells() if s == C.GENERALISATION_SCENARIO]
        out0 = os.path.join(tmp, 'g4_empty')
        _, t_def = run(gate + ['--out', out0], root)
        _, t_skip = run(gate + ['--out', out0, '--skip_heldout'], root)
        assert all(c in t_def for c in held), 'held-out cells not reported by default:\n' + t_def[-1500:]
        assert not any(c in t_skip for c in held), '--skip_heldout still reports them:\n' + t_skip[-1500:]
        n_gated = sum(1 for line in t_def.splitlines() if re.match(r'\s+\S+@\S+\s+incomplete', line))
        assert n_gated == 18 and 'INCOMPLETE' in t_def, t_def[-1500:]

        def synth(name, bad):
            out = os.path.join(tmp, name)
            os.makedirs(out)
            pdr = {}
            for cell in ('sparse_fast@60', 'medium_slow@30'):
                for sd in range(1, 6):
                    ref = 0.30 + 0.01 * sd
                    pdr[f'reference|{cell}|{sd}'] = ref
                    for st in ('student_2000', 'student_2001'):
                        pdr[f'{st}|{cell}|{sd}'] = ref - (0.05 if cell == bad else 0.0) + 0.0001 * sd
            json.dump({'train': {}, 'pdr': pdr}, open(os.path.join(out, 'gate.json'), 'w'))
            return out
        args = ['--cells', 'sparse_fast@60', 'medium_slow@30']
        rc_h, t_h = run(gate + ['--out', synth('g4_heldbad', 'medium_slow@30')] + args, root)
        rc_g, t_g = run(gate + ['--out', synth('g4_gatedbad', 'sparse_fast@60')] + args, root)
        assert rc_h == 0 and 'G4 CHECK 4 (v3): PASS' in t_h and 'held out: reported, not gated' in t_h, t_h[-1500:]
        assert rc_g == 1 and 'G4 CHECK 4 (v3): FAIL' in t_g, t_g[-1500:]
        return ('4 held-out cells reported by default (none with --skip_heldout), 18 rows in all; a held-out '
                'cell 5 pp below the reference is reported but the verdict stays PASS, while the same '
                'shortfall in a gated cell FAILS')
    c4()

    @check('5 .gitignore: smoke folders and raw episode recordings stay out of git')
    def c5():
        txt = open(os.path.join(root, '.gitignore'), encoding='utf-8-sig').read()
        assert 'results/*_smoke/' in txt and 'results/*/raw/' in txt
        rc, _ = run(['git', '-C', root, 'rev-parse', '--is-inside-work-tree'], root)
        if rc != 0:
            return 'patterns present (not a git work tree here, so not exercised)'
        probe = {'results/teacher_choice_v28/raw/__probe__.npz': 0, 'results/foo_smoke/__probe__.json': 0,
                 'results/teacher_choice_v28/__probe__.json': 1}
        for path, want in probe.items():
            rc, out = run(['git', '-C', root, 'check-ignore', '-q', path], root)
            assert rc == want, f'{path}: git check-ignore exit {rc}, expected {want} ({out.strip()})'
        return 'raw/ and *_smoke/ paths ignored; an ordinary result file is not'
    c5()

    @check('6 the spec grid table says what config_v2 says')
    def c6():
        spec = open(os.path.join(root, 'docs', 'DATASET_V3_SPEC.md'), encoding='utf-8').read()
        for sc in C.DATASET_GRID:
            m = re.search(r'^\| ' + sc + r'[^|]*\| [AC] \| ([\d.]+) \| ([^|]+) \|', spec, re.M)
            assert m, f'no grid row for {sc} in DATASET_V3_SPEC §2'
            anchor = [float(m.group(1))]
            band = [float(x) for x in re.findall(r'[\d.]+', m.group(2))]
            assert {'anchor': anchor, 'band': band} == grid_of(sc), (sc, anchor, band, grid_of(sc))
        return 'all five rows of DATASET_V3_SPEC §2 match DATASET_GRID'
    c6()

    print('\n' + '=' * 78)
    print('  v29 VERIFICATION')
    print('=' * 78)
    for name, ok, msg in RESULTS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}\n         {msg}")
    passed = all(ok for _, ok, _ in RESULTS)
    print(f"\n  {'ALL PASS' if passed else 'FAILURES ABOVE'} ({sum(ok for _, ok, _ in RESULTS)}/{len(RESULTS)})")
    if a.keep:
        print(f'  kept {tmp}')
    else:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
