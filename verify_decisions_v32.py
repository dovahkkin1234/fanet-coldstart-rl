"""verify_decisions_v32.py -- verify v32 by EXECUTION.

    python verify_decisions_v32.py --root .

Runs record_warmstart_v3.py on copies of the committed G4 v3m record (gate.json and
student_eval_val.json) with stand-in checkpoints that carry the recorded seeds and validation
accuracies, and on deliberately broken copies: the rule must select student_2001 with the
figures the docs quote, follow the evidence when it changes, and refuse every input it exists
to stop. Touches nothing in the repo. Under a minute.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback

RESULTS = []
SIGNATURE = '92e0c265044dd292'     # code signature of data\v3m -- v32 must not change it
EXPORT_SHA = '41449c931c20abf1'    # manifest of data\phaseB_v3m, as student_eval_val.json records it


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


def run(cmd, cwd, timeout=600):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, encoding='utf-8',
                           errors='replace', timeout=timeout)
    except subprocess.TimeoutExpired:
        return -9, f'TIMEOUT after {timeout}s: {cmd}'
    return r.returncode, r.stdout + r.stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.')
    ap.add_argument('--keep', action='store_true', help='keep the temporary folder')
    a = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(errors='replace')
    root = os.path.abspath(a.root)
    src = os.path.join(root, 'src')
    sys.path.insert(0, src)
    tmp = tempfile.mkdtemp(prefix='v32_verify_')
    py = sys.executable
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    rec_py = os.path.join(src, 'record_warmstart_v3.py')
    real_gate = os.path.join(root, 'results', 'g4_v3m')

    import torch
    import generate_dataset_v3 as G

    def load_json(p):
        with open(p) as f:
            return json.load(f)

    def save_json(p, obj):
        with open(p, 'w') as f:
            json.dump(obj, f, indent=1)

    def sha256(p):
        with open(p, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()

    def repo_state():
        _, st = run(['git', '-C', root, 'status', '--porcelain', '-uall'], root)
        _, df = run(['git', '-C', root, 'diff', 'HEAD', '--binary'], root)
        return hashlib.sha256((st + '\0' + df).encode('utf-8', 'replace')).hexdigest(), st

    state0 = repo_state()

    def make_gate(name, edit_gate=None, edit_ck=None, skip=()):
        """A gate folder: copies of the committed record + stand-in checkpoints."""
        d = os.path.join(tmp, name)
        os.makedirs(d)
        g = load_json(os.path.join(real_gate, 'gate.json'))
        if edit_gate:
            edit_gate(g)
        save_json(os.path.join(d, 'gate.json'), g)
        shutil.copy(os.path.join(real_gate, 'student_eval_val.json'), d)
        for seed in ('2000', '2001'):
            if f'student_{seed}.pt' in skip:
                continue
            ck = {'seed': int(seed), 'val_contested': g['train'][seed]['val_contested'],
                  'mask': ['hop_distance_to_dst', 'cand_hop_distance', 'cand_reachable'],
                  'state_dict': {'w': torch.full((4,), float(seed))}}
            if edit_ck:
                edit_ck(seed, ck)
            torch.save(ck, os.path.join(d, f'student_{seed}.pt'))
        return d

    def record(d, *extra):
        return run([py, rec_py, '--gate', d, *extra], tmp)

    @check('1 nothing in the code signature changed: data\\v3m, its export and G4 v3m stay valid')
    def c1():
        assert G.CODE_SIGNATURE == SIGNATURE, (G.CODE_SIGNATURE, SIGNATURE)
        assert 'record_warmstart_v3' not in G.SIGNATURE_MODULES
        for f in ('gate.json', 'student_eval_val.json'):
            assert os.path.isfile(os.path.join(real_gate, f)), f'results/g4_v3m/{f} missing'
        return f'code signature still {SIGNATURE}; the committed G4 v3m record is present'
    c1()

    @check('2 the rule on the committed G4 v3m record selects student_2001, with the figures the docs quote')
    def c2():
        d = make_gate('real')
        rc, txt = record(d)
        assert rc == 0, txt[-1500:]
        w = load_json(os.path.join(d, 'warmstart.json'))
        c = w['candidates']
        p = w['paired']['student_2000.pt']
        assert w['selected'] == 'student_2001.pt', w['selected']
        assert round(c['student_2001.pt']['g4_mean_diff_pp'], 3) == 0.009 \
            and round(c['student_2000.pt']['g4_mean_diff_pp'], 3) == -0.035, c
        assert round(p['selected_minus_this_mean_pp'], 3) == 0.044 and p['cells_ahead'] == 12 \
            and p['cells'] == 14 and round(p['sign_test_p_two_sided'], 3) == 0.013, p
        assert c['student_2000.pt']['val_contested_at_selection'] > c['student_2001.pt']['val_contested_at_selection']
        assert all(c[n]['sha256'] == sha256(os.path.join(d, n)) for n in c)
        assert w['export_manifest_sha256_16'] == EXPORT_SHA and w['gate']['evaluation_seeds'] == [1, 2, 3, 4, 5] \
            and len(w['gate']['gated_cells']) == 14 and len(w['gate']['held_out_cells']) == 4
        assert max(abs(c[n][k]) for n in c for k in ('g4_min_episode_diff_pp', 'g4_max_episode_diff_pp')) < 0.26
        assert w['decided'] == '2026-10-10' and set(w['decisions']) == {'suite_c_weight', 'early_stopping'}
        return ('student_2001.pt: +0.009 pp vs -0.035 pp; paired +0.044 pp, ahead in 12/14 cells, sign '
                'test p = 0.013; val accuracy ranks them the other way; every episode within +-0.26 pp; '
                f'both sha256 recorded; export {EXPORT_SHA}')
    c2()

    @check('3 the rule follows the evidence: swapped delivery selects student_2000; a tie goes to the lower seed')
    def c3():
        def swap(g):
            p = g['pdr']
            for k in [k for k in p if k.startswith('student_2000|')]:
                k2 = 'student_2001|' + k.split('|', 1)[1]
                p[k], p[k2] = p[k2], p[k]
        d = make_gate('swapped', edit_gate=swap)
        rc, txt = record(d)
        assert rc == 0 and load_json(os.path.join(d, 'warmstart.json'))['selected'] == 'student_2000.pt', txt[-800:]

        def tie(g):
            p = g['pdr']
            for k in [k for k in p if k.startswith('student_2001|')]:
                p[k] = p['student_2000|' + k.split('|', 1)[1]]
        d = make_gate('tie', edit_gate=tie)
        rc, txt = record(d)
        assert rc == 0 and load_json(os.path.join(d, 'warmstart.json'))['selected'] == 'student_2000.pt', txt[-800:]
        return 'swapping the two students\' rollouts selects student_2000; identical rollouts select the lower seed'
    c3()

    @check('4 refusals: recorded twice, a gate that did not pass, a smoke gate, a foreign or missing checkpoint, a missing episode')
    def c4():
        msgs = []
        d = os.path.join(tmp, 'real')
        before = sha256(os.path.join(d, 'warmstart.json'))
        rc, txt = record(d)
        assert rc == 1 and 'recorded once' in txt and sha256(os.path.join(d, 'warmstart.json')) == before, txt[-500:]
        msgs.append('a second recording')
        cases = [
            ('fail', dict(edit_gate=lambda g: g['verdict'].update(verdict='FAIL')), 'gate that passed'),
            ('smoke', dict(edit_gate=lambda g: g.update(provenance={'smoke': True})), 'smoke run'),
            ('foreign', dict(edit_ck=lambda s, ck: ck.update(val_contested=0.5) if s == '2001' else None),
             'is not the student gate.json trained'),
            ('missing', dict(skip=('student_2001.pt',)), 'every candidate must be present'),
            ('episode', dict(edit_gate=lambda g: g['pdr'].pop('reference|sink_50@30|4')), 'misses 1 episode'),
        ]
        for name, kw, why in cases:
            d = make_gate('bad_' + name, **kw)
            rc, txt = record(d)
            assert rc == 1 and why in txt and not os.path.exists(os.path.join(d, 'warmstart.json')), (name, txt[-500:])
            msgs.append(name)
        rc, txt = record(os.path.join(tmp, 'real'), '--ckpt', 'x.pt')
        assert rc == 1 and '--check only' in txt, txt[-300:]
        return 'refused, nothing written: ' + ', '.join(msgs) + '; --ckpt without --check'
    c4()

    @check('5 --check and selected_checkpoint() accept the recorded file or an identical copy, and nothing else')
    def c5():
        import record_warmstart_v3 as W
        d = os.path.join(tmp, 'real')
        sel = os.path.join(d, 'student_2001.pt')
        rc, txt = record(d, '--check')
        assert rc == 0 and 'OK' in txt, txt[-500:]
        cp = os.path.join(tmp, 'backup_copy.pt')
        shutil.copy(sel, cp)
        rc, txt = record(d, '--check', '--ckpt', cp)
        assert rc == 0 and 'OK' in txt, txt[-500:]
        assert os.path.abspath(W.selected_checkpoint(d)) == os.path.abspath(sel)
        with open(cp, 'ab') as f:
            f.write(b'\0')
        rc, txt = record(d, '--check', '--ckpt', cp)
        assert rc == 1 and 'MISMATCH' in txt, txt[-500:]
        rc, txt = record(d, '--check', '--ckpt', os.path.join(d, 'student_2000.pt'))
        assert rc == 1 and 'MISMATCH' in txt, txt[-500:]
        os.replace(cp, sel)                                   # the warmstart itself changed
        try:
            W.selected_checkpoint(d)
            raise AssertionError('a modified warmstart was loaded')
        except SystemExit as e:
            assert 'not the recorded warmstart' in str(e), e
        return ('the selected file and a byte-identical copy pass; a copy one byte longer, the other '
                'student and a modified warmstart are refused')
    c5()

    @check('6 docs record the decisions; the v30 and v31 doc guards still hold')
    def c6():
        def text(*p):
            with open(os.path.join(root, *p), encoding='utf-8') as f:
                return f.read()
        spec = text('docs', 'DATASET_V3_SPEC.md')
        for s in ('**decided 2026-10-10 (v32): the natural share**', '**decided 2026-10-10 (v32): keep it.**',
                  '15. **Warmstart: `results\\g4_v3m\\student_2001.pt`**',
                  '**closed 2026-10-10 — the natural share**', '**closed 2026-10-10 — unweighted, as G4 v3m**',
                  '9  v32: python src\\record_warmstart_v3.py --gate results\\g4_v3m',
                  # v31 guards
                  'restated G4 **PASS** on 2026-10-10', '| 17 | **The SP-BP picker breaks exact ties',
                  '**v31 — every check shown to fail.**', '8  v31: python src\\eval_students_v3.py',
                  '9. SP-BP tie order', '14. Per-behaviour imitation accuracy',
                  # v30 guards
                  '| 16 | **The audit', 'v30 — exact re-check',
                  'audit_dataset_v3.py --data data\\v3m --max_workers 12'):
            assert s in spec, s
        cl = text('docs', 'CLAUDE.md')
        for s in ('decided 2026-10-10 (v32): the warmstart is student_2001.pt',
                  'record_warmstart_v3.selected_checkpoint()',
                  'Warmstart candidates: results\\g4_v3m\\student_2000.pt', 'v30 (2026-10-08)'):
            assert s in cl, s
        assert 'the Suite C weight, the\nearly-stopping metric' not in cl, 'CLAUDE.md still lists them as open'
        rd = text('results', 'README.md')
        for s in ('> **v32 (2026-10-10).**', '`g4_v3m/warmstart.json`', '`g4_v3m/student_eval_test.json`',
                  '> **v31 (2026-10-10).**', '`g4_v3m/gate.json`', '`g4_v3m/student_eval_val.json`',
                  '> **v30 (2026-10-08).**', 'dataset_v3/v3m_audit_v3.json'):
            assert s in rd, s
        return 'spec (status, V3-10, V3-12, §10, §12), CLAUDE.md and results/README.md'
    c6()

    @check('7 this verification changed nothing in the repo')
    def c7():
        state1 = repo_state()
        assert state1[0] == state0[0], f'git status / diff changed:\n{state0[1]}\n->\n{state1[1]}'
        return 'git status and git diff identical before and after'
    c7()

    print('\n' + '=' * 78)
    print('  v32 VERIFICATION')
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
