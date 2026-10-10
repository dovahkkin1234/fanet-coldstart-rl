"""verify_records_checks_v31.py -- verify v31 by EXECUTION.

    python verify_records_checks_v31.py --root .

Builds a small Dataset V3 in a temporary folder (6 episodes of 20 s: all four behaviour
policies, train, val and test seeds) and three exports of it, then runs the v31 code against
them and against deliberately broken copies: every new guard and every new negative
control is also shown to fire, and the pre-v31 code (git history, commit 9c4ddac) is shown
to miss what v31 catches. Touches nothing in the repo (every output goes to the temporary
folder). About 5-10 minutes on CPU.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

RESULTS = []
SIGNATURE = '92e0c265044dd292'     # code signature of data\v3m (v29 tree) -- v31 must not change it
BASE_COMMIT = '9c4ddac'            # last commit before v31 (v3m audit PASS)
EPISODES = [('dense_slow', 60.0, 101), ('sink_50', 30.0, 114), ('very_dense', 80.0, 119),
            ('dense_slow', 60.0, 136), ('dense_slow', 60.0, 141), ('dense_slow', 60.0, 150)]
DURATION = 20.0
V31_CODE = ['rollout_gate_v3', 'train_supervised_v2', 'eval_students_v3',
            'preflight_dataset_v3_check', 'audit_dataset_v3']
CELLS = ['dense_slow@60', 'sink_50@30']


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


def run(cmd, cwd, timeout=1200, env=None):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, encoding='utf-8',
                           errors='replace', timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return -9, f'TIMEOUT after {timeout}s: {cmd}'
    return r.returncode, r.stdout + r.stderr


def _episode(job):
    src, out, sc, r, sd = job
    sys.path.insert(0, src)
    import numpy as np
    import generate_dataset_v3 as G
    beh = G.behaviour_table()[sd]
    arrs, meta = G.run_episode(sc, r, sd, beh, duration=DURATION)
    base = os.path.join(out, G.shard_name(sc, r, sd))
    os.makedirs(os.path.dirname(base), exist_ok=True)
    np.savez_compressed(base + '.npz', **arrs)
    meta['shard'] = os.path.relpath(base + '.npz', out).replace('\\', '/')
    meta['bytes'] = os.path.getsize(base + '.npz')
    with open(base + '.json', 'w') as f:
        json.dump(meta, f, indent=1)
    return meta['key'], beh


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
    tmp = tempfile.mkdtemp(prefix='v31_verify_')
    py = sys.executable
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    gate = os.path.join(src, 'rollout_gate_v3.py')
    evs = os.path.join(src, 'eval_students_v3.py')
    data = os.path.join(tmp, 'syn')
    pb1, pb2, pbt = os.path.join(tmp, 'pb1'), os.path.join(tmp, 'pb2'), os.path.join(tmp, 'pb_test')
    g4 = os.path.join(tmp, 'g4')

    import numpy as np
    import audit_dataset_v3 as A
    import generate_dataset_v3 as G

    def load_json(p):
        with open(p) as f:
            return json.load(f)

    def save_json(p, obj):
        with open(p, 'w') as f:
            json.dump(obj, f, indent=1)

    def sha16(p):
        with open(p, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]

    def base_module(rel, name):
        """The pre-v31 version of src/<rel> from the git history, imported as `name`."""
        d = os.path.join(tmp, 'base')
        os.makedirs(d, exist_ok=True)
        rc, txt = run(['git', '-C', root, 'show', f'{BASE_COMMIT}:src/{rel}'], root)
        assert rc == 0, f'git show {BASE_COMMIT}:src/{rel} failed (needs the repo history): {txt[-300:]}'
        p = os.path.join(d, name + '.py')
        with open(p, 'w', encoding='utf-8') as f:
            f.write(txt)
        spec = importlib.util.spec_from_file_location(name, p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod, p

    def repo_state():
        _, st = run(['git', '-C', root, 'status', '--porcelain', '-uall'], root)
        _, df = run(['git', '-C', root, 'diff', 'HEAD', '--binary'], root)
        return hashlib.sha256((st + '\0' + df).encode('utf-8', 'replace')).hexdigest(), st

    state0 = repo_state()

    def gate_run(*args, timeout=1200, script=None, env=None):
        # cwd = the temporary folder: the gate's relative defaults (data\phaseB_v3, and before v31
        # --out results\g4_v3) must never resolve into the repo
        return run([py, script or gate, *args], tmp, timeout=timeout, env=env)

    @check('1 nothing in the code signature changed: data\\v3m, its export and G4 v3m stay valid')
    def c1():
        import rollout_gate_v3 as R
        assert A.AUDIT_VERSION == 'v31' and getattr(R, 'GATE_VERSION', None) == 'v31', \
            'the v31 code is not in place (apply_records_checks_v31.py first)'
        assert G.CODE_SIGNATURE == SIGNATURE, (G.CODE_SIGNATURE, SIGNATURE)
        assert not set(V31_CODE) & set(G.SIGNATURE_MODULES), set(V31_CODE) & set(G.SIGNATURE_MODULES)
        assert A.SCORE_TOL == 1e-3 and A.LABEL_MIN_AGREE == 0.99 and A.FEAT_TOL == 1e-4 \
            and A.FEAT_MIN_OK == 0.999, 'an audit tolerance changed'
        assert R.MARGIN_PP == 1.0 and R.TUNED == dict(lr=1e-3, attn_dropout=0.1, max_epochs=100), \
            'a G4 setting changed'
        return (f'code signature still {SIGNATURE} (no signature module touched); audit tolerances '
                f'and the G4 margin / tuned config unchanged')
    c1()

    @check('2 a small dataset (6 episodes: all four behaviour policies; train, val, test) and three exports')
    def c2():
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=min(5, os.cpu_count() or 2)) as ex:
            got = list(ex.map(_episode, [(src, data, sc, r, sd) for sc, r, sd in EPISODES]))
        man = G.build_manifest(data, smoke=True)
        behs = sorted({b for _, b in got})
        assert behs == ['da_gpsr', 'dijkstra_nodrop', 'gpsr', 'spbp'], behs
        ex = os.path.join(src, 'export_phaseb_v3.py')
        for out, frac, splits in ((pb1, '1.0', ['train', 'val']), (pb2, '0.5', ['train', 'val']),
                                  (pbt, '1.0', ['test'])):
            rc, txt = run([py, ex, '--data', data, '--out', out, '--splits', *splits,
                           '--ctx_frac', frac], root)
            assert rc == 0, txt[-1500:]
        m1, m2 = load_json(os.path.join(pb1, 'manifest.json')), load_json(os.path.join(pb2, 'manifest.json'))
        mt = load_json(os.path.join(pbt, 'manifest.json'))
        assert m1['code_signature'] == SIGNATURE and m2['n_rows'] < m1['n_rows'] and mt['episodes'] \
            == ['dense_slow@60#150'], (m1['n_rows'], m2['n_rows'], mt['episodes'])
        return (f"{man['totals']['episodes']} episodes, {man['totals']['contexts']:,} contexts, "
                f"behaviours {behs}; train+val exports of {m1['n_rows']:,} and {m2['n_rows']:,} rows, "
                f"a test export of {mt['n_rows']:,}")
    c2()

    gpath = os.path.join(g4, 'gate.json')
    ckpt = os.path.join(g4, 'student_2000.pt')

    @check('3 gate: provenance recorded; a resume with another export or setting is refused')
    def c3():
        import torch
        msgs = []
        rc, txt = gate_run('--data', pb1, '--stage', 'analyze')
        assert rc == 2 and '--out' in txt, txt[-800:]
        msgs.append('--out is required')
        rc, txt = gate_run('--data', os.path.join(tmp, 'nothing'), '--out', os.path.join(tmp, 'g4x'),
                           '--stage', 'train', '--smoke', timeout=300)
        assert rc == 1 and 'no export at' in txt, txt[-800:]
        msgs.append('a missing export is named as such')
        args = ['--data', pb1, '--out', g4, '--stage', 'train', '--smoke', '--device', 'cpu',
                '--cells', CELLS[0]]
        rc, txt = gate_run(*args)
        assert rc == 3 and '[student 2000]' in txt, txt[-1500:]       # trained; no rollouts -> INCOMPLETE
        res = load_json(gpath)
        p = res['provenance']
        want_export = {'manifest_sha256_16': sha16(os.path.join(pb1, 'manifest.json')),
                       'code_signature': SIGNATURE}
        assert p['gate_version'] == 'v31' and p['mask'] == 'hop' and p['smoke'] is True \
            and p['es_metric'] == 'accuracy_contested' \
            and p['tuned'] == {'lr': 1e-3, 'attn_dropout': 0.1, 'max_epochs': 100} \
            and all(p['export'][k] == v for k, v in want_export.items()), p
        ck = torch.load(ckpt, map_location='cpu', weights_only=False)
        assert ck['export_manifest_sha256_16'] == want_export['manifest_sha256_16'] \
            and ck['es_metric'] == 'accuracy_contested' \
            and res['train']['2000']['es_metric'] == 'accuracy_contested', sorted(ck)
        msgs.append('provenance in gate.json and the checkpoint')
        mt = os.path.getmtime(ckpt)
        rc, txt = gate_run(*args)
        assert rc == 3 and '[student 2000]' not in txt and os.path.getmtime(ckpt) == mt, txt[-1500:]
        msgs.append('same export and settings: resumed, nothing retrained')
        with open(gpath) as f:
            before = f.read()
        for extra, why in ((['--data', pb2, '--smoke'], 'export manifest_sha256_16'),
                           (['--data', pb1, '--smoke', '--es_metric', 'accuracy_contested_w'], 'es_metric'),
                           (['--data', pb1], 'smoke')):
            for stage in ('train', 'rollout'):
                rc, txt = gate_run('--out', g4, '--stage', stage, '--device', 'cpu', '--cells', CELLS[0],
                                   *extra, timeout=300)
                assert rc == 1 and 'holds results from another export or setting' in txt \
                    and why in txt, (extra, stage, txt[-800:])
        with open(gpath) as f:
            assert f.read() == before, 'a refused run changed gate.json'
        msgs.append('another export / early-stopping metric / smoke flag refused at train and rollout, '
                    'gate.json untouched')
        return '; '.join(msgs)
    c3()

    @check('4 gate: a pre-v31 gate.json is analysed as it is, resumed only with --adopt_legacy_results')
    def c4():
        leg = os.path.join(tmp, 'g4_legacy')
        shutil.copytree(g4, leg)
        lp = os.path.join(leg, 'gate.json')
        res = load_json(lp)
        res.pop('provenance')
        save_json(lp, res)
        args = ['--data', pb1, '--out', leg, '--stage', 'train', '--smoke', '--device', 'cpu',
                '--cells', CELLS[0]]
        rc, txt = gate_run(*args, timeout=300)
        assert rc == 1 and 'written before v31' in txt, txt[-800:]
        rc, txt = gate_run('--out', leg, '--stage', 'analyze', '--smoke', '--cells', CELLS[0])
        assert rc == 3 and 'predates v31' in txt and 'provenance' not in load_json(lp), txt[-800:]
        rc, txt = gate_run(*args, '--adopt_legacy_results')
        assert rc == 3 and '[student 2000]' not in txt, txt[-1500:]
        p = load_json(lp)['provenance']
        assert p['adopted_legacy_results'] is True, p
        rc, txt = gate_run(*args)
        assert rc == 3, txt[-800:]
        return ('refused for train/rollout; analysed with a note; adopted only on request (recorded '
                'in its provenance), then resumed normally')
    c4()

    @check('5 gate analysis: a cell missing any seed is INCOMPLETE (the pre-v31 gate passed it on fewer seeds)')
    def c5():
        rng = np.random.default_rng(0)
        pdr = {}
        for cell in CELLS:
            for sd in range(1, 6):
                ref = 0.80 + 0.02 * rng.random()
                pdr[f'reference|{cell}|{sd}'] = ref
                for i in range(2):
                    pdr[f'student_{2000 + i}|{cell}|{sd}'] = ref + 0.001 * rng.standard_normal()
        cases = {'full': dict(pdr),
                 'no_ref': {k: v for k, v in pdr.items() if k != f'reference|{CELLS[1]}|5'},
                 'no_student': {k: v for k, v in pdr.items() if k != f'student_2001|{CELLS[0]}|3'}}
        out = {}
        for name, p in cases.items():
            d = os.path.join(tmp, 'syn_' + name)
            os.makedirs(d)
            save_json(os.path.join(d, 'gate.json'), {'train': {}, 'pdr': p})
            rc, txt = gate_run('--out', d, '--stage', 'analyze', '--cells', *CELLS)
            out[name] = (rc, load_json(os.path.join(d, 'gate.json'))['verdict'])
        assert out['full'][0] == 0 and out['full'][1]['verdict'] == 'PASS', out['full']
        assert out['no_ref'][0] == 3 and out['no_ref'][1]['incomplete'] == [CELLS[1]], out['no_ref']
        assert out['no_student'][0] == 3 and out['no_student'][1]['incomplete'] == [CELLS[0]], out['no_student']
        # negative control: the pre-v31 gate analysed the cell on the seeds it had
        _, base_gate = base_module('rollout_gate_v3.py', 'rollout_gate_v30')
        d = os.path.join(tmp, 'syn_no_ref')
        env = dict(os.environ, PYTHONPATH=src)
        rc, txt = gate_run('--out', d, '--stage', 'analyze', '--cells', *CELLS, script=base_gate, env=env)
        v = load_json(os.path.join(d, 'gate.json'))['verdict']
        assert rc == 0 and v['verdict'] == 'PASS' and v['incomplete'] == [], (rc, v, txt[-800:])
        return ('all seeds: PASS (rc 0); a missing reference or student seed: INCOMPLETE (rc 3) -- the '
                'pre-v31 gate reported PASS on 4 of 5 seeds for the same file')
    c5()

    @check('6 trainer: --es_metric picks the epoch; default evaluation unchanged; per-behaviour accuracy exact')
    def c6():
        import torch
        import train_supervised_v2 as T
        from model_gnn_attn import FANETRouter
        ds = T.PhaseB(pb1, mask=T.MASK_PRESETS['hop'], use_weights=True)
        hp = dict(T.SEARCH_SPACE)
        hp.update(d=32, max_epochs=4, patience=2)
        seq = {'accuracy_contested': [0.50, 0.60, 0.70, 0.80],
               'accuracy_contested_w': [0.90, 0.80, 0.70, 0.60]}
        calls = []

        def stub(model, ds_, split, device, fpb, breakdown=False):
            e = len(calls)
            calls.append(e)
            return {k: v[e] for k, v in seq.items()}
        real = T.evaluate
        T.evaluate = stub
        try:
            calls.clear()
            _, best_u, ep_u = T.train_one(ds, 'attention', 2000, 'cpu', hp=hp)
            calls.clear()
            _, best_w, ep_w = T.train_one(ds, 'attention', 2000, 'cpu', hp=hp, es_metric='accuracy_contested_w')
        finally:
            T.evaluate = real
        assert (best_u, ep_u) == (0.80, 4) and (best_w, ep_w) == (0.90, 3), (best_u, ep_u, best_w, ep_w)
        for bad_metric, dsx in (('bogus', ds), ('accuracy_contested_w', copy.copy(ds))):
            if dsx is not ds:
                dsx.dec = {k: v for k, v in ds.dec.items() if k != 'weight'}
            try:
                T.train_one(dsx, 'attention', 2000, 'cpu', hp=hp, es_metric=bad_metric)
                raise AssertionError(f'es_metric {bad_metric} accepted')
            except ValueError:
                pass
        # the default evaluation is the pre-v31 one, key for key
        Tb, _ = base_module('train_supervised_v2.py', 'train_supervised_v30')
        torch.manual_seed(0)
        m = FANETRouter(d=32, layers=hp['layers'], heads=hp['heads'], dropout=hp['dropout'],
                        mixer='attention', attn_dropout=0.1)
        m.eval()
        new = T.evaluate(m, ds, 'train', 'cpu', 48)
        old = Tb.evaluate(m, ds, 'train', 'cpu', 48)
        assert json.dumps(new, sort_keys=True) == json.dumps(old, sort_keys=True), (new, old)
        # breakdown=True against an independent recount
        out = T.evaluate(m, ds, 'train', 'cpu', 48, breakdown=True)
        acc = {}
        with torch.no_grad():
            fs = ds.by_frame['train']
            for i in range(0, len(fs), 48):
                bt = ds.batch('train', fs[i:i + 48], 'cpu')
                ok = (T.run_batch(m, bt).argmax(-1) == bt['label']).tolist()
                for r, c in zip(bt['ids'].tolist(), ok):
                    if not ds.contested[r]:
                        continue
                    w = float(ds.dec['weight'][r])
                    for tag, col in (('beh', 'behaviour'), ('sc', 'scenario')):
                        s = acc.setdefault((tag, str(ds.dec[col][r])), [0, 0, 0.0, 0.0])
                        s[0] += 1
                        s[1] += int(c)
                        s[2] += w
                        s[3] += w * int(c)
        keys = {k for k in out if k.startswith(('n_contested_', 'accuracy_contested_beh_',
                                                'accuracy_contested_sc_', 'accuracy_contested_w_beh_',
                                                'accuracy_contested_w_sc_'))}
        assert len(keys) == 3 * len(acc), (sorted(keys), sorted(acc))
        for (tag, b), (n, c, w, wc) in acc.items():
            assert out[f'n_contested_{tag}_{b}'] == n
            assert abs(out[f'accuracy_contested_{tag}_{b}'] - c / n) < 1e-12
            assert abs(out[f'accuracy_contested_w_{tag}_{b}'] - wc / w) < 1e-9
        behs = sorted(b for t, b in acc if t == 'beh')
        assert behs == ['dijkstra_nodrop', 'gpsr', 'spbp'], behs
        return ('early stopping follows es_metric (stubbed curves: best 0.80 after 4 epochs unweighted, '
                '0.90 after 3 weighted); a bad metric or a missing weight column raises; default '
                f'evaluate() identical to the pre-v31 one; breakdown matches a recount for {behs}')
    c6()

    @check('7 eval_students_v3: accuracy per behaviour; refuses a student from another export')
    def c7():
        import torch
        msgs = []
        rc, txt = run([py, evs, '--data', pb1, '--gate', g4, '--device', 'cpu'], tmp)
        assert rc == 0, txt[-1500:]
        rep = load_json(os.path.join(g4, 'student_eval_val.json'))
        st = rep['students']['student_2000.pt']
        ck = torch.load(ckpt, map_location='cpu', weights_only=False)
        mt = st['metrics']
        n_con = sum(v for k, v in mt.items() if k.startswith('n_contested_beh_'))
        # same weights, same rows: equal up to at most one decision (thread-count rounding at a near-tie)
        assert abs(mt['accuracy_contested'] - ck['val_contested']) <= 1.0 / n_con + 1e-12, \
            (mt['accuracy_contested'], ck['val_contested'])
        assert 'accuracy_contested_beh_da_gpsr' in mt and 'accuracy_contested_beh_gpsr' in mt \
            and 'accuracy_contested_w_sc_dense_slow' in mt and rep['legacy_checkpoints_matched_by_path'] == []
        msgs.append(f"val contested {mt['accuracy_contested']:.4f} = the value the gate selected on")
        rc, txt = run([py, evs, '--data', pb2, '--gate', g4, '--device', 'cpu'], tmp)
        assert rc == 1 and 'trained on another export' in txt, txt[-800:]
        rc, txt = run([py, evs, '--data', pb1, '--eval_data', pb2, '--gate', g4, '--device', 'cpu'], tmp)
        assert rc == 0 and load_json(os.path.join(g4, 'student_eval_val.json'))['evaluated_on'][
            'manifest_sha256_16'] == sha16(os.path.join(pb2, 'manifest.json')), txt[-800:]
        other = os.path.join(tmp, 'pb2_other_dataset')
        shutil.copytree(pb2, other)
        om = load_json(os.path.join(other, 'manifest.json'))
        om['source_created'] = 'another dataset'
        save_json(os.path.join(other, 'manifest.json'), om)
        rc, txt = run([py, evs, '--data', pb1, '--eval_data', other, '--gate', g4, '--device', 'cpu'], tmp)
        assert rc == 1 and 'does not come from the dataset' in txt, txt[-800:]
        rc, txt = run([py, evs, '--data', pb1, '--gate', g4, '--split', 'test', '--device', 'cpu'], tmp)
        assert rc == 1 and 'has no test rows' in txt, txt[-800:]
        rc, txt = run([py, evs, '--data', pb1, '--eval_data', pbt, '--split', 'test', '--gate', g4,
                       '--device', 'cpu'], tmp)
        tm = load_json(os.path.join(g4, 'student_eval_test.json'))['students']['student_2000.pt']['metrics']
        assert rc == 0 and 'accuracy_contested_beh_spbp' in tm \
            and os.path.isfile(os.path.join(g4, 'student_eval_val.json')), txt[-800:]
        msgs.append('another export refused; another export of the same dataset accepted (--eval_data; '
                    'the test-split export gives the SP-BP-driven accuracy val cannot); one from another '
                    'dataset refused; an empty split refused')
        # a pre-v31 checkpoint (as in G4 v3m): matched by path, export older, same training rows
        pbc = os.path.join(tmp, 'pb1_copy')
        shutil.copytree(pb1, pbc)
        leg = os.path.join(tmp, 'g4_legacy_ckpt')
        os.makedirs(leg)
        ck2 = {k: v for k, v in ck.items() if k not in ('export_manifest_sha256_16', 'es_metric')}
        ck2['data'] = os.path.abspath(pbc)
        lck = os.path.join(leg, 'student_2000.pt')
        torch.save(ck2, lck)
        res = load_json(gpath)
        res.pop('provenance')
        save_json(os.path.join(leg, 'gate.json'), res)
        t_old = time.time() - 3600
        os.utime(os.path.join(pbc, 'manifest.json'), (t_old, t_old))
        rc, txt = run([py, evs, '--data', pbc, '--gate', leg, '--device', 'cpu'], tmp)
        assert rc == 0 and 'predates v31' in txt and load_json(os.path.join(leg, 'student_eval_val.json'))[
            'legacy_checkpoints_matched_by_path'] == ['student_2000.pt'], txt[-800:]
        rc, txt = run([py, evs, '--data', pb1, '--gate', leg, '--device', 'cpu'], tmp)
        assert rc == 1 and 'was trained on' in txt, txt[-800:]
        res['train']['2000']['n_train'] += 1
        save_json(os.path.join(leg, 'gate.json'), res)
        rc, txt = run([py, evs, '--data', pbc, '--gate', leg, '--device', 'cpu'], tmp)
        assert rc == 1 and 'training rows' in txt, txt[-800:]
        res['train']['2000']['n_train'] -= 1
        save_json(os.path.join(leg, 'gate.json'), res)
        t_new = time.time() + 3600
        os.utime(os.path.join(pbc, 'manifest.json'), (t_new, t_new))
        rc, txt = run([py, evs, '--data', pbc, '--gate', leg, '--device', 'cpu'], tmp)
        assert rc == 1 and 'rewritten after' in txt, txt[-800:]
        msgs.append('a pre-v31 checkpoint is accepted only on its recorded path, with an older export and '
                    "gate.json's training-row count")
        return '; '.join(msgs)
    c7()

    @check('8 G3.5: the duplicate-key control fires (the v26 one never reached the key test); report keeps the evidence')
    def c8():
        import preflight_dataset_v3_check as P
        man = load_json(os.path.join(data, 'manifest.json'))
        e = man['episodes'][0]
        meta = load_json(os.path.join(data, e['shard'].replace('.npz', '.json')))
        d = P.load_shard(data, e['shard'])
        nc = P.negative_controls(d, meta, man)
        assert all(nc.values()) and nc.get('9 dedup (duplicate key)') and nc.get('9 dedup (multiplicity sum)'), nc

        def without_key_test(mod):
            real = mod.check_dedup
            mod.check_dedup = lambda dd, mm: [x for x in real(dd, mm) if 'duplicate context keys' not in x]
            try:
                return mod.negative_controls(d, meta, man)
            finally:
                mod.check_dedup = real
        dead = [k for k, v in without_key_test(P).items() if not v]
        assert dead == ['9 dedup (duplicate key)'], dead
        Pb, base_pf = base_module('preflight_dataset_v3_check.py', 'preflight_v30')
        base_dead = [k for k, v in without_key_test(Pb).items() if not v]
        assert base_dead == [], base_dead
        # the whole gate, pre-v31 and v31, on the same data: same verdict per check
        env = dict(os.environ, PYTHONPATH=src)
        rc0, txt0 = run([py, base_pf, '--data', data, '--report_dir', os.path.join(tmp, 'rep_pf_base')],
                        root, env=env)
        old = load_json(os.path.join(tmp, 'rep_pf_base', 'syn_preflight_v3.json'))
        rc, txt = run([py, os.path.join(src, 'preflight_dataset_v3_check.py'), '--data', data,
                       '--report_dir', os.path.join(tmp, 'rep_pf')], root)
        rep = load_json(os.path.join(data, 'preflight_v3.json'))
        assert rep['results'] == old['results'] and rc == rc0, (rep['results'], old['results'])
        cells = sorted({f"{x['scenario']}@{x['rate']:g}" for x in man['episodes']})
        assert rep['negative_controls'] == nc and sorted(rep['diagnostics']) == cells \
            and isinstance(rep['saturation'], list) and rep['saturation'], (sorted(rep), txt[-1500:])
        # a 20 s toy episode leaves every candidate reachable: check 6 calls cand_reachable dead.
        # Anything else failing here would be real.
        failing = sorted(k for k, v in rep['results'].items() if not v)
        assert failing in ([], ['6 features']) and all(
            'cand_reachable' in m for m in rep['details'].get('6 features', []) if 'DEAD' in m), \
            (failing, rep['details'])
        verdict = 'PASS' if not failing else 'FAIL in check 6 only (cand_reachable constant in 20 s episodes)'
        return (f'{len(nc)}/{len(nc)} controls fire; with the key test disabled only the new control '
                f'reports it (the v26 controls all still passed); report: negative_controls, diagnostics '
                f'for {len(cells)} cells, saturation; verdict per check identical to the pre-v31 gate on '
                f'the same data ({verdict})')
    c8()

    @check('9 audit: A, D (dijkstra, spbp), E and F controls fire and die with their check; F requires equality')
    def c9():
        man = load_json(os.path.join(data, 'manifest.json'))
        names = A._norm_names(man)
        e0 = man['episodes'][0]
        meta0 = load_json(os.path.join(data, e0['shard'].replace('.npz', '.json')))
        d0 = dict(np.load(os.path.join(data, e0['shard'])))
        nc = man['norm_constants_per_scenario'][meta0['scenario']]
        assert A._negative_controls(d0, meta0, nc, names) == []
        assert A._negative_controls_manifest(man, e0, meta0, d0) == []
        real = A.derive
        for key, want in (('a_bad', ['A candidates']), ('d_dij_bad', ['D dijkstra votes']),
                          ('d_spbp_bad', ['D spbp votes'])):
            def off(*args, _k=key, **kw):
                r = real(*args, **kw)
                r[_k] = 0
                return r
            A.derive = off
            try:
                dead = A._negative_controls(d0, meta0, nc, names)
            finally:
                A.derive = real
            assert dead == want, (key, dead)
        for fn, want in (('check_splits', ['E splits (seed outside the plan)', 'E splits (seed in two splits)']),
                         ('check_hashing', ['F hashing (behaviour table)', 'F hashing (packet sample)'])):
            keep = getattr(A, fn)
            setattr(A, fn, lambda *args, **kw: True)
            try:
                dead = A._negative_controls_manifest(man, e0, meta0, d0)
            finally:
                setattr(A, fn, keep)
            assert dead == want, (fn, dead)
        # F: a sampled set that satisfies the rule but misses a packet passed v30, fails v31
        frac, salt = man['rl_packet_frac'][0], man['hash']['salts']['packet']
        sub = dict(d0, p_pid=d0['p_pid'][1:])
        assert all(A._sm64(e0['seed'], int(p), salt) < frac for p in sub['p_pid'])
        assert A.check_hashing(man, e0, meta0, d0) and not A.check_hashing(man, e0, meta0, sub)
        # E and F (behaviour table) on the committed v3m manifest
        vm = load_json(os.path.join(root, 'results', 'dataset_v3', 'v3m_manifest.json'))
        ev = vm['episodes'][0]
        fv, sv = vm['rl_packet_frac'][0], vm['hash']['salts']['packet']
        dv = {'p_pid': np.array([p for p in range(3000) if A._sm64(ev['seed'], p, sv) < fv], np.int32)}
        mv = {'metrics': {'n_generated': 3000}}
        assert A.check_splits(vm) and A.check_hashing(vm, ev, mv, dv)
        assert A._negative_controls_manifest(vm, ev, mv, dv) == []
        # the audit end to end, v30 and v31 on the same data: same sample, counts and verdict
        _, base_au = base_module('audit_dataset_v3.py', 'audit_v30')
        env = dict(os.environ, PYTHONPATH=src)
        rc0, txt0 = run([py, base_au, '--data', data, '--report_dir', os.path.join(tmp, 'rep_au_base'),
                         '--max_workers', '2'], root, env=env)
        old = load_json(os.path.join(tmp, 'rep_au_base', 'syn_audit_v3.json'))
        rc, txt = run([py, os.path.join(src, 'audit_dataset_v3.py'), '--data', data, '--report_dir',
                       os.path.join(tmp, 'rep_au'), '--max_workers', '2'], root)
        rep = load_json(os.path.join(data, 'audit_v3.json'))
        assert rc == 0 and 'ALL detected' in txt and 'AUDIT v3 PASS' in txt \
            and rep['audit_version'] == 'v31' and rep['results']['E splits'] and rep['results']['F hashing'], txt[-1500:]
        assert rc0 == 0 and old['audit_version'] == 'v30' and rep['results'] == old['results'] \
            and rep['counts'] == old['counts'] and rep['counts_float32'] == old['counts_float32'], \
            (rep['results'], old['results'])
        return ('each new control fires on a real shard and reports exactly its own check when that check '
                'is disabled; a packet sample missing one packet passes the v30 rule and fails v31; E and F '
                'pass on the committed v3m manifest (900 episodes, behaviour table); audit PASS, ALL detected, '
                'counts and verdict identical to the v30 audit on the same data')
    c9()

    @check('10 docs record v31; the v30 doc guards still hold')
    def c10():
        def text(*p):
            with open(os.path.join(root, *p), encoding='utf-8') as f:
                return f.read()
        spec = text('docs', 'DATASET_V3_SPEC.md')
        for s in ('restated G4 **PASS** on 2026-10-10', '| 17 | **The SP-BP picker breaks exact ties',
                  '| 18 | **The held-out scenario shares', '| 19 | **The anchors are the lowest swept rate',
                  '**v31 — every check shown to fail.**', '0.6 × (21.5 − frames GB) / decisions GB',
                  '6  DONE 2026-10-09', '7  DONE 2026-10-10', '8  v31: python src\\eval_students_v3.py',
                  '9. SP-BP tie order', '14. Per-behaviour imitation accuracy',
                  # v30 guards
                  '| 16 | **The audit', 'v30 — exact re-check',
                  'audit_dataset_v3.py --data data\\v3m --max_workers 12'):
            assert s in spec, s
        cl = text('docs', 'CLAUDE.md')
        for s in ('Warmstart candidates: results\\g4_v3m\\student_2000.pt', 'v30 (2026-10-08)',
                  'Never edit a module in generate_dataset_v3.SIGNATURE_MODULES'):
            assert s in cl, s
        rp = text('docs', 'FANET_Full_Project_Report.md')
        for s in ('> **Dated record (2026-09-30)', '**54 ledger entries', '| 54 | Checks claiming more than they test'):
            assert s in rp, s
        rd = text('results', 'README.md')
        for s in ('> **v31 (2026-10-10).**', '`g4_v3m/gate.json`', '`g4_v3m/student_eval_val.json`',
                  '> **v30 (2026-10-08).**', 'dataset_v3/v3m_audit_v3.json'):
            assert s in rd, s
        return 'spec (status, §0 #16-#19, §7, §9, §10, §11, §12), CLAUDE.md, the report and results/README.md'
    c10()

    @check('11 this verification changed nothing in the repo')
    def c11():
        state1 = repo_state()
        assert state1[0] == state0[0], f'git status / diff changed:\n{state0[1]}\n->\n{state1[1]}'
        return 'git status and git diff identical before and after'
    c11()

    print('\n' + '=' * 78)
    print('  v31 VERIFICATION')
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
