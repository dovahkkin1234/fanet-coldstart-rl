"""verify_audit_exact_v30.py -- verify the v30 audit by EXECUTION.

    python verify_audit_exact_v30.py --root .

Builds a small Dataset V3 in a temporary folder (5 episodes of 30 s, one per
behaviour policy), then checks the new audit against the v26 audit and against
deliberately corrupted copies. Every guard is also shown to fire on the input it
exists to stop. Touches nothing in the repo. About 5 minutes.
"""
import argparse
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback

RESULTS = []
V29_SIGNATURE = '92e0c265044dd292'      # code signature of data\v3m (v29 tree)
V26_COMMIT = '5d8baef'                  # last commit that changed src/audit_dataset_v3.py before v30
EPISODES = [('sink_50', 40.0, 114), ('dense_slow', 100.0, 101), ('very_dense', 80.0, 119),
            ('sparse_fast', 60.0, 102), ('medium_slow', 60.0, 116)]
DURATION = 30.0


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
    tmp = tempfile.mkdtemp(prefix='v30_verify_')
    py = sys.executable
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    audit = os.path.join(src, 'audit_dataset_v3.py')
    data = os.path.join(tmp, 'syn')

    import numpy as np
    import audit_dataset_v3 as A
    import generate_dataset_v3 as G

    def load_json(p):
        with open(p) as f:
            return json.load(f)

    def sampled(man, shard, per_shard=1500):
        """The audit's sample of `shard` (same generator, same order as the audit)."""
        rng = np.random.default_rng(5)
        for e in man['episodes']:
            n = int(e['n_contexts'])
            idx = rng.choice(n, size=min(n, per_shard), replace=False) if n else []
            if e['shard'] == shard:
                return [int(i) for i in idx]
        raise KeyError(shard)

    def copy_data(name):
        dst = os.path.join(tmp, name)
        shutil.copytree(data, dst)
        return dst

    def rewrite(dpath, shard, fn):
        p = os.path.join(dpath, shard)
        d = dict(np.load(p))
        fn(d)
        np.savez_compressed(p, **d)

    def audit_cli(dpath, *extra):
        rc, txt = run([py, audit, '--data', dpath, '--report_dir', os.path.join(dpath, 'rep'),
                       '--max_workers', '2', *extra], root)
        return rc, txt, load_json(os.path.join(dpath, 'audit_v3.json'))

    def audit_inproc(dpath, score_tol, *extra):
        old = A.SCORE_TOL
        A.SCORE_TOL = score_tol
        argv = sys.argv
        sys.argv = ['audit_dataset_v3.py', '--data', dpath, '--report_dir', os.path.join(dpath, 'rep'),
                    '--max_workers', '1', *extra]
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                rc = A.main()
        finally:
            A.SCORE_TOL, sys.argv = old, argv
        return rc, buf.getvalue(), load_json(os.path.join(dpath, 'audit_v3.json'))

    @check('1 the audit is outside the code signature: data\\v3m stays valid')
    def c1():
        assert 'audit_dataset_v3' not in G.SIGNATURE_MODULES
        assert G.CODE_SIGNATURE == V29_SIGNATURE, (G.CODE_SIGNATURE, V29_SIGNATURE)
        assert A.AUDIT_VERSION in ('v30', 'v31') and A.SCORE_TOL == 1e-3 and A.LABEL_MIN_AGREE == 0.99 \
            and A.FEAT_TOL == 1e-4 and A.FEAT_MIN_OK == 0.999, 'a tolerance changed'
        return f'code signature still {V29_SIGNATURE}; tolerances as in v26 (score 1e-3, label 0.99, spbp 1%)'
    c1()

    @check('2 a small dataset: 5 episodes, one per behaviour policy')
    def c2():
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=min(5, os.cpu_count() or 2)) as ex:
            got = list(ex.map(_episode, [(src, data, sc, r, sd) for sc, r, sd in EPISODES]))
        man = G.build_manifest(data, smoke=True)
        behs = sorted({b for _, b in got})
        assert behs == ['da_gpsr', 'dijkstra_nodrop', 'gpsr', 'spbp'], behs
        return f"{man['totals']['episodes']} episodes, {man['totals']['contexts']:,} contexts, behaviours {behs}"
    c2()

    @check('3 same sample and same float32 counts as the v26 audit, any number of workers')
    def c3():
        v26 = os.path.join(tmp, 'audit_v26.py')
        rc, txt = run(['git', '-C', root, 'show', f'{V26_COMMIT}:src/audit_dataset_v3.py'], root)
        assert rc == 0, f'git show {V26_COMMIT} failed (needs the repo history): {txt[-500:]}'
        with open(v26, 'w', encoding='utf-8') as f:
            f.write(txt)
        rc, t26 = run([py, v26, '--data', data, '--report_dir', os.path.join(tmp, 'rep26')], root)
        r26 = load_json(os.path.join(tmp, 'rep26', 'syn_audit_v3.json'))
        assert 'AUDIT v3 PASS' in t26, t26[-1500:]
        out = {}
        for w in ('1', '3'):
            rc, txt = run([py, audit, '--data', data, '--report_dir', os.path.join(tmp, 'rep30_' + w),
                           '--max_workers', w], root)
            assert rc == 0 and 'AUDIT v3 PASS' in txt, txt[-1500:]
            out[w] = load_json(os.path.join(tmp, 'rep30_' + w, 'syn_audit_v3.json'))
            assert out[w]['counts_float32'] == r26['counts'], (out[w]['counts_float32'], r26['counts'])
            assert out[w]['results_float32'] == r26['results'] == out[w]['results']
        assert out['1']['flagged'] == out['3']['flagged']
        # negative control: the comparison sees a different sample
        rc, _ = run([py, audit, '--data', data, '--report_dir', os.path.join(tmp, 'rep30_x'),
                     '--per_shard', '1000'], root)
        other = load_json(os.path.join(tmp, 'rep30_x', 'syn_audit_v3.json'))['counts_float32']
        assert other != r26['counts'], 'the comparison cannot tell two samples apart'
        return (f"counts identical to the v26 audit ({r26['counts']['n']:,} contexts, max score diff "
                f"{r26['counts']['b_score_maxdiff']:.2e}) with 1 and 3 workers; a different sample differs")
    c3()

    man = load_json(os.path.join(data, 'manifest.json'))
    shard_of = {e['key']: e for e in man['episodes']}
    e_sk = shard_of['sink_50@40#114']            # spbp-driven
    e_ds = shard_of['dense_slow@100#101']        # gpsr-driven
    e_sf = shard_of['sparse_fast@60#102']        # da_gpsr-driven, fastest replay

    @check('4 the replay reproduces the stored episode byte for byte -- and only that episode')
    def c4():
        d = dict(np.load(os.path.join(data, e_sf['shard'])))
        n_fr = len(d['f_n_edges'])
        fr, rep = A.replay_exact(data, e_sf, list(range(n_fr)))
        assert rep['exact'] and rep['arrays_differing_from_stored'] == [] and len(fr) == n_fr, rep
        # negative control 1: another seed (config made consistent) replays another episode
        bad = copy_data('bad_seed')
        mp = os.path.join(bad, e_sf['shard'].replace('.npz', '.json'))
        m = load_json(mp)
        m['seed'] += 1
        m['config']['seed'] += 1
        with open(mp, 'w') as f:
            json.dump(m, f)
        _, rep1 = A.replay_exact(bad, e_sf, [0, 1])
        assert not rep1['exact'] and 'f_node_feat' in rep1['why'], rep1
        # negative control 2: a shard written by other code is not replayed
        m = load_json(os.path.join(data, e_sf['shard'].replace('.npz', '.json')))
        m['code_signature'] = '0' * 16
        with open(mp, 'w') as f:
            json.dump(m, f)
        _, rep2 = A.replay_exact(bad, e_sf, [0])
        assert not rep2['exact'] and 'code' in rep2['why'], rep2
        return (f"{n_fr} frames, all {len(d)} arrays byte-identical; another seed is rejected "
                f"(frames differ), another code signature is refused")
    c4()

    @check('5 on the exact frame every stored score, label, vote and behaviour choice is reproduced')
    def c5():
        p = os.path.join(data, e_sk['shard'])
        d = dict(np.load(p))
        meta = load_json(p.replace('.npz', '.json'))
        nc = man['norm_constants_per_scenario'][meta['scenario']]
        names = A._norm_names(man)
        frames, rep = A.replay_exact(data, e_sk, list(range(len(d['f_n_edges']))))
        assert rep['exact'], rep
        get32, _ = A.build_frames(d, nc, names)
        off = np.concatenate([[0], np.cumsum(d['c_k'].astype(np.int64))])
        diag = float(np.hypot(nc['area_x'], nc['area_y']))

        def worst(dd):
            o2 = np.concatenate([[0], np.cumsum(dd['c_k'].astype(np.int64))])
            ratio, e32, bad = 0.0, 0.0, 0
            for i in range(len(dd['c_mult'])):
                f = int(dd['c_frame'][i])
                rx = A.derive(frames[f], dd, i, o2, nc, names['query'], diag, meta['behaviour'], features=False)
                r3 = A.derive(get32(f), dd, i, o2, nc, names['query'], diag, meta['behaviour'], features=False)
                st = dd['c_scores'][o2[i]:o2[i + 1]].astype(np.float64)
                half_ulp = np.spacing(np.abs(st.astype(np.float32))).astype(np.float64) / 2
                ratio = max(ratio, rx['b_score_diff'] / (half_ulp.max() + 1e-12))
                e32 = max(e32, r3['b_score_diff'])
                bad += sum(rx[k] for k in ('a_bad', 'd_gpsr_bad', 'd_dij_bad', 'd_spbp_bad', 'd_beh_bad'))
                bad += 1 - rx['b_agree']
            return ratio, e32, bad
        ratio, e32, bad = worst(d)
        assert ratio <= 1.0 and bad == 0, (ratio, bad)
        # negative control: a stored score off by 1e-6 is visible on the exact frame
        i0 = int(np.where(d['c_k'] > 1)[0][0])
        d2 = dict(d, c_scores=d['c_scores'].copy())
        d2['c_scores'][off[i0]] += np.float32(1e-6)
        ratio2, _, _ = worst(d2)
        assert ratio2 > 1.0, ratio2
        return (f"{len(d['c_mult']):,} contexts of {e_sk['key']} (spbp-driven): exact-frame score error "
                f"<= {ratio:.2f} x the float32 rounding of the stored score (float32 frame: up to "
                f"{e32:.1e}); 0 label / vote / behaviour mismatches; a 1e-6 change in one stored score is caught")
    c5()

    @check('6 flagged contexts are re-checked end to end (tolerance lowered for the test only)')
    def c6():
        base = load_json(os.path.join(tmp, 'rep30_1', 'syn_audit_v3.json'))
        tol = base['counts_float32']['b_score_maxdiff'] / 2      # forces flags on real contexts
        dpath = copy_data('forced')
        rc, txt, rep = audit_inproc(dpath, tol)
        assert rep['n_flagged'] >= 1 and all(r['exact'] for r in rep['replays']), txt[-2000:]
        assert all(r['exact'] is not None and not r['exact']['fails'] for r in rep['flagged']), rep['flagged']
        assert rc == 0 and not rep['results_float32']['B label'] and rep['results']['B label'], txt[-2000:]
        assert rep['counts']['b_score_maxdiff'] < tol <= rep['counts_float32']['b_score_maxdiff']
        # --exact_replay off keeps the float32 verdict
        rc_off, _, rep_off = audit_inproc(dpath, tol, '--exact_replay', 'off')
        assert rc_off == 1 and not rep_off['results']['B label'] and rep_off['replays'] == []
        # a flagged shard written by other code keeps its float32 result
        sh = rep['flagged'][0]['shard']
        mp = os.path.join(dpath, sh.replace('.npz', '.json'))
        m = load_json(mp)
        m['code_signature'] = '0' * 16
        with open(mp, 'w') as f:
            json.dump(m, f)
        rc_sig, _, rep_sig = audit_inproc(dpath, tol)
        assert rc_sig == 1 and any(not r['exact'] and 'code' in r['why'] for r in rep_sig['replays'])
        return (f"tolerance {tol:.1e}: {rep['n_flagged']} context(s) flagged in {len(rep['replays'])} shard(s), "
                f"all replays byte-identical, all resolved (float32 verdict B FAIL, re-checked PASS); "
                f"--exact_replay off keeps FAIL; a shard from other code is not replayed and keeps FAIL")
    c6()

    @check('7 real errors survive the re-check (production tolerance)')
    def c7():
        msgs = []

        def first_multi(e):
            dd = dict(np.load(os.path.join(data, e['shard'])))
            return next(i for i in sampled(man, e['shard']) if dd['c_k'][i] > 1), dd

        # a) a stored score 5e-3 off
        i, dd = first_multi(e_sf)
        dpath = copy_data('bad_score')
        o = int(np.concatenate([[0], np.cumsum(dd['c_k'].astype(np.int64))])[i])
        rewrite(dpath, e_sf['shard'], lambda d: d['c_scores'].__setitem__(o, d['c_scores'][o] + np.float32(5e-3)))
        rc, txt, rep = audit_cli(dpath)
        assert rc == 1 and not rep['results']['B label'], txt[-1500:]
        rec = [r for r in rep['flagged'] if r['context'] == i]
        assert rec and rec[0]['exact'] and rec[0]['exact']['score_diff'] > 4e-3, rec
        assert 'c_scores' in rep['replays'][0]['arrays_differing_from_stored'], rep['replays']
        msgs.append('score +5e-3 -> B FAIL after the replay')
        # b) a stored gpsr vote changed
        i, dd = first_multi(e_sf)
        dpath = copy_data('bad_vote')
        rewrite(dpath, e_sf['shard'],
                lambda d: d['c_votes'].__setitem__((i, 1), (d['c_votes'][i, 1] + 1) % d['c_k'][i]))
        rc, txt, rep = audit_cli(dpath)
        assert rc == 1 and not rep['results']['D votes'] and rep['counts']['d_gpsr_bad'] == 1, txt[-1500:]
        msgs.append('gpsr vote -> D FAIL')
        # c) a behaviour choice changed in the spbp-driven and the gpsr-driven episode
        for e in (e_sk, e_ds):
            i, dd = first_multi(e)
            dpath = copy_data('bad_beh_' + e['behaviour'])
            rewrite(dpath, e['shard'], lambda d: d['c_beh'].__setitem__(i, (d['c_beh'][i] + 1) % d['c_k'][i]))
            rc, txt, rep = audit_cli(dpath)
            assert rc == 1 and not rep['results']['D votes'] and rep['counts']['d_beh_bad'] >= 1, txt[-1500:]
            msgs.append(f"{e['behaviour']} behaviour choice -> D FAIL")
        # d) a stored frame that is not the episode's: the replay is not used
        i, dd = first_multi(e_sf)
        f, dst = int(dd['c_frame'][i]), int(dd['c_dst'][i])
        N = int(dd['f_n_nodes'][f])
        dpath = copy_data('bad_frame')
        col = man['node_features'].index('x')
        rewrite(dpath, e_sf['shard'],
                lambda d: d['f_node_feat'].__setitem__((f * N + dst, col), d['f_node_feat'][f * N + dst, col]
                                                       + np.float32(5.0 / man['norm_constants_per_scenario']
                                                                    ['sparse_fast']['area_x'])))
        rc, txt, rep = audit_cli(dpath)
        assert rc == 1 and rep['replays'] and not rep['replays'][0]['exact'] \
            and 'f_node_feat' in rep['replays'][0]['why'], txt[-1500:]
        msgs.append('a stored frame moved 5 m -> replay not used, FAIL kept')
        return '; '.join(msgs)
    c7()

    @check("8 the audit's own negative controls can fail (behaviour check disabled)")
    def c8():
        p = os.path.join(data, e_sk['shard'])
        d = dict(np.load(p))
        meta = load_json(p.replace('.npz', '.json'))
        nc = man['norm_constants_per_scenario'][meta['scenario']]
        assert A._negative_controls(d, meta, nc, A._norm_names(man)) == []
        real = A.derive

        def no_beh(*args, **kw):
            r = real(*args, **kw)
            r['d_beh_bad'] = 0
            return r
        A.derive = no_beh
        try:
            dead = A._negative_controls(d, meta, nc, A._norm_names(man))
        finally:
            A.derive = real
        assert sorted(dead) == ['D behaviour (dijkstra_nodrop)', 'D behaviour (gpsr)',
                                'D behaviour (spbp)'], dead
        return 'all controls fire on the real shard; with the behaviour check disabled the three new ones report it'
    c8()

    @check('9 docs: spec §0 #16, §7 and the run order describe v30')
    def c9():
        spec = open(os.path.join(root, 'docs', 'DATASET_V3_SPEC.md'), encoding='utf-8').read()
        assert '| 16 | **The audit' in spec and 'v30 — exact re-check' in spec
        assert 'audit_dataset_v3.py --data data\\v3m --max_workers 12' in spec
        cl = open(os.path.join(root, 'docs', 'CLAUDE.md'), encoding='utf-8').read()
        assert 'v30 (2026-10-08)' in cl
        rd = open(os.path.join(root, 'results', 'README.md'), encoding='utf-8').read()
        assert '> **v30 (2026-10-08).**' in rd and 'dataset_v3/v3m_audit_v3.json' in rd
        return 'spec, CLAUDE.md and results/README.md updated'
    c9()

    print('\n' + '=' * 78)
    print('  v30 VERIFICATION')
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
