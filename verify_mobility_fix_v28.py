"""verify_mobility_fix_v28.py -- verify the v28 patch by EXECUTION.

    python verify_mobility_fix_v28.py --root .

Nine checks; every guard is also shown to FIRE on the input it exists to stop
(negative controls). Runs in a temporary folder; touches nothing in the repo.
About 6-10 minutes (check 1 simulates 2,600 drone-hours of mobility).
"""
import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback

import numpy as np

RESULTS = []


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


def original_step(self, dt):
    """DroneRWP.step BEFORE v28, verbatim -- the reference the fix is compared with."""
    if self.pause_remaining > 0:
        self.pause_remaining -= dt
        self.vx = self.vy = self.vz = 0.0
        return
    dx = self.dest_x - self.x
    dy = self.dest_y - self.y
    dz = self.dest_z - self.z
    dist = np.sqrt(dx * dx + dy * dy + dz * dz)
    if dist < 1.0:
        self.pause_remaining = self.rng.uniform(0, self.pause_max)
        self.vx = self.vy = self.vz = 0.0
        self._pick_new_waypoint()
        return
    vertical_fraction = min(abs(dz) / max(dist, 1e-6), 0.3)
    scale = self.current_speed / dist
    self.vx = dx * scale * (1.0 - vertical_fraction)
    self.vy = dy * scale * (1.0 - vertical_fraction)
    self.vz = dz * scale * vertical_fraction
    self.x += self.vx * dt
    self.y += self.vy * dt
    self.z += self.vz * dt
    hit_boundary = False
    if self.x < 0 or self.x > self.area_x:
        self.x = float(np.clip(self.x, 0, self.area_x))
        hit_boundary = True
    if self.y < 0 or self.y > self.area_y:
        self.y = float(np.clip(self.y, 0, self.area_y))
        hit_boundary = True
    if self.z < self.z_min or self.z > self.z_max:
        self.z = float(np.clip(self.z, self.z_min, self.z_max))
        hit_boundary = True
    if hit_boundary:
        self._pick_new_waypoint()


def would_miss(d, dt):
    """Independent restatement of the bug: would the ORIGINAL step pass the waypoint
    without landing within 1 m of it?"""
    if d.pause_remaining > 0:
        return False
    dv = np.array([d.dest_x - d.x, d.dest_y - d.y, d.dest_z - d.z])
    dist = float(np.sqrt(dv @ dv))
    if dist < 1.0:
        return False
    vf = min(abs(dv[2]) / max(dist, 1e-6), 0.3)
    s = dv * (d.current_speed / dist * dt) * np.array([1 - vf, 1 - vf, vf])
    return float(s @ s) >= dist * dist and float((dv - s) @ (dv - s)) >= 1.0


def trapped_share(step, scen, seeds, t_end):
    """Share of drones that at some point hovered within 0.75 * speed_max + 1 m of
    their waypoint for >= 30 s without arriving, and median arrivals per drone."""
    from mobility import DroneRWP
    n = trapped = 0
    arrivals = []
    near_r = 0.75 * scen['speed_max'] + 1.0
    for seed in seeds:
        for i in range(scen['num_drones']):
            d = DroneRWP(i, scen['area_x'], scen['area_y'], 100, 300, scen['speed_min'],
                         scen['speed_max'], scen['pause_max'], seed=seed)
            near_since, last_arr, n_arr, is_trapped = None, 0.0, 0, False
            for k in range(int(t_end / 0.5)):
                was_paused = d.pause_remaining > 0
                step(d, 0.5)
                t = (k + 1) * 0.5
                if not was_paused and d.pause_remaining > 0:
                    last_arr, n_arr, near_since = t, n_arr + 1, None
                dd = (d.x - d.dest_x) ** 2 + (d.y - d.dest_y) ** 2 + (d.z - d.dest_z) ** 2
                if dd < near_r * near_r:
                    near_since = t if near_since is None else near_since
                else:
                    near_since = None
                if near_since is not None and t - near_since >= 30 and t - last_arr >= 30:
                    is_trapped = True
            n += 1
            trapped += is_trapped
            arrivals.append(n_arr)
    return trapped / n, float(np.median(arrivals))


def run(cmd, cwd):
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, encoding='utf-8', errors='replace')
    return r.returncode, r.stdout + r.stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.')
    ap.add_argument('--keep', action='store_true', help='keep the temporary folder')
    a = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):      # a redirected Windows console is cp1252
        stream.reconfigure(errors='replace')
    root = os.path.abspath(a.root)
    src = os.path.join(root, 'src')
    sys.path.insert(0, src)
    tmp = tempfile.mkdtemp(prefix='v28_verify_')
    py = sys.executable
    os.environ['PYTHONIOENCODING'] = 'utf-8'

    import mobility
    import config_v2 as C
    MOB = getattr(mobility, 'MOBILITY_VERSION', None)
    scen = dict(C.SCENARIOS)
    scen['sink_50'] = C.dataset_scenario_cfg('sink_50')

    @check('1 no drone is ever trapped (detector shown to fire on the original code)')
    def c1():
        assert MOB == 'v28-arrive-on-pass', f'MOBILITY_VERSION = {MOB} -- is the patch applied?'
        sh, _ = trapped_share(original_step, scen['sparse_fast'], range(101, 104), 300)
        assert sh >= 0.9, f'negative control: the original code traps only {sh:.0%} of sparse_fast drones'
        rows = []
        for name, s in scen.items():
            share, arr = trapped_share(mobility.DroneRWP.step, s, range(101, 111), 1000)
            assert share == 0.0, f'{name}: {share:.1%} of drones still trapped with the fix'
            rows.append(f'{name} {arr:.0f}')
        return (f'original code: {sh:.0%} of sparse_fast drones trapped by 300 s (the detector fires); '
                f'fixed: 0 trapped, 5 scenarios x 10 seeds x 1000 s; waypoints reached per drone '
                f'per 1000 s (median): ' + ', '.join(rows))
    c1()

    @check('2 the fix changes nothing but the overshooting step')
    def c2():
        from mobility import DroneRWP
        n_div = n_drones = 0
        for name, s in scen.items():
            for seed in range(101, 106):
                for i in range(s['num_drones']):
                    args = (i, s['area_x'], s['area_y'], 100, 300, s['speed_min'], s['speed_max'],
                            s['pause_max'])
                    o, f = DroneRWP(*args, seed=seed), DroneRWP(*args, seed=seed)
                    n_drones += 1
                    for k in range(2000):
                        miss = would_miss(o, 0.5)
                        original_step(o, 0.5)
                        f.step(0.5)
                        same = (o.x, o.y, o.z, o.pause_remaining) == (f.x, f.y, f.z, f.pause_remaining)
                        if not same:
                            assert miss, f'{name} seed {seed} drone {i}: diverged at step {k} without an overshoot'
                            n_div += 1
                            break
                        assert not miss, f'{name} seed {seed} drone {i}: overshoot at step {k} not caught'
        return (f'{n_drones} drones x 1000 s: every trajectory is bit-identical to the original until its '
                f'first would-be overshoot ({n_div} drones reach one), and only there does it change')
    c2()

    @check('3 the convergecast sink stays pinned')
    def c3():
        from simulator_v2 import FANETSimulatorV2

        class S(FANETSimulatorV2):
            sink_pos = None

            def _build_graph(self):
                if self.sink_pos is None:
                    self.sink_pos = []
                d = self.drones[self.sink_node]
                self.sink_pos.append((d.x, d.y, d.z))
                return super()._build_graph()
        sim = S(C.dataset_episode_config('sink_50', 20.0, 101, actor='spbp', duration=20.0))
        sim.run()
        p = np.array(sim.sink_pos)
        assert len(p) > 10 and float(np.ptp(p, axis=0).max()) == 0.0, np.ptp(p, axis=0)
        return f'sink at {tuple(round(float(v), 1) for v in p[0])} for all {len(p)} frames'
    c3()

    @check('4 G2 parity gate still passes, regression anchors unchanged')
    def c4():
        rc, txt = run([py, os.path.join(src, 'preflight_simulator_v2_check.py')], root)
        assert rc == 0 and 'G2 PASS' in txt and txt.count('vs anchor OK') == 2, txt[-2500:]
        return 'G2 PASS; dijkstra and spbp anchors match'
    c4()

    @check('5 band scripts: self-tests pass; --phase1_only stops after phase 1; mobility recorded')
    def c5():
        out = []
        # 20 s, one map seed: the second rate of each pair is usable, so phase 2 has work to skip.
        # The self-tests (determinism, paired traffic, no collapse) run here, at the short
        # duration they were designed at, so the 1000 s re-measure can skip them.
        for script, extra in (('find_usable_band.py', ['--scenario', 'very_dense', '--rates', '40', '50']),
                              ('find_congestion_band_convergecast.py',
                               ['--scenario', 'sink_50', '--rates', '25', '30'])):
            base = [py, os.path.join(src, script), '--duration', '20', '--initial_energy', '8000',
                    '--map-seeds', '1', '--measure-seeds', '1', '--max_workers', '2'] + extra
            outp = os.path.join(tmp, script.replace('.py', '.json'))
            rc, txt = run(base + ['--phase1_only', '--out', outp], root)
            assert rc == 0 and os.path.isfile(outp), txt[-2000:]
            assert 'SELF-TESTS PASSED' in txt, txt[-2000:]
            b = json.load(open(outp))
            rp = b['run_params']
            assert rp.get('mobility') == MOB and rp.get('phase1_only') is True, rp
            n_use = sum(bool(c.get('usable')) for c in b['curve'])
            assert n_use >= 1, f'{script}: no usable rate in the test run, the skip is not exercised'
            assert 'phase 2:' not in txt and 'phase 2 (matched queue ablation) skipped' in txt, txt[-1500:]
            rc, txt = run(base + ['--skip-self-test', '--out',
                                  os.path.join(tmp, 'p2_' + script.replace('.py', '.json'))], root)
            assert rc == 0 and 'phase 2:' in txt, f'{script}: negative control -- phase 2 did not run without the flag'
            out.append(f"{script.split('.')[0]} ({n_use} usable)")
        return ('self-tests pass on both scripts; with --phase1_only phase 2 is skipped (and runs without '
                'it); mobility recorded: ' + '; '.join(out))
    c5()

    @check('6 grid verifier refuses pre-v28 band files; the generator refuses a pre-v28 grid file')
    def c6():
        import verify_dataset_grid_v3 as V
        import generate_dataset_v3 as G
        res_old, res_new = os.path.join(tmp, 'res_old'), os.path.join(tmp, 'res_new')
        os.makedirs(res_old)
        os.makedirs(res_new)
        covered, src_curves = [], {}
        for sc, names in V.BAND_FILES.items():
            assert names[0].endswith('_v28.json'), names
            # v29: test data = the newest band file present (once re-measured, the v28 one,
            # which matches the current grid); without its mobility marker, under the
            # pre-v28 name, it plays a pre-v28 file
            have = [n for n in names if os.path.isfile(os.path.join(root, 'results', n))]
            if not have:
                continue
            b = json.load(open(os.path.join(root, 'results', have[0])))
            b.setdefault('run_params', {}).pop('mobility', None)
            json.dump(b, open(os.path.join(res_old, names[1]), 'w'))
            src_curves[sc] = b['curve']
            b['run_params']['mobility'] = MOB
            json.dump(b, open(os.path.join(res_new, names[0]), 'w'))
            covered.append(sc)
        assert covered, 'no band files in results\\ to test with'
        gvo, gvn = os.path.join(tmp, 'gv_old.json'), os.path.join(tmp, 'gv_new.json')
        _, t_old = run([py, os.path.join(src, 'verify_dataset_grid_v3.py'), '--results', res_old, '--out', gvo], root)
        _, t_new = run([py, os.path.join(src, 'verify_dataset_grid_v3.py'), '--results', res_new, '--out', gvn], root)
        g_old, g_new = json.load(open(gvo)), json.load(open(gvn))
        assert g_old.get('mobility') == g_new.get('mobility') == MOB, 'grid file does not record the mobility'
        for k, v in g_old['cells'].items():
            if k.split('@')[0] in covered:
                assert v['status'] == 'UNVERIFIED' and 'pre-v28' in v['reason'], (k, v)
        for sc in covered:
            assert V.REMEASURE[sc] in t_old, f'no re-measure command printed for {sc}'
        assert not any('mobility' in v['reason'] for v in g_new['cells'].values()), t_new[-1500:]
        n_pass = sum(v['status'] == 'PASS' for v in g_new['cells'].values())
        assert n_pass >= 1, t_new[-1500:]
        # v29: the BAND MOVED report must equal the rule applied to the test data ({} when
        # DATASET_GRID follows those bands) -- restated here, not taken from the verifier
        chosen = {k.split('@')[0] for k, v in g_new['cells'].items() if 'band_file' in v}
        expect = {}
        for sc in chosen:
            swept = sorted(float(c['rate']) for c in src_curves[sc])
            use = sorted(float(c['rate']) for c in src_curves[sc] if c.get('usable'))
            rule = {'anchor': [swept[0]] if use and swept[0] < use[0] else [], 'band': use}
            gr = C.DATASET_GRID[sc]
            if rule != {'anchor': [float(x) for x in gr['anchor']], 'band': [float(x) for x in gr['band']]}:
                expect[sc] = rule
        got = {sc: v['by_rule'] for sc, v in g_new.get('band_moved', {}).items()}
        assert got == expect, f'BAND MOVED report {got} != the rule {expect}'
        moved = ''
        if 'dense_slow' in covered:      # negative control: a band that grew is reported
            res_mv = os.path.join(tmp, 'res_moved')
            shutil.copytree(res_new, res_mv)
            p = os.path.join(res_mv, V.BAND_FILES['dense_slow'][0])
            b = json.load(open(p))
            for c in b['curve']:
                c['usable'] = c['usable'] or float(c['rate']) == 120.0
            json.dump(b, open(p, 'w'))
            gvm = os.path.join(tmp, 'gv_moved.json')
            _, t_mv = run([py, os.path.join(src, 'verify_dataset_grid_v3.py'), '--results', res_mv,
                           '--out', gvm], root)
            g_mv = json.load(open(gvm))
            assert 120.0 in g_mv['band_moved']['dense_slow']['by_rule']['band'], g_mv['band_moved']
            assert 'BAND MOVED  dense_slow' in t_mv, t_mv[-1500:]
            moved = '; a band that grew is reported as moved (DATASET_GRID must follow)'
        cells = C.dataset_cells()
        ok_cells = [(s, r) for s, r in cells if g_new['cells'][f'{s}@{r:g}']['status'] == 'PASS']
        assert G.unverified_cells(g_new, ok_cells) == []
        stale = G.unverified_cells(dict(g_new, mobility=None), ok_cells)
        assert len(stale) == len(ok_cells) and 'pre-v28' in stale[0], stale[:2]
        return (f'pre-v28 band files: every cell of {len(covered)} scenario(s) UNVERIFIED, re-measure '
                f'commands printed; the same data marked v28: {n_pass}/{len(g_new["cells"])} PASS (the '
                f'rest for other reasons, e.g. altitude), BAND MOVED report = the rule '
                f'({len(expect)} moved){moved}; the generator '
                f'counts all {len(ok_cells)} PASS cells of a pre-v28 grid file as unverified')
    c6()

    @check('7 generator: code signature; resume and the manifest refuse shards from other code')
    def c7():
        import generate_dataset_v3 as G
        sig = G.CODE_SIGNATURE
        assert len(sig) == 16
        alt = os.path.join(tmp, 'src_alt')
        shutil.copytree(src, alt, ignore=shutil.ignore_patterns('__pycache__'))
        with io.open(os.path.join(alt, 'mobility.py'), 'a', encoding='utf-8', newline='') as f:
            f.write('\n# one changed line\n')
        assert G.code_signature(alt) != sig, 'signature blind to a change in mobility.py'
        crlf = os.path.join(tmp, 'src_crlf')
        shutil.copytree(src, crlf, ignore=shutil.ignore_patterns('__pycache__'))
        for m in G.SIGNATURE_MODULES:
            p = os.path.join(crlf, m + '.py')
            data = open(p, 'rb').read().replace(b'\r\n', b'\n')
            open(p, 'wb').write(data.replace(b'\n', b'\r\n'))
        assert G.code_signature(crlf) == sig, 'signature changes with line endings'
        out = os.path.join(tmp, 'data')
        for seed in (101, 136):
            m = G._job((out, 'dense_slow', 60.0, seed, 'da_gpsr', G.EPSILON, G.RL_PACKET_FRAC, 12.0))
            assert m['code_signature'] == sig and m['mobility'] == MOB
            assert G.shard_complete(out, 'dense_slow', 60.0, seed)
        js = {sd: os.path.join(out, G.shard_name('dense_slow', 60.0, sd)) + '.json' for sd in (101, 136)}
        good = {sd: json.load(open(p)) for sd, p in js.items()}
        json.dump(dict(good[101], code_signature='0' * 16), open(js[101], 'w'))
        assert not G.shard_complete(out, 'dense_slow', 60.0, 101), 'resume accepted a shard from other code'
        assert G.build_manifest(out, smoke=True)['totals']['episodes'] == 1, 'manifest kept a stale shard'
        json.dump(dict(good[136], code_signature=None), open(js[136], 'w'))
        try:
            G.build_manifest(out, smoke=True)
            raise AssertionError('an all-stale manifest rebuild was written')
        except SystemExit as e:
            assert 'NOT written' in str(e)
        assert json.load(open(os.path.join(out, 'manifest.json')))['totals']['episodes'] == 1
        for sd in (101, 136):
            json.dump(good[sd], open(js[sd], 'w'))
        man = G.build_manifest(out, smoke=True)
        assert man['totals']['episodes'] == 2 and man['code_signature'] == sig and man['mobility'] == MOB
        gv = os.path.join(tmp, 'gv_pre.json')
        json.dump({'cells': {'dense_slow@60': {'status': 'PASS'}}}, open(gv, 'w'))
        rc, txt = run([py, os.path.join(src, 'generate_dataset_v3.py'), '--out', os.path.join(tmp, 'gen'),
                       '--cells', 'dense_slow@60', '--seeds', '101', '--grid_verification', gv], root)
        assert f'code signature {sig}' in txt, 'stray-module check failed in a fresh process:\n' + txt[-1500:]
        assert rc != 0 and 'GRID GATE' in txt and 'pre-v28' in txt, txt[-1500:]
        return (f'signature {sig}: changes with mobility.py, not with line endings; resume and the manifest '
                f'drop a shard from other code; an all-stale rebuild is refused (manifest untouched); a fresh '
                f'generator covers every module it loads and refuses a pre-v28 grid file')
    c7()

    @check('8 G3.5 v3, export and gate refuse a dataset from other code')
    def c8():
        import generate_dataset_v3 as G
        import preflight_dataset_v3_check as P
        data = os.path.join(tmp, 'data')
        man = json.load(open(os.path.join(data, 'manifest.json')))
        assert P.check_schema(man) == [], P.check_schema(man)
        assert any('other code' in x for x in P.check_schema(dict(man, code_signature=None))), \
            'G3.5 accepted a pre-v28 manifest'
        assert P.check_signatures([G.CODE_SIGNATURE] * 2, man) == []
        assert P.check_signatures([G.CODE_SIGNATURE, 'f' * 16], man), 'mixed shards not flagged'
        exp = os.path.join(tmp, 'pb')
        rc, txt = run([py, os.path.join(src, 'export_phaseb_v3.py'), '--data', data, '--out', exp,
                       '--splits', 'train', 'val'], root)
        assert rc == 0, txt[-2000:]
        pm = json.load(open(os.path.join(exp, 'manifest.json')))
        assert pm['code_signature'] == G.CODE_SIGNATURE and pm['stale_code_allowed'] is False
        stale = os.path.join(tmp, 'data_stale')
        shutil.copytree(data, stale)
        json.dump(dict(man, code_signature=None), open(os.path.join(stale, 'manifest.json'), 'w'))
        rc, txt = run([py, os.path.join(src, 'export_phaseb_v3.py'), '--data', stale, '--out',
                       os.path.join(tmp, 'pb_stale'), '--splits', 'train', 'val'], root)
        assert rc != 0 and 'other code' in txt, 'export accepted a dataset from other code:\n' + txt[-1500:]
        rc, txt = run([py, os.path.join(src, 'export_phaseb_v3.py'), '--data', stale, '--out',
                       os.path.join(tmp, 'pb_stale'), '--splits', 'train', 'val', '--allow_stale_code'], root)
        assert rc == 0 and 'WARNING' in txt, txt[-1500:]
        assert json.load(open(os.path.join(tmp, 'pb_stale', 'manifest.json')))['stale_code_allowed'] is True
        gate = [py, os.path.join(src, 'rollout_gate_v3.py'), '--stage', 'train', '--smoke', '--device', 'cpu']
        rc, txt = run(gate + ['--data', os.path.join(tmp, 'pb_stale'), '--out', os.path.join(tmp, 'g4_bad')], root)
        assert rc != 0 and 'other code' in txt and '[student' not in txt, \
            'gate accepted an export from other code:\n' + txt[-1500:]
        # the gate's exit code is its verdict (INCOMPLETE = 3 after a train-only stage), so
        # "accepted" means: it trained
        rc, txt = run(gate + ['--data', exp, '--out', os.path.join(tmp, 'g4')], root)
        assert '[student' in txt and 'other code' not in txt, txt[-2000:]
        return ('G3.5 flags a pre-v28 manifest and mixed shards; the export and the gate refuse other '
                'code, accept current code, and --allow_stale_code works and is recorded')
    c8()

    @check('9 the 300 s tests never mix simulators')
    def c9():
        tc_old = os.path.join(tmp, 'tc_old')
        os.makedirs(tc_old)
        json.dump({'duration': 300.0, 'record_prob': 0.05}, open(os.path.join(tc_old, 'config.json'), 'w'))
        scl = [py, os.path.join(src, 'test_bc_data_scaling.py'), '--device', 'cpu']
        rc, txt = run(scl + ['--tc', tc_old, '--out', os.path.join(tmp, 'bc1'), '--stage', 'rollout'], root)
        assert rc != 0 and 'pre-v28' in txt, 'replayed pre-v28 teacher-choice episodes:\n' + txt[-1500:]
        tct = [py, os.path.join(src, 'test_teacher_choice.py'), '--stage', 'analyze']
        rc, txt = run(tct + ['--out', tc_old], root)
        assert rc != 0 and 'pre-v28' in txt, 'resumed a pre-v28 teacher-choice folder:\n' + txt[-1500:]
        tc_new = os.path.join(tmp, 'tc_new')
        run(tct + ['--out', tc_new], root)
        cfg = json.load(open(os.path.join(tc_new, 'config.json')))
        assert cfg.get('mobility') == MOB, cfg
        bc_old = os.path.join(tmp, 'bc_old')
        os.makedirs(bc_old)
        json.dump({'hard': {'x': {}}, 'curve': {}, 'pdr': {}}, open(os.path.join(bc_old, 'scaling.json'), 'w'))
        rc, txt = run(scl + ['--tc', tc_new, '--out', bc_old, '--stage', 'analyze'], root)
        assert rc != 0 and 'pre-v28' in txt, 'mixed pre-v28 and v28 results in one scaling.json:\n' + txt[-1500:]
        return ('test_bc_data_scaling refuses to replay or pair with pre-v28 teacher-choice runs and to mix '
                'simulators in one scaling.json; test_teacher_choice refuses a pre-v28 folder and records '
                'the mobility in a new one')
    c9()

    print('\n' + '=' * 78)
    print('  v28 VERIFICATION')
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
