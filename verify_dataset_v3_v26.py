"""verify_dataset_v3_v26.py -- verify the v26 patch by EXECUTION (not inspection).

    python verify_dataset_v3_v26.py --src src

Each check exercises the patched code; the ones that guard against silent failure
are also shown to fail on a broken input. Takes ~2-4 minutes (one 20 s episode is
generated, exported, trained for one epoch and rolled out).
"""
import argparse
import inspect
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback

RESULTS = []


def check(name):
    def deco(fn):
        def run():
            try:
                msg = fn() or ''
                RESULTS.append((name, True, msg))
            except Exception as e:                      # noqa: BLE001
                RESULTS.append((name, False, f'{type(e).__name__}: {e}'))
                traceback.print_exc()
        run.__name__ = fn.__name__
        return run
    return deco


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='src')
    ap.add_argument('--keep', action='store_true', help='keep the temporary data directory')
    a = ap.parse_args()
    src = os.path.abspath(a.src)
    sys.path.insert(0, src)
    tmp = tempfile.mkdtemp(prefix='v26_verify_')

    import numpy as np
    import networkx as nx

    @check('1 features: schema v6, live own queue, v5 kept')
    def c1():
        import features_v2 as F
        assert F.FEATURE_SCHEMA_VERSION == 6
        assert F.QUERY_FEATURES[2] == 'own_queue_live'
        assert F.LEGACY_FEATURE_LISTS[5]['query_features'][2] == 'current_queue_occupancy'
        cfg = {'area_x': 800, 'area_y': 800, 'z_min': 100, 'z_max': 300, 'comm_range': 250,
               'num_drones': 3, 'initial_energy': 8000.0}
        n6, n5 = F.norm_constants(cfg), F.norm_constants(cfg, schema_version=5)
        assert n6['schema_version'] == 6 and n5['schema_version'] == 5
        assert n6['buffered_ref'] == F.BUFFERED_REF_V6 and n5['buffered_ref'] == 500.0
        G = nx.Graph()
        for i, (x, q) in enumerate(((0.0, 0.4), (100.0, 0.2), (200.0, 0.0))):
            G.add_node(i, x=x, y=0.0, z=150.0, queue_occupancy=q)
        G.add_edge(0, 1, link_quality=0.9); G.add_edge(1, 2, link_quality=0.8)

        class P:
            current, dst, hops, path = 0, 2, 0, [0]
        h = F.hop_distances_to(G, 2)
        try:
            F.extract_decision(G, P(), [1], n6, h, 0, 0.0, ttl_const=20)
            raise AssertionError('v6 accepted a missing live own queue')
        except ValueError:
            pass
        q6, _ = F.extract_decision(G, P(), [1], n6, h, 0, 0.0, ttl_const=20, own_queue_occ=0.7)
        q5, _ = F.extract_decision(G, P(), [1], n5, h, 0, 0.0, ttl_const=20)
        assert abs(q6[2] - 0.7) < 1e-7 and abs(q5[2] - 0.4) < 1e-7, (q6[2], q5[2])
        return 'v6 refuses a missing own queue; v6 reads it live (0.7), v5 reads the snapshot (0.4)'
    c1()

    @check('2 manifest compatibility: v5 only on request, still column-checked')
    def c2():
        import features_v2 as F
        m5 = {'feature_schema_version': 5, 'node_features': F.NODE_FEATURES,
              'edge_features': F.EDGE_FEATURES, 'candidate_features': F.CANDIDATE_FEATURES,
              'query_features': F.LEGACY_FEATURE_LISTS[5]['query_features'], 'local_horizon': 2}
        assert F.assert_manifest_compatible(m5), 'v5 accepted without accept_legacy'
        assert not F.assert_manifest_compatible(m5, accept_legacy=True)
        bad = dict(m5, query_features=list(F.QUERY_FEATURES))       # v5 tag, v6 columns
        assert F.assert_manifest_compatible(bad, accept_legacy=True), 'broken legacy manifest accepted'
        return 'refused by default, accepted with accept_legacy, wrong columns still refused'
    c2()

    @check('3 config: dataset grid added, parity reference untouched')
    def c3():
        import config_v2 as C
        assert C.BASE['duration'] == 40.0 and (C.BASE['z_min'], C.BASE['z_max']) == (50, 150)
        assert C.RATES == [0.5, 2.0, 4.0]
        assert C.OPERATING_POINT['duration'] == 1000.0 and C.OPERATING_POINT['initial_energy'] == 8000.0
        cells = C.dataset_cells()
        assert len(cells) == 16, len(cells)
        assert C.load_bucket_rel('dense_slow', 40) == 'low' and C.load_bucket_rel('dense_slow', 100) == 'high'
        assert C.load_bucket_rel('dense_slow', 60) == 'medium'
        cfg = C.dataset_episode_config('sink_50', 30, 101)
        assert cfg['z_min'] == 100 and cfg['duration'] == 1000.0 and cfg['sink_node'] == 0
        v8b = os.path.join(os.path.dirname(src), 'apply_v8b_operating_point_STAGED.py')
        note = ''
        if os.path.isfile(v8b):
            out = subprocess.run([sys.executable, v8b, '--src', src, '--dry-run'],
                                 capture_output=True, text=True)
            assert 'DRY RUN OK' in out.stdout or 'ALREADY APPLIED' in out.stdout, out.stdout
            note = '; v8b staged patch still matches'
        return f'16 cells, BASE/RATES unchanged, explicit operating point{note}'
    c3()

    @check('4 teacher pickers pinned (and pins able to fail)')
    def c4():
        import teacher_pickers_v3 as T
        T.assert_pins()
        return 'da_gpsr / gpsr / dijkstra / spbp == real teachers; broken pickers detected'
    c4()

    @check('5 generator self-test + one episode + G3.5 v3 checks on it')
    def c5():
        import generate_dataset_v3 as G
        import preflight_dataset_v3_check as P
        G.self_test()
        arrs, meta = G.run_episode('dense_slow', 60.0, 136, 'da_gpsr', duration=20.0)
        man = {'behaviour': {'table': {str(k): v for k, v in G.behaviour_table().items()}}}
        probs = []
        for _, fn in P.SHARD_CHECKS:
            probs += fn(arrs, meta)
        probs += P.check_behaviour(arrs, meta, man)
        assert not probs, probs
        nc = P.negative_controls(arrs, meta, man)
        assert all(nc.values()), nc
        os.makedirs(os.path.join(tmp, 'data'), exist_ok=True)
        meta['shard'] = 'shards/dense_slow/dense_slow_r60_s136.npz'
        os.makedirs(os.path.join(tmp, 'data', 'shards', 'dense_slow'), exist_ok=True)
        np.savez_compressed(os.path.join(tmp, 'data', meta['shard']), **arrs)
        meta['bytes'] = os.path.getsize(os.path.join(tmp, 'data', meta['shard']))
        json.dump(meta, open(os.path.join(tmp, 'data', meta['shard'].replace('.npz', '.json')), 'w'))
        G.build_manifest(os.path.join(tmp, 'data'), smoke=True)
        _a2, m2 = G.run_episode('dense_slow', 60.0, 136, 'da_gpsr', duration=20.0)
        assert m2['sha256_16'] == meta['sha256_16'], 'regenerated episode differs'
        c = meta['counts']
        return (f"{c['n_recorded_decisions']} decisions -> {c['n_contexts']} contexts "
                f"({c['n_ownq_pairs']} own-queue pairs), {c['n_steps']} steps; all shard checks "
                f"pass, {len(nc)}/{len(nc)} corruptions detected, regeneration byte-identical")
    c5()

    @check('6 independent audit on that episode')
    def c6():
        out = subprocess.run([sys.executable, os.path.join(src, 'audit_dataset_v3.py'),
                              '--data', os.path.join(tmp, 'data'), '--per_shard', '400',
                              '--report_dir', os.path.join(tmp, 'reports')],
                             capture_output=True, text=True)
        assert 'AUDIT v3 PASS' in out.stdout, out.stdout[-1500:] + out.stderr[-1500:]
        return 'labels, features, votes re-derived independently; corruptions detected'
    c6()

    @check('7 export (own queue drawn / expanded) -> weighted PhaseB -> one training epoch')
    def c7():
        n_dec = json.load(open(os.path.join(tmp, 'data', 'manifest.json')))['totals']['decisions']
        means = {}
        for mode in ('expand', 'sample'):
            out = subprocess.run([sys.executable, os.path.join(src, 'export_phaseb_v3.py'),
                                  '--data', os.path.join(tmp, 'data'), '--ownq', mode,
                                  '--out', os.path.join(tmp, 'pb')], capture_output=True, text=True)
            assert out.returncode == 0, out.stdout[-1500:] + out.stderr[-1500:]
            pm = json.load(open(os.path.join(tmp, 'pb', 'manifest.json')))
            assert round(pm['decisions_represented']) == n_dec, (mode, pm['decisions_represented'], n_dec)
            means[mode] = pm['own_queue_check']
        st = means['expand']['stream_mean_len']
        assert abs(means['expand']['exported_mean_len'] - st) < 1e-9, means
        assert abs(means['sample']['exported_mean_len'] - st) < 1.0, means
        import torch
        from train_supervised_v2 import PhaseB, train_one, SEARCH_SPACE, MASK_PRESETS, evaluate
        ds = PhaseB(os.path.join(tmp, 'pb'), mask=MASK_PRESETS['hop'], use_weights=True)
        assert ds.schema_version == 6
        bt = ds.batch('val', ds.by_frame['val'][:4], 'cpu')
        assert bt['weight'] is not None and float(bt['weight'].min()) >= 1.0
        hp = dict(SEARCH_SPACE, max_epochs=1, patience=1)
        # the only episode is a val seed (136): train on it for plumbing only
        ds.idx['train'], ds.by_frame['train'] = ds.idx['val'], ds.by_frame['val']
        model, val, _ = train_one(ds, 'attention', 2000, 'cpu', hp=hp)
        ev = evaluate(model, ds, 'val', 'cpu', 48)
        assert 'accuracy_raw_w' in ev
        torch.save({'state_dict': model.state_dict(), 'mixer': 'attention', 'hp': hp,
                    'mask': MASK_PRESETS['hop'], 'schema': ds.schema_version}, os.path.join(tmp, 'm.pt'))
        return (f"both exports represent all {n_dec} decisions; mean own queue: stream {st:.2f}, "
                f"expand {means['expand']['exported_mean_len']:.2f}, sample "
                f"{means['sample']['exported_mean_len']:.2f} (first occurrence "
                f"{means['sample']['first_occurrence_mean_len']:.2f}); weighted loss runs; val acc "
                f"{ev['accuracy_raw']:.3f} (weighted {ev['accuracy_raw_w']:.3f})")
    c7()

    @check('8 rollout actor follows the training data schema')
    def c8():
        import rollout_eval_v2 as R
        assert 'schema_version' in inspect.signature(R.ModelActorSimulator.__init__).parameters
        for v in (5, 6):
            R.assert_mask_applied(['hop_distance_to_dst', 'cand_hop_distance', 'cand_reachable'],
                                  'cpu', schema_version=v)
        import torch
        import config_v2 as C
        from model_gnn_attn import FANETRouter
        ck = torch.load(os.path.join(tmp, 'm.pt'), weights_only=False)
        hp = ck['hp']
        m = FANETRouter(d=hp['d'], layers=hp['layers'], heads=hp['heads'], dropout=hp['dropout'],
                        mixer='attention', attn_dropout=hp.get('attn_dropout', 0.0))
        m.load_state_dict(ck['state_dict']); m.eval()
        cfg = C.dataset_episode_config('dense_slow', 60.0, 1, actor='da_gpsr', duration=12.0)
        res = R.ModelActorSimulator(cfg, m, 'cpu', mask=ck['mask'], schema_version=6).run()
        return f"v5 and v6 decision paths run; a v6 student rolled out (pdr {res['network_pdr']:.3f})"
    c8()

    @check('9 legacy guards: v2 generator needs --legacy_v5; band script has altitude flags')
    def c9():
        out = subprocess.run([sys.executable, os.path.join(src, 'generate_dataset_v2.py'),
                              '--out', os.path.join(tmp, 'v2')], capture_output=True, text=True)
        assert out.returncode != 0 and 'LEGACY' in (out.stdout + out.stderr), out.stdout + out.stderr
        txt = io.open(os.path.join(src, 'find_congestion_band_convergecast.py'), encoding='utf-8').read()
        assert "'--z_min'" in txt and "'z_min': args.z_min" in txt
        import generate_dataset_v2 as G2
        sim = G2.DatasetSimulator({**__import__('config_v2').BASE,
                                   **__import__('config_v2').SCENARIOS['dense_slow'],
                                   'packet_rate': 0.5, 'seed': 101, 'actor': 'spbp'})
        assert sim.nc['schema_version'] == 5
        return 'v2 generator refuses without --legacy_v5 and stays v5; z flags present'
    c9()

    print('\n' + '=' * 78)
    print('  v26 VERIFICATION')
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
