"""
verify_dataset_grid_v3.py  --  grid-in-band verifier for Dataset V3 (spec §5).

    python src\\verify_dataset_grid_v3.py
    python src\\verify_dataset_grid_v3.py --diagnose_seeds 3 --max_workers 8

Checks config_v2.DATASET_GRID against the band-search result files and writes
results\\grid_verification.json, which generate_dataset_v3.py requires (every cell
PASS, or --allow_unverified_cells).

  band cell    PASS iff the band file was measured at the dataset operating point
               (duration 1000 s, battery 8000, altitude 100-300 m, and -- v28 --
               the current MOBILITY_VERSION) and marks the
               rate usable under the band criteria
  anchor cell  PASS iff measured in that same file and below its lowest usable
               rate (the anchor is below the band on purpose)
  otherwise    UNVERIFIED, with the reason

The band is defined under the band search's fixed reference actor
(spbp_ab_noqueue), deliberately not under the dataset's behaviour policy.
--diagnose_seeds N additionally runs N seeds per cell with the restricted
da_gpsr policy and REPORTS its queue overflow, energy share and dead nodes --
measured 2026-10-02: dense_slow@60 0.019 and sink_50@30 0.015 overflow under
da_gpsr vs 0.099 / 0.144 under the reference. Reported, never used to move cells.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_v2 as C                                        # noqa: E402
from mobility import MOBILITY_VERSION                        # noqa: E402

BAND_FILES = {
    # v28: the re-measurement with the fixed mobility goes FIRST (suffix _v28, so the
    # pre-v28 files are kept for comparison); those stay listed only so the verifier
    # can say why it refuses them
    'dense_slow': ['band_dense_slow_1000s_v28.json', 'band_dense_slow_1000s.json'],
    'very_dense': ['band_very_dense_1000s_v28.json', 'band_very_dense_1000s.json'],
    'medium_slow': ['band_medium_slow_1000s_v28.json', 'band_medium_slow_1000s.json'],
    'sparse_fast': ['band_sparse_fast_1000s_v28.json', 'band_sparse_fast_1000s.json'],
    # the re-measurement at 100-300 m goes FIRST; the original (50-150 m) is the fallback
    'sink_50': ['band_sink50_1000s_z100_300_v28.json', 'band_sink50_1000s_z100_300.json',
                'band_sink50_1000s.json'],
}

# v28: the commands that produce each band file at the operating point with the
# current mobility. Phase 1 decides which rates are usable -- all the grid needs.
# --skip-self-test: the self-tests check the CODE (determinism, paired traffic, no
# collapse) in 8 serial full-length episodes -- hours at 1000 s;
# verify_mobility_fix_v28.py check 5 runs them on both band scripts.
REMEASURE = {
    'dense_slow': r'python src\find_usable_band.py --scenario dense_slow --duration 1000 '
                  r'--initial_energy 8000 --rates 40 60 80 100 120 160 --phase1_only --skip-self-test '
                  r'--max_workers 12 --out results\band_dense_slow_1000s_v28.json',
    'very_dense': r'python src\find_usable_band.py --scenario very_dense --duration 1000 '
                  r'--initial_energy 8000 --rates 40 60 80 100 120 --phase1_only --skip-self-test '
                  r'--max_workers 12 --out results\band_very_dense_1000s_v28.json',
    'medium_slow': r'python src\find_usable_band.py --scenario medium_slow --duration 1000 '
                   r'--initial_energy 8000 --rates 30 40 60 80 --phase1_only --skip-self-test '
                   r'--max_workers 12 --out results\band_medium_slow_1000s_v28.json',
    'sparse_fast': r'python src\find_usable_band.py --scenario sparse_fast --duration 1000 '
                   r'--initial_energy 8000 --rates 40 60 80 100 120 --phase1_only --skip-self-test '
                   r'--max_workers 12 --out results\band_sparse_fast_1000s_v28.json',
    'sink_50': r'python src\find_congestion_band_convergecast.py --scenario sink_50 '
               r'--duration 1000 --initial_energy 8000 --rates 20 30 40 50 60 --phase1_only --skip-self-test '
               r'--max_workers 8 --out results\band_sink50_1000s_z100_300_v28.json',
}


def band_operating_point(b):
    rp = b.get('run_params') or {}
    z = None
    if rp.get('z_min') is not None:
        z = (float(rp['z_min']), float(rp['z_max']))
    elif b.get('alt'):
        z = tuple(float(v) for v in b['alt'])
    else:
        prov = b.get('provenance') or {}
        if prov.get('z_min') is not None:
            z = (float(prov['z_min']), float(prov['z_max']))   # BASE at run time
    return {'duration': float(rp.get('duration', b.get('duration', 0)) or 0),
            'initial_energy': (float(rp['initial_energy']) if rp.get('initial_energy') is not None
                               else float(b['initial_energy']) if b.get('initial_energy') is not None
                               else None),
            'z': z, 'mobility': rp.get('mobility')}


def matches(op):
    o = C.OPERATING_POINT
    why = []
    if op['duration'] != o['duration']:
        why.append(f"duration {op['duration']} != {o['duration']}")
    if op['initial_energy'] != o['initial_energy']:
        why.append(f"battery {op['initial_energy']} != {o['initial_energy']}")
    if op.get('mobility') != MOBILITY_VERSION:
        why.append(f"mobility {op.get('mobility') or 'pre-v28 (waypoint trapping)'} "
                   f"!= {MOBILITY_VERSION}")
    if op['z'] != (float(o['z_min']), float(o['z_max'])):
        why.append(f"altitude {op['z']} != ({o['z_min']}, {o['z_max']})")
    return why


def diagnose(job):
    scenario, rate, seed, duration = job
    import teacher_pickers_v3 as T
    from simulator_v2 import FANETSimulatorV2
    from generate_dataset_v2 import canonical_candidates

    class R(FANETSimulatorV2):
        def _select_next_hop(self, G, pkt, nb):
            cs = canonical_candidates(G, pkt.current, pkt.dst, set(pkt.path))
            return T.da_gpsr_pick(G, pkt.current, pkt.dst, cs) if cs else None
    over = {'actor': 'da_gpsr'}
    if duration:
        over['duration'] = duration
    sim = R(C.dataset_episode_config(scenario, rate, seed, **over))
    m = sim.run()
    g = max(m['n_generated'], 1)
    return scenario, rate, seed, {'pdr': m['network_pdr'],
                                  'q_ovf': m['drop_reasons'].get('queue_overflow', 0) / g,
                                  'energy': m['drop_reasons'].get('energy_depleted', 0) / g,
                                  'dead': int(sum(1 for e in sim.energy if e <= 0))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='results')
    ap.add_argument('--out', default=os.path.join('results', 'grid_verification.json'))
    ap.add_argument('--diagnose_seeds', type=int, default=0)
    ap.add_argument('--diagnose_duration', type=float, default=None)
    ap.add_argument('--max_workers', type=int, default=8,
                    help='default 8: on Windows 16 spawn workers stalled a run (project report)')
    args = ap.parse_args()

    out = {'operating_point': C.OPERATING_POINT, 'grid': C.DATASET_GRID, 'cells': {},
           'mobility': MOBILITY_VERSION, 'band_moved': {},
           'created': time.strftime('%Y-%m-%d %H:%M:%S')}
    print('=' * 78)
    print('  DATASET V3 GRID-IN-BAND VERIFICATION')
    print('=' * 78)
    for sc, g in C.DATASET_GRID.items():
        chosen, notes = None, []
        for fn in BAND_FILES[sc]:
            p = os.path.join(args.results, fn)
            if not os.path.isfile(p):
                notes.append(f'{fn}: missing')
                continue
            b = json.load(open(p))
            why = matches(band_operating_point(b))
            if why:
                notes.append(f"{fn}: {'; '.join(why)}")
                continue
            chosen = (fn, b)
            break
        if chosen is not None:
            # v28: the grid follows the band (decision 2026-10-02: band = the usable rates,
            # anchor = the lowest swept rate, below the band). Reported here, never applied.
            swept = sorted(float(c['rate']) for c in chosen[1]['curve'])
            use = sorted(float(c['rate']) for c in chosen[1]['curve'] if c.get('usable'))
            rule = {'anchor': [swept[0]] if use and swept[0] < use[0] else [], 'band': use}
            if rule != {'anchor': [float(x) for x in g['anchor']], 'band': [float(x) for x in g['band']]}:
                out['band_moved'][sc] = {'grid': {'anchor': g['anchor'], 'band': g['band']},
                                         'by_rule': rule, 'band_file': chosen[0]}
        for r in C.dataset_rates(sc):
            key = f'{sc}@{r:g}'
            role = C.rate_role(sc, r)
            if chosen is None:
                out['cells'][key] = {'status': 'UNVERIFIED', 'role': role,
                                     'reason': 'no band file at the operating point: ' + ' | '.join(notes)}
                continue
            fn, b = chosen
            curve = {float(c['rate']): c for c in b['curve']}
            usable = sorted(float(c['rate']) for c in b['curve'] if c.get('usable'))
            c = curve.get(float(r))
            if c is None:
                st, why = 'FAIL', 'rate not measured in the band file'
            elif role == 'band':
                st, why = (('PASS', 'usable under the band criteria') if c.get('usable')
                           else ('FAIL', 'not usable under the band criteria'))
            else:
                st, why = (('PASS', f'below the band (lowest usable {usable[0] if usable else None})')
                           if usable and float(r) < usable[0] else ('FAIL', 'anchor not below the band'))
            out['cells'][key] = {'status': st, 'role': role, 'reason': why, 'band_file': fn,
                                 'usable_rates': usable,
                                 'reference_q_ovf': c.get('q_ovf') if c else None,
                                 'reference_pdr': c.get('pdr') if c else None}
    for key, v in out['cells'].items():
        print(f"    {key:<18} {v['role']:<7} {v['status']:<11} {v['reason']}")

    if args.diagnose_seeds:
        jobs = [(s, r, sd, args.diagnose_duration) for s, r in C.dataset_cells()
                for sd in range(1, args.diagnose_seeds + 1)]
        print(f'\n  DIAGNOSTIC under the dataset behaviour policy (restricted da_gpsr): {len(jobs)} episodes')
        agg = {}
        with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
            for fu in as_completed([ex.submit(diagnose, j) for j in jobs]):
                s, r, sd, m = fu.result()
                agg.setdefault(f'{s}@{r:g}', []).append(m)
        for key, ms in sorted(agg.items()):
            d = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
            out['cells'][key]['da_gpsr_diagnostic'] = d
            print(f"    {key:<18} pdr {d['pdr']:.3f}  q_ovf {d['q_ovf']:.3f}  energy {d['energy']:.3f}  "
                  f"dead {d['dead']:.1f}")
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    json.dump(out, open(args.out, 'w'), indent=1)
    n_pass = sum(v['status'] == 'PASS' for v in out['cells'].values())
    print(f"\n  {n_pass}/{len(out['cells'])} cells PASS  -> {args.out}")
    if n_pass < len(out['cells']):
        print('  The generator refuses non-PASS cells unless --allow_unverified_cells.')
    todo = [sc for sc in REMEASURE if any(v['status'] == 'UNVERIFIED' for k, v in out['cells'].items()
                                          if k.split('@')[0] == sc)]
    if todo:
        print('  Measure these bands at the operating point with the current mobility:')
        for sc in todo:
            print('    ' + REMEASURE[sc])
    for sc, v in out['band_moved'].items():
        r = v['by_rule']
        note = ('  -- no usable rate: widen --rates' if not r['band'] else
                '  -- no swept rate below the band: add a lower rate' if not r['anchor'] else '')
        print(f"  BAND MOVED  {sc}: DATASET_GRID {v['grid']}  ->  by the rule {r}  ({v['band_file']}){note}")
    if out['band_moved']:
        print('  DATASET_GRID in config_v2 must follow the band before generating (decision '
              '2026-10-02: band = the usable rates, anchor = the lowest swept rate, below the band).')
    return 0


if __name__ == '__main__':
    sys.exit(main())
