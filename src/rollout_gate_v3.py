"""
rollout_gate_v3.py  --  G4 check 4, RESTATED for Dataset V3 (docs/DATASET_V3_SPEC.md §8).

    python src\\rollout_gate_v3.py --data data\\phaseB_v3 --out results\\g4_v3 --max_workers 8
    python src\\rollout_gate_v3.py --data data\\phaseB_v3_smoke --out results\\g4_v3_smoke --smoke

The M4 gate compared the student against SP-BP at >= 90% of its PDR. SP-BP is no
longer the oracle and the label teacher is da_gpsr, so the gate becomes a paired
NON-INFERIORITY test against the policy the student imitates:

  reference  da_gpsr acting through the restricted picker (canonical, visited-
             excluded candidates -- exactly the policy the labels come from),
             epsilon 0, at the dataset operating point
  student    the masked GNN (mask 'hop', tuned M4 config) trained on the export,
             with the loss weights the export declares (context multiplicity)
  PASS iff   in EVERY evaluated training cell, the 95% CI lower bound of the
             paired PDR difference (student - reference, mean over model seeds)
             is above -1.0 pp. Held-out medium_slow cells are REPORTED, not gated.
  A cell whose CI straddles -1 pp with a half-width above 1 pp is INCONCLUSIVE
  (too few seeds to decide, not a failure): re-run the gate with
  --eval_seeds 1 2 3 4 5 6 7 8 9 10 and decide on those. A cell whose whole CI
  is below -1 pp, or whose narrow CI straddles it, FAILS. Power: the teacher-
  choice test measured a per-seed sd of (student - da_gpsr) of 0.09-0.42 pp at
  300 s, so 5 seeds at 1000 s should give half-widths well under 1 pp.

Pre-registered here, before any v3 result exists. Stages are resumable.
"""

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config_v2 as C                                            # noqa: E402

MARGIN_PP = 1.0
TUNED = dict(lr=1e-3, attn_dropout=0.1, max_epochs=100)          # M4 decision D-10


def reference_episode(scenario, rate, seed, duration=None):
    import teacher_pickers_v3 as T
    from simulator_v2 import FANETSimulatorV2
    from generate_dataset_v2 import canonical_candidates

    class RestrictedTeacherSim(FANETSimulatorV2):
        def _select_next_hop(self, G, pkt, neighbors):
            cands = canonical_candidates(G, pkt.current, pkt.dst, set(pkt.path))
            if not cands:
                return None
            return T.da_gpsr_pick(G, pkt.current, pkt.dst, cands)

    over = {'actor': 'da_gpsr'}
    if duration:
        over['duration'] = duration
    sim = RestrictedTeacherSim(C.dataset_episode_config(scenario, rate, seed, **over))
    return float(sim.run()['network_pdr'])


_MODELS = {}


def student_episode(ckpt, scenario, rate, seed, duration=None):
    import torch
    from model_gnn_attn import FANETRouter
    from rollout_eval_v2 import ModelActorSimulator
    torch.set_num_threads(1)
    if ckpt not in _MODELS:
        ck = torch.load(ckpt, map_location='cpu', weights_only=False)
        hp = ck['hp']
        m = FANETRouter(d=hp['d'], layers=hp['layers'], heads=hp['heads'], dropout=hp['dropout'],
                        mixer=ck['mixer'], attn_dropout=hp.get('attn_dropout', 0.0))
        m.load_state_dict(ck['state_dict'])
        m.eval()
        _MODELS[ckpt] = (m, ck)
    m, ck = _MODELS[ckpt]
    over = {'actor': 'da_gpsr'}
    if duration:
        over['duration'] = duration
    sim = ModelActorSimulator(C.dataset_episode_config(scenario, rate, seed, **over), m, 'cpu',
                              mask=ck['mask'], schema_version=ck['schema'])
    return float(sim.run()['network_pdr'])


def _roll(job):
    arm, ckpt, scenario, rate, seed, duration = job
    t0 = time.time()
    pdr = (reference_episode(scenario, rate, seed, duration) if arm == 'reference'
           else student_episode(ckpt, scenario, rate, seed, duration))
    return arm, scenario, rate, seed, pdr, time.time() - t0


def paired(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    mean = float(d.mean())
    if n < 2:
        return mean, float('nan'), float('nan')
    se = float(d.std(ddof=1)) / math.sqrt(n)
    try:
        from scipy import stats as st
        crit = float(st.t.ppf(0.975, n - 1))
    except ImportError:
        crit = 2.262 if n == 10 else 2.776 if n == 5 else 1.96
    return mean, mean - crit * se, mean + crit * se


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join('data', 'phaseB_v3'))
    ap.add_argument('--out', default=os.path.join('results', 'g4_v3'))
    ap.add_argument('--model_seeds', type=int, default=2)
    ap.add_argument('--eval_seeds', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    ap.add_argument('--cells', nargs='+', default=None, help="'scenario@rate' (default: all training cells)")
    ap.add_argument('--include_heldout', action='store_true')
    ap.add_argument('--max_workers', type=int, default=8,
                    help='default 8: on Windows 16 spawn workers stalled a run (project report)')
    ap.add_argument('--device', default=None)
    ap.add_argument('--smoke', action='store_true', help='2 epochs, 60 s rollouts, 1 seed')
    ap.add_argument('--stage', choices=('all', 'train', 'rollout', 'analyze'), default='all')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    bad = [s for s in args.eval_seeds if 101 <= s <= 150]
    if bad:
        raise SystemExit(f'eval seeds {bad} are dataset seeds -- that would measure memorisation')
    duration = 60.0 if args.smoke else None
    if args.smoke:
        args.eval_seeds, args.model_seeds = args.eval_seeds[:1], 1

    cells = [(s, r) for s, r in C.dataset_cells() if s != C.GENERALISATION_SCENARIO]
    if args.cells:
        cells = [(s, r) for s, r in C.dataset_cells() if f'{s}@{r:g}' in args.cells]
    held = ([(s, r) for s, r in C.dataset_cells() if s == C.GENERALISATION_SCENARIO]
            if args.include_heldout else [])
    res_path = os.path.join(args.out, 'gate.json')
    res = json.load(open(res_path)) if os.path.isfile(res_path) else {'train': {}, 'pdr': {}}

    # ---- train
    if args.stage in ('all', 'train'):
        import torch
        from train_supervised_v2 import PhaseB, train_one, SEARCH_SPACE, MASK_PRESETS
        dman = json.load(open(os.path.join(args.data, 'manifest.json')))
        ds = PhaseB(args.data, mask=MASK_PRESETS['hop'], use_weights=bool(dman.get('use_weights')))
        hp = dict(SEARCH_SPACE)
        hp.update(TUNED)
        if args.smoke:
            hp.update(max_epochs=2, patience=1)
        dev = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
        for i in range(args.model_seeds):
            seed = 2000 + i
            ck = os.path.join(args.out, f'student_{seed}.pt')
            if str(seed) in res['train'] and os.path.isfile(ck):
                continue
            t0 = time.time()
            model, val, epochs = train_one(ds, 'attention', seed, dev, hp=hp)
            torch.save({'state_dict': model.state_dict(), 'mixer': 'attention', 'hp': hp,
                        'mask': MASK_PRESETS['hop'], 'schema': ds.schema_version,
                        'seed': seed, 'data': os.path.abspath(args.data),
                        'weighting': dman.get('weighting'), 'val_contested': val}, ck)
            res['train'][str(seed)] = {'val_contested': val, 'epochs': epochs,
                                       'n_train': int(len(ds.idx['train'])),
                                       'seconds': time.time() - t0, 'schema': ds.schema_version}
            json.dump(res, open(res_path, 'w'), indent=1)
            print(f'  [student {seed}] val contested {val:.4f}  epochs {epochs}  '
                  f"n_train {len(ds.idx['train']):,}  ({time.time() - t0:.0f}s)")
        del ds                        # a full export can be ~20 GB: free it before the
        import gc                     # rollout workers start
        gc.collect()

    # ---- rollout
    if args.stage in ('all', 'rollout'):
        jobs = []
        for s, r in cells + held:
            for sd in args.eval_seeds:
                k = f'reference|{s}@{r:g}|{sd}'
                if k not in res['pdr']:
                    jobs.append(('reference', None, s, r, sd, duration))
                for i in range(args.model_seeds):
                    k = f'student_{2000 + i}|{s}@{r:g}|{sd}'
                    if k not in res['pdr']:
                        jobs.append((f'student_{2000 + i}',
                                     os.path.join(args.out, f'student_{2000 + i}.pt'), s, r, sd, duration))
        print(f'  ROLLOUT: {len(jobs)} episodes (resume-aware)')
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
            futs = [ex.submit(_roll, j) for j in jobs]
            for n, fu in enumerate(as_completed(futs), 1):
                arm, s, r, sd, pdr, secs = fu.result()
                res['pdr'][f'{arm}|{s}@{r:g}|{sd}'] = pdr
                json.dump(res, open(res_path, 'w'), indent=1)
                el = time.time() - t0
                print(f'    {n}/{len(jobs)}  {arm:<14} {s}@{r:g} s{sd}  pdr={pdr:.4f}  {secs:.0f}s '
                      f'(~{el / n * (len(jobs) - n) / 60:.0f} min left)')

    # ---- analyze
    print('\n  G4 CHECK 4 (v3): student - restricted da_gpsr, paired, mean over model seeds')
    rows, incon, failed, incomplete = [], [], [], []
    for s, r in cells + held:
        cell = f'{s}@{r:g}'
        seeds = [sd for sd in args.eval_seeds if f'reference|{cell}|{sd}' in res['pdr']]
        studs = [f'student_{2000 + i}' for i in range(args.model_seeds)]
        if not seeds or not all(f'{a}|{cell}|{sd}' in res['pdr'] for a in studs for sd in seeds):
            print(f'    {cell:<18} incomplete')
            if (s, r) in cells:
                incomplete.append(cell)
            continue
        ref = [res['pdr'][f'reference|{cell}|{sd}'] for sd in seeds]
        stu = [np.mean([res['pdr'][f'{a}|{cell}|{sd}'] for a in studs]) for sd in seeds]
        mean, lo, hi = paired(stu, ref)
        gated = (s, r) in cells
        m = MARGIN_PP / 100
        if math.isnan(lo):
            status = 'PASS' if mean > -m else 'FAIL'
        elif lo > -m:
            status = 'PASS'
        elif hi < -m:
            status = 'FAIL'
        elif (hi - lo) / 2 > m:
            status = 'INCONCLUSIVE'
        else:
            status = 'FAIL'
        if gated and status == 'FAIL':
            failed.append(cell)
        if gated and status == 'INCONCLUSIVE':
            incon.append(cell)
        rows.append({'cell': cell, 'gated': gated, 'mean_pp': 100 * mean, 'ci_pp': [100 * lo, 100 * hi],
                     'ref_pdr': float(np.mean(ref)), 'n_seeds': len(seeds), 'status': status})
        print(f"    {cell:<18} ref {np.mean(ref):.4f}  diff {100 * mean:+.2f} pp "
              f"CI[{100 * lo:+.2f}, {100 * hi:+.2f}]  {status}"
              f"{'' if gated else '   (held out: reported, not gated)'}")
    verdict = ('FAIL' if failed else 'INCOMPLETE' if incomplete else
               'INCONCLUSIVE' if incon else 'PASS')
    res['verdict'] = {'verdict': verdict, 'pass': verdict == 'PASS', 'margin_pp': MARGIN_PP,
                      'failed': failed, 'inconclusive': incon, 'incomplete': incomplete,
                      'rows': rows}
    json.dump(res, open(res_path, 'w'), indent=1)
    print(f"\n  G4 CHECK 4 (v3): {verdict}   (non-inferiority margin {MARGIN_PP} pp)")
    if verdict == 'INCONCLUSIVE':
        print(f'  CI too wide in {incon}: re-run with --eval_seeds 1 2 3 4 5 6 7 8 9 10 (pre-registered)')
    return {'PASS': 0, 'FAIL': 1, 'INCONCLUSIVE': 2, 'INCOMPLETE': 3}[verdict]


if __name__ == '__main__':
    sys.exit(main())
