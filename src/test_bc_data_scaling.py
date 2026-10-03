"""
test_bc_data_scaling.py  --  does behaviour cloning of da_gpsr need more data? (Point 1)

    python src\\test_bc_data_scaling.py --stage all --max_workers 8
    python src\\test_bc_data_scaling.py --stage all --extend --rollout --max_workers 8

THE QUESTION. Dataset V3 keeps every distinct decision context (60-270k per
1000 s episode, ~100-150M over the grid -- DATASET_V3_SPEC section 9), 200-300x the
449k rows on which the teacher-choice test's GLOBAL student already reached
99.4-99.5% held-out accuracy and PDR parity with da_gpsr. Is training on more of
it worth the GPU time and RAM, or is imitation already saturated? The decisive rows are the HARD ones -- label != slot 0,
i.e. where da_gpsr's queue and link-quality terms override "go to the node closest
to the destination". They are 4-25% of contexts and they carry the congestion-aware
behaviour the warmstart is supposed to inherit.

Uses only results\\teacher_choice (its raw episodes, assembled datasets, models and
rollout.json) -- no change to the M4 pipeline.

STAGES
  hard     accuracy of the existing teacher-test students (GLOBAL, GLOBAL_HO,
           S_da_ms) on the held-out validation seeds, split into easy / hard rows
  curve    retrain GLOBAL on NESTED subsets of its 449k training rows (3/10/30/100%)
           and evaluate the same way
  extend   (--extend) re-simulate the same 30 training episodes recording 4x the
           original record_prob (20% instead of 5%) -- same trajectories (recording
           uses its own random stream; verified: identical metrics, and the 5% rows
           are an exact subset of the 20% rows, features and actions included) --
           then train at ~4x the data (~1.8M rows). The trainer batches by FRAMES
           (48 per batch), so 4x rows also means 4x rows per batch at the same
           number of steps per epoch -- exactly how it will consume a denser V3
           export, which is the decision this test informs.
  rollout  (--rollout) roll every curve/extend model out on the first 5 rollout
           seeds of the teacher-choice run and compare with da_gpsr's PDR already
           stored in its rollout.json (paired, same seeds)
  analyze  table + the pre-registered reading below

PRE-REGISTERED READING (fixed before running)
  DATA-LIMITED  if, in any cell, hard-row accuracy rises by >= 0.5 pp from the 30%
                subset to 100%, or from 100% to the 4x extension, or rollout PDR vs
                da_gpsr improves by >= 0.5 pp over the same steps
  SATURATED     otherwise: report the smallest training size within 0.5 pp of the
                best on every cell, overall AND hard rows
Settings (duration, altitude, battery, epsilon, record_prob, seeds) are read from
<tc>\config.json, i.e. exactly what the teacher-choice run used.
Feature semantics: the teacher-test data are schema v5 (frame-start snapshot own
queue). Every model here is trained and rolled out with v5 semantics, whether or
not the v26 patch (schema v6) has been applied. What it measures -- how imitation
accuracy grows with the amount of training data -- does not depend on that.
"""

import argparse
import inspect
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CELLS = ('medium_slow', 'dense_slow', 'very_dense')
FRACS = (0.03, 0.10, 0.30, 1.0)
STEP_PP = 0.5
X4 = 4.0
# test_teacher_choice.py defaults, used only if <tc>\config.json is missing
TC_DEFAULTS = dict(duration=300.0, z_min=100.0, z_max=300.0, initial_energy=8000.0,
                   epsilon=0.10, record_prob=0.05, val_seeds=[136, 137],
                   rollout_seeds=list(range(1, 11)))


def tc_settings(tc_dir):
    """The teacher-choice run's own arguments. The extension must reproduce its
    trajectories exactly (same duration / altitude / battery / epsilon -- a different
    epsilon gives a different trajectory) and the rollouts must be paired with its
    da_gpsr arm, so nothing here is hard-coded when config.json exists."""
    p = os.path.join(tc_dir, 'config.json')
    c = json.load(open(p)) if os.path.isfile(p) else {}
    s = {k: c.get(k, v) for k, v in TC_DEFAULTS.items()}
    for k in ('duration', 'z_min', 'z_max', 'initial_energy', 'epsilon', 'record_prob'):
        s[k] = float(s[k])
    return s, (p if c else 'test_teacher_choice defaults (config.json not found)')


def _sm64(i, salt=97):
    m = (1 << 64) - 1
    x = (int(i) * 0x9E3779B97F4A7C15 + salt) & m
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & m
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & m
    return ((x ^ (x >> 31)) >> 11) / 2.0 ** 53


def legacy_phaseb(path, mask):
    from train_supervised_v2 import PhaseB
    if 'accept_legacy' in inspect.signature(PhaseB.__init__).parameters:
        return PhaseB(path, mask=mask, accept_legacy=True)
    return PhaseB(path, mask=mask)


class Subset:
    """A PhaseB view whose TRAIN split keeps only the selected rows. Validation and
    frames are untouched, so every subset is evaluated on identical data."""

    def __init__(self, ds, keep_mask):
        self.__dict__.update(ds.__dict__)
        self.idx = dict(ds.idx)
        self.idx['train'] = ds.idx['train'][keep_mask[ds.idx['train']]]
        self.by_frame = dict(ds.by_frame)
        fs = []
        for fid, ids in ds.by_frame['train']:
            k = ids[keep_mask[ids]]
            if len(k):
                fs.append((fid, k))
        self.by_frame['train'] = fs
        self._ds = ds

    def batch(self, split, frame_slice, device):
        return type(self._ds).batch(self, split, frame_slice, device)


def per_row_eval(model, ds, split, device, fpb=48):
    """Per-row correctness (train_supervised_v2.evaluate only returns means)."""
    import torch
    from train_supervised_v2 import run_batch
    model.eval()
    ok, ids = [], []
    with torch.no_grad():
        fs = ds.by_frame[split]
        for i in range(0, len(fs), fpb):
            bt = ds.batch(split, fs[i:i + fpb], device)
            ok.append((run_batch(model, bt).argmax(-1) == bt['label']).cpu().numpy())
            ids.append(bt['ids'])
    return np.concatenate(ok), np.concatenate(ids)


def summarise(ok, ids, ds):
    d = ds.dec
    out = {}
    for sc in CELLS:
        m = d['scenario'][ids] == sc
        if not m.any():
            continue
        hard = d['label'][ids] != 0
        con = ds.contested[ids]
        out[sc] = {'acc': float(ok[m].mean()), 'acc_contested': float(ok[m & con].mean()),
                   'acc_hard': float(ok[m & hard].mean()) if (m & hard).any() else float('nan'),
                   'acc_easy': float(ok[m & ~hard].mean()),
                   'hard_share': float(hard[m].mean()), 'n': int(m.sum())}
    return out


def load_model(path, device):
    import torch
    from model_gnn_attn import FANETRouter
    ck = torch.load(path, map_location=device, weights_only=False)
    hp = ck['hp']
    m = FANETRouter(d=hp['d'], layers=hp['layers'], heads=hp['heads'], dropout=hp['dropout'],
                    mixer=ck['mixer'], attn_dropout=hp.get('attn_dropout', 0.0)).to(device)
    m.load_state_dict(ck['state_dict'])
    m.eval()
    return m


SMOKE = {'on': False}


def hp_tuned():
    from train_supervised_v2 import SEARCH_SPACE
    hp = dict(SEARCH_SPACE)
    hp.update(lr=1e-3, attn_dropout=0.1, max_epochs=100)
    if SMOKE['on']:
        # smoke records up to every decision in the extension: small batches keep a
        # CPU-only plumbing run inside a few GB (a batch is a set of frames)
        hp.update(max_epochs=2, patience=1, frames_per_batch=8)
    return hp


# ─────────────────────────────────────────────────────────────────────────────
def stage_hard(a, res, device):
    from train_supervised_v2 import MASK_PRESETS
    mask = MASK_PRESETS['hop']
    evals = {sc: legacy_phaseb(os.path.join(a.tc, 'data', f'EVAL_{sc}_da_gpsr'), mask) for sc in CELLS}
    for name in ('GLOBAL', 'GLOBAL_HO', 'S_da_ms'):
        for ms in (2000, 2001):
            key = f'{name}:{ms}'
            p = os.path.join(a.tc, 'models', f'{name}_{ms}.pt')
            if key in res['hard'] or not os.path.isfile(p):
                continue
            model = load_model(p, device)
            out = {}
            for sc, ds in evals.items():
                ok, ids = per_row_eval(model, ds, 'val', device)
                out.update(summarise(ok, ids, ds))
            res['hard'][key] = out
            print(f'  [{key:<13}] ' + '  '.join(
                f"{sc}: all {v['acc']:.4f} hard {v['acc_hard']:.4f} (share {v['hard_share']:.3f})"
                for sc, v in out.items()))
    return res


def train_eval(ds, keep, seed, device, evals, hp, ckpt):
    import torch
    from train_supervised_v2 import train_one, MASK_PRESETS
    sub = Subset(ds, keep)
    t0 = time.time()
    model, val, epochs = train_one(sub, 'attention', seed, device, hp=hp)
    torch.save({'state_dict': model.state_dict(), 'mixer': 'attention', 'hp': hp,
                'mask': MASK_PRESETS['hop'], 'schema': 5, 'seed': seed}, ckpt)
    out = {}
    for sc, eds in evals.items():
        ok, ids = per_row_eval(model, eds, 'val', device)
        out.update(summarise(ok, ids, eds))
    return {'n_train': int(len(sub.idx['train'])), 'epochs': epochs, 'val_contested': val,
            'seconds': time.time() - t0, 'cells': out}


def stage_curve(a, res, device, data_dir, tag, fracs):
    from train_supervised_v2 import MASK_PRESETS
    mask = MASK_PRESETS['hop']
    ds = legacy_phaseb(data_dir, mask)
    evals = {sc: legacy_phaseb(os.path.join(a.tc, 'data', f'EVAL_{sc}_da_gpsr'), mask) for sc in CELLS}
    u = np.array([_sm64(i) for i in range(ds.n)])
    hp = hp_tuned()
    os.makedirs(os.path.join(a.out, 'models'), exist_ok=True)
    for f in fracs:
        for ms in range(a.model_seeds):
            key = f'{tag}:{f:g}:{2000 + ms}'
            if key in res['curve']:
                continue
            ck = os.path.join(a.out, 'models', f"{tag}_{f:g}_{2000 + ms}.pt")
            r = train_eval(ds, u < f, 2000 + ms, device, evals, hp, ck)
            r['ckpt'] = ck
            res['curve'][key] = r
            json.dump(res, open(os.path.join(a.out, 'scaling.json'), 'w'), indent=1)
            print(f"  [{key:<16}] n_train {r['n_train']:>9,}  epochs {r['epochs']:>3}  "
                  + '  '.join(f"{sc}: {v['acc']:.4f}/{v['acc_hard']:.4f}" for sc, v in r['cells'].items())
                  + f"  ({r['seconds']:.0f}s)")
    return res


# ── extend: same trajectories, 4x the recorded decisions ─────────────────────
def _gen_x4(job):
    scen, seed, rec_p, eps, raw_dir, tc_args = job
    import features_v2 as F
    import test_teacher_choice as TC
    args = argparse.Namespace(**tc_args)
    sim = TC.TeacherSim(TC.episode_config(scen, seed, args), 'da_gpsr', epsilon=eps, record_prob=rec_p)
    if 'schema_version' in inspect.signature(F.norm_constants).parameters:
        sim.nc = F.norm_constants(TC.episode_config(scen, seed, args), schema_version=5)
    t0 = time.time()
    sim.run()
    used = sorted(sim._used)
    remap = {f: i for i, f in enumerate(used)}
    fr = [sim._frames[f] for f in used]
    D = sim.dec
    np.savez_compressed(
        os.path.join(raw_dir, f'{scen}_da_gpsr_s{seed}.npz'),
        node_feat=np.concatenate([f[1] for f in fr]), node_counts=np.array([f[1].shape[0] for f in fr]),
        edge_index=np.concatenate([f[2] for f in fr], axis=1), edge_feat=np.concatenate([f[3] for f in fr]),
        edge_counts=np.array([f[2].shape[1] for f in fr]),
        frame=np.array([remap[d['frame']] for d in D]), current=np.array([d['current'] for d in D]),
        dst=np.array([d['dst'] for d in D]), cand_flat=np.concatenate([d['cands'] for d in D]),
        cand_counts=np.array([len(d['cands']) for d in D]), cand_feat=np.concatenate([d['cf'] for d in D]),
        query=np.array([d['qf'] for d in D], np.float32),
        lab_da_gpsr=np.array([d['lab_da_gpsr'] for d in D], np.int32))
    return scen, seed, len(D), time.time() - t0


def assemble(rows, out_dir, split_plan):
    """rows: [(scenario, seed, raw_npz_path)] -> a PhaseB directory (schema v5)."""
    import features_v2 as F
    NF, NC, EI, EF, EC, CF, CI, CK = [], [], [], [], [], [], [], []
    cols = {k: [] for k in ('frame_id', 'current', 'dst', 'query_feat', 'label', 'scenario',
                            'seed', 'load_bucket')}
    off = 0
    for scen, sd, p in rows:
        z = dict(np.load(p))
        NF.append(z['node_feat']); NC.append(z['node_counts'])
        EI.append(z['edge_index']); EF.append(z['edge_feat']); EC.append(z['edge_counts'])
        co = np.concatenate([[0], np.cumsum(z['cand_counts'])])
        CI.append(z['cand_flat']); CF.append(z['cand_feat']); CK.append(z['cand_counts'])
        n = len(z['lab_da_gpsr'])
        cols['frame_id'].append(z['frame'] + off); cols['current'].append(z['current'])
        cols['dst'].append(z['dst']); cols['query_feat'].append(z['query'])
        cols['label'].append(z['lab_da_gpsr']); cols['scenario'].append(np.array([scen] * n))
        cols['seed'].append(np.full(n, sd, np.int32)); cols['load_bucket'].append(np.array(['band'] * n))
        off += len(z['node_counts'])
        assert co[-1] == len(z['cand_flat'])
    os.makedirs(out_dir, exist_ok=True)
    nc_, ec_ = np.concatenate(NC), np.concatenate(EC)
    np.savez(os.path.join(out_dir, 'frames.npz'),
             node_feat_flat=np.concatenate(NF).astype(np.float32),
             node_ids_flat=np.concatenate([np.arange(k) for k in nc_]).astype(np.int32),
             node_offsets=np.concatenate([[0], np.cumsum(nc_)]).astype(np.int64),
             edge_index_flat=np.concatenate(EI, axis=1),
             edge_feat_flat=np.concatenate(EF).astype(np.float32),
             edge_offsets=np.concatenate([[0], np.cumsum(ec_)]).astype(np.int64))
    ck = np.concatenate(CK)
    np.savez(os.path.join(out_dir, 'decisions.npz'),
             cand_flat=np.concatenate(CI).astype(np.int32),
             cand_offsets=np.concatenate([[0], np.cumsum(ck)]).astype(np.int64),
             cand_feat_flat=np.concatenate(CF).astype(np.float32),
             **{k: np.concatenate(v) for k, v in cols.items()})
    v5 = getattr(F, 'LEGACY_FEATURE_LISTS', {}).get(5, {'query_features': F.QUERY_FEATURES})
    json.dump({'purpose': 'test_bc_data_scaling x4 extension', 'feature_schema_version': 5,
               'node_features': F.NODE_FEATURES, 'edge_features': F.EDGE_FEATURES,
               'query_features': v5['query_features'], 'candidate_features': F.CANDIDATE_FEATURES,
               'local_horizon': F.LOCAL_HORIZON,
               'split_plan': split_plan},
              open(os.path.join(out_dir, 'manifest.json'), 'w'), indent=1)


def stage_extend(a, res, device):
    # bulky intermediates live under <out>/data/, which .gitignore's 'data/' already ignores
    raw4 = os.path.join(a.out, 'data', 'raw_x4')
    os.makedirs(raw4, exist_ok=True)
    tman = json.load(open(os.path.join(a.tc, 'data', 'GLOBAL', 'manifest.json')))
    tr_lo, tr_hi = tman['split_plan']['train_seeds']
    va_lo, va_hi = tman['split_plan']['val_seeds']
    train_seeds = [s for s in tman['seeds'] if tr_lo <= s <= tr_hi]
    val_seeds = [s for s in tman['seeds'] if va_lo <= s <= va_hi]
    rp0 = a.tc_cfg['record_prob']
    rp = min(1.0, X4 * rp0)
    jobs = [(sc, sd, rp, a.tc_cfg['epsilon'], raw4, a.tc_args) for sc in CELLS for sd in train_seeds
            if not os.path.isfile(os.path.join(raw4, f'{sc}_da_gpsr_s{sd}.npz'))]
    print(f'  EXTEND: {len(jobs)} episodes at record_prob {rp:g} (original {rp0:g}, x{rp / rp0:.2f}); '
          f'same trajectories, original rows an exact subset')
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.max_workers) as ex:
        for i, fu in enumerate(as_completed([ex.submit(_gen_x4, j) for j in jobs]), 1):
            sc, sd, n, secs = fu.result()
            print(f'    {i}/{len(jobs)}  {sc:<12} s{sd}  decisions={n:>7}  {secs:.0f}s  '
                  f'(elapsed {time.time() - t0:.0f}s)')
    d4 = os.path.join(a.out, 'data', 'GLOBAL_x4')
    if not os.path.isfile(os.path.join(d4, 'manifest.json')):
        rows = [(sc, sd, os.path.join(raw4, f'{sc}_da_gpsr_s{sd}.npz')) for sc in CELLS for sd in train_seeds]
        rows += [(sc, sd, os.path.join(a.tc, 'raw', f'{sc}_da_gpsr_s{sd}.npz')) for sc in CELLS for sd in val_seeds]
        assemble(rows, d4, tman['split_plan'])
    return stage_curve(a, res, device, d4, 'x4', (1.0,))


# ── rollouts ─────────────────────────────────────────────────────────────────
_MC = {}


def _roll(job):
    ckpt, scen, seed, tc_args = job
    import torch
    import test_teacher_choice as TC
    from rollout_eval_v2 import ModelActorSimulator
    from train_supervised_v2 import MASK_PRESETS
    torch.set_num_threads(1)
    if ckpt not in _MC:
        _MC[ckpt] = load_model(ckpt, 'cpu')
    cfg = TC.episode_config(scen, seed, argparse.Namespace(**tc_args))
    kw = {'mask': MASK_PRESETS['hop']}
    if 'schema_version' in inspect.signature(ModelActorSimulator.__init__).parameters:
        kw['schema_version'] = 5
    t0 = time.time()
    m = ModelActorSimulator(cfg, _MC[ckpt], 'cpu', **kw).run()
    return ckpt, scen, seed, float(m['network_pdr']), time.time() - t0


def stage_rollout(a, res):
    jobs = []
    for key, r in res['curve'].items():
        for sc in CELLS:
            for sd in a.rollout_seeds:
                if f'{key}|{sc}|{sd}' not in res['pdr']:
                    jobs.append((r['ckpt'], sc, sd, a.tc_args))
    inv = {r['ckpt']: k for k, r in res['curve'].items()}
    print(f'  ROLLOUT: {len(jobs)} episodes')
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.max_workers) as ex:
        for i, fu in enumerate(as_completed([ex.submit(_roll, j) for j in jobs]), 1):
            ck, sc, sd, pdr, secs = fu.result()
            res['pdr'][f'{inv[ck]}|{sc}|{sd}'] = pdr
            json.dump(res, open(os.path.join(a.out, 'scaling.json'), 'w'), indent=1)
            print(f'    {i}/{len(jobs)}  {inv[ck]:<18} {sc:<12} s{sd}  pdr={pdr:.4f}  {secs:.0f}s '
                  f'(~{(time.time() - t0) / i * (len(jobs) - i) / 60:.0f} min left)')
    return res


# ── analyze ──────────────────────────────────────────────────────────────────
def stage_analyze(a, res):
    tc_roll = json.load(open(os.path.join(a.tc, 'rollout.json')))
    groups = {}
    for key, r in res['curve'].items():
        tag, f, ms = key.split(':')
        groups.setdefault((tag, float(f)), []).append((key, r))
    order = sorted(groups, key=lambda g: (g[0] != 'GLOBAL', g[1]))
    print('\n  BEHAVIOUR-CLONING DATA SCALING  (mean over model seeds; acc = all rows / HARD rows)')
    if res['hard']:
        print(f"  existing teacher-test students on validation seeds {a.tc_cfg['val_seeds']}:")
        for k, v in sorted(res['hard'].items()):
            print(f'    {k:<14} ' + '  '.join(f"{sc} {c['acc']:.4f}/{c['acc_hard']:.4f}" for sc, c in v.items()))
    hdr = '  '.join(f'{sc:>21}' for sc in CELLS)
    print(f"\n    {'data':<16} {'n_train':>10}  {hdr}   PDR vs da_gpsr (pp): " + ' '.join(f'{c[:6]:>7}' for c in CELLS))
    table = {}
    for g in order:
        rows = groups[g]
        n = int(np.mean([r['n_train'] for _, r in rows]))
        acc = {sc: (float(np.mean([r['cells'][sc]['acc'] for _, r in rows])),
                    float(np.mean([r['cells'][sc]['acc_hard'] for _, r in rows]))) for sc in CELLS}
        pdd, by_seed = {}, {}
        for sc in CELLS:
            by_seed[sc] = {}
            for sd in a.rollout_seeds:
                st = [res['pdr'].get(f'{k}|{sc}|{sd}') for k, _ in rows]
                ref = tc_roll.get(f'da_gpsr|{sc}|{sd}', {}).get('pdr')
                if ref is not None and all(x is not None for x in st):
                    by_seed[sc][str(sd)] = 100 * (float(np.mean(st)) - ref)
            v = list(by_seed[sc].values())
            pdd[sc] = (float(np.mean(v)), len(v)) if v else (float('nan'), 0)
        table[f'{g[0]}:{g[1]:g}'] = {'n_train': n, 'acc': acc, 'pdr_diff_pp': pdd,
                                     'pdr_diff_pp_by_seed': by_seed}
        label = f'{g[0]} {g[1]:g}'
        print(f'    {label:<16} {n:>10,}  ' + '  '.join(f'{acc[sc][0]:.4f} / {acc[sc][1]:.4f}' + ' ' * 4
                                                   for sc in CELLS)
              + '   ' + ' '.join(f'{pdd[sc][0]:+7.2f}' for sc in CELLS))
    # pre-registered reading
    steps = [('GLOBAL:0.3', 'GLOBAL:1'), ('GLOBAL:1', 'x4:1')]
    limited = []
    print('\n  steps (hard-row accuracy; PDR paired over rollout seeds, se = seed noise of the step):')
    for lo, hi in steps:
        if lo not in table or hi not in table:
            continue
        for sc in CELLS:
            dh = 100 * (table[hi]['acc'][sc][1] - table[lo]['acc'][sc][1])
            bl, bh = table[lo]['pdr_diff_pp_by_seed'][sc], table[hi]['pdr_diff_pp_by_seed'][sc]
            d = [bh[s] - bl[s] for s in bh if s in bl]
            dp = float(np.mean(d)) if d else float('nan')
            se = float(np.std(d, ddof=1) / math.sqrt(len(d))) if len(d) > 1 else float('nan')
            msg = f'{sc}: {lo}->{hi} hard acc {dh:+.2f} pp, PDR {dp:+.2f} pp (se {se:.2f}, n={len(d)})'
            print(f'    {msg}')
            if dh >= STEP_PP or (not math.isnan(dp) and dp >= STEP_PP):
                limited.append(msg)
    if limited:
        verdict = 'DATA-LIMITED: ' + '; '.join(limited)
    else:
        best = {sc: max(t['acc'][sc][1] for t in table.values()) for sc in CELLS}
        ok = [k for k, t in sorted(table.items(), key=lambda kv: kv[1]['n_train'])
              if all(100 * (best[sc] - t['acc'][sc][1]) < STEP_PP for sc in CELLS)]
        verdict = (f"SATURATED: {ok[0]} ({table[ok[0]]['n_train']:,} rows) is within {STEP_PP} pp of "
                   f'the best hard-row accuracy in every cell' if ok else 'SATURATED (no single size within margin)')
    print(f'\n  PRE-REGISTERED READING: {verdict}')
    res['analysis'] = {'table': table, 'verdict': verdict}
    json.dump(res, open(os.path.join(a.out, 'scaling.json'), 'w'), indent=1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tc', default=os.path.join('results', 'teacher_choice'))
    ap.add_argument('--out', default=os.path.join('results', 'bc_data_scaling'))
    ap.add_argument('--stage', choices=('all', 'hard', 'curve', 'extend', 'rollout', 'analyze'), default='all')
    ap.add_argument('--extend', action='store_true', help='include the 4x-data stage in --stage all')
    ap.add_argument('--rollout', action='store_true', help='include rollouts in --stage all')
    ap.add_argument('--model_seeds', type=int, default=2)
    ap.add_argument('--rollout_seeds', type=int, nargs='+', default=None,
                    help="default: the first 5 of the teacher-choice run's rollout seeds "
                         '(paired with its da_gpsr arm)')
    ap.add_argument('--max_workers', type=int, default=8,
                    help='default 8: on Windows 16 spawn workers stalled a run (project report)')
    ap.add_argument('--device', default=None)
    ap.add_argument('--smoke', action='store_true',
                    help='plumbing check against a test_teacher_choice --smoke result (2 epochs, 1 seed)')
    a = ap.parse_args()
    a.tc_cfg, cfg_src = tc_settings(a.tc)
    a.tc_args = {k: a.tc_cfg[k] for k in ('duration', 'z_min', 'z_max', 'initial_energy')}
    if a.rollout_seeds is None:
        a.rollout_seeds = list(a.tc_cfg['rollout_seeds'])[:5]
    unpaired = [s for s in a.rollout_seeds if s not in a.tc_cfg['rollout_seeds']]
    if a.smoke:
        SMOKE['on'] = True
        a.rollout_seeds = a.rollout_seeds[:1]
        a.model_seeds = 1
    import torch
    device = a.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(a.out, exist_ok=True)
    p = os.path.join(a.out, 'scaling.json')
    res = json.load(open(p)) if os.path.isfile(p) else {'hard': {}, 'curve': {}, 'pdr': {}}
    print('=' * 78)
    print(f'  BC DATA SCALING  tc={a.tc}  device={device}'
          f'{"   *** SMOKE: numbers meaningless ***" if a.smoke else ""}')
    print('=' * 78)
    print(f"  teacher-choice settings from {cfg_src}: {a.tc_cfg['duration']:g}s, "
          f"z {a.tc_cfg['z_min']:g}-{a.tc_cfg['z_max']:g} m, battery {a.tc_cfg['initial_energy']:g}, "
          f"epsilon {a.tc_cfg['epsilon']:g}, record_prob {a.tc_cfg['record_prob']:g}; "
          f'rollout seeds {a.rollout_seeds}')
    if unpaired:
        print(f'  WARNING: rollout seeds {unpaired} are not in the teacher-choice rollout.json -- '
              f'no paired da_gpsr reference for them (they are left out of the PDR columns)')
    t0 = time.time()
    if a.stage in ('all', 'hard'):
        res = stage_hard(a, res, device)
    if a.stage in ('all', 'curve'):
        res = stage_curve(a, res, device, os.path.join(a.tc, 'data', 'GLOBAL'), 'GLOBAL', FRACS)
    if a.stage == 'extend' or (a.stage == 'all' and a.extend):
        res = stage_extend(a, res, device)
    json.dump(res, open(p, 'w'), indent=1)
    if a.stage == 'rollout' or (a.stage == 'all' and a.rollout):
        res = stage_rollout(a, res)
    if a.stage in ('all', 'analyze', 'rollout'):
        stage_analyze(a, res)
    print(f'\n  saved {p}   total {time.time() - t0:.0f}s')
    return 0


if __name__ == '__main__':
    sys.exit(main())
