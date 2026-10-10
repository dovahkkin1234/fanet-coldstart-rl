"""
eval_students_v3.py  --  imitation accuracy of the G4 students (DATASET_V3_SPEC §6).

    python src\\eval_students_v3.py --data data\\phaseB_v3m --gate results\\g4_v3m
    python src\\eval_students_v3.py --data data\\phaseB_v3m --gate results\\g4_v3m --eval_data data\\phaseB_v3m_test --split test

For every student_*.pt in the gate folder, on one split: raw and contested accuracy overall,
per behaviour policy, per scenario and per load bucket, unweighted (one row per distinct
context) and weighted by multiplicity (= the raw decision stream). Writes
<gate>/student_eval_<split>.json.

--data is the export the students were trained on. A checkpoint trained on another export is
refused: v31 checkpoints record the export manifest's hash; older ones (G4 v3m) are matched by
the export path they record, must be newer than the export's manifest and, when evaluated on
that export, must have trained on exactly its number of training rows (gate.json).

By default the students are evaluated on the val split of --data (val also chose their epoch).
The behaviour mix (§6) gives val no SP-BP-driven seed and test no gpsr-driven one. For
SP-BP-driven states, export the test split (export_phaseb_v3.py --splits test) and pass it as
--eval_data with --split test; it must come from the same dataset. Decide on val; the test
numbers are for the record.

Needs the RAM of the evaluated export (PhaseB loads it whole); uses a GPU if present.
"""

import argparse
import glob
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

VERSION = 'v31'
# what two exports must share to come from the same dataset
SAME_DATASET = ('source_created', 'code_signature', 'split_plan', 'norm_constants_per_scenario',
                'feature_schema_version', 'record_schema_version', 'label_teacher')


def read_manifest(d):
    p = os.path.join(d, 'manifest.json')
    if not os.path.isfile(p):
        raise SystemExit(f'  no export at {d} (manifest.json missing)')
    with open(p, 'rb') as f:
        raw = f.read()
    return hashlib.sha256(raw).hexdigest()[:16], json.loads(raw.decode('utf-8')), p


def same_path(a, b):
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='the export the students were trained on')
    ap.add_argument('--gate', required=True, help='the gate folder holding student_*.pt')
    ap.add_argument('--eval_data', default=None,
                    help='the export to evaluate on (default: --data); it must come from the same '
                         'dataset, e.g. its test split')
    ap.add_argument('--split', default='val', choices=('train', 'val', 'test', 'generalisation'))
    ap.add_argument('--device', default=None)
    args = ap.parse_args()

    cks = sorted(glob.glob(os.path.join(args.gate, 'student_*.pt')))
    if not cks:
        raise SystemExit(f'  no student_*.pt in {args.gate}')
    sha, man, mpath = read_manifest(args.data)
    eval_dir = args.eval_data or args.data
    same_export = same_path(eval_dir, args.data)
    if same_export:
        esha, eman = sha, man
    else:
        esha, eman, _ = read_manifest(eval_dir)
        diff = [k for k in SAME_DATASET if eman.get(k) != man.get(k)]
        if diff:
            raise SystemExit(f'  {eval_dir} does not come from the dataset behind {args.data} '
                             f'(differs in {diff})')

    import torch
    from model_gnn_attn import FANETRouter
    from train_supervised_v2 import PhaseB, evaluate

    gpath = os.path.join(args.gate, 'gate.json')
    gtrain = {}
    if os.path.isfile(gpath):
        with open(gpath) as f:
            gtrain = json.load(f).get('train') or {}
    loaded, legacy = [], []
    for p in cks:
        name = os.path.basename(p)
        ck = torch.load(p, map_location='cpu', weights_only=False)
        rec = ck.get('export_manifest_sha256_16')
        if rec is not None:
            if rec != sha:
                raise SystemExit(f'  {p} was trained on another export (manifest {rec}, '
                                 f'{args.data} has {sha})')
        else:
            if not same_path(str(ck.get('data', '')), args.data):
                raise SystemExit(f"  {p} was trained on {ck.get('data')}, not {args.data}")
            if os.path.getmtime(mpath) > os.path.getmtime(p):
                raise SystemExit(f'  {args.data} was rewritten after {name} was trained '
                                 f'(its manifest is newer than the checkpoint)')
            legacy.append(name)
            print(f'  note: {name} predates v31 -- matched to {args.data} by its recorded path')
        loaded.append((p, ck))
    masks = {tuple(ck['mask']) for _, ck in loaded}
    if len(masks) != 1:
        raise SystemExit(f'  the students were trained with different masks: {masks}')
    dev = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    t0 = time.time()
    ds = PhaseB(eval_dir, mask=list(masks.pop()), use_weights=bool(eman.get('use_weights')))
    n_split = int(len(ds.idx[args.split]))
    print(f'  loaded {eval_dir}: {ds.n:,} rows, {n_split:,} in {args.split} ({time.time() - t0:.0f}s)')
    if same_export:
        for p, ck in loaded:
            name = os.path.basename(p)
            want = (gtrain.get(str(ck.get('seed'))) or {}).get('n_train')
            if name in legacy and want is not None and int(want) != len(ds.idx['train']):
                raise SystemExit(f"  {name} trained on {int(want):,} rows (gate.json); "
                                 f"{args.data} has {len(ds.idx['train']):,} training rows")
    if not n_split:
        held = sorted(k for k, v in ds.idx.items() if len(v))
        raise SystemExit(f'  {eval_dir} has no {args.split} rows (it holds: {held})')
    if args.split == 'train':
        print('  note: the training split -- in-sample accuracy')
    if args.split == 'generalisation':
        print("  note: medium_slow seeds share each seed's mobility and flows with dense_slow "
              "(DATASET_V3_SPEC §0 #18) -- not a clean generalisation number")

    report = {'version': VERSION, 'split': args.split,
              'trained_on': {'path': os.path.abspath(args.data), 'manifest_sha256_16': sha,
                             'n_rows': man.get('n_rows'), 'ctx_frac': man.get('ctx_frac')},
              'evaluated_on': {'path': os.path.abspath(eval_dir), 'manifest_sha256_16': esha,
                               'n_rows': eman.get('n_rows'), 'ctx_frac': eman.get('ctx_frac'),
                               'rows_in_split': n_split},
              'legacy_checkpoints_matched_by_path': legacy, 'students': {}}
    for p, ck in loaded:
        name = os.path.basename(p)
        hp = ck['hp']
        m = FANETRouter(d=hp['d'], layers=hp['layers'], heads=hp['heads'], dropout=hp['dropout'],
                        mixer=ck['mixer'], attn_dropout=hp.get('attn_dropout', 0.0)).to(dev)
        m.load_state_dict(ck['state_dict'])
        t1 = time.time()
        out = evaluate(m, ds, args.split, dev, hp['frames_per_batch'], breakdown=True)
        report['students'][name] = {
            'seed': ck.get('seed'), 'es_metric': ck.get('es_metric', 'accuracy_contested'),
            'val_at_selection': ck.get('val_contested'), 'metrics': dict(sorted(out.items()))}
        print(f"\n  {name}  ({time.time() - t1:.0f}s)   rows {out['n']:,}")
        print(f"    overall    raw {out['accuracy_raw']:.4f}   contested {out['accuracy_contested']:.4f}"
              + (f"   contested, weighted {out['accuracy_contested_w']:.4f}"
                 if 'accuracy_contested_w' in out else ''))
        for tag, label in (('beh', 'behaviour'), ('sc', 'scenario')):
            pre = f'accuracy_contested_{tag}_'
            for k in sorted(k for k in out if k.startswith(pre)):
                b = k[len(pre):]
                w = out.get(f'accuracy_contested_w_{tag}_{b}')
                print(f"    {label:<10} {b:<16} contested {out[k]:.4f}"
                      + (f"   weighted {w:.4f}" if w is not None else '')
                      + f"   (n {out[f'n_contested_{tag}_{b}']:,})")
        for b in ('low', 'medium', 'high'):
            if f'accuracy_{b}' in out:
                print(f"    load       {b:<16} contested {out[f'accuracy_contested_{b}']:.4f}"
                      f"   raw {out[f'accuracy_{b}']:.4f}")
    dst = os.path.join(args.gate, f'student_eval_{args.split}.json')
    with open(dst, 'w') as f:
        json.dump(report, f, indent=1)
    print(f'\n  written: {dst}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
