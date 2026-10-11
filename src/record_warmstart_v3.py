"""
record_warmstart_v3.py  --  pin the M5 warmstart (DATASET_V3_SPEC §12 item 15).

    python src\\record_warmstart_v3.py --gate results\\g4_v3m
    python src\\record_warmstart_v3.py --gate results\\g4_v3m --check
    python src\\record_warmstart_v3.py --gate results\\g4_v3m --check --ckpt <a copy of the checkpoint>

Records the decision once, in <gate>/warmstart.json:
  * the warmstart, chosen by the RULE below from the students of a PASSED G4 gate;
  * the evidence the rule was applied to (gate.json, and student_eval_val.json if present);
  * the training decisions of 2026-10-10 that define its recipe (DECISIONS);
  * the sha256 of every candidate. The checkpoints are not in git, so the hash is what
    identifies the warmstart later: M5 loads it through selected_checkpoint(), which
    refuses a file whose hash differs.

Refuses: an existing warmstart.json (a decision is recorded once -- delete the file
deliberately to record another), a gate whose verdict is not PASS or that was a smoke run, a
candidate whose checkpoint is missing or is not the student gate.json trained (its recorded
validation accuracy must equal gate.json's), and a gate missing any episode the rule needs.

--check: verify the selected checkpoint (or --ckpt, e.g. a backup copy) against the
recorded sha256; exit code 1 on a mismatch.
"""

import argparse
import datetime
import hashlib
import json
import math
import os
import sys

VERSION = 'v32'
DECIDED = '2026-10-10'
RULE = ('of the students of a passed G4 gate, the one with the highest mean paired delivery '
        'difference (student - restricted da_gpsr, in pp) over every gated cell and evaluation '
        'seed of gate.json; ties go to the lower model seed. Delivery is the objective M5 '
        'optimises; imitation accuracy ranks students that are this close unreliably.')
DECISIONS = {
    'suite_c_weight': 'natural share: sink_50 = 18.6% of the training weight (22.8% of the '
                      'exported rows), as G4 v3m trained (DATASET_V3_SPEC V3-10)',
    'early_stopping': "unweighted contested validation accuracy ('accuracy_contested'), as "
                      'G4 v3m trained (DATASET_V3_SPEC V3-12)',
}
CAVEAT = ("chosen on the gate's own evaluation seeds, so the selected student's G4 numbers are "
          "slightly optimistic: measure M5's starting point on fresh seeds")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def selected_checkpoint(gate):
    """The warmstart checkpoint's path, after checking its sha256 against warmstart.json.
    M5 code loads the warmstart through this function, never by a hard-coded path."""
    wpath = os.path.join(gate, 'warmstart.json')
    if not os.path.isfile(wpath):
        raise SystemExit(f'  no warmstart recorded in {gate} (run record_warmstart_v3.py first)')
    with open(wpath) as f:
        rec = json.load(f)
    path = os.path.join(gate, rec['selected'])
    if not os.path.isfile(path):
        raise SystemExit(f'  the warmstart {path} is missing (restore it from a backup; '
                         f'--check --ckpt verifies a copy)')
    got, want = sha256_file(path), rec['candidates'][rec['selected']]['sha256']
    if got != want:
        raise SystemExit(f'  {path} is not the recorded warmstart (sha256 {got[:16]} != {want[:16]})')
    return path


def sign_test_two_sided(k, n):
    tail = sum(math.comb(n, i) for i in range(max(k, n - k), n + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def check(args):
    wpath = os.path.join(args.gate, 'warmstart.json')
    if not os.path.isfile(wpath):
        raise SystemExit(f'  no warmstart recorded in {args.gate}')
    with open(wpath) as f:
        rec = json.load(f)
    path = args.ckpt or os.path.join(args.gate, rec['selected'])
    if not os.path.isfile(path):
        raise SystemExit(f'  {path} is missing')
    got, want = sha256_file(path), rec['candidates'][rec['selected']]['sha256']
    if got != want:
        print(f'  MISMATCH: {path} has sha256 {got[:16]}..., the recorded warmstart '
              f"{rec['selected']} has {want[:16]}...")
        return 1
    print(f"  OK: {path} is the recorded warmstart {rec['selected']} (sha256 {got[:16]}..., "
          f"decided {rec['decided']})")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gate', required=True, help='the G4 gate folder (gate.json + student_*.pt)')
    ap.add_argument('--check', action='store_true', help='verify a checkpoint against the record')
    ap.add_argument('--ckpt', default=None, help='with --check: the file to verify (default: the '
                                                 'selected checkpoint in --gate)')
    args = ap.parse_args()
    if args.check:
        return check(args)
    if args.ckpt:
        raise SystemExit('  --ckpt is for --check only')

    wpath = os.path.join(args.gate, 'warmstart.json')
    if os.path.isfile(wpath):
        raise SystemExit(f'  {wpath} exists: a decision is recorded once (verify it with --check; '
                         f'to record another, delete the file deliberately)')
    gpath = os.path.join(args.gate, 'gate.json')
    if not os.path.isfile(gpath):
        raise SystemExit(f'  no gate.json in {args.gate}')
    with open(gpath) as f:
        g = json.load(f)
    ver = g.get('verdict') or {}
    if ver.get('verdict') != 'PASS':
        raise SystemExit(f"  {gpath}: verdict {ver.get('verdict')} -- a warmstart comes only from a "
                         f'gate that passed')
    if (g.get('provenance') or {}).get('smoke'):
        raise SystemExit(f'  {gpath} is a smoke run')
    gated = [r['cell'] for r in ver['rows'] if r['gated']]
    held = [r['cell'] for r in ver['rows'] if not r['gated']]
    pdr = g['pdr']
    seeds = sorted({int(k.split('|')[2]) for k in pdr if k.startswith('reference|')})
    students = sorted({k.split('|')[0] for k in pdr if k.startswith('student_')})
    if not gated or not seeds or not students:
        raise SystemExit(f'  {gpath} holds no gated cells, evaluation seeds or students')

    def diffs(stud, cells):
        miss = [f'{a}|{c}|{s}' for c in cells for s in seeds for a in ('reference', stud)
                if f'{a}|{c}|{s}' not in pdr]
        if miss:
            raise SystemExit(f'  {gpath} misses {len(miss)} episode(s) the rule needs, e.g. {miss[0]}')
        return [100 * (pdr[f'{stud}|{c}|{s}'] - pdr[f'reference|{c}|{s}']) for c in cells for s in seeds]

    evp = os.path.join(args.gate, 'student_eval_val.json')
    ev = {}
    export_sha = None
    if os.path.isfile(evp):
        with open(evp) as f:
            evj = json.load(f)
        ev = evj.get('students') or {}
        export_sha = (evj.get('trained_on') or {}).get('manifest_sha256_16')

    import torch
    cands = {}
    for stud in students:
        name = stud + '.pt'
        path = os.path.join(args.gate, name)
        if not os.path.isfile(path):
            raise SystemExit(f'  {path} is missing: every candidate must be present to be compared')
        ck = torch.load(path, map_location='cpu', weights_only=False)
        tr = (g.get('train') or {}).get(str(ck.get('seed')))
        if tr is None or ck.get('val_contested') != tr.get('val_contested'):
            raise SystemExit(f"  {path} is not the student gate.json trained (seed {ck.get('seed')}, "
                             f"val {ck.get('val_contested')} vs {tr and tr.get('val_contested')})")
        if f"student_{ck.get('seed')}" != stud:
            raise SystemExit(f"  {path} holds seed {ck.get('seed')}, not {stud}")
        d = diffs(stud, gated)
        h = diffs(stud, held) if held else []
        m = (ev.get(name) or {}).get('metrics') or {}
        cands[name] = {
            'sha256': sha256_file(path), 'bytes': os.path.getsize(path), 'seed': int(ck['seed']),
            'epochs': tr.get('epochs'), 'n_train': tr.get('n_train'),
            'es_metric': ck.get('es_metric', 'accuracy_contested'),
            'val_contested_at_selection': tr['val_contested'],
            'val_contested_w': m.get('accuracy_contested_w'),
            'g4_mean_diff_pp': sum(d) / len(d), 'g4_min_episode_diff_pp': min(d),
            'g4_max_episode_diff_pp': max(d),
            'g4_held_out_mean_diff_pp': (sum(h) / len(h)) if h else None,
        }
    best = max(cands, key=lambda k: (cands[k]['g4_mean_diff_pp'], -cands[k]['seed']))
    paired = {}
    for name in cands:
        if name == best:
            continue
        cell = [sum(pdr[f"{best[:-3]}|{c}|{s}"] - pdr[f"{name[:-3]}|{c}|{s}"] for s in seeds)
                * 100 / len(seeds) for c in gated]
        k = sum(1 for x in cell if x > 0)
        paired[name] = {'selected_minus_this_mean_pp': sum(cell) / len(cell), 'cells_ahead': k,
                        'cells': len(cell), 'sign_test_p_two_sided': sign_test_two_sided(k, len(cell))}
    with open(gpath, 'rb') as f:
        gsha = hashlib.sha256(f.read()).hexdigest()[:16]
    rec = {'version': VERSION, 'decided': DECIDED, 'recorded': datetime.date.today().isoformat(),
           'selected': best, 'rule': RULE, 'decisions': DECISIONS, 'caveat': CAVEAT,
           'gate': {'path': os.path.abspath(gpath), 'sha256_16': gsha, 'verdict': 'PASS',
                    'margin_pp': ver.get('margin_pp'), 'gated_cells': gated, 'held_out_cells': held,
                    'evaluation_seeds': seeds},
           'export_manifest_sha256_16': export_sha, 'candidates': cands, 'paired': paired}

    print(f'  candidates: {len(gated)} gated cells x {len(seeds)} evaluation seeds of {gpath}')
    for name, c in sorted(cands.items()):
        print(f"    {name:<16} epochs {c['epochs']:>3}  val contested {c['val_contested_at_selection']:.5f}  "
              f"G4 mean diff {c['g4_mean_diff_pp']:+.3f} pp  [{c['g4_min_episode_diff_pp']:+.3f}, "
              f"{c['g4_max_episode_diff_pp']:+.3f}]  sha256 {c['sha256'][:16]}...")
    print(f'  rule: {RULE}')
    print(f'  selected: {best}')
    for name, p in paired.items():
        print(f"    vs {name}: {p['selected_minus_this_mean_pp']:+.3f} pp, ahead in {p['cells_ahead']}/"
              f"{p['cells']} cells (sign test p = {p['sign_test_p_two_sided']:.3f})")
    with open(wpath, 'w') as f:
        json.dump(rec, f, indent=1)
    print(f'  written: {wpath}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
