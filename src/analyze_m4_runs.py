"""
analyze_m4_runs.py — read what is already on disk. No GPU, no retraining.

    python src/analyze_m4_runs.py --runs results/m4/runs_L2_encoded.json --data data/phaseB

train_supervised_v2.py records far more per run than its summary prints: both
splits, raw and contested accuracy, and a per-load-bucket breakdown. Three of
the six G4 checks are answerable from that file without spending another
second of GPU time.

  check 1  beats the trivial baseline -- compare against the CONTESTED floor
           (0.5946), not the raw one (0.7114). 28.8% of decisions have the
           destination as the label; those are free wins that inflate every
           model equally and hide the model-vs-model difference.
  check 3  ORIGINAL FORM IS NOW MOOT. It asked whether the GNN's advantage
           grows with load. There is no advantage. The honest replacement,
           stated before looking: does the DEFICIT vary with load? A deficit
           concentrated at high load means something different from a flat one.
  check 5  generalisation to held-out medium_slow. Report reachability
           alongside it: that scenario is 54.7% reachable, and the locality
           experiment found hop-distance information is a genuine crutch
           exactly there (+0.011..+0.013, p<0.05 in all three cells). A drop
           here is partly partition structure, not model quality.

HOLM CORRECTION IS APPLIED ACROSS THE WHOLE FAMILY.
G3 Holm-corrects its 12 cells and reports it; M-7 is a standing reviewer item.
Reporting ~14 paired t-tests here uncorrected, in the same paper, would be an
internal inconsistency a reviewer will find before you do.
"""

import argparse
import json
import math
import os


def paired(a, b):
    n = len(a)
    d = [x - y for x, y in zip(a, b)]
    mean = sum(d) / n
    var = sum((x - mean) ** 2 for x in d) / (n - 1) if n > 1 else 0.0
    sd = math.sqrt(var)
    se = sd / math.sqrt(n) if n > 1 else 0.0
    t = mean / se if se > 0 else 0.0
    try:
        from scipy import stats
        p = float(2 * stats.t.sf(abs(t), df=n - 1))
        crit = float(stats.t.ppf(0.975, df=n - 1))
    except ImportError:
        p, crit = float('nan'), 1.96
    return dict(n=n, mean=mean, sd=sd, t=t, p=p,
                lo=mean - crit * se, hi=mean + crit * se,
                d=(mean / sd if sd > 0 else float('nan')),
                wins=sum(1 for x in d if x > 0))


def holm(ps):
    """Holm-Bonferroni. Returns adjusted p in the ORIGINAL order."""
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    m, adj, run = len(ps), [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * ps[i]
        run = max(run, val)                 # enforce monotonicity
        adj[i] = min(run, 1.0)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True)
    ap.add_argument('--data', default='data/phaseB')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    runs = json.load(open(args.runs))
    man_path = os.path.join(args.data, 'manifest.json')
    man = json.load(open(man_path)) if os.path.isfile(man_path) else {}

    # Mixer names are READ FROM THE FILE, not hardcoded: an
    # attention_edgekey-vs-mlp run would otherwise KeyError here.
    mixers = sorted({k.split(':')[0] for k in runs})
    if len(mixers) < 2:
        raise SystemExit(f'need two mixers in the file, found {mixers}')
    m_a, m_b = mixers[0], mixers[1]
    seeds = sorted({int(k.split(':')[1]) for k in runs})
    seeds = [s for s in seeds if f'{m_a}:{s}' in runs and f'{m_b}:{s}' in runs]
    if not seeds:
        raise SystemExit('no seeds with BOTH models present')

    metrics = [k for k in runs[f'{m_a}:{seeds[0]}']
               if k.startswith(('test_', 'generalisation_'))
               and not k.endswith('_n')]
    metrics = [m for m in metrics if 'above_trivial' not in m]

    print('=' * 78)
    print('  M4 RUN ANALYSIS — from stored results, no retraining')
    print('=' * 78)
    print(f'  {args.runs}')
    print(f'  {len(seeds)} paired seeds  |  schema v{man.get("feature_schema_version","?")}'
          f'  local_horizon={man.get("local_horizon","?")}')
    ep = [runs[f'{m}:{s}'].get('epochs_run') for s in seeds for m in ('attention', 'mlp')]
    ep = [e for e in ep if e]
    if ep:
        print(f'  epochs to early stop: min={min(ep)} median={sorted(ep)[len(ep)//2]} max={max(ep)}')

    rows, ps = [], []
    for k in sorted(metrics):
        a = [runs[f'{m_a}:{s}'][k] for s in seeds]
        b = [runs[f'{m_b}:{s}'][k] for s in seeds]
        if any(x != x for x in a + b):
            continue
        st = paired(a, b)
        st['metric'] = k
        st['gnn'] = sum(a) / len(a)
        st['mlp'] = sum(b) / len(b)
        rows.append(st)
        ps.append(st['p'])
    for r, q in zip(rows, holm(ps)):
        r['p_holm'] = q

    print('\n' + '-' * 78)
    print(f'  {m_a} vs {m_b}  (positive = {m_a} better)')
    print('-' * 78)
    print(f"  {'metric':<40}{m_a[:8]:>8}{m_b[:8]:>8}{'diff':>9}{'p_holm':>10}  sig")
    for r in rows:
        star = '*' if r['p_holm'] < 0.05 else ' '
        print(f"  {r['metric']:<40}{r['gnn']:8.4f}{r['mlp']:8.4f}"
              f"{r['mean']:+9.4f}{r['p_holm']:10.2e}  {star}")
    print(f"\n  {len(ps)} paired tests, Holm-corrected as a family (M-7). "
          f"* = significant after correction.")

    # ---- check 1 --------------------------------------------------------
    # Compute the floor from the DATASET, never a hardcoded constant: 0.5946
    # is this dataset's value and would be silently wrong on any other.
    tr_c = man.get('trivial_contested')
    # PER-SPLIT floors, not one pooled number.
    # The pooled contested floor (0.5946) is an average over scenarios whose own
    # floors differ enormously, because the floor is driven by candidate count:
    #   very_dense 0.2529 (17.9 candidates)   dense_slow  0.4147 (9.1)
    #   medium_slow 0.6832 (4.0)              sparse_fast 0.8035 (2.7)
    # Scoring the held-out medium_slow result against the pooled floor makes
    # generalisation look BETTER than test (+0.02), which is an artifact of
    # having a quarter as many candidates to choose between -- not evidence of
    # good generalisation. Each split gets its own floor.
    floors, scen_floors = {}, {}
    try:
        import numpy as _np
        _d = _np.load(os.path.join(args.data, 'decisions.npz'))
        _co, _cf = _d['cand_offsets'], _d['cand_flat']
        _con = _cf[_co[:-1] + _d['label']] != _d['dst']
        _sc, _sd = _d['scenario'], _d['seed'].astype(int)
        _ks = _co[1:] - _co[:-1]
        for _s in _np.unique(_sc):
            _m = (_sc == _s) & _con
            scen_floors[str(_s)] = (float((_d['label'][_m] == 0).mean()),
                                    float(_ks[_sc == _s].mean()))
        _pl = man.get('split_plan', {})
        _hoscen = _pl.get('generalisation_scenario', 'medium_slow')
        _gen = _sc == _hoscen
        _te = ~_gen
        if 'test_seeds' in _pl:
            _te = _te & (_sd >= _pl['test_seeds'][0]) & (_sd <= _pl['test_seeds'][1])
        for _nm, _m in (('test', _te), ('generalisation', _gen)):
            _mm = _m & _con
            if _mm.sum():
                floors[_nm] = float((_d['label'][_mm] == 0).mean())
        tr_c = floors.get('test')
    except Exception as _e:
        tr_c = None
    print('\n' + '-' * 78)
    print('  CHECK 1 — beats the trivial baseline (contested floor)')
    print('-' * 78)
    base = next((r for r in rows if r['metric'] == 'test_accuracy_contested'), None)
    if base:
        if scen_floors:
            print("  per-scenario contested floors (driven by candidate count):")
            for _s, (_f, _k) in sorted(scen_floors.items()):
                print(f"    {_s:<14} floor {_f:.4f}   {_k:.2f} mean candidates")
        if tr_c is None:
            print("  test-split floor unavailable (pass --data); skipping check 1")
            floor = None
        else:
            floor = tr_c
            print(f"  TEST-SPLIT floor {floor:.4f}   "
                  f"GNN {base['gnn']:.4f} ({base['gnn']-floor:+.4f})   "
                  f"MLP {base['mlp']:.4f} ({base['mlp']-floor:+.4f})")
        if floor is not None:
            print(f"  -> {'PASS for both' if min(base['gnn'], base['mlp']) > floor + 0.02 else 'REVIEW'}")

    # ---- check 3 (replacement form) -------------------------------------
    print('\n' + '-' * 78)
    print('  CHECK 3 (replacement) — does the GNN deficit vary with LOAD?')
    print('  Original form asked whether the ADVANTAGE grows with load. There')
    print('  is no advantage, so the pre-registered question is answered NO.')
    print('  This is the informative remainder, stated before looking.')
    print('-' * 78)
    for split in ('test', 'generalisation'):
        got = [(b, next((r for r in rows
                         if r['metric'] == f'{split}_accuracy_contested_{b}'), None))
               for b in ('low', 'medium', 'high')]
        got = [(b, r) for b, r in got if r]
        if not got:
            continue
        print(f"  {split}:")
        for b, r in got:
            print(f"    {b:<8} GNN {r['gnn']:.4f}  MLP {r['mlp']:.4f}  "
                  f"diff {r['mean']:+.4f}  [{r['lo']:+.4f},{r['hi']:+.4f}]  "
                  f"p_holm={r['p_holm']:.2e}")
        ds = [r['mean'] for _, r in got]
        print(f"    spread low->high: {ds[0]:+.4f} -> {ds[-1]:+.4f}  "
              f"({'deficit GROWS with load' if ds[-1] < ds[0] - 0.002 else 'deficit SHRINKS with load' if ds[-1] > ds[0] + 0.002 else 'FLAT across load'})")

    # ---- check 5 --------------------------------------------------------
    print('\n' + '-' * 78)
    print('  CHECK 5 — generalisation to held-out medium_slow')
    print('-' * 78)
    t = next((r for r in rows if r['metric'] == 'test_accuracy_contested'), None)
    g = next((r for r in rows if r['metric'] == 'generalisation_accuracy_contested'), None)
    if t and g:
        ft, fg = floors.get('test'), floors.get('generalisation')
        print(f"  RAW accuracy   GNN test {t['gnn']:.4f} -> gen {g['gnn']:.4f}  "
              f"({g['gnn']-t['gnn']:+.4f})")
        print(f"                 MLP test {t['mlp']:.4f} -> gen {g['mlp']:.4f}  "
              f"({g['mlp']-t['mlp']:+.4f})")
        if ft and fg:
            print(f"\n  ABOVE OWN FLOOR (test {ft:.4f}, generalisation {fg:.4f}) "
                  f"-- this is the honest comparison:")
            print(f"    GNN  {t['gnn']-ft:+.4f} (test) -> {g['gnn']-fg:+.4f} (gen)"
                  f"   {'DROP' if g['gnn']-fg < t['gnn']-ft else 'gain'} "
                  f"{abs((g['gnn']-fg)-(t['gnn']-ft)):.4f}")
            print(f"    MLP  {t['mlp']-ft:+.4f} (test) -> {g['mlp']-fg:+.4f} (gen)"
                  f"   {'DROP' if g['mlp']-fg < t['mlp']-ft else 'gain'} "
                  f"{abs((g['mlp']-fg)-(t['mlp']-ft)):.4f}")
            print("    The raw numbers above show generalisation BEATING test.")
            print("    That is a candidate-count artifact: medium_slow averages")
            print("    ~4 candidates against ~18 in very_dense. Against its own")
            print("    floor the picture is different -- report THIS pair.")
        print(f"  GNN-MLP on generalisation: {g['mean']:+.4f} "
              f"[{g['lo']:+.4f},{g['hi']:+.4f}]  p_holm={g['p_holm']:.2e}")
        print()
        print("  REPORT REACHABILITY WITH THIS NUMBER. medium_slow is 54.7%")
        print("  reachable, and the locality experiment found hop-distance")
        print("  information is a real crutch there (+0.011..+0.013, p<0.05 in")
        print("  all three cells) while being ~free elsewhere. Part of any drop")
        print("  is partition structure, not model quality.")
        if g['mean'] > 0 and g['p_holm'] < 0.05:
            print()
            print("  ** THE SIGN FLIPS ON THE HELD-OUT SCENARIO: the GNN wins")
            print("     here while losing on test. That makes the result")
            print("     REGIME-DEPENDENT, not a flat loss, and it is the single")
            print("     most important thing in this output. Do not report the")
            print("     pooled number alone.")

    if args.out:
        json.dump({'seeds': seeds, 'rows': rows}, open(args.out, 'w'), indent=2)
        print(f"\n  wrote {args.out}")
    print('=' * 78 + '\n')


if __name__ == '__main__':
    main()
