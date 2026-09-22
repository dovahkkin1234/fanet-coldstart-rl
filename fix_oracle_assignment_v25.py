"""fix_oracle_assignment_v25.py -- recompute the oracle from stored per-seed data.

THE BUG (in panel_contenders_v2.py's decision rule, not in the measurements)

The leading-group rule was written as:

    excluded from the leading group only if BOTH significantly worse
    AND worse by >= MIN_PRACTICAL_DELTA

That reads sensibly and is wrong. A teacher 4.40pp worse than the top SURVIVES
the filter whenever the paired test fails to reach significance -- which at
n=10 with real variance happens routinely. Parsimony then selects it, because
it is the simplest member of a "leading group" it should never have been in.

Measured consequence on results/panel_contenders_v2.json -- 4 of 9
oracle-setting cells picked a teacher that is materially worse:

    dense_slow@60    picked dijkstra   -3.90pp vs da_gpsr   (p_holm 0.106)
    dense_slow@80    picked dijkstra   -3.29pp vs da_gpsr   (p_holm 0.245)
    dense_slow@100   picked dijkstra   -2.65pp vs gpsr      (p_holm 0.347)
    sink_50@30       picked dijkstra   -4.40pp vs da_gpsr   (p_holm 0.469)

"Not significantly worse" at n=10 does not mean "equally good". Parsimony was
intended to break GENUINE ties, never to promote an underpowered loser.

THE CORRECTED RULE

    leading group = every teacher whose POINT ESTIMATE is within
                    MIN_PRACTICAL_DELTA of the top scorer

Significance becomes confirmatory, reported alongside, rather than the gate.
Parsimony then breaks ties inside a group that is genuinely tied by effect
size. The medium_slow cells are unaffected -- dijkstra really is tied there
(0.10-0.17pp behind spbp_ab_noqueue), which is what the rule is for.

NO RE-RUN IS NEEDED. panel_contenders_v2 stores per_seed_pdr, so this
recomputes everything from the existing file. That per-seed storage was added
precisely so a decision-rule error would cost a recompute rather than 11 hours.

USAGE
    python fix_oracle_assignment_v25.py results\\panel_contenders_v2.json
"""
import json, math, sys

MIN_PRACTICAL_DELTA = 0.01          # 1pp
ALPHA = 0.05
COMPLEXITY = {'dijkstra': 0, 'gpsr': 1, 'da_gpsr': 2,
              'spbp_ab_noqueue': 3, 'spbp': 4}
QUALIFIED = {'sparse_fast'}


def paired_ttest(x, y):
    d = [xi - yi for xi, yi in zip(x, y)]
    n = len(d)
    if n < 2:
        return float('nan')
    mean = sum(d) / n
    var = sum((v - mean) ** 2 for v in d) / (n - 1)
    if var == 0:
        return 0.0 if mean != 0 else 1.0
    t = mean / math.sqrt(var / n)
    try:
        from scipy import stats
        return float(2 * stats.t.sf(abs(t), n - 1))
    except Exception:
        return float('nan')


def holm_dict(pmap):
    keys = list(pmap)
    vals = [pmap[k] for k in keys]
    idx = sorted(range(len(vals)), key=lambda i: vals[i])
    m, adj, run = len(vals), [0.0] * len(vals), 0.0
    for k, i in enumerate(idx):
        run = max(run, (m - k) * vals[i])
        adj[i] = min(1.0, run)
    return dict(zip(keys, adj))


def main():
    if len(sys.argv) != 2:
        print('usage: fix_oracle_assignment_v25.py <panel_contenders_v2.json>')
        return 1
    path = sys.argv[1]
    d = json.load(open(path))

    per_seed = d.get('per_seed_pdr')
    if not per_seed:
        print('  ERROR: no per_seed_pdr in this file -- it predates v2 storage.')
        print('  The assignment cannot be recomputed without re-running the panel.')
        return 1

    print('=' * 92)
    print(f'  RECOMPUTING ORACLE ASSIGNMENT -- {path}')
    print(f'  corrected rule: leading group = within {100*MIN_PRACTICAL_DELTA:.0f}pp of top')
    print('=' * 92)

    # rebuild comparisons from per-seed data, top vs all others
    praw, tags, cells = [], [], {}
    for key, per in per_seed.items():
        means = {t: sum(v.values()) / len(v) for t, v in per.items() if v}
        top = max(means, key=means.get)
        comps = {}
        for t in means:
            if t == top:
                continue
            seeds = sorted(set(per[top]) & set(per[t]))
            if len(seeds) < 2:
                continue
            pv = paired_ttest([per[t][s] for s in seeds], [per[top][s] for s in seeds])
            comps[t] = {'delta_vs_top': means[t] - means[top], 'p_raw': pv}
            if pv == pv:
                praw.append(pv); tags.append((key, t))
        cells[key] = {'means': means, 'top': top, 'comparisons': comps}

    adj = holm_dict(dict(zip(tags, praw))) if praw else {}
    for (key, t), _ in zip(tags, praw):
        cells[key]['comparisons'][t]['p_holm'] = adj.get((key, t), float('nan'))

    print(f"\n  {'cell':<22}{'old rec':<18}{'NEW rec':<18}{'gap closed'}")
    print('  ' + '-' * 70)
    assignment, changed = {}, []
    for key in sorted(cells):
        cr = cells[key]
        means, top = cr['means'], cr['top']
        lead = [t for t, m in means.items() if m >= means[top] - MIN_PRACTICAL_DELTA]
        rec = min(lead, key=lambda t: COMPLEXITY.get(t, 99))
        cr['leading_group'] = lead
        cr['recommended'] = rec
        old = d['results'].get(key, {}).get('recommended', '?')
        scenario = key.split('|')[0]
        if scenario not in QUALIFIED:
            assignment[key] = rec
        gap = ''
        if rec != old:
            og = 100 * (means.get(old, means[top]) - means[top])
            gap = f'{og:+.2f}p -> 0.00p'
            changed.append(key)
        print(f'  {key:<22}{old:<18}{rec:<18}{gap}')

    print(f'\n  {len(changed)} cell(s) corrected: {changed}')

    print('\n' + '=' * 92)
    print('  CORRECTED ORACLE ASSIGNMENT')
    print('=' * 92)
    for key in sorted(assignment):
        cr = cells[key]
        lead = cr['leading_group']
        tie = f"  (tied with {[t for t in lead if t != cr['recommended']]})" if len(lead) > 1 else ''
        print(f"  {key:<22}-> {assignment[key]}{tie}")
    for key in sorted(cells):
        if key.split('|')[0] in QUALIFIED:
            print(f"  {key:<22}   FLAGGED -- best {cells[key]['top']}, no oracle set")

    teachers_used = sorted(set(assignment.values()))
    print(f'\n  distinct oracles across {len(assignment)} cells: {teachers_used}')

    nq = [k for k, v in assignment.items() if v == 'spbp_ab_noqueue']
    print(f"  cells recommending spbp_ab_noqueue: {len(nq)}/{len(assignment)}")
    if not nq:
        print('  -> H2 stands: the SP-BP family is not the oracle anywhere.')
        tops = [k for k, c in cells.items() if c['top'] == 'spbp_ab_noqueue']
        if tops:
            print(f'     (though it IS the top scorer in {len(tops)} cell(s): {tops})')
            print('      -- tied with dijkstra there, i.e. stripped of its queue term')
            print('      SP-BP converges to shortest-path behaviour.)')

    d['results'] = {**d.get('results', {}), **{k: {**d.get('results', {}).get(k, {}), **v}
                                              for k, v in cells.items()}}
    d['oracle_assignment'] = assignment
    d['decision_rule'] = {
        'version': 'v25',
        'leading_group': f'point estimate within {MIN_PRACTICAL_DELTA} of top',
        'tiebreak': 'parsimony (simplest teacher in the leading group)',
        'superseded': 'v2 rule excluded only if significant AND >=1pp worse, '
                      'which let underpowered losers into the leading group',
    }
    out = path.replace('.json', '_corrected.json')
    with open(out, 'w') as f:
        json.dump(d, f, indent=2)
    print(f'\n  saved to {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
