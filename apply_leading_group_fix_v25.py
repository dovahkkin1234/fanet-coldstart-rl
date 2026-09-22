"""apply_leading_group_fix_v25.py -- fix the leading-group rule at source.

Assertion-guarded, idempotent, two anchors. Companion to
fix_oracle_assignment_v25.py, which repairs the already-produced JSON; this
stops future runs reproducing the same error.

THE BUG

    c['sig_worse_than_top'] = bool(c['significant'] and c['practically_worse'])

A teacher is excluded from the leading group only when it is BOTH significant
AND >=1pp worse. So a teacher 4.40pp worse SURVIVES whenever the paired test is
underpowered -- and parsimony then selects it as the simplest member of a group
it should never have entered. Measured: 4 of 9 oracle-setting cells picked a
teacher 2.65-4.40pp worse than the top.

THE FIX

Membership is decided by EFFECT SIZE, not by failure to reject:

    in_leading_group = delta_vs_top > -MIN_PRACTICAL_DELTA

Significance is still computed and printed, but as confirmation rather than as
the gate. Parsimony then breaks ties within a group that is genuinely tied.

USAGE
    python apply_leading_group_fix_v25.py --src src --dry-run
    python apply_leading_group_fix_v25.py --src src
"""
import argparse, io, os, sys

TARGET = 'panel_contenders_v2.py'
GUARD = 'v25: membership by EFFECT SIZE'

EDITS = [
    ("""        c['significant'] = bool(c['p_holm'] < ALPHA)
        c['practically_worse'] = bool(c['delta_vs_top'] <= -MIN_PRACTICAL_DELTA)
        # excluded from the leading group only if BOTH hold
        c['sig_worse_than_top'] = bool(c['significant'] and c['practically_worse'])""",
     """        c['significant'] = bool(c['p_holm'] < ALPHA)
        c['practically_worse'] = bool(c['delta_vs_top'] <= -MIN_PRACTICAL_DELTA)
        # v25: membership by EFFECT SIZE, not by failure to reject.
        # The previous rule (significant AND practically_worse) let a teacher
        # 4.40pp worse stay in the leading group whenever the paired test was
        # underpowered at n=10 -- parsimony then picked it for being simplest.
        # "Not significantly worse" is not "equally good". Significance is
        # still computed and printed, but as confirmation, not as the gate.
        c['sig_worse_than_top'] = bool(c['practically_worse'])"""),

    ("""        lead = [cr['top']] + [t for t, c in cr['comparisons'].items()
                              if not c.get('sig_worse_than_top', False)]""",
     """        # v25: leading group = every teacher whose POINT ESTIMATE sits within
        # MIN_PRACTICAL_DELTA of the top scorer.
        lead = [cr['top']] + [t for t, c in cr['comparisons'].items()
                              if c.get('delta_vs_top', -1) > -MIN_PRACTICAL_DELTA]"""),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='src')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    path = os.path.join(a.src, TARGET)
    if not os.path.exists(path):
        print(f'  ERROR: {path} not found'); return 1
    text = io.open(path, encoding='utf-8').read()
    if GUARD in text:
        print('  ALREADY APPLIED. Nothing to do.'); return 0

    staged, ok = text, True
    for i, (old, new) in enumerate(EDITS, 1):
        n = staged.count(old)
        if n != 1:
            print(f'  anchor {i}: matched {n} times, expected 1  <-- ABORT')
            ok = False
        else:
            print(f'  anchor {i}: OK')
            staged = staged.replace(old, new, 1)

    if not ok:
        print('\n  NO FILE WRITTEN.'); return 1
    if a.dry_run:
        print(f'\n  DRY RUN OK -- {len(EDITS)}/{len(EDITS)} anchors matched.'); return 0

    io.open(path, 'w', encoding='utf-8').write(staged)
    print(f'\n  WROTE {path}')
    print('  Future runs use the corrected rule. The EXISTING result file is')
    print('  repaired separately by fix_oracle_assignment_v25.py -- no re-run.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
