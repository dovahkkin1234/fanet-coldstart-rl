"""apply_panel_delta_fix_v16.py -- fix the delta sign bug in the oracle panel.

Assertion-guarded str.replace, idempotent, one anchor.

THE BUG

Two versions of the comparison logic were written in sequence. An early draft
picked the argmax `winner` and computed:

    comps[t] = {'delta': means[winner] - means[t], ...}

That was later reframed to compare every challenger against a FIXED incumbent
('spbp') instead of the argmax winner -- to fix a different bug, where an
all-tied cell let one outlier seed get declared a "winner" on noise. The loop
condition, the seed-pairing, and the paired_ttest(x, y) call were all updated
correctly for that reframing. The `delta` line was not.

CONFIRMED, not suspected: every stored delta in a real 4-cell, 8-teacher run
matched `means[winner] - means[t]` exactly (7/7 exact matches, arithmetic
verified independently from the raw 'means' dict). The reported p-values are
NOT affected -- the t-test itself used the correct incumbent -- only the sign
and magnitude of `delta`, and therefore the printed "who beats whom" direction,
were wrong. This inverted a verdict: "backpressure beats spbp, do not pin it"
should have read "spbp beats backpressure, keep it."

FIX

    'delta': means[winner] - means[t]   ->   'delta': means[t] - means[incumbent]

USAGE
    python apply_panel_delta_fix_v16.py --src src --dry-run
    python apply_panel_delta_fix_v16.py --src src
"""
import argparse, io, os, sys

TARGET = 'panel_convergecast_oracle.py'
GUARD = "'delta': means[t] - means[incumbent]"

OLD = "            comps[t] = {'delta': means[winner] - means[t], 'p_raw': float(p),"
NEW = ("            # v16 FIX: this was 'means[winner] - means[t]' -- a leftover\n"
       "            # from an earlier argmax-winner design. The loop below already\n"
       "            # correctly pairs seeds and runs the t-test against `incumbent`;\n"
       "            # only this delta line still referenced the wrong comparator.\n"
       "            # Confirmed by exact arithmetic match on a real run: every stored\n"
       "            # delta equalled means[winner]-means[t], not means[t]-means[incumbent].\n"
       "            comps[t] = {'delta': means[t] - means[incumbent], 'p_raw': float(p),")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='src')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    path = os.path.join(a.src, TARGET)
    if not os.path.exists(path):
        print(f"  ERROR: {path} not found"); return 1
    text = io.open(path, encoding='utf-8').read()
    if GUARD in text:
        print("  ALREADY APPLIED. Nothing to do."); return 0

    n = text.count(OLD)
    if n != 1:
        print(f"  anchor matched {n} times, expected 1  <-- ABORT")
        return 1
    print("  anchor 1: OK")
    if a.dry_run:
        print("\n  DRY RUN OK. Nothing written."); return 0

    io.open(path, 'w', encoding='utf-8').write(text.replace(OLD, NEW, 1))
    print(f"\n  WROTE {path}")
    print("  Any FUTURE panel run will report the correct direction.")
    print("  Your EXISTING results/panel_cc_v2.json does not need re-running --")
    print("  use fix_panel_json_v16.py to recompute its verdict from the saved data.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
