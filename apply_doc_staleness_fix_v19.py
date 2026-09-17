"""apply_doc_staleness_fix_v19.py -- mark §6.3's frozen grid as superseded.

Assertion-guarded str.replace, idempotent, one anchor. Documentation only --
no code, no simulator behaviour change.

WHY

Three generations of "the" rate grid exist in the written record with no
marker chain between them:

  1. PROBE_PREREGISTRATION.md §6.3 -- RATES = [0.02, 0.05, 0.10, 0.25, 0.40].
     This was itself measured on data later found to be leak-contaminated
     (the v12 destination-queue leak). Void.
  2. The v13 congestion-window rewrite of probe_rate_grid.py -- per-flow
     rates [20, 35, 50, 65, 80, 100, 120]. Measured post-v12, but at the OLD
     duration/energy operating point, before the energy-ceiling investigation.
     Superseded by (3).
  3. LOAD_DENSITY_DESIGN.md -- per-SCENARIO grids derived from each
     scenario's own measured usable band at the FINAL operating point
     (1000 s, INITIAL_ENERGY=8000). This is current.

Someone reading §6.3 today sees a "FROZEN GRID" heading with no indication
it is two generations stale. This patch adds a superseding note directly
above it -- the original text is left untouched below, so the historical
record (what was decided, when, and why it turned out wrong) stays intact,
matching the project's convention of recording amendments rather than
deleting superseded decisions.

USAGE
    python apply_doc_staleness_fix_v19.py --dry-run
    python apply_doc_staleness_fix_v19.py
"""
import argparse, io, os, sys

TARGET = os.path.join('docs', 'PROBE_PREREGISTRATION.md')
GUARD = 'SUPERSEDED (v19)'

OLD = """### 6.3 FROZEN GRID

Selected by **coverage**, not peak-finding, per A3:

```
RATES = [0.02, 0.05, 0.10, 0.25, 0.40]
```"""

NEW = """### 6.3 FROZEN GRID

> **SUPERSEDED (v19).** This grid was measured on data later found to be
> leak-contaminated (the v12 destination-queue leak -- see
> `apply_delivery_leak_fix_v12.py`). It is void, not merely outdated.
>
> A second generation (v13's congestion-window rates, per-flow
> `[20, 35, 50, 65, 80, 100, 120]`) replaced it post-leak-fix, measured at the
> operating point BEFORE the energy-ceiling investigation. That is also
> superseded.
>
> **Current truth is `LOAD_DENSITY_DESIGN.md`** -- per-SCENARIO grids derived
> from each scenario's own measured usable band at the final operating point
> (1000 s, `INITIAL_ENERGY=8000`, per `apply_v8b_operating_point_STAGED.py`).
> The text below is kept for history; do not use it to configure anything.

Selected by **coverage**, not peak-finding, per A3 (historical; see note above):

```
RATES = [0.02, 0.05, 0.10, 0.25, 0.40]   # VOID -- see LOAD_DENSITY_DESIGN.md
```"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    path = os.path.join(a.root, TARGET)
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
    return 0


if __name__ == '__main__':
    sys.exit(main())
