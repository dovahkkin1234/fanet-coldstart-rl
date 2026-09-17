"""apply_norm_constants_fix_v18.py -- close the remaining norm_constants duplication.

Assertion-guarded str.replace, idempotent, three independent anchors.

WHY

v17 fixed 'initial_energy' in norm_constants() -- it was hardcoded from
INITIAL_ENERGY_REF while every OTHER entry in the same dict read from cfg.
Scanning the same dict afterward found three more entries with the identical
shape, currently correct only because nothing has changed their source yet:

    'max_queue':    MAX_QUEUE_REF     (50.0)  vs simulator_v2.MAX_QUEUE (50)
    'ttl':          TTL_REF           (20.0)  vs simulator_v2.TTL (20)
    'lifetime_ref': LIFETIME_REF      (60.0)  vs link_model_v2's max_lifetime=60.0

The comment directly above these constants already says "mirrors simulator_v2 /
models.py" -- the intent to keep them in sync was stated from the start, it was
just never wired. Not bugs today. Same latent-drift risk v17 closed, left open.

WHY THREE ANCHORS, NOT ONE BLOCK

A first draft anchored a single 4-line block assuming 'max_queue',
'initial_energy', 'ttl', 'lifetime_ref' were still consecutive lines. They are
not, once v17 is applied: v17 inserted a 6-line comment directly above its own
line, breaking that adjacency. The block anchor matched zero times against a
v17-patched file. Three independent single-line anchors instead -- each unique
in the file regardless of what v17 (or anything else) put between them.

FIX -- same shape as v17

    'max_queue': MAX_QUEUE_REF,    -> 'max_queue': float(cfg.get('max_queue', MAX_QUEUE_REF)),
    'ttl': TTL_REF,                -> 'ttl': float(cfg.get('ttl', TTL_REF)),
    'lifetime_ref': LIFETIME_REF,  -> 'lifetime_ref': float(cfg.get('lifetime_ref', LIFETIME_REF)),

Defaults unchanged -- a config that omits these keys is bit-identical to
before. HOP_CAP and BUFFERED_REF are deliberately untouched: HOP_CAP is a
documented, intentional scoping decision, not a physics mirror; BUFFERED_REF is
ungrounded rather than duplicated (nothing in simulator_v2 defines a
"buffered" constant for it to drift out of sync with) -- a different, lower
priority issue, noted but not fixed here.

USAGE
    python apply_norm_constants_fix_v18.py --src src --dry-run
    python apply_norm_constants_fix_v18.py --src src
"""
import argparse, io, os, sys

TARGET = 'features_v2.py'
GUARD = "cfg.get('lifetime_ref'"

EDITS = [
    ("        'max_queue': MAX_QUEUE_REF,",
     "        # v18: follow cfg, same fix as v17 applied to 'initial_energy'.\n"
     "        'max_queue': float(cfg.get('max_queue', MAX_QUEUE_REF)),"),
    ("        'ttl': TTL_REF,",
     "        'ttl': float(cfg.get('ttl', TTL_REF)),"),
    ("        'lifetime_ref': LIFETIME_REF,",
     "        'lifetime_ref': float(cfg.get('lifetime_ref', LIFETIME_REF)),"),
]


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

    staged, ok = text, True
    for i, (old_line, new_line) in enumerate(EDITS, 1):
        n = staged.count(old_line)
        if n != 1:
            print(f"  anchor {i} ({old_line.strip()[:30]}...): "
                  f"matched {n} times, expected 1  <-- ABORT")
            ok = False
        else:
            print(f"  anchor {i}: OK")
            staged = staged.replace(old_line, new_line, 1)

    if not ok:
        print("\n  NO FILE WRITTEN."); return 1
    if a.dry_run:
        print("\n  DRY RUN OK -- 3/3 anchors matched. Nothing written."); return 0

    io.open(path, 'w', encoding='utf-8').write(staged)
    print(f"\n  WROTE {path}")
    print("  Defaults unchanged. NEXT: python verify_norm_constants_v18.py --src " + a.src)
    return 0


if __name__ == '__main__':
    sys.exit(main())
