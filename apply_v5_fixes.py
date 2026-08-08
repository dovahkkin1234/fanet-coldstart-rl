"""
apply_v5_fixes.py — the redundancy check found something on REAL data.

Applies ON TOP OF v1-v4. This is still ONE cycle: the standing rule is "if the
correlation check surfaces further redundancy, fold it into that same pass and
re-run the whole thing." That is exactly what happened, so this is the fold-in,
not a twelfth audit round.

════════════════════════════════════════════════════════════════════════════
HOW THIS WAS FOUND
════════════════════════════════════════════════════════════════════════════
A full v1-v4 stack was built from a clean baseline and a 34,400-decision
dataset was regenerated end to end. G3.5 then FAILED check 8:

    ** edge: link_quality <-> packet_error_rate  |r|=0.9982 |rho|=0.9987

Measured directly on the 526,038 stored edge rows:

    lq       mean=0.4073  std=0.2942
    per      mean=0.5836  std=0.2974
    lq + per mean=0.9909  std=0.0181     <- they are COMPLEMENTS

per is not merely correlated with lq. It is approximately 1 - lq. This is the
ttl_left / hops_so_far defect again, in the edge block, and it went undetected
through every previous milestone because check 8 did not exist until v1.

Every other edge pair is clean, which is what makes this a duplication rather
than a general property of the block:

    distance <-> link_quality              |r|=0.5362  |rho|=0.4734
    distance <-> packet_error_rate         |r|=0.4973  |rho|=0.4363
    link_quality <-> estimated_lifetime    |r|=0.1410  |rho|=0.0649
    link_quality <-> relative_velocity     |r|=0.0107  |rho|=0.0005

WHY THEY ARE COMPLEMENTS. In simulator_v2._measured_link both are built from
the same three quantities -- sinr_clean, p_clear, p_coll:

    lq  = clip(sinr_db/30, 0, 1) * p_clear * (1 - p_coll)
    per = 1 - (1 - per_phy) * (1 - p_interf) * (1 - p_coll)

per_phy is a monotone decreasing function of the same sinr that drives
lq_clean, and the two share the p_coll factor outright. In the measured
operating range the composition lands within 0.018 of exact complementarity.

  J1  EDGE_FEATURES drops 'packet_error_rate'. 5 -> 4. Schema 3 -> 4.
      link_quality is kept rather than per because it is the signal every
      teacher scores on (da_gpsr, backpressure, spbp all read link_quality),
      so keeping it preserves the correspondence between what the model sees
      and what produced its labels. per stays on the graph and in the
      simulator; only the model's view of it is removed.
      The audit's only positional dependency is
      F.EDGE_FEATURES.index('link_quality'), which is name-based and
      unaffected.

════════════════════════════════════════════════════════════════════════════
J2  A PREDICTION I GOT WRONG — the cand_reachable exemption is not needed
════════════════════════════════════════════════════════════════════════════
v4 added a NAMED EXEMPTION for cand_hop_distance <-> cand_reachable, on my
reasoning that cand_hop_distance already saturates at 1.0 for unreachable
candidates and the two would therefore be near-perfectly collinear. Measured
on the real dataset:

    cand_hop_distance <-> cand_reachable   |r|=0.8391  |rho|=0.2597

Both far below the 0.98 / 0.995 thresholds. The exemption never fires.

The reason my prediction failed: 95.8% of candidates ARE reachable, and among
those, cand_hop_distance varies across {0, 0.1, 0.2, 0.3, ...}. Rank
correlation is dominated by that variation, not by the binary reachable/not
split. The saturation argument was right about the mechanism and wrong about
the magnitude.

  J2  The exemption is REMOVED. Leaving a standing exemption for a pair that
      is not actually redundant is worse than useless: if they ever did become
      collinear, the gate would stay silent about it. The machinery stays
      (empty dict, _exempt(), and the handling in redundancy_report) so a
      future genuine case can be added with a written reason -- but nothing is
      exempted on speculation.

      This does NOT undo the decision to add cand_reachable. That column was
      justified by ABLATABILITY, not by information content, and the
      measurement changes nothing about that argument.

════════════════════════════════════════════════════════════════════════════
J3  Split completeness — fail fast, not after 261 seconds
════════════════════════════════════════════════════════════════════════════
_split_sizes assigns by hardcoded seed ranges (train 101-135, val 136-142,
test 143-150) while --seeds is a free CLI argument. Running with seeds outside
101-150 produces a dataset where decisions match no branch and are silently
counted into no split. Observed with --seeds 1 2 3: 22,416 of 34,400
decisions (65.2%) orphaned.

TO BE ACCURATE ABOUT THE SEVERITY: the independent audit DOES catch this --
"[FAIL] Split viable, all decisions assigned". G3.5 does not; its check 5
("seeds=3") and check 7 ("3 unique seeds") both pass happily. So the layered
defence works and this was never a silent corruption path, as long as the
audit is run -- which the plan requires. It is a footgun, not a hole.

  J3  Generation-time assertion so the failure costs seconds instead of a
      full generation plus two gates. Refuses to write a dataset whose splits
      do not account for every decision, and names the seed range.

USAGE
    python apply_v5_fixes.py --src src [--dry-run]
"""

import argparse
import os
import sys

VERSION = "v5"


class PatchError(RuntimeError):
    pass


def sub(text, old, new, label, path):
    n = text.count(old)
    if n != 1:
        raise PatchError(
            f"[{path}] anchor for '{label}' matched {n} times, expected 1.\n"
            f"  anchor starts: {old[:110]!r}")
    return text.replace(old, new, 1)


# ── J1 ───────────────────────────────────────────────────────────────────────
J1A_OLD = """EDGE_FEATURES = [
    'distance',
    'link_quality',
    'packet_error_rate',
    'estimated_link_lifetime',
    'relative_velocity',
]"""

J1A_NEW = """EDGE_FEATURES = [
    'distance',
    'link_quality',
    'estimated_link_lifetime',
    'relative_velocity',
]
# REMOVED FEATURE -- 'packet_error_rate' sat between 'link_quality' and
# 'estimated_link_lifetime'. It is very nearly 1 - link_quality.
# Measured on 526,038 stored edge rows from a full regeneration:
#     lq       mean=0.4073  std=0.2942
#     per      mean=0.5836  std=0.2974
#     lq + per mean=0.9909  std=0.0181     <- COMPLEMENTS
#     |r| = 0.9982   |rho| = 0.9987
# Every other edge pair is clean (next highest is distance <-> link_quality at
# |r|=0.5362), so this is a duplication, not a property of the block.
#
# The cause is shared construction in simulator_v2._measured_link: both are
# built from the same sinr_clean, p_clear and p_coll --
#     lq  = clip(sinr_db/30,0,1) * p_clear * (1 - p_coll)
#     per = 1 - (1 - per_phy) * (1 - p_interf) * (1 - p_coll)
# with per_phy monotone decreasing in the same sinr, and (1 - p_coll) shared
# outright.
#
# link_quality is the one kept because it is the signal every teacher scores
# on -- da_gpsr, backpressure and spbp all read link_quality -- so the model's
# view stays aligned with what produced its labels. packet_error_rate remains
# on the graph and in the simulator; only the model's view of it is removed.
#
# Found by G3.5 check 8 on a real regeneration, not by inspection. This is the
# third instance of the same defect class after snr <-> distance and
# ttl_left <-> hops_so_far, and the first one no human predicted."""

J1B_OLD = """FEATURE_SCHEMA_VERSION = 3"""
J1B_NEW = """FEATURE_SCHEMA_VERSION = 4"""

J1C_OLD = """            e.get('packet_error_rate', 0.0),
"""
J1C_NEW = ""


# ── J2 ───────────────────────────────────────────────────────────────────────
J2_OLD = """ALLOWED_REDUNDANT_PAIRS = {
    ('candidate', 'cand_hop_distance', 'cand_reachable'):
        "DELIBERATE. cand_reachable adds no information -- cand_hop_distance "
        "already saturates at 1.0 for unreachable candidates, and with mean "
        "hops 1.73-3.15 against HOP_CAP=10 a legitimate 1.0 essentially never "
        "occurs. The column exists so reachability can be MASKED "
        "INDEPENDENTLY in the M4 deployability ablation, which is impossible "
        "while the flag is folded inside a distance column. Reachability-from-"
        "destination is a global property, so this is the column that has to "
        "come out to support a decentralisation claim.",
}"""

J2_NEW = """# EMPTY ON PURPOSE. An entry was added here for
# cand_hop_distance <-> cand_reachable on the PREDICTION that they would be
# near-collinear, since cand_hop_distance saturates at 1.0 for unreachable
# candidates. Measured on the real dataset, they are not:
#     cand_hop_distance <-> cand_reachable   |r|=0.8391  |rho|=0.2597
# -- both far below threshold, because 95.8% of candidates are reachable and
# the rank correlation is dominated by hop variation among those rather than
# by the binary split. The exemption never fired, and a standing exemption for
# a pair that is not redundant is worse than none: if they ever did become
# collinear, this gate would stay silent about it. So it is removed.
#
# The machinery stays for a future GENUINE case. Nothing gets exempted on a
# prediction -- only on a measurement plus a written reason.
ALLOWED_REDUNDANT_PAIRS = {}"""


# ── J3 ───────────────────────────────────────────────────────────────────────
J3_OLD = """    tot = max(tr + va + te + ge, 1)
    return {'train': tr, 'val': va, 'test': te, 'generalisation': ge,
            'generalisation_share': round(ge / tot, 4)}"""

J3_NEW = """    tot = max(tr + va + te + ge, 1)

    # FAIL FAST ON ORPHANED DECISIONS.
    # The split is assigned by HARDCODED seed ranges (train 101-135, val
    # 136-142, test 143-150) while --seeds is a free CLI argument. Seeds
    # outside 101-150 match no branch and are counted into no split, silently.
    # Observed with --seeds 1 2 3: 22,416 of 34,400 decisions (65.2%) orphaned.
    #
    # The independent audit does catch this ("Split viable, all decisions
    # assigned"), and G3.5 does not -- its check 5 and check 7 both pass. So
    # this was never a silent corruption path provided the audit is run. But
    # discovering it after a full generation plus two gates costs a great deal
    # more than discovering it here.
    assigned = tr + va + te + ge
    if assigned != len(all_dec):
        raise AssertionError(
            f"{len(all_dec) - assigned} of {len(all_dec)} decisions "
            f"({100 * (len(all_dec) - assigned) / max(len(all_dec), 1):.1f}%) "
            f"belong to NO split.\\n"
            f"  The split is assigned by seed range: train 101-135, "
            f"val 136-142, test 143-150, and any seed for the held-out "
            f"scenario {heldout!r}.\\n"
            f"  Seeds seen: {sorted({int(d['seed']) for d in all_dec})}\\n"
            f"  Either pass --seeds within 101-150, or update the ranges in "
            f"_split_sizes to match.")

    return {'train': tr, 'val': va, 'test': te, 'generalisation': ge,
            'generalisation_share': round(ge / tot, 4)}"""


PATCHES = {
    'features_v2.py': [
        (J1A_OLD, J1A_NEW, 'J1 drop packet_error_rate'),
        (J1B_OLD, J1B_NEW, 'J1 schema version 4'),
        (J1C_OLD, J1C_NEW, 'J1 extract_frame'),
    ],
    'preflight_dataset_v2_check.py': [
        (J2_OLD, J2_NEW, 'J2 remove speculative exemption'),
    ],
    'generate_dataset_v2.py': [
        (J3_OLD, J3_NEW, 'J3 split completeness assertion'),
    ],
}

MARKER = "REMOVED FEATURE -- 'packet_error_rate'"
GUARD = 'features_v2.py'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='src')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    g = os.path.join(args.src, GUARD)
    if not os.path.isfile(g):
        print(f"ERROR: {g} not found. Run from the repo root or pass --src.")
        return 2
    txt = open(g, encoding='utf-8').read()
    if MARKER in txt:
        print(f"ERROR: {GUARD} already patched by v5. Restore from git first.")
        return 2
    if 'cand_reachable' not in txt:
        print(f"ERROR: {GUARD} has no cand_reachable.\n"
              f"  v5 applies on top of v1-v4. Apply those first.")
        return 2

    staged = {}
    print(f"\n{'=' * 78}\n  V5 FIXES — assertion-guarded\n{'=' * 78}")
    for fname, edits in PATCHES.items():
        path = os.path.join(args.src, fname)
        if not os.path.isfile(path):
            raise PatchError(f'missing file: {path}')
        text = open(path, encoding='utf-8').read()
        for old, new, label in edits:
            text = sub(text, old, new, label, fname)
            print(f"    [ok] {fname:<32} {label}")
        staged[path] = text

    if args.dry_run:
        print(f"\n  DRY RUN — {len(staged)} file(s) would change, nothing written.")
        return 0

    for path, text in staged.items():
        open(path, 'w', encoding='utf-8').write(text)
    print(f"\n  WROTE {len(staged)} file(s).")
    print("\n  Feature schema is now v4:")
    print("    NODE      9   unchanged")
    print("    EDGE      4   was 6 -- dropped snr (v1), packet_error_rate (v5)")
    print("    QUERY     6   was 7 -- dropped hops_so_far (v1)")
    print("    CANDIDATE 4   was 3 -- added cand_reachable (v4)")
    print("=" * 78 + "\n")
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except PatchError as e:
        print(f"\nPATCH FAILED — nothing was written.\n{e}\n")
        sys.exit(1)
