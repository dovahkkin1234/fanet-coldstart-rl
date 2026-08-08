"""
apply_v6_fixes.py — a normaliser I failed to recalibrate in v1.

Applies ON TOP OF v1-v5. Requires ONE more regeneration (~220 s + gates).

════════════════════════════════════════════════════════════════════════════
K1  neigh_buffered_packets is CLIPPED on 38.8% of decisions
════════════════════════════════════════════════════════════════════════════
v1's A3 changed this feature's semantics from a GLOBAL in-flight count to a
LOCAL count of packets buffered within LOCAL_HORIZON hops. It did not
recalibrate the normaliser, which stayed at the value written for the global
quantity:

        min(neigh_buffered / 100.0, 1.0)

Measured on a regeneration instrumented to store the RAW value (28,492
decisions):

        mean = 102.12   std = 101.76   max = 393
        p50 = 63   p75 = 155   p90 = 267   p95 = 321   p99 = 368

The mean of the raw quantity is 102 against a normaliser of 100. So:

        /100  -> 38.77% of decisions sit at EXACTLY 1.0
                 p75 through p99 are ALL exactly 1.0

Over a third of decisions carry no information in that column, and the dead
region is precisely the HIGH-LOAD tail -- the regime the entire congestion
thesis is about. A model cannot distinguish a neighbourhood holding 100
packets from one holding 393.

This is a genuine information loss in the stored dataset, not a display
issue: the clip happens at extraction, so it CANNOT be undone in the
dataloader. It requires regeneration. That is my error, introduced in v1 and
missed by the v1 verification because that harness checked the feature's
SEMANTICS (is it the k-hop mean? does it ignore the global argument?) and
never its RANGE.

  K1  Normaliser becomes a persisted constant, BUFFERED_REF = 500.0.
      Sized from data, not guessed: zero clipping against an observed max of
      393, with 27% headroom. Schema 4 -> 5.

      WHY LINEAR /500 RATHER THAN THE ALTERNATIVES.
        - /400 also clips 0.00%, but sits exactly at the observed max with no
          headroom; one denser scenario and it saturates again.
        - log1p(v)/log1p(1000) spreads better (mean 0.545 std 0.253 vs
          0.204/0.204) and never clips, but it introduces a second magic
          constant and makes the column nonlinear in packet count while
          queue_occupancy and neigh_mean_occupancy stay linear. Mixed
          transforms in one block are harder to interpret in the M4 feature
          analysis, which is the milestone this exists to serve.
        - Normalising by neighbourhood size would turn this into mean
          occupancy per node -- which IS neigh_mean_occupancy. The two
          features are meant to be volume vs intensity; collapsing them would
          make the pair redundant and G3.5 check 8 would (correctly) fail.

      Redundancy against its sibling was checked at both candidate
      normalisers before choosing: |rho| = 0.8045 either way, well under the
      0.995 threshold. The choice does not create a new redundancy.

════════════════════════════════════════════════════════════════════════════
K2  A saturation diagnostic, so the CLASS is closed and not the instance
════════════════════════════════════════════════════════════════════════════
Check 6 catches DEAD columns (zero variance). Check 8 catches DUPLICATED
columns. Neither catches a column that is alive, unique, and CLIPPED across a
third of its range -- which is why this survived a full G3.5 pass, a clean
independent audit, and a purpose-built verification harness.

  K2  G3.5 now reports, per column, the fraction of mass sitting at exactly
      the observed min and exactly the observed max.

      DIAGNOSTIC, NOT A GATE FAILURE, and deliberately so. Several columns
      are legitimately concentrated at a boundary: is_destination and
      cand_reachable are binary; ttl_left is 1.0 whenever hops == 0, which is
      the single most common state. Automatically distinguishing "genuine
      mode at the boundary" from "clipped by a mis-sized normaliser" is not
      reliable, and a gate that fires on the former would be trained away
      rather than heeded. Binary columns are skipped; everything else is
      reported with its distinct-value count so a human can tell the two
      apart in one glance.

USAGE
    python apply_v6_fixes.py --src src [--dry-run]
"""

import argparse
import os
import sys

VERSION = "v6"


class PatchError(RuntimeError):
    pass


def sub(text, old, new, label, path):
    n = text.count(old)
    if n != 1:
        raise PatchError(
            f"[{path}] anchor for '{label}' matched {n} times, expected 1.\n"
            f"  anchor starts: {old[:110]!r}")
    return text.replace(old, new, 1)


# ── K1 ───────────────────────────────────────────────────────────────────────
K1A_OLD = """# cost to claiming it.
LOCAL_HORIZON = 2"""

K1A_NEW = """# cost to claiming it.
LOCAL_HORIZON = 2

# Normaliser for neigh_buffered_packets: the count of packets buffered within
# LOCAL_HORIZON hops. SIZED FROM DATA, not guessed.
#
# v1 changed this feature from a GLOBAL in-flight count to a LOCAL one and
# left the old /100.0 in place. Measured on a regeneration instrumented to
# store the raw value (28,492 decisions):
#     mean=102.12  std=101.76  max=393
#     p50=63  p75=155  p90=267  p95=321  p99=368
# The raw MEAN was 102 against a normaliser of 100, so 38.77% of decisions
# clipped to exactly 1.0 and p75..p99 were all exactly 1.0 -- a dead column
# across the entire high-load tail, which is the regime the congestion thesis
# is about. The clip happens at extraction, so it could not be repaired in the
# dataloader; it cost a regeneration.
#
# 500 gives zero clipping against the observed max of 393 with 27% headroom.
# IF THE SCENARIO GRID GETS DENSER, RE-MEASURE. G3.5's saturation diagnostic
# reports the mass at each column's max precisely so this is noticed rather
# than rediscovered.
BUFFERED_REF = 500.0"""

K1B_OLD = """FEATURE_SCHEMA_VERSION = 4"""
K1B_NEW = """FEATURE_SCHEMA_VERSION = 5"""

K1C_OLD = """        'hop_cap': HOP_CAP,"""
K1C_NEW = """        'hop_cap': HOP_CAP,
        'buffered_ref': BUFFERED_REF,"""

K1D_OLD = """        min(neigh_buffered / 100.0, 1.0),"""
K1D_NEW = """        min(neigh_buffered / nc['buffered_ref'], 1.0),"""


# ── K2 ───────────────────────────────────────────────────────────────────────
K2A_OLD = """def redundancy_report(block_name, arr, names):"""

K2A_NEW = '''SATURATION_REPORT_MIN = 0.10     # only mention columns above this


def saturation_report(block_name, arr, names):
    """Fraction of each column sitting at EXACTLY its observed min / max.

    Check 6 catches dead columns (zero variance); check 8 catches duplicated
    ones. Neither catches a column that is alive, unique, and CLIPPED -- which
    is how neigh_buffered_packets shipped with 38.77% of its mass at exactly
    1.0, through a full G3.5 pass, a clean independent audit, and a
    purpose-built verifier.

    DIAGNOSTIC ONLY. Binary columns (is_destination, cand_reachable) are
    legitimately massed at a boundary, and ttl_left is 1.0 whenever hops == 0,
    which is the commonest state. Separating "genuine mode" from "mis-sized
    normaliser" automatically is not reliable, and a gate that fired on the
    former would get tuned away instead of heeded. The distinct-value count is
    printed so a human can tell them apart at a glance.
    """
    if arr is None or arr.ndim != 2 or arr.shape[0] < 2:
        return []
    step = max(arr.shape[0] // REDUNDANCY_SAMPLE, 1)
    x = np.asarray(arr[::step], dtype=np.float64)
    if len(names) != x.shape[1]:
        return []
    lines = []
    for j in range(x.shape[1]):
        col = x[:, j]
        nuniq = len(np.unique(col))
        if nuniq <= 2:
            continue                      # binary: boundary mass is the point
        at_max = float((col == col.max()).mean())
        at_min = float((col == col.min()).mean())
        worst = max(at_max, at_min)
        if worst < SATURATION_REPORT_MIN:
            continue
        where = 'max' if at_max >= at_min else 'min'
        lines.append(
            f'    {block_name}.{names[j]:<24} {100*worst:5.1f}% at {where} '
            f'({col.max():.4g} / {col.min():.4g}), {nuniq} distinct values')
    return lines


def redundancy_report(block_name, arr, names):'''

K2B_OLD = """    print(f"    feature redundancy: thresholds |r|>{PEARSON_MAX} or "
          f"|rho|>{SPEARMAN_MAX}")"""

K2B_NEW = """    sat_lines = []
    for _bn, _arr, _names in (
            ('node', frm['node_feat_flat'], F.NODE_FEATURES),
            ('edge', frm['edge_feat_flat'], F.EDGE_FEATURES),
            ('query', qf, F.QUERY_FEATURES),
            ('candidate', dec['cand_feat_flat'], F.CANDIDATE_FEATURES)):
        sat_lines += saturation_report(_bn, _arr, _names)
    print(f"    saturation (non-binary columns >= "
          f"{int(100*SATURATION_REPORT_MIN)}% at a boundary):")
    if sat_lines:
        for _l in sat_lines:
            print(_l)
        print("      A high share at MAX on a many-valued column usually means")
        print("      a mis-sized normaliser, and the lost range cannot be")
        print("      recovered in the dataloader -- the clip happens at")
        print("      extraction. A share at MIN is often a genuine mode.")
    else:
        print("      none")
    print(f"    feature redundancy: thresholds |r|>{PEARSON_MAX} or "
          f"|rho|>{SPEARMAN_MAX}")"""


PATCHES = {
    'features_v2.py': [
        (K1A_OLD, K1A_NEW, 'K1 BUFFERED_REF constant'),
        (K1B_OLD, K1B_NEW, 'K1 schema version 5'),
        (K1C_OLD, K1C_NEW, 'K1 norm_constants'),
        (K1D_OLD, K1D_NEW, 'K1 extract_decision'),
    ],
    'preflight_dataset_v2_check.py': [
        (K2A_OLD, K2A_NEW, 'K2 saturation_report'),
        (K2B_OLD, K2B_NEW, 'K2 diagnostic output'),
    ],
}

MARKER = 'BUFFERED_REF'
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
        print(f"ERROR: {GUARD} already patched by v6. Restore from git first.")
        return 2
    if 'FEATURE_SCHEMA_VERSION = 4' not in txt:
        print(f"ERROR: {GUARD} is not at schema 4.\n"
              f"  v6 applies on top of v1-v5. Apply those first.")
        return 2

    staged = {}
    print(f"\n{'=' * 78}\n  V6 FIXES — assertion-guarded\n{'=' * 78}")
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
    print("\n  REGENERATION IS REQUIRED. The clip happened at extraction, so the")
    print("  lost range is not recoverable from data\\phaseB. Schema 4 -> 5, so")
    print("  both checkers will abort on the existing dataset until you do.")
    print("=" * 78 + "\n")
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except PatchError as e:
        print(f"\nPATCH FAILED — nothing was written.\n{e}\n")
        sys.exit(1)
