"""
apply_mechanism_fixes_v3.py — the SP-BP mechanism experiment's baseline is wrong.

Applies ON TOP OF apply_pre_m4_fixes_v1.py and apply_anchor_fixes_v2.py.

────────────────────────────────────────────────────────────────────────────
G1  experiment_spbp_mechanism.py — spbp_ab_full does not reproduce panel SP-BP
────────────────────────────────────────────────────────────────────────────
The two experiments disagree about SP-BP while agreeing about DA-GPSR:

    experiment_spbp_mechanism :  spbp_ab_full = 0.4061   da_gpsr = 0.3770
    experiment_queue_weight   :  spbp         = 0.4128   da_gpsr = 0.3770

Identical grids, identical seeds -- DA-GPSR matches to four decimals. Only the
SP-BP reconstruction differs, by 0.0067.

That is EXACTLY the defect experiment_queue_weight.py already found and fixed,
documented in its own comment at the _spbp_qscale unreachability guard:

    "The first version used h.get(n, 999.0) and kept going when `current` was
     unreachable; panel SP-BP instead RETURNS None for an unreachable current
     node and SKIPS unreachable candidates. That divergence cost 0.0067 PDR
     (0.4056 vs 0.4123) and slipped past a control that only checked src->dst
     reachability, never candidate reachability."

queue_weight fixed it and added assert_controls() at import. spbp_mechanism
never got either. _spbp_generic still does h.get(current, 999.0) and
h.get(n, 999.0), which scores unreachable candidates as if they were 999 hops
away instead of excluding them.

PER-CELL FINGERPRINT (ablation `full` vs G3 panel `spbp`):

    very-dense  all three loads   identical      (0.242 / 0.752 / 0.410)
    dense       all three loads   identical      (0.273 / 0.798 / 0.479)
    medium      high     0.192 -> 0.178          -0.014
    medium      medium   0.326 -> 0.306          -0.020
    sparse-fast high     0.214 -> 0.190          -0.024
    sparse-fast medium   0.324 -> 0.299          -0.025

Divergence appears ONLY in the partitioned regimes. Part A of this same script
measures medium_slow at 54.7% reachable and sparse_fast at 21.4% -- so the
untested branch is being hit constantly, precisely where the numbers move.

WHAT THIS DOES AND DOES NOT INVALIDATE
  - "queue term entirely = +0.0646" is measured against a baseline ~0.0067 low.
    Expect roughly +0.071 after the fix. Direction and magnitude survive.
  - "queue DIFFERENTIAL -> candidate-only = +0.0000" is a WITHIN-ablation
    comparison: both sides use the same reconstruction, so the bias cancels
    exactly. Finding 2's core claim -- the differential is inoperative -- is
    UNAFFECTED. Same for the additive/multiplicative result.

  G1a  _spbp_generic mirrors panel spbp_next_hop's unreachability handling.
  G1b  assert_controls() ported from queue_weight, adapted to spbp_ab_full,
       including the partitioned-case counter that stops the control passing
       without exercising the branch that broke.

────────────────────────────────────────────────────────────────────────────
G2  experiment_queue_weight.py — the UP and DOWN sweeps are not comparable
────────────────────────────────────────────────────────────────────────────
    DOWN (SP-BP q_scale) : 0.02, 0.1, 0.3, 1, 3     dense below and near ref
    UP   (DA-GPSR w)     : 1, 5, 15, 50, 150        nothing below 1, step 5x

"Only one direction moves" is therefore partly "only one direction was sampled
near the reference." At w=5 DA-GPSR is already at -216% of the gap, so the
sweep cannot distinguish

    (a) queue weight CANNOT help DA-GPSR                    from
    (b) DA-GPSR's queue weight is ALREADY AT OR ABOVE its optimum

and those have opposite implications for the mechanism story. Two hints that
(b) is live: nothing tests w < 1 at all, and on the DOWN side q=3 scores
+0.0010 ABOVE q=1, i.e. SP-BP's own optimum is above its reference and the
sweep stops at 3.

  G2a  Grids mirrored. UP gains sub-reference resolution; DOWN is extended
       upward past its apparent optimum. w=15 is retained so the collapse
       evidence survives in the same table; 50 and 150 are dropped as they
       only re-demonstrate it (they remain in the committed JSON).
           DA_WEIGHTS: 5 -> 10 values
           SP_SCALES : 5 ->  9 values
       RUNTIME: configs go 12 -> 21, so runs go 4320 -> 7560 (~1.75x).
  G2b  The verdict now reports the argmax of each sweep and says explicitly
       whether the reference sits at, above, or below the optimum -- the
       question the old grid could not answer.

USAGE
    python apply_mechanism_fixes_v3.py --src src [--dry-run]

Every replacement is anchored and asserted; edits are staged in memory and
committed only if EVERY anchor matched exactly once.
"""

import argparse
import os
import sys

VERSION = "v3"


class PatchError(RuntimeError):
    pass


def sub(text, old, new, label, path):
    n = text.count(old)
    if n != 1:
        raise PatchError(
            f"[{path}] anchor for '{label}' matched {n} times, expected 1.\n"
            f"  anchor starts: {old[:110]!r}")
    return text.replace(old, new, 1)


# ── G1a: unreachability handling ─────────────────────────────────────────────
G1A_OLD = """    h_cur = float(h.get(current, 999.0))
    q_cur = float(G.nodes[current].get('queue_len', 0.0))

    best, best_score = None, -float('inf')
    for n in neighbors:
        q_n = float(G.nodes[n].get('queue_len', 0.0))
        lq = float(G.edges[current, n].get('link_quality', 0.0))
        hop = v_bias * (h_cur - float(h.get(n, 999.0)))"""

G1A_NEW = """    # MIRROR panel spbp_next_hop EXACTLY on unreachability. The previous
    # h.get(current, 999.0) / h.get(n, 999.0) form kept scoring when `current`
    # was disconnected from the destination, and treated an unreachable
    # CANDIDATE as merely 999 hops away instead of excluding it. Panel SP-BP
    # returns None in the first case and `continue`s in the second.
    #
    # This is the same defect experiment_queue_weight.py documents and fixed;
    # it cost 0.0067 PDR there (0.4056 vs 0.4123) and cost the same here
    # (spbp_ab_full 0.4061 vs panel spbp 0.4128). It is invisible in dense and
    # very-dense -- those cells matched panel SP-BP exactly -- and shows up
    # only in medium (54.7% reachable) and sparse-fast (21.4%), which is where
    # the ablation's numbers diverged by up to 0.025.
    if current not in h:
        return None                       # disconnected from dst this frame
    h_cur = float(h[current])
    q_cur = float(G.nodes[current].get('queue_len', 0.0))

    best, best_score = None, -float('inf')
    for n in neighbors:
        if n not in h:
            continue                      # unreachable -> exclude, not score
        q_n = float(G.nodes[n].get('queue_len', 0.0))
        lq = float(G.edges[current, n].get('link_quality', 0.0))
        hop = v_bias * (h_cur - float(h[n]))"""


# ── G1b: the control ─────────────────────────────────────────────────────────
G1B_OLD = """ABLATIONS = {
    'spbp_ab_full':      spbp_ab_full,"""

G1B_NEW = '''def assert_controls():
    """spbp_ab_full must reproduce panel spbp_next_hop EXACTLY.

    Ported from experiment_queue_weight.assert_controls, which exists because
    an earlier control passed while the implementation under it was wrong: it
    only ran when nx.has_path(src, dst) held, so it never exercised the case
    where some CANDIDATE is unreachable from the destination -- which is
    exactly where the two implementations diverged. Part A of this very script
    then measured that unreachability is the COMMON case here (medium_slow 55%
    reachable, sparse_fast 21%), so the untested branch was being hit
    constantly.

    Sparse connection probabilities are used ON PURPOSE to force partitions,
    the number of genuinely partitioned cases is counted, and a non-trivial
    count is REQUIRED -- so this control cannot pass by only ever walking the
    easy path again.

    Only spbp_ab_full is checked against the panel. The other three ablations
    are deliberate structural deviations and have no panel counterpart; their
    correctness rests on sharing _spbp_generic with the one variant that IS
    pinned to a reference implementation.
    """
    from routing_teachers_v2 import spbp_next_hop
    rng = np.random.default_rng(0)
    n_sp = n_partitioned = 0
    for trial in range(300):
        n = int(rng.integers(5, 14))
        p_edge = 0.12 + 0.23 * rng.random()      # sparse -> reliable partitions
        G = nx.Graph(); G.graph['comm_range'] = 250.0
        for i in range(n):
            G.add_node(i, x=float(rng.integers(0, 900)),
                       y=float(rng.integers(0, 900)), z=100.0, energy=90.0,
                       queue_occupancy=float(rng.random()),
                       queue_len=float(rng.integers(0, 6)))
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < p_edge:
                    G.add_edge(i, j, distance=float(rng.integers(50, 250)),
                               link_quality=float(rng.random()),
                               packet_error_rate=float(rng.random()) * 0.3)
        src, dst = 0, n - 1
        if not list(G.neighbors(src)) or dst not in G:
            continue
        try:
            reach = set(nx.single_source_shortest_path_length(G, dst))
        except nx.NodeNotFound:
            continue
        if any(nb not in reach for nb in G.neighbors(src)) or src not in reach:
            n_partitioned += 1
        if spbp_ab_full(G, src, dst) != spbp_next_hop(G, src, dst):
            raise AssertionError(
                f"spbp_ab_full diverged from panel spbp (trial {trial}) -- the "
                f"whole component ablation is then measured against the wrong "
                f"baseline, which is what produced 0.4061 vs 0.4128")
        n_sp += 1
    if n_sp < 50:
        raise AssertionError(f"control checked too few cases: {n_sp}")
    if n_partitioned < 20:
        raise AssertionError(
            f"control only exercised {n_partitioned} partitioned cases -- too "
            f"few to trust; it would pass without testing the branch that broke")
    assert_controls.n_partitioned = n_partitioned


assert_controls()


ABLATIONS = {
    'spbp_ab_full':      spbp_ab_full,'''

G1C_OLD = """    print("  (follows up the locality result; addresses reviewer finding M-3)")
    print("=" * 78)"""

G1C_NEW = """    print("  (follows up the locality result; addresses reviewer finding M-3)")
    print(f"  control passed at import: spbp_ab_full == panel spbp "
          f"({getattr(assert_controls, 'n_partitioned', 0)} PARTITIONED "
          f"graphs exercised)")
    print("=" * 78)"""


# ── G2a: mirrored grids ──────────────────────────────────────────────────────
G2A_OLD = """DA_WEIGHTS = [1.0, 5.0, 15.0, 50.0, 150.0]
SP_SCALES = [0.02, 0.1, 0.3, 1.0, 3.0]"""

G2A_NEW = """# MIRRORED GRIDS.
# The previous pair was not comparable: DOWN sampled 0.02-3x (dense below and
# around the reference) while UP sampled 1-150x (nothing below the reference,
# smallest step 5x). Since DA-GPSR is already at -216% of the gap by w=5, that
# grid could not distinguish "queue weight cannot help DA-GPSR" from "DA-GPSR's
# queue weight is already at or above its optimum" -- opposite conclusions.
# UP now has sub-reference resolution; DOWN now extends past its own apparent
# optimum (q=3 scored +0.0010 ABOVE q=1, so the old grid stopped too early).
# w=15 kept so the collapse stays visible in the same table; 50 and 150 dropped
# as pure re-demonstration -- they remain in the previously committed JSON.
# Configs 12 -> 21, so runs 4320 -> 7560.
DA_WEIGHTS = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 15.0]
SP_SCALES = [0.02, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 3.0, 10.0]"""


# ── G2b: verdict reports the argmax of each sweep ────────────────────────────
G2B_OLD = """    best_w = max(DA_WEIGHTS, key=lambda w: means[f'dagpsr_w{w:g}'])
    if best_da > sp_ref + 0.002:"""

G2B_NEW = """    best_w = max(DA_WEIGHTS, key=lambda w: means[f'dagpsr_w{w:g}'])
    best_q = max(SP_SCALES, key=lambda q: means[f'spbp_q{q:g}'])

    # WHERE DOES EACH REFERENCE SIT ON ITS OWN CURVE? The old grid could not
    # answer this for DA-GPSR at all: w=1 was its smallest value, so an
    # optimum below the reference was unobservable by construction.
    print()
    print(f"  sweep optima:  DA-GPSR best at w={best_w:g} ({best_da:.4f});  "
          f"SP-BP best at q={best_q:g} ({means[f'spbp_q{best_q:g}']:.4f})")
    if best_w < 1.0:
        print("  ** DA-GPSR's reference weight is ABOVE its optimum -- the panel")
        print("     default is over-weighting the queue term, so the old UP sweep")
        print("     was climbing away from the peak from its first step. Any")
        print("     'queue weight cannot help DA-GPSR' claim is REFUTED.")
    elif best_w > 1.0:
        print("  ** DA-GPSR improves with MORE queue weight -- scaling is part of")
        print("     the gap after all. Report the swept curve, not a single point.")
    else:
        print("  ** DA-GPSR's reference weight IS its optimum on this grid, now")
        print("     bracketed on both sides. The gap is not a weight choice.")
    if best_q != 1.0:
        print(f"  ** SP-BP's reference is NOT its optimum either (best q={best_q:g}).")
        print("     The panel comparison is partly a comparison of tuning; say so.")

    if best_da > sp_ref + 0.002:"""


PATCHES = {
    'experiment_spbp_mechanism.py': [
        (G1A_OLD, G1A_NEW, 'G1a unreachability mirrors panel SP-BP'),
        (G1B_OLD, G1B_NEW, 'G1b assert_controls ported'),
        (G1C_OLD, G1C_NEW, 'G1c control banner'),
    ],
    'experiment_queue_weight.py': [
        (G2A_OLD, G2A_NEW, 'G2a mirrored grids'),
        (G2B_OLD, G2B_NEW, 'G2b sweep-optimum reporting'),
    ],
}

MARKER = 'assert_controls'
GUARD_FILE = 'experiment_spbp_mechanism.py'
PRECONDITION = "add_argument('--out'"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='src')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    guard = os.path.join(args.src, GUARD_FILE)
    if not os.path.isfile(guard):
        print(f"ERROR: {guard} not found. Run from the repo root or pass --src.")
        return 2
    g = open(guard, encoding='utf-8').read()
    if MARKER in g:
        print(f"ERROR: {GUARD_FILE} already contains {MARKER!r} — this patch has "
              f"already been applied.\n  Restore from git before re-running.")
        return 2
    if PRECONDITION not in g:
        print(f"ERROR: {GUARD_FILE} has no --out argument.\n"
              f"  This patch applies ON TOP OF apply_pre_m4_fixes_v1.py.\n"
              f"  Apply that first.")
        return 2

    staged = {}
    print(f"\n{'=' * 78}\n  MECHANISM + GRID FIXES {VERSION} — assertion-guarded\n{'=' * 78}")
    for fname, edits in PATCHES.items():
        path = os.path.join(args.src, fname)
        if not os.path.isfile(path):
            raise PatchError(f'missing file: {path}')
        text = open(path, encoding='utf-8').read()
        for old, new, label in edits:
            text = sub(text, old, new, label, fname)
            print(f"    [ok] {fname:<34} {label}")
        staged[path] = text

    if args.dry_run:
        print(f"\n  DRY RUN — {len(staged)} file(s) would change, nothing written.")
        return 0

    for path, text in staged.items():
        open(path, 'w', encoding='utf-8').write(text)
    print(f"\n  WROTE {len(staged)} file(s).")
    print("\n  NEXT:")
    print("    1. python verify_mechanism_fixes_v3.py --src src")
    print("    2. python src/experiment_spbp_mechanism.py --out results/spbp_mechanism.json")
    print("       -> `full` must now match G3's spbp per cell; mean ~0.4128,")
    print("          and the queue-term cost should rise to roughly +0.071.")
    print("    3. python src/experiment_queue_weight.py --out results/queue_weight.json")
    print("       -> ~7560 runs, up from 4320. Read the new 'sweep optima' line.")
    print("=" * 78 + "\n")
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except PatchError as e:
        print(f"\nPATCH FAILED — nothing was written.\n{e}\n")
        sys.exit(1)
