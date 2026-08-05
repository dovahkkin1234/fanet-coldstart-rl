"""
apply_anchor_fixes_v2.py — follow-up to apply_pre_m4_fixes_v1.py.

Fixes the four issues found reviewing the M-4 collision-flip diffs. Applies to
the LOCAL working tree (post-flip), not to origin/main @ bb1f0a1.

F1  simulator_v2.py — stale cross-reference.
    The flip comment says the new regression constant is recorded in
    `preflight_simulator_v2_check.REGRESSION_PDR`. That constant does not
    exist; the float-PDR anchor was deliberately abandoned for integer counts
    after the 2.9e-7 rounding failure. The comment would send a future reader
    looking for the design that was rejected.

F2  preflight_simulator_v2_check.py — the anchor is UNSET and ACTOR-DEPENDENT.

    F2a  ACTOR PINNING. The anchored run was
             run({**base, ..., 'actor': args.actor})
         with `--actor` defaulting to 'dijkstra'. Recording the anchor under
         one actor and re-running the gate under another reports drift that is
         not drift, and the failure message does not say why. The anchor runs
         are now PINNED to named constants and are independent of --actor.
         Determinism (r1 vs r2) still runs under --actor, because that is a
         property of the build, not of the anchor.

    F2b  SECOND, LINK-QUALITY-SENSITIVE ANCHOR — closes the §3.4 open item.
         The primary anchor runs 'dijkstra', which ignores link_quality
         entirely, so it CANNOT catch a link-model change. The M-4 flip is
         exactly such a change, and this anchor would have slept through it.
         A second anchor under 'spbp' — link-quality- AND queue-sensitive, and
         the oracle teacher M3.5 labels with — covers that blind spot. Two
         extra integers, no new machinery.
         This becomes load-bearing at M5, when a learned model takes the actor
         slot and link-quality-sensitive paths are the thing under test.

    F2c  UNSET STATE IS LOUDER. `anchored = True` when the constants are None
         is correct for bootstrapping — a gate that cannot pass on a fresh
         checkout is useless — but it means check 6 currently reports PASS
         while providing ZERO drift protection. It now prints a banner and
         marks the verdict line ARMED / **UNSET**, so an unarmed gate cannot be
         mistaken for a passing one at a glance.

F3  preflight_interference_check.py — RNG stream parity between models.
    The unsaturated branch skipped `rng.random(len(cs_nodes))`, so the two
    collision models consumed the shared generator at different rates and their
    subsequent shadowing draws diverged. Saturated still reproduced its
    pre-flip numbers (its branch consumed identically), so nothing published is
    wrong — but a saturated-vs-unsaturated comparison was UNPAIRED, with
    different shadowing realisations on each side, and a small difference could
    not be attributed to the collision model. The draw is now unconditional and
    only its USE is branch-dependent, which makes the comparison paired and
    leaves saturated byte-identical.

    Also documents that this gate is HOMOGENEOUS in activity: it passes
    [activity] * len(cs_nodes), while simulator_v2 passes each node's own
    activity because heterogeneity IS the congestion-collapse mechanism. G1
    therefore validates the collision model's SHAPE, not that mechanism, and
    should not be cited as evidence about heterogeneous congestion.

F5  preflight_interference_check.py — lq did not respond to the collision
    model AT ALL. compute_link_features_v2 returns link_quality =
    clip(sinr_db/30,0,1), a pure SINR quantity that p_coll never touches;
    only `per` carries the collision term. simulator_v2._measured_link does its
    own fold-in (lq = lq_clean * p_clear * (1 - p_coll)) as the M3 audit fix,
    but that fix never reached this gate. Measured: switching
    --collision_model left mean lq IDENTICAL to six decimals at every activity
    level (0.812726 / 0.617758 / 0.295316 at a = 0.02 / 0.06 / 0.20) while
    `per` moved. The flag was steering half the output, and G1's headline
    metric is link quality. Now folded in.

F4  link_model_v2.py — silent clamp becomes an assertion.
    `float(min(max(p_collision, 0.0), 1.0))` absorbs an out-of-range value,
    which can only arise from an upstream bug, into a plausible number. That is
    the failure signature this project keeps catching. NaN is caught too:
    min/max propagate it silently.

USAGE
    python apply_anchor_fixes_v2.py --src src [--dry-run]

Every replacement is anchored and asserted; edits are staged in memory and
committed only if EVERY anchor matched exactly once.
"""

import argparse
import os
import sys

VERSION = "v2"


class PatchError(RuntimeError):
    pass


def sub(text, old, new, label, path):
    n = text.count(old)
    if n != 1:
        raise PatchError(
            f"[{path}] anchor for '{label}' matched {n} times, expected 1.\n"
            f"  anchor starts: {old[:110]!r}")
    return text.replace(old, new, 1)


# ── F1 ───────────────────────────────────────────────────────────────────────
F1_OLD = """        # NOTE: G2's regression constant changes with this flip. The new value
        # is recorded in preflight_simulator_v2_check.REGRESSION_PDR, which now
        # asserts it explicitly rather than only checking run-to-run equality."""

F1_NEW = """        # NOTE ON G2's REGRESSION CONSTANT. Measured at that gate's default
        # config (30 drones, rate 1.0, seed 42, interference on):
        #     dijkstra : 87/280 saturated  ->  87/280 unsaturated   (NO CHANGE)
        #     spbp     : 111/280 saturated -> 112/280 unsaturated   (CHANGED)
        # The primary anchor runs dijkstra, which routes on hop count and never
        # reads link_quality, so it is EMPIRICALLY BLIND to this flip. An
        # earlier draft of this comment said the constant changes with the flip;
        # for that actor it does not. This is the concrete reason a second,
        # link-quality-sensitive anchor exists.
        # Both pairs are recorded in preflight_simulator_v2_check as INTEGER
        # pairs -- REGRESSION_DELIVERED / REGRESSION_GENERATED for dijkstra and
        # REGRESSION_LQ_* for spbp -- which that gate now asserts explicitly
        # rather than only checking run-to-run equality.
        # (There is no REGRESSION_PDR: the float-PDR anchor was abandoned after
        # a 2.9e-7 rounding failure made its error message self-contradictory.)"""


# ── F2 ───────────────────────────────────────────────────────────────────────
F2A_OLD = """REGRESSION_DELIVERED = None
REGRESSION_GENERATED = None"""

F2A_NEW = '''#
# THE ANCHOR RUNS ARE ACTOR-PINNED. They previously used `args.actor`, so
# recording the anchor under one actor and re-running the gate under another
# reported drift that was not drift -- and the message did not say why. The
# anchors now ignore --actor entirely. Determinism (r1 vs r2) still runs under
# --actor, because that is a property of the build, not of the anchor.
REGRESSION_ACTOR = 'dijkstra'
REGRESSION_DELIVERED = None
REGRESSION_GENERATED = None

# SECOND ANCHOR, LINK-QUALITY-SENSITIVE.
# 'dijkstra' routes on hop count and ignores link_quality completely, so the
# primary anchor above CANNOT detect a change to the link model. The M-4
# collision-model flip is precisely such a change, and this gate would have
# slept through it. 'spbp' scores on link_quality AND queue occupancy, and is
# the oracle teacher the M3.5 labels come from, so drift here is drift in the
# thing every downstream milestone depends on.
# Two integers. No new machinery. Closes the second-anchor item.
#
# THIS IS NOT HYPOTHETICAL. Measured at this gate's default config:
#     dijkstra : 87/280 saturated  ->  87/280 unsaturated   (blind)
#     spbp     : 111/280 saturated -> 112/280 unsaturated   (sees it)
# The M-4 collision-model flip -- a deliberate, documented physics change --
# moves the dijkstra anchor by EXACTLY ZERO. Of the two anchors, only the
# link-quality-sensitive one can detect the change that was just made.
REGRESSION_LQ_ACTOR = 'spbp'
REGRESSION_LQ_DELIVERED = None
REGRESSION_LQ_GENERATED = None


def _anchor_block(label, res, delivered, generated, actor, fname):
    """Compare one run against one recorded pair. Returns (ok, armed).

    Unset => ok=True, armed=False. A gate that cannot pass on a fresh checkout
    is useless, so bootstrapping must not FAIL -- but an unarmed gate must
    never LOOK like a passing one, hence the banner and the armed flag that
    reaches the verdict line.
    """
    if delivered is None or generated is None:
        print(f"\\n    ┌─ [{label} ANCHOR NOT SET] — check 6 is NOT protecting "
              f"against drift ─┐")
        print(f"    │ Copy these lines verbatim into {fname}:")
        print(f"    │     {label}_DELIVERED = {res['n_delivered']}")
        print(f"    │     {label}_GENERATED = {res['n_generated']}")
        print(f"    │ (actor={actor}; PDR {res['n_delivered']}/"
              f"{res['n_generated']} = {res['network_pdr']:.6f}, shown for")
        print(f"    │  reference only -- the anchor is the integer counts, so "
              f"there is")
        print(f"    │  no rounding to get wrong.)")
        print(f"    └{'─' * 68}┘")
        return True, False
    ok = (res['n_delivered'] == delivered and res['n_generated'] == generated)
    if not ok:
        print(f"\\n    *** REGRESSION [{label}, actor={actor}]: "
              f"delivered/generated = {res['n_delivered']}/{res['n_generated']}"
              f", anchor = {delivered}/{generated} ***")
        print("    The simulator's behaviour changed. If intentional (a")
        print("    documented physics change), update BOTH pairs and say so in")
        print("    the commit. If not, something drifted silently.")
        if label == 'REGRESSION_LQ':
            print("    This anchor reads link_quality; the dijkstra anchor does")
            print("    not. If only this one moved, the change is in the LINK")
            print("    MODEL, not in topology or queueing.")
    return ok, True'''

F2B_OLD = """    determinism = (r1['n_delivered'] == r2['n_delivered'] and
                   r1['n_dropped'] == r2['n_dropped'] and
                   abs(r1['network_pdr'] - r2['network_pdr']) < 1e-12)
    # Drift check against the recorded anchor, not just run-to-run equality.
    if REGRESSION_DELIVERED is None or REGRESSION_GENERATED is None:
        anchored = True
        print(f"\\n    [REGRESSION ANCHOR NOT SET]")
        print(f"    Copy these two lines verbatim into {os.path.basename(__file__)}:")
        print(f"        REGRESSION_DELIVERED = {r1['n_delivered']}")
        print(f"        REGRESSION_GENERATED = {r1['n_generated']}")
        print(f"    (PDR {r1['n_delivered']}/{r1['n_generated']} = "
              f"{r1['network_pdr']:.6f} -- shown for reference only; the anchor is")
        print(f"     the integer counts, so there is no rounding to get wrong.)")
    else:
        anchored = (r1['n_delivered'] == REGRESSION_DELIVERED and
                    r1['n_generated'] == REGRESSION_GENERATED)
        if not anchored:
            print(f"\\n    *** REGRESSION: delivered/generated = "
                  f"{r1['n_delivered']}/{r1['n_generated']}, anchor = "
                  f"{REGRESSION_DELIVERED}/{REGRESSION_GENERATED} ***")
            print("    The simulator's behaviour changed. If intentional (a")
            print("    documented physics change), update both constants and say so")
            print("    in the commit. If not, something drifted silently.")
    c6 = determinism and anchored"""

F2B_NEW = """    determinism = (r1['n_delivered'] == r2['n_delivered'] and
                   r1['n_dropped'] == r2['n_dropped'] and
                   abs(r1['network_pdr'] - r2['network_pdr']) < 1e-12)

    # ---- drift, against two ACTOR-PINNED anchors ----
    # Pinned so that --actor cannot make a matching build look like a
    # regression. r1 is reused only when its actor already equals the pinned
    # one, which is the default path and saves a redundant episode.
    _fname = os.path.basename(__file__)
    ra = (r1 if args.actor == REGRESSION_ACTOR else
          run({**base, 'packet_rate': mid, 'interference_on': True,
               'actor': REGRESSION_ACTOR}))
    ok_a, armed_a = _anchor_block('REGRESSION', ra, REGRESSION_DELIVERED,
                                  REGRESSION_GENERATED, REGRESSION_ACTOR, _fname)
    rb = run({**base, 'packet_rate': mid, 'interference_on': True,
              'actor': REGRESSION_LQ_ACTOR})
    ok_b, armed_b = _anchor_block('REGRESSION_LQ', rb, REGRESSION_LQ_DELIVERED,
                                  REGRESSION_LQ_GENERATED, REGRESSION_LQ_ACTOR,
                                  _fname)
    anchored = ok_a and ok_b
    fully_armed = armed_a and armed_b
    c6 = determinism and anchored"""

F2C_OLD = """        ("6. Bit-reproducible + no drift vs anchor",
         c6, f"{r1['n_delivered']}/{r1['n_generated']} (PDR {r1['network_pdr']:.6f}); "
             f"determinism={determinism}; anchor="
             + ('unset' if REGRESSION_DELIVERED is None
                else f"{REGRESSION_DELIVERED}/{REGRESSION_GENERATED}")),"""

F2C_NEW = """        ("6. Bit-reproducible + no drift vs anchors",
         c6, f"determinism={determinism}; "
             f"{REGRESSION_ACTOR}={ra['n_delivered']}/{ra['n_generated']} "
             + ('(**UNSET**)' if not armed_a else 'vs anchor OK' if ok_a else 'DRIFT')
             + f"; {REGRESSION_LQ_ACTOR}={rb['n_delivered']}/{rb['n_generated']} "
             + ('(**UNSET**)' if not armed_b else 'vs anchor OK' if ok_b else 'DRIFT')
             + ('' if fully_armed else '  <-- NOT PROTECTING AGAINST DRIFT')),"""


# ── F3 ───────────────────────────────────────────────────────────────────────
F3A_OLD = """    \"\"\"For each feasible link, compute link_quality/PER at a given offered-load
    `activity` (probability a node transmits this slot). Returns arrays
    (link_quality, per) over all links, one sample per link.\"\"\""""

F3A_NEW = """    \"\"\"For each feasible link, compute link_quality/PER at a given offered-load
    `activity` (probability a node transmits this slot). Returns arrays
    (link_quality, per) over all links, one sample per link.

    SCOPE -- this gate is HOMOGENEOUS in activity. Every carrier-sense peer is
    given the same `activity`, because G1 sweeps a single scalar offered-load
    level. simulator_v2 instead passes each node's OWN activity, and its
    comment is explicit that the heterogeneity is deliberate: congested and
    idle nodes coexisting is the congestion-collapse mechanism, and a mean
    blurs it away. So G1 validates the collision model's SHAPE -- monotone in
    load, no hard-zero plateau, correct v1 reduction at activity 0 -- and NOT
    that mechanism. Do not cite G1 as evidence about heterogeneous congestion;
    that evidence comes from G2's load sweep.
    \"\"\""""

F3B_OLD = """        if collision_model == 'unsaturated':
            # Every carrier-sense peer is a potential contender that transmits
            # with probability `activity`; no Bernoulli draw and no rounding to
            # an integer count, so no hard-zero plateau.
            p_coll_override = v2.bianchi_collision_prob_unsaturated(
                [activity] * len(cs_nodes))
            n_contenders = 1
        else:
            n_active_cs = int(np.sum(rng.random(len(cs_nodes)) < activity)) if cs_nodes else 0
            n_contenders = 1 + n_active_cs   # tx itself + active carrier-sense peers
            p_coll_override = None"""

F3B_NEW = """        # RNG PARITY. The draw is UNCONDITIONAL; only its USE is branch-
        # dependent. Previously the unsaturated branch skipped it, so the two
        # models consumed the shared generator at different rates and every
        # subsequent shadowing draw diverged. Saturated still reproduced its
        # pre-flip numbers (its branch consumed identically), so nothing
        # published was wrong -- but a saturated-vs-unsaturated comparison was
        # UNPAIRED, running on different shadowing realisations, and a small
        # difference could not be attributed to the collision model. Drawing
        # unconditionally leaves the saturated path byte-identical and makes
        # the comparison paired.
        cs_draw = rng.random(len(cs_nodes)) if cs_nodes else np.empty(0)
        if collision_model == 'unsaturated':
            # Every carrier-sense peer is a potential contender that transmits
            # with probability `activity`; no Bernoulli thresholding and no
            # rounding to an integer count, so no hard-zero plateau. The peer
            # list excludes tx, matching simulator_v2's
            # [a for k, a in enumerate(self.activity) if cs_mask[k] and k != i]:
            # p_coll is P(some OTHER station transmits in this slot).
            p_coll_override = v2.bianchi_collision_prob_unsaturated(
                [activity] * len(cs_nodes))
            n_contenders = 1
        else:
            n_active_cs = int(np.sum(cs_draw < activity)) if cs_nodes else 0
            n_contenders = 1 + n_active_cs   # tx itself + active carrier-sense peers
            p_coll_override = None"""


# ── F5 ───────────────────────────────────────────────────────────────────────
F5_OLD = """        sh_link = rng.normal(0.0, shadowing_sigma)
        _, _, lq, per = v2.compute_link_features_v2(
            d, interference_mw=interf_mw, n_contenders=n_contenders,
            shadowing_db=sh_link, p_collision=p_coll_override)
        lq_out.append(lq)
        per_out.append(per)"""

F5_NEW = """        sh_link = rng.normal(0.0, shadowing_sigma)
        _, _, lq_sinr, per = v2.compute_link_features_v2(
            d, interference_mw=interf_mw, n_contenders=n_contenders,
            shadowing_db=sh_link, p_collision=p_coll_override)

        # FOLD MAC CONTENTION INTO lq -- the M3 audit fix, which had been
        # applied in simulator_v2 but NOT here.
        #
        # compute_link_features_v2 returns link_quality = clip(sinr_db/30,0,1).
        # That is a pure SINR quantity: p_coll never touches it, only `per`.
        # simulator_v2._measured_link therefore does its own fold-in --
        #     lq = lq_clean * p_clear * (1 - p_coll)
        # -- with the comment that leaving MAC contention out made lq and per
        # 'describe different physics'. That fix landed in the simulator only.
        #
        # MEASURED CONSEQUENCE, before this change: switching --collision_model
        # moved `per` but left `lq` identical to six decimal places at every
        # activity level (0.812726 at a=0.02, 0.617758 at 0.06, 0.295316 at
        # 0.20 -- the SAME under both models). The new --collision_model flag
        # was steering only half the output, and G1's headline metric IS link
        # quality. The gate and the simulator were still reporting different
        # physics for the same scenario, which is exactly what the flag was
        # added to prevent.
        #
        # The hidden-terminal term is deliberately NOT reconciled: this gate
        # models it as continuous interference power entering SINR, the
        # simulator as a Bernoulli lethal-interferer indicator (p_clear). Those
        # are different modelling choices, not an inconsistency, and lq_sinr
        # already carries this gate's version through the SINR denominator.
        p_coll_eff = (p_coll_override if p_coll_override is not None
                      else v2.bianchi_collision_prob(n_contenders))
        lq = float(np.clip(lq_sinr * (1.0 - p_coll_eff), 0.0, 1.0))
        lq_out.append(lq)
        per_out.append(per)"""


# ── F4 ───────────────────────────────────────────────────────────────────────
F4_OLD = """    p_coll = (bianchi_collision_prob(n_contenders) if p_collision is None
              else float(min(max(p_collision, 0.0), 1.0)))"""

F4_NEW = """    if p_collision is None:
        p_coll = bianchi_collision_prob(n_contenders)
    else:
        # ASSERT, do not clamp. A p_collision outside [0,1] can only come from
        # an upstream bug, and silently folding it to a boundary turns that bug
        # into a plausible number -- the exact failure signature this project
        # keeps catching. NaN is caught here too: min/max propagate it without
        # complaint, so a NaN activity would have flowed straight into per.
        p_collision = float(p_collision)
        assert p_collision == p_collision, "p_collision is NaN"
        assert -1e-9 <= p_collision <= 1.0 + 1e-9, \\
            f"p_collision out of range: {p_collision!r}"
        p_coll = min(max(p_collision, 0.0), 1.0)   # float-noise clamp only"""


PATCHES = {
    'simulator_v2.py': [
        (F1_OLD, F1_NEW, 'F1 stale REGRESSION_PDR cross-reference'),
    ],
    'preflight_simulator_v2_check.py': [
        (F2A_OLD, F2A_NEW, 'F2a/F2b actor pinning + lq anchor'),
        (F2B_OLD, F2B_NEW, 'F2 check 6 body'),
        (F2C_OLD, F2C_NEW, 'F2c verdict row'),
    ],
    'preflight_interference_check.py': [
        (F3A_OLD, F3A_NEW, 'F3 homogeneity scope note'),
        (F3B_OLD, F3B_NEW, 'F3 RNG parity'),
        (F5_OLD, F5_NEW, 'F5 fold MAC contention into gate lq'),
    ],
    'link_model_v2.py': [
        (F4_OLD, F4_NEW, 'F4 assert instead of silent clamp'),
    ],
}

MARKER = 'REGRESSION_LQ_ACTOR'
PRECONDITION = ("REGRESSION_DELIVERED", 'preflight_simulator_v2_check.py')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='src')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    guard = os.path.join(args.src, PRECONDITION[1])
    if not os.path.isfile(guard):
        print(f"ERROR: {guard} not found. Run from the repo root or pass --src.")
        return 2
    g = open(guard, encoding='utf-8').read()
    if MARKER in g:
        print(f"ERROR: {PRECONDITION[1]} already contains {MARKER!r} — this "
              f"patch has already been applied.\n"
              f"  Restore from git before re-running.")
        return 2
    if PRECONDITION[0] not in g:
        print(f"ERROR: {PRECONDITION[1]} has no {PRECONDITION[0]!r}.\n"
              f"  This patch applies ON TOP OF the M-4 collision-flip work.\n"
              f"  Commit or restore that first.")
        return 2

    staged = {}
    print(f"\n{'=' * 78}\n  ANCHOR + PARITY FIXES {VERSION} — assertion-guarded\n{'=' * 78}")
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
    print("    1. python src/preflight_simulator_v2_check.py")
    print("       -> prints TWO anchor blocks (dijkstra and spbp). Paste all")
    print("          four integers in, then re-run and confirm the verdict")
    print("          shows 'vs anchor OK' twice and NO '**UNSET**'.")
    print("    2. python src/preflight_interference_check.py")
    print("=" * 78 + "\n")
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except PatchError as e:
        print(f"\nPATCH FAILED — nothing was written.\n{e}\n")
        sys.exit(1)
