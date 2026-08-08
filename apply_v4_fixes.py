"""
apply_v4_fixes.py — the last pre-regeneration pass.

Applies ON TOP OF v1, v2 and v3. After this: regenerate once, re-gate once,
build model_gnn_attn.py.

════════════════════════════════════════════════════════════════════════════
FIRST, A CORRECTION TO SOMETHING BEING SAID ABOUT THIS BUG
════════════════════════════════════════════════════════════════════════════
It has been claimed that leaving spbp_khop_next_hop unfixed would corrupt the
533k dataset's labels and force a second regeneration. THAT IS NOT TRUE, and
it was checked rather than assumed:

    generate_dataset_v2.py imports:  simulator_v2, teacher_panel,
                                     routing_teachers_v2, features_v2
    It does NOT import routing_teachers_v3_local.

Labels come from routing_teachers_v2.spbp_next_hop via spbp_pick_restricted --
the panel implementation, which is correct. The k-hop variants are only
reachable by explicitly selecting spbp_k1..k4/kinf as the actor, which dataset
generation never does. The dataset is NOT at risk from this defect, and no
regeneration is being forced by it.

What IS at risk: Part B of experiment_spbp_mechanism, and the locality result
in experiment_locality_cost -- i.e. two PUBLISHED CLAIMS, not training data.
That is a good enough reason to fix it, and the right reason to state.

════════════════════════════════════════════════════════════════════════════
H1/H2  routing_teachers_v3_local.py — spbp_khop(k=inf) != panel SP-BP
════════════════════════════════════════════════════════════════════════════
Measured on sparse random graphs: 279/345 agreement, 66 disagreements, and
exactly 66 partitioned cases. The divergence is ENTIRELY on partitioned
graphs. Third occurrence of the same defect class.

WHY THE EXISTING CONTROL PASSED. assert_kinf_matches_global has:

    if not list(G.neighbors(src)) or not nx.has_path(G, src, dst):
        continue

plus p_edge = 0.4, which is dense enough that partitions are rare. This is the
IDENTICAL weakness experiment_queue_weight.py documents in its own control:
it checks src->dst reachability and never CANDIDATE reachability -- the branch
where the implementations actually differ. Three files, three controls, the
same blind spot in all three.

THE FIX, AND WHY IT IS NOT SYMMETRIC ACROSS k.
The k-hop variants must differ from panel SP-BP in HORIZON ONLY. So:

  k = None / inf  -> exact panel semantics. Return None if `current` is
                     outside the destination's component; SKIP candidates that
                     are outside it. This is what "full global knowledge"
                     means, and it includes knowing what is unreachable.
  finite k        -> UNCHANGED. The geographic proxy stays. A router with a
                     k-hop horizon genuinely CANNOT distinguish "unreachable"
                     from "far away", and modelling that is the entire point
                     of the experiment. Giving finite-k variants global
                     reachability knowledge would be the real corruption.

PRE-REGISTERED PREDICTION, recorded before running (§3.6 discipline):
kinf currently scores unreachable candidates via the proxy instead of
excluding them, exactly as the broken spbp_ab_full did -- and that defect was
worth +0.0067 PDR there. So kinf should get BETTER after this fix, and the
measured locality cost (kinf - k1) should GROW from its recorded +0.0005.
If it does not move at all, this model of the defect is wrong and the
locality conclusion needs rethinking rather than restating.

  H1  spbp_khop_next_hop: panel semantics at k = None/inf only.
  H2  assert_kinf_matches_global: sparse p_edge, no has_path early-continue,
      partitioned-case counter with a floor -- same shape as the v3 control.

════════════════════════════════════════════════════════════════════════════
H3/H4  features_v2.py + preflight — cand_reachable, and the exemption it needs
════════════════════════════════════════════════════════════════════════════
H3  CANDIDATE_FEATURES gains 'cand_reachable'. 3 -> 4. Schema 2 -> 3.

    BE CLEAR ABOUT WHY. This adds NO information the model lacked:
      - SP-BP excludes unreachable candidates, so the label never points at
        one; the model can learn "never pick these" from labels alone.
      - cand_hop_distance already saturates at 1.0 for unreachable nodes, and
        with mean hops 1.73-3.15 against HOP_CAP=10 a legitimate 1.0 
        essentially never occurs -- so it is ALREADY a de facto flag.
    The argument for the column is ABLATABILITY, not information: with the
    flag folded inside a distance column you cannot run the model with and
    without reachability. With it separate you can, and "does the model need
    global connectivity knowledge?" becomes a measurable result.

    It should also be said plainly that reachability-from-destination is a
    GLOBAL property. A drone cannot compute it from a 2-hop neighbourhood.
    So this column moves AWAY from decentralisation, not toward it -- which is
    precisely why it needs to be separately maskable now that LOCAL_HORIZON=2
    is the declared scoping choice.

H4  G3.5 check 8 will FLAG the new column, and it is RIGHT to.
    cand_reachable == 0 iff cand_hop_distance == 1.0 in these scenarios, so
    Spearman |rho| will be at or near 1.0. That is the redundancy check
    correctly reporting what H3's own justification admits: no new
    information. Loosening the threshold to hide it would defeat the check
    that was just added to catch snr and hops_so_far.
    Instead: an EXPLICIT, NAMED exemption list. The pair is reported as INFO
    with its justification, every other pair still FAILS the gate, and adding
    a future exemption requires writing down a reason.

════════════════════════════════════════════════════════════════════════════
H5  experiment_queue_weight.py — my verdict threshold fires on noise
════════════════════════════════════════════════════════════════════════════
The G2b block I added in v3 triggers on `best_q != 1.0` with no magnitude
test, so it printed "SP-BP's reference is NOT its optimum either" on a
+0.0010 difference against a gap of +0.0359 -- and against a curve that is
flat at 0.4138 / 0.4138 / 0.4139 for q = 1.5 / 3 / 10. That is SATURATION,
not an optimum. Left as-is it would put "the panel comparison is partly a
comparison of tuning" into the paper on a 2.8%-of-gap difference. My bug;
now requires the gain to exceed 10% of the gap.

════════════════════════════════════════════════════════════════════════════
H6  experiment_spbp_mechanism.py — the variant that closes Finding 4
════════════════════════════════════════════════════════════════════════════
The corrected ablation says something the old numbers could not:

    spbp_ab_candqueue  0.4128   <- candidate-only queue, DA-GPSR's structure
    da_gpsr            0.3770
    gap                0.0358   <- essentially the whole 0.0359 spbp gap

Both use candidate-only queue avoidance. Both were just shown to be at their
own optimal queue weight. So the residual is NOT in the queue term at all --
the only structural difference left is the PROGRESS METRIC:

    SP-BP   : v_bias * (h_cur - h_n)                 integer BFS hops
    DA-GPSR : normalised geographic progress

That is the original Finding 4 hypothesis. experiment_queue_weight refuted the
SCALING version of it; the METRIC version was never tested and is now the only
candidate left standing. Part A supports it: geo and hop distance correlate
0.89-0.93, leaving 7-11% disagreement -- ample room for 0.036 PDR.

  H6  New variant spbp_ab_geohop: identical to full SP-BP except the hop term
      uses geographic progress converted to hop-equivalent units,
          (geo(cur,dst) - geo(n,dst)) / comm_range
      so the SCALE matches and only the METRIC changes. Without that
      conversion the variant would confound metric with scale, which is the
      mistake the queue-weight sweep already had to correct for.

  PRE-REGISTERED: if spbp_ab_geohop lands near 0.377, the progress metric
  explains the gap and Finding 4 is closed. If it stays near 0.4128, the
  metric is NOT the mechanism and the residual is somewhere not yet proposed
  -- report that as an open question rather than inventing a third hypothesis.

USAGE
    python apply_v4_fixes.py --src src [--dry-run]
"""

import argparse
import os
import sys

VERSION = "v4"


class PatchError(RuntimeError):
    pass


def sub(text, old, new, label, path):
    n = text.count(old)
    if n != 1:
        raise PatchError(
            f"[{path}] anchor for '{label}' matched {n} times, expected 1.\n"
            f"  anchor starts: {old[:110]!r}")
    return text.replace(old, new, 1)


# ── H1 ───────────────────────────────────────────────────────────────────────
H1_OLD = """    comm_range = _comm_range_of(G)
    dpos = _pos(G, destination)
    kk = float(k) if (k is not None and k != float('inf')) else 0.0

    def h_of(n):
        if n in h_known:
            return float(h_known[n])
        # beyond the horizon: locally-computable geographic proxy
        geo = float(np.linalg.norm(dpos - _pos(G, n)))
        return kk + geo / max(comm_range, 1e-6)

    q_cur = float(G.nodes[current].get('queue_len', 0.0))
    h_cur = h_of(current)
    best, best_score = None, -float('inf')
    for n in neighbors:
        q_n = float(G.nodes[n].get('queue_len', 0.0))
        lq = float(G.edges[current, n].get('link_quality', 0.0))
        score = lq * ((q_cur - q_n) + v_bias * (h_cur - h_of(n)))
        if score > best_score:
            best_score, best = score, n
    return best if best is not None else _progress_fallback(G, current, destination)"""

H1_NEW = """    comm_range = _comm_range_of(G)
    dpos = _pos(G, destination)
    is_global = (k is None or k == float('inf'))
    kk = 0.0 if is_global else float(k)

    # UNREACHABILITY MUST BE HANDLED EXACTLY AS PANEL SP-BP DOES AT k=inf.
    # Previously h_of() fell through to the geographic proxy for ANY node
    # missing from h_known -- including nodes that are genuinely outside the
    # destination's connected component. At finite k that is correct and
    # deliberate. At k=inf it is a BUG: h_known is then the destination's whole
    # component, so a missing node is unreachable, and panel spbp_next_hop
    # returns None for an unreachable `current` and SKIPS unreachable
    # candidates rather than scoring them.
    #
    # Measured before this fix: 279/345 agreement with panel SP-BP, with
    # exactly 66 disagreements against 66 partitioned cases -- the divergence
    # is entirely on partitioned graphs. Same defect that cost spbp_ab_full
    # 0.0067 PDR.
    #
    # THE ASYMMETRY IS INTENTIONAL. Finite-k variants keep the proxy: a router
    # with a k-hop horizon genuinely cannot tell "unreachable" from "far away",
    # and modelling that is the whole point of the locality experiment.
    # Handing finite-k variants global reachability knowledge would corrupt
    # the very quantity being measured.
    if is_global:
        if current not in h_known:
            return None
        skip = lambda n: n not in h_known          # noqa: E731
    else:
        skip = lambda n: False                     # noqa: E731

    def h_of(n):
        if n in h_known:
            return float(h_known[n])
        # beyond the horizon: locally-computable geographic proxy
        geo = float(np.linalg.norm(dpos - _pos(G, n)))
        return kk + geo / max(comm_range, 1e-6)

    q_cur = float(G.nodes[current].get('queue_len', 0.0))
    h_cur = h_of(current)
    best, best_score = None, -float('inf')
    for n in neighbors:
        if skip(n):
            continue
        q_n = float(G.nodes[n].get('queue_len', 0.0))
        lq = float(G.edges[current, n].get('link_quality', 0.0))
        score = lq * ((q_cur - q_n) + v_bias * (h_cur - h_of(n)))
        if score > best_score:
            best_score, best = score, n
    return best if best is not None else _progress_fallback(G, current, destination)"""


# ── H2 ───────────────────────────────────────────────────────────────────────
H2_OLD = """    from routing_teachers_v2 import spbp_next_hop
    rng = np.random.default_rng(seed)
    checked = 0
    for _ in range(trials):
        n = int(rng.integers(5, 14))
        G = nx.Graph()
        G.graph['comm_range'] = 250.0
        for i in range(n):
            G.add_node(i, x=float(rng.integers(0, 900)), y=float(rng.integers(0, 900)),
                       z=100.0, energy=90.0,
                       queue_occupancy=float(rng.random()),
                       queue_len=float(rng.integers(0, 6)))
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < 0.4:
                    G.add_edge(i, j, distance=float(rng.integers(50, 250)),
                               link_quality=float(rng.random()),
                               packet_error_rate=float(rng.random()) * 0.3)
        src, dst = 0, n - 1
        if not list(G.neighbors(src)) or not nx.has_path(G, src, dst):
            continue
        ref = spbp_next_hop(G, src, dst)
        got = spbp_kinf_next_hop(G, src, dst)
        checked += 1
        if ref != got:
            raise AssertionError(
                f"spbp_khop(k=inf) diverged from spbp_next_hop: {got} vs {ref}")
    if checked == 0:
        raise AssertionError("assert_kinf_matches_global checked 0 cases")"""

H2_NEW = """    STRENGTHENED. The previous version passed for 345 runs while the function
    it guards disagreed with panel SP-BP on 66 of them. Two reasons, both
    fixed here:

      1. `if not nx.has_path(G, src, dst): continue` -- it only ever checked
         src->dst reachability, never CANDIDATE reachability, which is the
         branch where the two implementations actually differ. This is the
         identical blind spot documented in experiment_queue_weight's control
         and fixed again in experiment_spbp_mechanism's. Three files, three
         controls, the same hole in all three.
      2. p_edge = 0.4 is dense enough that partitions are rare, so even
         without (1) the branch was seldom reached.

    Now: sparse p_edge in [0.12, 0.35] to force partitions, no reachability
    early-continue, and a REQUIRED floor on genuinely partitioned cases so
    this control cannot pass again by only walking the easy path.
    \"\"\"
    from routing_teachers_v2 import spbp_next_hop
    rng = np.random.default_rng(seed)
    checked = n_partitioned = 0
    for _ in range(max(trials, 300)):
        n = int(rng.integers(5, 14))
        p_edge = 0.12 + 0.23 * rng.random()
        G = nx.Graph()
        G.graph['comm_range'] = 250.0
        for i in range(n):
            G.add_node(i, x=float(rng.integers(0, 900)), y=float(rng.integers(0, 900)),
                       z=100.0, energy=90.0,
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
        ref = spbp_next_hop(G, src, dst)
        got = spbp_kinf_next_hop(G, src, dst)
        checked += 1
        if ref != got:
            raise AssertionError(
                f"spbp_khop(k=inf) diverged from spbp_next_hop: {got} vs {ref} "
                f"-- the locality experiment would then be measuring this "
                f"reimplementation, not the information horizon")
    if checked < 50:
        raise AssertionError(
            f"assert_kinf_matches_global checked only {checked} cases")
    if n_partitioned < 20:
        raise AssertionError(
            f"control exercised only {n_partitioned} partitioned cases -- too "
            f"few to trust; it would pass without testing the branch that broke")
    assert_kinf_matches_global.n_partitioned = n_partitioned"""

H2_DOC_OLD = """    Without this, a difference measured across k could be an artifact of this
    reimplementation rather than of the information horizon -- which would
    invalidate the entire locality experiment. Runs at import.
    \"\"\""""

H2_DOC_NEW = """    Without this, a difference measured across k could be an artifact of this
    reimplementation rather than of the information horizon -- which would
    invalidate the entire locality experiment. Runs at import.
"""


# ── H3 ───────────────────────────────────────────────────────────────────────
H3A_OLD = """CANDIDATE_FEATURES = [
    'progress',            # normalised geographic progress toward destination
    'cand_hop_distance',   # candidate's BFS hop distance to destination
    'is_destination',      # candidate IS the destination
]"""

H3A_NEW = """CANDIDATE_FEATURES = [
    'progress',            # normalised geographic progress toward destination
    'cand_hop_distance',   # candidate's BFS hop distance to destination
    'is_destination',      # candidate IS the destination
    'cand_reachable',      # candidate lies in the destination's component
]
# ADDED 'cand_reachable'. Be honest about what it does and does not do.
#
# It adds NO INFORMATION the model lacked. SP-BP excludes unreachable
# candidates, so the label never points at one and the model could learn
# "never pick these" from labels alone; and cand_hop_distance already
# saturates at 1.0 for unreachable nodes, which with mean hops 1.73-3.15
# against HOP_CAP=10 makes it a de facto unreachability flag already.
# G3.5 check 8 will therefore flag this pair as redundant, and it will be
# RIGHT to -- see ALLOWED_REDUNDANT_PAIRS in preflight_dataset_v2_check.
#
# The argument for the column is ABLATABILITY. Folded inside a distance
# column, reachability cannot be masked independently; as its own column it
# can, and "does the model need global connectivity knowledge?" becomes a
# measurable result instead of an assumption.
#
# Note also that reachability-from-destination is a GLOBAL property -- a drone
# cannot compute it from a 2-hop neighbourhood. This column moves AWAY from
# decentralisation, which is exactly why it must be separately maskable now
# that LOCAL_HORIZON = 2 is the declared scoping choice."""

H3B_OLD = """FEATURE_SCHEMA_VERSION = 2"""
H3B_NEW = """FEATURE_SCHEMA_VERSION = 3"""

# ORDER MATTERS AND WAS GOT WRONG ONCE ALREADY. The first draft of this patch
# inserted cand_reachable directly after the cand_hop_distance line, i.e.
# BEFORE is_destination -- so the tuple became
# (progress, cand_hop, reachable, is_dest) while CANDIDATE_FEATURES declared
# (progress, cand_hop, is_dest, reachable). Every name then resolved to the
# wrong column, silently, with plausible values. The verification harness
# caught it (3d/3e), which is the whole reason it tests VALUES and not just
# widths. Anchor on the closing of the tuple so the append is unambiguous.
H3C_OLD = """            1.0 if u == dst else 0.0,
        )
    return qf, cf"""
H3C_NEW = """            1.0 if u == dst else 0.0,
            1.0 if u in h_map else 0.0,       # cand_reachable (see note above)
        )
    return qf, cf"""


# ── H4 ───────────────────────────────────────────────────────────────────────
H4A_OLD = """PEARSON_MAX = 0.98      # linear duplication
SPEARMAN_MAX = 0.995    # monotone (possibly nonlinear) duplication
REDUNDANCY_SAMPLE = 200000"""

H4A_NEW = '''PEARSON_MAX = 0.98      # linear duplication
SPEARMAN_MAX = 0.995    # monotone (possibly nonlinear) duplication
REDUNDANCY_SAMPLE = 200000

# NAMED EXEMPTIONS. A pair listed here is reported as INFO instead of failing
# the gate. Every entry needs a written reason, and the reason has to be that
# the duplication is DELIBERATE -- not that it is inconvenient. Loosening
# PEARSON_MAX/SPEARMAN_MAX instead would defeat the check that was added to
# catch snr <-> distance and ttl_left <-> hops_so_far in the first place.
ALLOWED_REDUNDANT_PAIRS = {
    ('candidate', 'cand_hop_distance', 'cand_reachable'):
        "DELIBERATE. cand_reachable adds no information -- cand_hop_distance "
        "already saturates at 1.0 for unreachable candidates, and with mean "
        "hops 1.73-3.15 against HOP_CAP=10 a legitimate 1.0 essentially never "
        "occurs. The column exists so reachability can be MASKED "
        "INDEPENDENTLY in the M4 deployability ablation, which is impossible "
        "while the flag is folded inside a distance column. Reachability-from-"
        "destination is a global property, so this is the column that has to "
        "come out to support a decentralisation claim.",
}


def _exempt(block, a, b):
    """Exemptions are unordered: (block, a, b) matches (block, b, a)."""
    return (ALLOWED_REDUNDANT_PAIRS.get((block, a, b))
            or ALLOWED_REDUNDANT_PAIRS.get((block, b, a)))'''

H4B_OLD = """            if p > PEARSON_MAX or s > SPEARMAN_MAX:
                offenders.append((block_name, names[i], names[j], p, s))
                kind = ('linear' if p > PEARSON_MAX else
                        'monotone (nonlinear -- Pearson alone would MISS this)')
                lines.append(f'    ** {block_name}: {names[i]} <-> {names[j]}  '
                             f'|r|={p:.4f} |rho|={s:.4f}  [{kind}]')"""

H4B_NEW = """            if p > PEARSON_MAX or s > SPEARMAN_MAX:
                why = _exempt(block_name, names[i], names[j])
                if why:
                    lines.append(f'    [exempt] {block_name}: {names[i]} <-> '
                                 f'{names[j]}  |r|={p:.4f} |rho|={s:.4f}')
                    lines.append(f'             {why}')
                    continue
                offenders.append((block_name, names[i], names[j], p, s))
                kind = ('linear' if p > PEARSON_MAX else
                        'monotone (nonlinear -- Pearson alone would MISS this)')
                lines.append(f'    ** {block_name}: {names[i]} <-> {names[j]}  '
                             f'|r|={p:.4f} |rho|={s:.4f}  [{kind}]')"""


# ── H5 ───────────────────────────────────────────────────────────────────────
H5_OLD = """    if best_q != 1.0:
        print(f"  ** SP-BP's reference is NOT its optimum either (best q={best_q:g}).")
        print("     The panel comparison is partly a comparison of tuning; say so.")"""

H5_NEW = """    # MAGNITUDE MATTERS. The first version of this fired on best_q != 1.0 with
    # no size test and printed "not its optimum" on a +0.0010 difference
    # against a +0.0359 gap, with the curve flat at 0.4138/0.4138/0.4139 for
    # q = 1.5/3/10. That is SATURATION, not an optimum, and it would have put
    # "the panel comparison is partly a comparison of tuning" into the paper
    # on 2.8% of the gap. Require a tenth of the gap before saying it.
    sp_best = means[f'spbp_q{best_q:g}']
    sp_gain = sp_best - sp_ref
    if best_q != 1.0 and sp_gain > 0.1 * abs(gap):
        print(f"  ** SP-BP's reference is NOT its optimum either (best q={best_q:g}, "
              f"{sp_gain:+.4f} = {100*sp_gain/abs(gap):.0f}% of the gap).")
        print("     The panel comparison is partly a comparison of tuning; say so.")
    elif best_q != 1.0:
        print(f"  -- SP-BP's curve SATURATES above q~1.5 (best q={best_q:g} is only "
              f"{sp_gain:+.4f}, {100*sp_gain/abs(gap):.0f}% of the gap).")
        print("     Report as saturation, NOT as evidence the comparison is tuning.")"""


# ── H6 ───────────────────────────────────────────────────────────────────────
H6A_OLD = """ABLATIONS = {
    'spbp_ab_full':      spbp_ab_full,"""

H6A_NEW = '''def spbp_ab_geohop(G, current, destination, v_bias=SPBP_V_BIAS):
    """Full SP-BP with the hop term replaced by GEOGRAPHIC progress.

    This is the last standing hypothesis for the SP-BP / DA-GPSR residual.
    The corrected ablation shows spbp_ab_candqueue = 0.4128 against
    da_gpsr = 0.3770 -- both use candidate-only queue avoidance, and the
    queue-weight sweep showed both sit at their own optimal weight. So the
    residual is not in the queue term. The only structural difference left is
    the progress metric: integer BFS hops versus normalised geographic
    progress.

    SCALE IS CONVERTED, NOT DROPPED. Geographic progress is divided by
    comm_range so it is expressed in hop-equivalent units and enters the score
    at the same magnitude as (h_cur - h_n). Without that, this variant would
    confound METRIC with SCALE -- the exact confound the queue-weight sweep
    had to be re-gridded to avoid.

    PRE-REGISTERED: near 0.377 means the progress metric explains the gap and
    Finding 4 is closed. Near 0.4128 means it does not, and the residual is
    somewhere not yet proposed -- report that as open rather than inventing a
    third hypothesis after the fact.
    """
    neighbors = list(G.neighbors(current))
    if not neighbors:
        return None
    if destination in neighbors:
        return destination
    try:
        h = nx.single_source_shortest_path_length(G, destination)
    except nx.NodeNotFound:
        return None
    if current not in h:
        return None
    comm_range = float(G.graph.get('comm_range') or 250.0)
    dpos = _pos(G, destination)
    d_cur = float(np.linalg.norm(dpos - _pos(G, current)))
    q_cur = float(G.nodes[current].get('queue_len', 0.0))

    best, best_score = None, -float('inf')
    for n in neighbors:
        if n not in h:
            continue                      # same exclusion as panel SP-BP
        q_n = float(G.nodes[n].get('queue_len', 0.0))
        lq = float(G.edges[current, n].get('link_quality', 0.0))
        d_n = float(np.linalg.norm(dpos - _pos(G, n)))
        hop_geo = (d_cur - d_n) / max(comm_range, 1e-6)
        score = lq * ((q_cur - q_n) + v_bias * hop_geo)
        if score > best_score:
            best_score, best = score, n
    return best if best is not None else _progress_fallback(G, current, destination)


ABLATIONS = {
    'spbp_ab_full':      spbp_ab_full,
    'spbp_ab_geohop':    spbp_ab_geohop,'''

H6B_OLD = """                     ('spbp_ab_additive', 'lq MULTIPLICATIVE -> additive')]:"""
H6B_NEW = """                     ('spbp_ab_additive', 'lq MULTIPLICATIVE -> additive'),
                     ('spbp_ab_geohop', 'BFS hops -> geographic progress')]:"""


PATCHES = {
    'routing_teachers_v3_local.py': [
        (H1_OLD, H1_NEW, 'H1 panel semantics at k=inf'),
        (H2_DOC_OLD, H2_DOC_NEW, 'H2 control docstring'),
        (H2_OLD, H2_NEW, 'H2 strengthened control'),
    ],
    'features_v2.py': [
        (H3A_OLD, H3A_NEW, 'H3 cand_reachable'),
        (H3B_OLD, H3B_NEW, 'H3 schema version 3'),
        (H3C_OLD, H3C_NEW, 'H3 extract_decision'),
    ],
    'preflight_dataset_v2_check.py': [
        (H4A_OLD, H4A_NEW, 'H4 exemption list'),
        (H4B_OLD, H4B_NEW, 'H4 exemption handling'),
    ],
    'experiment_queue_weight.py': [
        (H5_OLD, H5_NEW, 'H5 verdict magnitude threshold'),
    ],
    'experiment_spbp_mechanism.py': [
        (H6A_OLD, H6A_NEW, 'H6 geo-progress variant'),
        (H6B_OLD, H6B_NEW, 'H6 cost reporting'),
    ],
}

MARKER = 'cand_reachable'
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
        print(f"ERROR: {GUARD} already contains {MARKER!r} — already applied.\n"
              f"  Restore from git before re-running.")
        return 2
    if 'FEATURE_SCHEMA_VERSION' not in txt:
        print(f"ERROR: {GUARD} has no FEATURE_SCHEMA_VERSION.\n"
              f"  v4 applies on top of v1/v2/v3. Apply those first.")
        return 2

    staged = {}
    print(f"\n{'=' * 78}\n  V4 FIXES — assertion-guarded\n{'=' * 78}")
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
    print("\n  NEXT:")
    print("    1. python verify_v4_fixes.py --src src")
    print("    2. python src/experiment_locality_cost.py --out results/locality_cost.json")
    print("       -> kinf should IMPROVE and the locality cost should GROW")
    print("          from +0.0005. Pre-registered; report whatever it does.")
    print("    3. python src/experiment_spbp_mechanism.py --out results/spbp_mechanism.json")
    print("       -> read spbp_ab_geohop. Near 0.377 closes Finding 4.")
    print("    4. THEN regenerate (schema 2 -> 3, both checkers will abort on")
    print("       the old dataset until you do).")
    print("=" * 78 + "\n")
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except PatchError as e:
        print(f"\nPATCH FAILED — nothing was written.\n{e}\n")
        sys.exit(1)
