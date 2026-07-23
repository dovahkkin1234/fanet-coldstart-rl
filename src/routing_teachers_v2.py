"""
routing_teachers_v2.py
Congestion-aware teachers for the multi-packet simulator (M3 panel).

Signature for all teachers: f(G, current, destination) -> next_hop id or None.
Pure functions of the graph snapshot; no persistent state.

Graph node attributes expected: x, y, z, energy, queue_occupancy, queue_len
Graph edge attributes expected: distance, link_quality, packet_error_rate,
                                estimated_link_lifetime

THE M3 PANEL (docs/M3_TEACHERS_ORACLE_DESIGN.md S2) - 6 teachers spanning four
decision principles, with congestion-aware teachers deliberately in the majority
so oracle labeling is not fighting panel composition:

  congestion-BLIND references (in routing_teachers.py):
    1. dijkstra          - hop count
    2. gpsr              - geographic progress
  congestion-AWARE (this file):
    3. backpressure      - queue differential
    4. spbp              - queue differential + hop-distance bias
    5. da_gpsr           - geographic progress penalized by queue occupancy
    6. etx_dijkstra      - expected transmission count (PER-weighted shortest path)

NOTE ON LOOP AVOIDANCE: teachers do NOT track visited nodes. simulator_v2
already masks already-visited nodes at the call site; duplicating that here
would make the two mechanisms interact confusingly (design spec S3.1).
"""

import numpy as np
import networkx as nx

# SP-BP: converts one hop of progress into queue-backlog units. Default 1.0
# chosen for interpretability (one hop of progress == one packet of backlog),
# deliberately not tuned. Sensitivity is an ablation candidate, not an M3 concern.
SPBP_V_BIAS = 1.0

# ETX: floor on delivery probability so a fully-broken link gets a large but
# finite weight rather than inf (keeps shortest_path well-behaved).
# ETX is retained as a REFERENCE implementation only -- it is no longer a panel
# member (see lq_dijkstra_next_hop for why).
ETX_EPS = 1e-3

# LQ-Weighted Dijkstra. eps=0.05 caps the worst-link cost at 20 perfect hops.
# eps=1e-3 was checked numerically and reproduces ETX's 1000x blow-up exactly,
# i.e. the well-behavedness comes from the floor, not from the choice of signal.
# alpha controls detour aggressiveness; 1.0 keeps cost linear in 1/quality.
LQ_EPS   = 0.05
LQ_ALPHA = 1.0

# DA-GPSR weights, pinned as validated in M2 (produced the +0.04-0.06 PDR actor
# spread that gate G2 confirmed). Do NOT retune without new evidence.
DAGPSR_W_PROGRESS = 1.0
DAGPSR_W_QUEUE    = 1.0
DAGPSR_W_QUALITY  = 0.5


# Backpressure diagnostics, measured DURING simulation (the previous G3
# diagnostic sampled random node pairs on a frozen post-episode graph, which is
# disconnected from the decisions that actually produced the PDR numbers).
_BP_STATS = {'calls': 0, 'zero_gradient': 0}


def reset_bp_stats():
    _BP_STATS['calls'] = 0
    _BP_STATS['zero_gradient'] = 0


def get_bp_stats():
    c = max(_BP_STATS['calls'], 1)
    return {'calls': _BP_STATS['calls'],
            'zero_gradient': _BP_STATS['zero_gradient'],
            'zero_gradient_rate': _BP_STATS['zero_gradient'] / c}


def _pos(G, n):
    return np.array([G.nodes[n]['x'], G.nodes[n]['y'], G.nodes[n]['z']])


def _progress_fallback(G, current, destination):
    """Destination-ward neighbor. Shared fallback for backpressure-family
    teachers when no queue gradient exists (empty/low-load network)."""
    neighbors = list(G.neighbors(current))
    if not neighbors:
        return None
    dest_pos = _pos(G, destination)
    dist_cd = float(np.linalg.norm(dest_pos - _pos(G, current)))
    best, best_prog = None, -float('inf')
    for n in neighbors:
        prog = dist_cd - float(np.linalg.norm(dest_pos - _pos(G, n)))
        if prog > best_prog:
            best_prog, best = prog, n
    return best


# -- 3. Backpressure (Tassiulas-Ephremides, single-commodity approximation) ---

def backpressure_next_hop(G, current, destination, return_zerodiff_flag=False):
    """Classical backpressure (Tassiulas-Ephremides, single-commodity).

        w(u) = link_quality(v,u) * (Q_v - Q_u)

    Picks argmax over ALL neighbours, including zero and negative weights.

    NO STRICT-POSITIVE GATE, NO PROGRESS FALLBACK -- deliberately. An earlier
    implementation required w > 0 before accepting a neighbour and otherwise fell
    back to destination-ward progress. Because measured queue occupancy in this
    environment sits at 0.02-0.14 (gate G2), most adjacent pairs have
    Q_v - Q_u == 0, so that gate fired at essentially every decision and the
    teacher silently collapsed into GPSR -- the two produced byte-identical PDR
    in all six oracle cells. The gate was added defensively against backpressure's
    known low-load wandering pathology, but it ate the entire algorithm.

    Tassiulas-Ephremides selects the max-weight link unconditionally; the
    throughput-optimality proof depends on always acting on the best available
    differential, however small. Restoring that is what makes this teacher
    actually backpressure.

    CONSEQUENCE, EXPECTED AND CORRECT: with no destination bias, an empty network
    gives all-zero weights and the argmax degenerates to the first neighbour in
    (deterministic) iteration order -- effectively a random walk. Pure
    backpressure IS expected to perform poorly at low load. That is the textbook
    weakness SP-BP exists to fix, and observing it here is a reportable finding,
    not a bug.

    The destination short-circuit is retained: the destination is a sink whose
    backlog is effectively zero, so backpressure would route to it anyway.

    `return_zerodiff_flag=True` also reports whether every neighbour differential
    was zero (i.e. the choice carried no backpressure information) -- the honest
    diagnostic replacing the old, and wrongly-measured, fallback rate.
    """
    _BP_STATS['calls'] += 1
    neighbors = list(G.neighbors(current))
    if not neighbors:
        return (None, False) if return_zerodiff_flag else None
    if destination in neighbors:
        return (destination, False) if return_zerodiff_flag else destination

    q_cur = float(G.nodes[current].get('queue_len', 0.0))
    best, best_w = None, -float('inf')
    max_abs_diff = 0.0
    for n in neighbors:
        q_n = float(G.nodes[n].get('queue_len', 0.0))
        lq = float(G.edges[current, n].get('link_quality', 0.0))
        diff = q_cur - q_n
        max_abs_diff = max(max_abs_diff, abs(diff))
        w = lq * diff
        if w > best_w:
            best_w, best = w, n

    zero_diff = (max_abs_diff == 0.0)
    if zero_diff:
        _BP_STATS['zero_gradient'] += 1
    return (best, zero_diff) if return_zerodiff_flag else best


# -- 4. SP-BP - Shortest-Path Backpressure -----------------------------------

def spbp_next_hop(G, current, destination, v_bias=SPBP_V_BIAS):
    """Hybrid backpressure that fixes classical backpressure's low-load
    pathology (packets random-walk when there is no queue gradient).

        score(u) = link_quality(v,u) * [ (Q_v - Q_u) + v_bias * (h(v) - h(u)) ]

    where h(n) is hop distance from n to the destination on the current graph.
    Behaves like Dijkstra at low load (hop term dominates) and like Backpressure
    at high load (queue term dominates), so it should perform well across the
    whole load range - the design spec predicts this is the teacher most likely
    to top the oracle ranking (S3.2).

    Unreachable candidates (h = inf) are excluded rather than scored.
    """
    neighbors = list(G.neighbors(current))
    if not neighbors:
        return None
    if destination in neighbors:
        return destination

    # Hop distances to destination, computed once per decision (BFS from dst).
    try:
        h = nx.single_source_shortest_path_length(G, destination)
    except nx.NodeNotFound:
        return None
    if current not in h:
        return None      # disconnected from destination this frame
    h_cur = h[current]

    q_cur = float(G.nodes[current].get('queue_len', 0.0))
    best, best_score = None, -float('inf')
    for n in neighbors:
        if n not in h:
            continue                      # unreachable -> exclude
        q_n = float(G.nodes[n].get('queue_len', 0.0))
        lq = float(G.edges[current, n].get('link_quality', 0.0))
        score = lq * ((q_cur - q_n) + v_bias * (h_cur - h[n]))
        if score > best_score:
            best_score, best = score, n

    return best if best is not None else _progress_fallback(G, current, destination)


# -- 5. DA-GPSR - Delay-Aware GPSR -------------------------------------------

def da_gpsr_next_hop(G, current, destination,
                     w_progress=DAGPSR_W_PROGRESS,
                     w_queue=DAGPSR_W_QUEUE,
                     w_quality=DAGPSR_W_QUALITY):
    """Geometric forwarding that steers away from congested neighbors:

        score(u) = w_progress * progress(u)
                 - w_queue    * queue_occupancy(u)
                 + w_quality  * link_quality(v,u)

    A neighbor with good geographic progress but a badly backed-up queue can
    lose to a slightly worse-positioned but empty neighbor. This is the teacher
    that produced the measurable actor spread validated in gate G2.
    """
    neighbors = list(G.neighbors(current))
    if not neighbors:
        return None
    dest_pos = _pos(G, destination)
    dist_cd = float(np.linalg.norm(dest_pos - _pos(G, current)))

    best, best_score = None, -float('inf')
    for n in neighbors:
        dist_nd = float(np.linalg.norm(dest_pos - _pos(G, n)))
        progress = (dist_cd - dist_nd) / max(dist_cd, 1.0)
        occ = float(G.nodes[n].get('queue_occupancy', 0.0))
        lq = float(G.edges[current, n].get('link_quality', 0.0))
        score = w_progress * progress - w_queue * occ + w_quality * lq
        if score > best_score:
            best_score, best = score, n
    return best


# Backward-compatible alias: M2's G2 harness and simulator_v2 reference this name.
queue_aware_greedy_next_hop = da_gpsr_next_hop


# -- 6. Dynamic-metric Dijkstra ----------------------------------------------
#
# WHY NOT ETX (design evolution, to be reported in the paper):
# ETX-Dijkstra was the original choice for the dynamic-metric slot. Exploratory
# testing (G3, full grid) showed a classic ETX failure mode under bursty
# hidden-terminal interference: the 1/(1-PER) weight grows without bound as PER
# rises, so the metric over-avoided moderate, localized PER by selecting much
# longer paths. Under per-hop Bernoulli trials the cumulative risk of those
# extended paths degraded performance BELOW congestion-blind Dijkstra in every
# oracle cell (e.g. dense/low 0.484 vs 0.532; dense/high 0.148 vs 0.179).
#
# Two bounded replacements are provided. Both keep edge weights strictly
# positive and additive, so Dijkstra stays mathematically sound and loop-free,
# and both normalise a perfect link to cost 1.0 == one hop.

LQ_LAMBDA = 1.0      # w = 1 + LQ_LAMBDA*(1-LQ); bounded in [1, 1+lambda]


def lq_dijkstra_additive_unverified(G, current, destination, lam=LQ_LAMBDA):
    """UNVERIFIED, NOT REGISTERED IN TEACHERS. Additively-bounded LQ-weighted
    shortest path: w(v,u) = 1 + lam*(1 - link_quality(v,u)).

    This function's docstring previously claimed (from context lost across a
    session compaction, never reproduced in this conversation) that this form
    beats Dijkstra at every load while the multiplicative form
    (lq_dijkstra_next_hop below) loses at rate>=2.0. A second definition of
    `lq_dijkstra_next_hop` was later added lower in this file with the SAME
    NAME, silently shadowing this one — meaning the panel and every G3 run to
    date has been calling the MULTIPLICATIVE form only; this additive form was
    dead code the whole time.

    Re-checked head-to-head (single seed, medium_slow) during the M3 teacher-
    expansion pass: dijkstra=0.352/0.286/0.202/0.132, multiplicative=
    0.362/0.248/0.195/0.127, additive=0.362/0.276/0.193/0.145 across
    rate=0.5/1/2/4. No form clearly dominates at n=1 — the original claim is
    NOT reproduced, but also not refuted; it was never controlled for seed
    noise either way. Since the dynamic-metric panel slot was already dropped
    on separate, well-controlled evidence (arq_etx and the multiplicative form
    both lost decisively to Dijkstra in the 30-seed grid), this is left as
    dead code with an honest paper trail rather than revived, to avoid
    reopening a settled question on single-seed noise.
    """
    try:
        for u, v, data in G.edges(data=True):
            lq = min(max(float(data.get('link_quality', 0.0)), 0.0), 1.0)
            data['lq_weight'] = 1.0 + lam * (1.0 - lq)
        path = nx.shortest_path(G, current, destination, weight='lq_weight')
        return path[1] if len(path) >= 2 else None
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def arq_etx_next_hop(G, current, destination, max_retx=5, drop_penalty=20.0):
    """ARQ-aware expected-transmission cost (bounded ETX).

        E[attempts] = sum_{k=0..K} PER^k = (1 - PER^(K+1)) / (1 - PER)
        P(drop)     = PER^(K+1)
        w(v,u)      = E[attempts] + drop_penalty * P(drop)

    Physically exact for the ARQ mechanism the simulator actually implements:
    with K retransmissions a bad link costs retries (delay/energy), and only
    loses the packet if all K+1 attempts fail. This bounds the weight by
    construction -- E[attempts] <= K+1 = 6 -- which is precisely what unbounded
    ETX lacks. A PER=0.5 link costs ~2.3 here rather than ETX's 2.0-with-
    unbounded-tail, and a PER=0.99 link costs ~26 rather than ETX's 100.

    Provided as the alternative dynamic-metric candidate; the panel member is
    chosen empirically (see the head-to-head in the M3 notes).
    """
    K = int(max_retx)
    try:
        for u, v, data in G.edges(data=True):
            per = min(max(float(data.get('packet_error_rate', 0.0)), 0.0), 0.999999)
            p_drop = per ** (K + 1)
            if per < 1e-9:
                e_att = 1.0
            else:
                e_att = (1.0 - per ** (K + 1)) / (1.0 - per)
            data['arq_weight'] = e_att + drop_penalty * p_drop
        path = nx.shortest_path(G, current, destination, weight='arq_weight')
        return path[1] if len(path) >= 2 else None
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def etx_dijkstra_next_hop(G, current, destination, eps=ETX_EPS):
    """Original unbounded ETX: w = 1/max(1-PER, eps).

    RETAINED FOR THE PAPER, NOT FOR THE PANEL. Its underperformance versus
    congestion-blind Dijkstra is a reportable finding about ETX under bursty
    interference, and keeping it runnable lets that claim be reproduced.
    """
    try:
        for u, v, data in G.edges(data=True):
            per = float(data.get('packet_error_rate', 0.0))
            data['etx_weight'] = 1.0 / max(1.0 - per, eps)
        path = nx.shortest_path(G, current, destination, weight='etx_weight')
        return path[1] if len(path) >= 2 else None
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


# -- 6. LQ-Weighted Dijkstra - the dynamic-metric panel teacher --------------

def lq_dijkstra_next_hop(G, current, destination, alpha=LQ_ALPHA, eps=LQ_EPS):
    """Shortest path weighted by inverse measured link quality:

        w(v,u) = 1 / max(link_quality(v,u), eps) ** alpha

    REPLACES ETX-Dijkstra in the panel. ETX was tried first and lost to plain
    hop-count Dijkstra in every oracle cell (0.484 vs 0.532, 0.272 vs 0.305,
    0.148 vs 0.179). Diagnosis: ETX's 1/(1-PER) blows up super-linearly as PER
    approaches 1, so it over-avoids moderate-PER links by taking longer paths --
    and under bursty per-slot interference each extra hop carries its own ARQ
    exposure, so the detour compounds risk instead of avoiding it.

    Two properties make this metric better behaved here:
      * Bounded. With eps=0.05 the worst link costs 20 perfect hops, not 1000
        (eps=1e-3 would reproduce exactly ETX's blow-up -- verified numerically,
        so the improvement comes from the FLOOR, not merely from swapping which
        probability is inverted).
      * A perfect link costs exactly 1.0, so the metric degrades gracefully from
        plain hop count rather than being a different objective entirely.

    Weights are strictly positive (lq <= 1 => w >= 1), so Dijkstra stays
    well-defined and loop-free.

    It also harmonises the panel: link_quality is the same measured signal
    da_gpsr, backpressure and spbp already consume, so the dynamic-metric teacher
    is grounded in the same operational reality rather than a second, differently
    -biased estimate.
    """
    try:
        for u, v, data in G.edges(data=True):
            lq = float(data.get('link_quality', 0.0))
            data['lq_weight'] = 1.0 / (max(lq, eps) ** alpha)
        path = nx.shortest_path(G, current, destination, weight='lq_weight')
        return path[1] if len(path) >= 2 else None
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


# ── Neighbourhood aggregates shared by CAR and Lookahead-SPBP ───────────────
# Both use "look one hop past the candidate" as their extra signal, applied to
# two different base families (pure geometric vs hybrid-backpressure), so a
# difference in outcome between them isolates FAMILY, not information depth.

def _neighborhood_mean(G, v, u, attr, default=0.0):
    """Mean of `attr` over u's neighbours, excluding v (where we came from).
    Falls back to u's own value if u has no other neighbours."""
    others = [n for n in G.neighbors(u) if n != v]
    if not others:
        return float(G.nodes[u].get(attr, default))
    return float(np.mean([G.nodes[n].get(attr, default) for n in others]))


# ── Drift-Plus-Penalty (Lyapunov) ────────────────────────────────────────────

DPP_V = 2.0   # penalty weight; deliberately DIFFERENT from SP-BP's v_bias=1.0
              # and a genuinely different penalty TYPE (retransmission cost,
              # not hop distance) so a DPP win/loss tests the backpressure-
              # hybrid IDEA broadly, not SP-BP's specific formula.


def dpp_next_hop(G, current, destination, V=DPP_V, max_retx=5):
    """Drift-Plus-Penalty (Lyapunov optimization): maximize queue backlog
    differential minus a tunable penalty on expected retransmission cost.

        score(u) = (Q_v - Q_u) - V * (E[attempts(v,u)] - 1)
        E[attempts] = (1 - PER^(K+1)) / (1 - PER)      [K = max_retx]

    SAME FAMILY as SP-BP (hybrid backpressure with a configurable penalty
    term), but structurally different in two ways: (1) the penalty axis is
    retransmission cost, not hop-distance-to-destination, and (2) the penalty
    is NOT weighted by link_quality (SP-BP's whole score is; DPP's penalty is
    a separate additive term, matching the standard Lyapunov drift-plus-
    penalty formulation). If DPP competes with SP-BP, that supports "hybrid
    backpressure hybrids generally do well here"; if it does not, that points
    to SP-BP's SPECIFIC hop-bias choice being what matters, not the hybrid
    idea in general.

    Destination short-circuit as in backpressure/SP-BP. No positivity gate on
    the score (see the fix note above the score loop) -- always argmax over
    all neighbours, matching the corrected backpressure_next_hop.
    """
    neighbors = list(G.neighbors(current))
    if not neighbors:
        return None
    if destination in neighbors:
        return destination

    K = int(max_retx)
    q_cur = float(G.nodes[current].get('queue_len', 0.0))
    best, best_score = None, -float('inf')
    for n in neighbors:
        q_n = float(G.nodes[n].get('queue_len', 0.0))
        per = min(max(float(G.edges[current, n].get('packet_error_rate', 0.0)),
                     0.0), 0.999999)
        e_att = 1.0 if per < 1e-9 else (1.0 - per ** (K + 1)) / (1.0 - per)
        score = (q_cur - q_n) - V * (e_att - 1.0)
        if score > best_score:
            best_score, best = score, n

    # NO best_score>0 GATE: an earlier version gated here, defensively, on the
    # theory that a negative score means "no useful signal, prefer progress."
    # That was the exact mistake already made and fixed once in backpressure
    # (M3 audit): e_att-1 >= 0 always (pure cost, never a bonus), and real
    # queue differentials are usually 0 or a small integer (measured occupancy
    # 0.02-0.14), so V=2.0 times the penalty exceeds a modest differential on
    # most decisions -- the gate fired ~85% of the time in testing, silently
    # collapsing DPP into plain destination-ward progress (byte-identical PDR
    # to GPSR). Always take the argmax, exactly like the corrected
    # backpressure_next_hop.
    return best


# ── CAR — Congestion-Aware Routing (geometric, DIFFERENT lineage) ───────────

def car_next_hop(G, current, destination,
                 w_progress=DAGPSR_W_PROGRESS, w_field=DAGPSR_W_QUEUE,
                 w_quality=DAGPSR_W_QUALITY):
    """Congestion-Aware Routing: geometric forwarding steered by a
    NEIGHBOURHOOD congestion field rather than the candidate's own queue.

        score(u) = w_progress * progress(u)
                 - w_field    * mean_occupancy(u's neighbours, excl. v)
                 + w_quality  * link_quality(v,u)

    NO backpressure term, no hop-distance term — a genuinely different design
    lineage from backpressure/SP-BP/DPP (pure geometric + congestion-zone
    avoidance, as in the original CAR literature), included specifically so a
    non-backpressure-family congestion-aware teacher is represented.

    Same weights as DA-GPSR by design: the ONLY difference from DA-GPSR is
    single-hop occupancy (DA-GPSR) vs neighbourhood-field occupancy (CAR), so
    a difference in outcome isolates lookahead BREADTH, holding everything
    else constant.
    """
    neighbors = list(G.neighbors(current))
    if not neighbors:
        return None
    dest_pos = _pos(G, destination)
    dist_cd = float(np.linalg.norm(dest_pos - _pos(G, current)))

    best, best_score = None, -float('inf')
    for n in neighbors:
        dist_nd = float(np.linalg.norm(dest_pos - _pos(G, n)))
        progress = (dist_cd - dist_nd) / max(dist_cd, 1.0)
        field = _neighborhood_mean(G, current, n, 'queue_occupancy')
        lq = float(G.edges[current, n].get('link_quality', 0.0))
        score = w_progress * progress - w_field * field + w_quality * lq
        if score > best_score:
            best_score, best = score, n
    return best


# ── Lookahead-SPBP — SP-BP extended with 2-hop backlog visibility ──────────

LA_SPBP_W2 = 0.5   # weight on the 2-hop term, deliberately HALF the immediate
                   # differential's implicit weight of 1.0 -- second-hop queue
                   # state is staler/less certain, so it should count for less
                   # if it counts at all. This is a stated design choice, not a
                   # tuned value.


def spbp_lookahead_next_hop(G, current, destination,
                            v_bias=SPBP_V_BIAS, w2=LA_SPBP_W2):
    """SP-BP extended with SECOND-HOP backlog visibility:

        score(u) = link_quality(v,u) * [ (Q_v - Q_u) + v_bias*(h_v - h_u)
                                          - w2 * mean_queue_len(u's neighbours, excl. v) ]

    Directly tests the staleness-vs-information tradeoff that explained why
    global path-optimization on link_quality (ETX/LQ-Dijkstra) lost to greedy
    one-hop use of the same signal: does looking one hop FURTHER AHEAD than
    SP-BP help (more information reaches the decision) or hurt (the extra
    information is staler and second-hand, since it describes neighbours of a
    node the packet has not reached yet)? If this wins, deeper-but-still-local
    lookahead helps and a GNN's multi-hop receptive field (M4) should too; if
    it ties or loses, one-hop-ahead is closer to the sweet spot in this
    environment and GNN depth should be treated as a real ablation variable,
    not assumed to help monotonically.
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
    h_cur = h[current]

    q_cur = float(G.nodes[current].get('queue_len', 0.0))
    best, best_score = None, -float('inf')
    for n in neighbors:
        if n not in h:
            continue
        q_n = float(G.nodes[n].get('queue_len', 0.0))
        lq = float(G.edges[current, n].get('link_quality', 0.0))
        q2 = _neighborhood_mean(G, current, n, 'queue_len')
        score = lq * ((q_cur - q_n) + v_bias * (h_cur - h[n]) - w2 * q2)
        if score > best_score:
            best_score, best = score, n

    return best if best is not None else _progress_fallback(G, current, destination)


# -- Random baseline (G3 sanity floor, not part of the panel) ----------------

def random_next_hop(G, current, destination, rng=None):
    """Uniform random neighbour. Every real teacher must beat this at every load
    (gate G3 check 1); if one does not, that teacher is broken.

    The RNG is supplied by the caller (simulator_v2 passes its own seeded
    generator). An earlier version used a module-level generator seeded once at
    import, which made the random baseline depend on call order rather than on
    the episode seed -- a genuine reproducibility hole in a number that check 1
    depends on.
    """
    neighbors = list(G.neighbors(current))
    if not neighbors:
        return None
    r = rng if rng is not None else np.random.default_rng()
    return neighbors[int(r.integers(len(neighbors)))]
