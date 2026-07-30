"""
routing_teachers_v3_local.py  —  LOCALITY-LIMITED teacher variants.

PURPOSE (addresses reviewer findings M-1 and M-2)
=================================================
Every teacher in the M3 panel queries a networkx graph with complete,
instantaneous, zero-cost knowledge of the whole network. SP-BP in particular
uses an EXACT BFS hop-distance from any node to any destination, recomputed
every frame. No distributed FANET router has this without continuous
link-state flooding -- and this project models interference from DATA traffic
rigorously while charging nothing for the CONTROL traffic that would buy the
global knowledge.

That asymmetry is the single most attackable point in the work. It is not
fixed by a disclaimer; it is fixed by MEASURING what the global knowledge is
worth, so the paper can either (a) declare SP-BP a genie-aided upper bound
with a quantified gap to realistic variants, or (b) show the gap is small and
locality is nearly free.

WHAT COUNTS AS REALISTIC INFORMATION (the load-bearing assumption)
------------------------------------------------------------------
Obtainable from standard periodic beaconing and packet headers:
  - own queue length and position
  - 1-hop neighbour queue lengths, positions, link qualities
  - the DESTINATION's position (carried in the packet header -- exactly
    GPSR's standing assumption, entirely standard in geographic routing)
Obtainable but costly (multi-hop control flooding, e.g. OLSR TC messages):
  - k-hop neighbourhood topology, hence exact hop-distance to destinations
    that lie within k hops
NOT obtainable without a central controller or full link-state flooding:
  - exact hop-distance to an ARBITRARY destination anywhere in the network

spbp_khop_next_hop models exactly this: exact hop-distance when the
destination is within k hops, and a geographic proxy otherwise.

WHY THIS ALSO SETTLES M-2
-------------------------
A k-layer GNN propagates information exactly k hops, so its achievable
knowledge of hop-distance is precisely the k-hop-limited case implemented
here. Measuring PDR(k) therefore answers directly whether the M4 design's
explicit `hop_distance_to_dst` input feature is:
  - a harmless convenience (if PDR(k=3) ~= PDR(k=inf), the GNN could have
    learned it anyway), or
  - a genie-aided crutch that must be declared as such (if PDR(k=3) is far
    below PDR(k=inf), the feature hands the model information no realistic
    3-layer architecture could derive).

A PREDICTION, RECORDED BEFORE RUNNING (so this is a test, not a story)
-----------------------------------------------------------------------
The M3.5 audit measured mean hop-distance-to-destination per scenario:
    very_dense  1.64 hops     dense_slow  2.09 hops
    medium_slow 5.32 hops     sparse_fast 6.74 hops
So a k=3 horizon should COVER the typical destination in the dense regimes
but MISS it in the sparse ones. The locality cost should therefore be small
in dense/very_dense and large in medium/sparse_fast. If instead the cost is
uniform across regimes, this model of locality is wrong and the conclusion
should not be trusted.
"""

import numpy as np
import networkx as nx

from routing_teachers_v2 import SPBP_V_BIAS, _pos, _progress_fallback

# Fallback hop-estimate when the destination is beyond the k-hop horizon.
# Must be locally computable: uses only the candidate's own position and the
# destination position from the packet header.
#   proxy(u) = k + geo_dist(u, dst) / comm_range
# The leading +k guarantees any node OUTSIDE the horizon ranks worse than any
# node known to be inside it, preserving monotonicity; the geographic term
# then orders the outside nodes sensibly among themselves.
DEFAULT_COMM_RANGE = 250.0


def _comm_range_of(G, fallback=DEFAULT_COMM_RANGE):
    """Radio range. A node knows its own radio's range, so treating this as
    known is realistic. Read from the graph if the simulator recorded it,
    otherwise estimated from the longest surviving link."""
    r = G.graph.get('comm_range')
    if r:
        return float(r)
    try:
        return max(float(d.get('distance', 0.0)) for _, _, d in G.edges(data=True)) or fallback
    except ValueError:
        return fallback


def khop_distance_map(G, destination, k):
    """Exact hop distance to `destination`, but ONLY for nodes within k hops.

    This is precisely the knowledge a k-layer GNN can propagate, and the
    knowledge a k-hop-limited distributed protocol can maintain. Nodes beyond
    the horizon are simply absent from the returned map -- the caller must
    supply a locally-computable proxy for them.
    """
    if k is None or k == float('inf'):
        return nx.single_source_shortest_path_length(G, destination)
    return nx.single_source_shortest_path_length(G, destination, cutoff=int(k))


def spbp_khop_next_hop(G, current, destination, k=3, v_bias=SPBP_V_BIAS):
    """SP-BP restricted to a k-hop information horizon.

    Identical scoring to spbp_next_hop:
        score(u) = lq(v,u) * [ (Q_v - Q_u) + v_bias * (h_v - h_u) ]
    but h is known EXACTLY only within k hops of the destination; beyond that
    it is replaced by a geographic proxy computed from the destination
    position (packet header) and the node's own position (GPS).

    k = None or float('inf') reproduces the original global SP-BP exactly,
    which the accompanying test asserts.
    """
    neighbors = list(G.neighbors(current))
    if not neighbors:
        return None
    if destination in neighbors:
        return destination

    try:
        h_known = khop_distance_map(G, destination, k)
    except nx.NodeNotFound:
        return None

    comm_range = _comm_range_of(G)
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
    return best if best is not None else _progress_fallback(G, current, destination)


# Convenience wrappers so these can be registered in the TEACHERS dict, which
# requires the standard (G, current, destination) signature.
def spbp_k1_next_hop(G, c, d):
    return spbp_khop_next_hop(G, c, d, k=1)


def spbp_k2_next_hop(G, c, d):
    return spbp_khop_next_hop(G, c, d, k=2)


def spbp_k3_next_hop(G, c, d):
    return spbp_khop_next_hop(G, c, d, k=3)


def spbp_k4_next_hop(G, c, d):
    return spbp_khop_next_hop(G, c, d, k=4)


def spbp_kinf_next_hop(G, c, d):
    """Global SP-BP expressed through the same code path, as a control.
    Should match routing_teachers_v2.spbp_next_hop exactly; asserted by
    assert_kinf_matches_global()."""
    return spbp_khop_next_hop(G, c, d, k=None)


def assert_kinf_matches_global(trials=80, seed=0):
    """The k=inf variant must reproduce the original global SP-BP exactly.

    Without this, a difference measured across k could be an artifact of this
    reimplementation rather than of the information horizon -- which would
    invalidate the entire locality experiment. Runs at import.
    """
    from routing_teachers_v2 import spbp_next_hop
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
        raise AssertionError("assert_kinf_matches_global checked 0 cases")


assert_kinf_matches_global()
