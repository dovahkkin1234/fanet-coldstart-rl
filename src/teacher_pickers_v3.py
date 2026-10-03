"""
teacher_pickers_v3.py  --  restricted teacher pickers for Dataset V3.

One definition per teacher, used for the LABEL (da_gpsr), the BEHAVIOUR policies
(da_gpsr / gpsr / spbp / dijkstra_nodrop) and the VOTES (dijkstra / gpsr / spbp).
Before this module the label, the votes and the behaviour each had their own code
path; the M3.5 audit found two of them disagreeing for purely mechanical reasons.

SEMANTICS (established for SP-BP in M3.5, re-used in test_teacher_choice):
  * distances / hop counts are computed on the FULL frame graph;
  * only the CHOICE is restricted to the legal, visited-excluded candidates.
Pruning visited nodes and re-running the teacher instead severed paths in sparse
topologies and sent 22.6% of M3.5's first labels to a fallback rule.

TIE ORDER. Every picker iterates candidates in G.neighbors(current) order -- the
order the real teacher iterates -- so exact ties break exactly as the teacher
breaks them. simulator_v2._build_graph adds edges in itertools.combinations
order, which makes that order ascending node id; audit_dataset_v3 relies on it
when it re-derives labels from stored frames.

PINS. assert_pins() checks every picker against its real teacher on 300 random
graphs (unrestricted case), checks the restriction semantics of the two purely
local teachers against the teacher run on a view with the visited edges hidden,
and then proves each pin can FAIL on a deliberately broken picker (project rule:
a check that only ever passes proves nothing). It runs at import.

NOTE ON da_gpsr AND AN ADJACENT DESTINATION. da_gpsr has NO destination
short-circuit: if the destination is a neighbour it is scored like any other
candidate (progress 1.0, minus ITS queue occupancy, plus link quality). Under the
v12 semantics a delivered packet never enters the destination's queue, so that
penalty is a modelling quirk of the teacher -- but it is the teacher the oracle
panels measured, so the label reproduces it exactly. generate_dataset_v3 counts
how often it happens (episode meta: n_dst_adjacent_not_chosen).
"""

import numpy as np
import networkx as nx

import routing_teachers_v2 as rt
from routing_teachers import dijkstra_next_hop, gpsr_next_hop
from generate_dataset_v2 import canonical_candidates, spbp_pick_restricted
import features_v2 as F

LABEL_TEACHER = 'da_gpsr'
VOTE_TEACHERS = ('dijkstra', 'gpsr', 'spbp')
BEHAVIOUR_TEACHERS = ('da_gpsr', 'gpsr', 'spbp', 'dijkstra_nodrop')


# ─────────────────────────────────────────────────────────────────────────────
# pickers.  Signature: (G, current, dst, cands, h_map) -> node or None
# ─────────────────────────────────────────────────────────────────────────────
def da_gpsr_scores(G, c, dst, cands):
    """da_gpsr's exact score for every candidate, in the order of `cands`.
    Arithmetic is written in the same order as rt.da_gpsr_next_hop so the floats
    are bit-identical."""
    dest_pos = rt._pos(G, dst)
    dist_cd = float(np.linalg.norm(dest_pos - rt._pos(G, c)))
    out = np.empty(len(cands), dtype=np.float64)
    for j, n in enumerate(cands):
        dist_nd = float(np.linalg.norm(dest_pos - rt._pos(G, n)))
        progress = (dist_cd - dist_nd) / max(dist_cd, 1.0)
        occ = float(G.nodes[n].get('queue_occupancy', 0.0))
        lq = float(G.edges[c, n].get('link_quality', 0.0))
        out[j] = (rt.DAGPSR_W_PROGRESS * progress - rt.DAGPSR_W_QUEUE * occ
                  + rt.DAGPSR_W_QUALITY * lq)
    return out


def da_gpsr_pick_scored(G, c, dst, cands):
    """(chosen node, scores in `cands` order). Argmax in G.neighbors order with a
    strict '>' -- exactly rt.da_gpsr_next_hop's loop, restricted to `cands`."""
    sc = da_gpsr_scores(G, c, dst, cands)
    pos = {n: j for j, n in enumerate(cands)}
    best, best_s = None, -float('inf')
    for n in G.neighbors(c):
        j = pos.get(n)
        if j is None:
            continue
        if sc[j] > best_s:
            best_s, best = sc[j], n
    return best, sc


def da_gpsr_pick(G, c, dst, cands, h_map=None):
    return da_gpsr_pick_scored(G, c, dst, cands)[0]


def gpsr_pick(G, c, dst, cands, h_map=None):
    """gpsr_next_hop restricted to `cands`: the neighbour closest to dst, strict
    '<', iterated in G.neighbors order."""
    cset = set(cands)
    dest_pos = rt._pos(G, dst)
    best, best_d = None, float('inf')
    for n in G.neighbors(c):
        if n not in cset:
            continue
        d = np.linalg.norm(dest_pos - rt._pos(G, n))
        if d < best_d:
            best_d, best = d, n
    return best


def dijkstra_pick(G, c, dst, cands, h_map):
    """dijkstra restricted to `cands`. Returns None iff dst is unreachable from c
    -- the real teacher then drops the packet (no_route). Used for VOTES only:
    a -1 vote therefore means 'dijkstra would drop here'."""
    if dst in cands:
        return dst
    if c not in h_map:
        return None
    try:
        path = nx.shortest_path(G, c, dst)
        ref = path[1] if len(path) >= 2 else None
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        ref = None
    if ref is not None and ref in cands:
        return ref
    reach = [u for u in cands if u in h_map]
    if not reach:
        return None
    return min(reach, key=lambda u: (h_map[u], cands.index(u)))


def dijkstra_nodrop_pick(G, c, dst, cands, h_map):
    """BEHAVIOUR form of dijkstra (decision 2026-10-02: the dataset ENCODES a drop
    action but no policy generates one). When dst is unreachable it forwards to
    the nearest-to-destination candidate (canonical slot 0) instead of dropping."""
    p = dijkstra_pick(G, c, dst, cands, h_map)
    return p if p is not None else cands[0]


def spbp_pick(G, c, dst, cands, h_map):
    """SP-BP restricted (the M3.5 picker; pinned by generate_dataset_v2 too)."""
    return spbp_pick_restricted(G, c, dst, cands, h_map)


PICKERS = {
    'da_gpsr': da_gpsr_pick,
    'gpsr': gpsr_pick,
    'dijkstra': dijkstra_pick,
    'dijkstra_nodrop': dijkstra_nodrop_pick,
    'spbp': spbp_pick,
}
REAL_TEACHERS = {
    'da_gpsr': rt.da_gpsr_next_hop,
    'gpsr': gpsr_next_hop,
    'dijkstra': dijkstra_next_hop,
    'spbp': rt.spbp_next_hop,
}


# ─────────────────────────────────────────────────────────────────────────────
# pins
# ─────────────────────────────────────────────────────────────────────────────
def _random_graph(rng):
    n = int(rng.integers(5, 13))
    G = nx.Graph()
    for i in range(n):
        G.add_node(i, x=float(rng.integers(0, 900)), y=float(rng.integers(0, 900)),
                   z=float(rng.integers(100, 300)), energy=90.0,
                   queue_occupancy=float(rng.random()),
                   queue_len=float(rng.integers(0, 50)))
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.4:
                G.add_edge(i, j, distance=float(rng.integers(50, 300)),
                           link_quality=float(rng.random()),
                           packet_error_rate=float(rng.random()) * 0.3)
    return G


def _pin_disagreements(pickers, trials=300, seed=7):
    """Count, per teacher, the cases where the restricted picker differs from the
    real teacher. Unrestricted case: only `src` is visited (src is never its own
    neighbour, so cands = all neighbours). Path-dependent teachers are compared
    only when a path exists, as generate_dataset_v2.assert_no_drift does."""
    rng = np.random.default_rng(seed)
    bad = {k: 0 for k in pickers}
    for _ in range(trials):
        G = _random_graph(rng)
        src, dst = 0, G.number_of_nodes() - 1
        if not list(G.neighbors(src)):
            continue
        cands = canonical_candidates(G, src, dst, {src})
        h = F.hop_distances_to(G, dst)
        has_path = nx.has_path(G, src, dst)
        for name, pick in pickers.items():
            if name in ('dijkstra', 'spbp') and not has_path:
                if name == 'dijkstra' and pick(G, src, dst, cands, h) is not None:
                    bad[name] += 1          # unreachable must mean "would drop"
                continue
            if pick(G, src, dst, cands, h) != REAL_TEACHERS[name](G, src, dst):
                bad[name] += 1
    return bad


def _restriction_disagreements(trials=300, seed=11):
    """For the two purely LOCAL teachers, the restricted pick must equal the real
    teacher run on a view in which the edges to visited neighbours are hidden
    (teacher_on_valid semantics). Also: every picker returns a legal candidate."""
    rng = np.random.default_rng(seed)
    bad = 0
    for _ in range(trials):
        G = _random_graph(rng)
        n = G.number_of_nodes()
        src, dst = 0, n - 1
        nb = list(G.neighbors(src))
        if len(nb) < 2:
            continue
        visited = {src} | set(rng.choice(nb, size=int(rng.integers(1, len(nb))),
                                         replace=False).tolist())
        visited.discard(dst)
        cands = canonical_candidates(G, src, dst, visited)
        if not cands:
            continue
        hide = [(src, v) for v in visited if G.has_edge(src, v)]
        H = nx.restricted_view(G, [], hide)
        for name in ('da_gpsr', 'gpsr'):
            if PICKERS[name](G, src, dst, cands, None) != REAL_TEACHERS[name](H, src, dst):
                bad += 1
        h = F.hop_distances_to(G, dst)
        for name, pick in PICKERS.items():
            got = pick(G, src, dst, cands, h)
            if got is not None and got not in cands:
                bad += 1
    return bad


def assert_pins(verbose=False):
    bad = _pin_disagreements({k: PICKERS[k] for k in REAL_TEACHERS})
    if any(bad.values()):
        raise AssertionError(f'restricted picker drifted from its real teacher: {bad}')
    r = _restriction_disagreements()
    if r:
        raise AssertionError(f'restriction semantics violated in {r} cases')

    # negative controls: every pin must be able to fail
    broken = {
        'da_gpsr': lambda G, c, d, cs, h: max(            # drops the queue term
            cs, key=lambda u: (da_gpsr_scores(G, c, d, [u])[0]
                               + float(G.nodes[u]['queue_occupancy']))),
        'gpsr': lambda G, c, d, cs, h: cs[-1],             # farthest from dst
        'dijkstra': lambda G, c, d, cs, h: (               # farthest reachable hop
            max([u for u in cs if u in h], key=lambda u: h[u])
            if [u for u in cs if u in h] else None),
        'spbp': lambda G, c, d, cs, h: cs[-1],
    }
    for name, fn in broken.items():
        b = _pin_disagreements({name: fn})
        if b[name] == 0:
            raise AssertionError(f'pin for {name} cannot detect a broken picker -- dead check')
    if verbose:
        print('  teacher pins: 4 pickers == real teachers (300 graphs), restriction '
              'semantics OK, every pin FAILS on a broken picker')
    return True


assert_pins()
