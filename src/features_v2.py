"""
features_v2.py  —  M4 feature extraction (replaces simulator_v2._make_obs placeholder).

Splits features into two groups, and the split is load-bearing rather than
cosmetic (see M4 spec section 3.1):

  FRAME-LEVEL  (node_feat, edge_feat) — depend ONLY on the network state, never
      on which packet is being routed. Encoded ONCE per frame by the GNN and
      reused for every routing decision in that frame.

  DECISION-LEVEL (query features, candidate features, hop_distance_to_dst) —
      depend on the packet / destination. Computed fresh per decision; cheap.

WHY THE SPLIT MATTERS: `very_dense` runs num_flows = N//4 = 11 concurrent flows
with potentially 11 different active destinations in a single frame. Any
destination-dependent value placed in node_feat would force the GNN encoder to
re-run once per active destination per frame instead of once per frame — up to
~11x the intended compute, silently, in the densest scenario.

NORMALISATION: scenarios differ in area (700-1500 m), so positions are
normalised per-scenario by that scenario's own extent. The constants used are
returned by `norm_constants(cfg)` and MUST be persisted with the dataset and
reused unchanged at training and evaluation time — a mismatch would not raise,
it would silently degrade the model.
"""

import numpy as np
import networkx as nx

# Feature name lists — single source of truth. Persisted with the dataset so a
# later column-order change cannot silently misalign training data.
# NOTE ON A REMOVED FEATURE: 'queue_len' used to sit between
# 'queue_occupancy' and 'energy'. It was dropped after the independent audit
# showed the two columns had IDENTICAL statistics across all four moments
# (min/max/mean/std to 3 decimals). Tracing it: NodeQueue.occupancy IS
# length/max_size, and MAX_QUEUE is a constant 50, so normalising queue_len by
# 50 reproduced queue_occupancy exactly. They were one signal stored twice.
# Harmless for training, but it would have split feature-importance across two
# identical columns in M4 -- the milestone whose whole purpose is justifying
# the architecture. Kept as queue_occupancy (already unit-scaled); the RAW
# queue length is still available to teachers via the graph attribute, which
# is what SP-BP's scoring actually uses.
NODE_FEATURES = [
    'x', 'y', 'z',
    'vx', 'vy', 'vz',
    'queue_occupancy',
    'energy',
    'degree',
]

EDGE_FEATURES = [
    'distance',
    'link_quality',
    'packet_error_rate',
    'estimated_link_lifetime',
    'relative_velocity',
]
# REMOVED FEATURE -- 'snr' sat between 'packet_error_rate' and
# 'estimated_link_lifetime'. The graph attribute it read is `sinr_clean`,
# returned by simulator_v2._measured_link from
# compute_link_features_v2(dist, interference_mw=0.0, n_contenders=1,
# shadowing_db=0.0). Both link directions pass the same dist_ij, so the
# 0.5*(s_ij+s_ji) average collapses to a deterministic, strictly monotone
# function of `distance`. Linear r is only about -0.93, which is why a
# correlation screen alone would have let it through -- but the map is
# invertible, so any nonlinear model recovers one column exactly from the
# other. It carried NO interference or load information despite its name.
# There is no measured SINR available to substitute: the interference model is
# a Bernoulli lethal-interferer indicator, not a continuous power sum (M-8).
# DEFERRED to M5: expose p_clear and p_coll as separate edge features. The
# model can currently recover only their product, since
# link_quality = lq_clean * p_clear * (1 - p_coll), and hidden-terminal loss
# and MAC contention are physically distinct channels.

QUERY_FEATURES = [
    'ttl_left',
    'dist_to_dest',
    'current_queue_occupancy',
    'neigh_buffered_packets',
    'neigh_mean_occupancy',
    'hop_distance_to_dst',
]
# REMOVED FEATURE -- 'hops_so_far' sat between 'ttl_left' and 'dist_to_dest'.
#   ttl_left    = max(TTL - hops, 0) / TTL
#   hops_so_far = min(hops / TTL, 1)
# These sum to exactly 1.0 on every row: max observed hops is 19 against
# TTL = 20, so neither clip ever engages. Exact LINEAR redundancy,
# Pearson r = -1.000. Same defect class as the queue_len removal below, and
# it would have split feature importance in the one milestone whose purpose is
# justifying the architecture.
#
# RENAMED -- 'n_inflight' -> 'neigh_buffered_packets',
#            'network_mean_occupancy' -> 'neigh_mean_occupancy'.
# Both are now computed over the LOCAL_HORIZON-hop neighbourhood of the
# current node (see LOCAL_HORIZON below). A column named
# `network_mean_occupancy` that holds a 2-hop mean is the same naming trap the
# `snr` removal above is about, so the names move with the semantics.

CANDIDATE_FEATURES = [
    'progress',            # normalised geographic progress toward destination
    'cand_hop_distance',   # candidate's BFS hop distance to destination
    'is_destination',      # candidate IS the destination
]

# Physical constants used for normalisation (mirrors simulator_v2 / models.py).
MAX_QUEUE_REF = 50.0
INITIAL_ENERGY_REF = 100.0
TTL_REF = 20.0
LIFETIME_REF = 60.0     # seconds, matches estimate_link_lifetime's max
HOP_CAP = 10.0          # hop distances clipped/normalised against this

# ── OBSERVABILITY SCOPING DECISION ───────────────────────────────────────────
# Two query features aggregate load beyond the current node. A distributed
# router cannot observe network-wide state, and the M4 encoder's dense N x N
# attention is likewise centralised. This constant is where that scoping
# decision is made explicit and persisted, rather than left implicit.
#
#   LOCAL_HORIZON = k    -> both features are computed over the k-hop
#                           neighbourhood. Locally computable: k-hop neighbour
#                           state is what OLSR-family protocols already
#                           exchange via HELLO messages, so the assumption is
#                           standard rather than novel.
#   LOCAL_HORIZON = None -> whole-network aggregation. This is the
#                           CONTROLLER-ASSISTED scoping choice and must be
#                           stated as such in the paper.
#
# Default 2 matches the shallow end of the M4 depth sweep L in {0,1,2,3} and
# the spbp_k1..k4 horizon pattern already established in
# routing_teachers_v3_local.py. It is also consistent with the measured result
# that global BFS hop-distance is worth only +0.0005 PDR: there is little
# evidence global aggregation buys anything here, and a large deployability
# cost to claiming it.
LOCAL_HORIZON = 2

# Bumped whenever the four feature lists change. Persisted in manifest.json and
# asserted by both checkers, so a checker can never resolve feature names
# against a module that disagrees with the dataset it is reading.
FEATURE_SCHEMA_VERSION = 2


def _assert_schema_sane():
    """Structural self-test at import. Cheap, and it fails loudly.

    This does NOT replace G3.5 check 8 -- name uniqueness is a property of the
    lists, redundancy is a property of the DATA and can only be measured on a
    generated dataset.
    """
    for nm, lst in (('NODE', NODE_FEATURES), ('EDGE', EDGE_FEATURES),
                    ('QUERY', QUERY_FEATURES), ('CANDIDATE', CANDIDATE_FEATURES)):
        assert len(set(lst)) == len(lst), f"{nm}_FEATURES contains a duplicate name"
        assert all(isinstance(x, str) and x for x in lst), f"{nm}_FEATURES malformed"
    assert LOCAL_HORIZON is None or (isinstance(LOCAL_HORIZON, int)
                                     and LOCAL_HORIZON >= 1), \
        "LOCAL_HORIZON must be None (controller-assisted) or an int >= 1"


_assert_schema_sane()


def norm_constants(cfg):
    """Per-scenario normalisation constants. Persist these with the dataset."""
    return {
        'area_x': float(cfg['area_x']),
        'area_y': float(cfg['area_y']),
        'z_min': float(cfg['z_min']),
        'z_max': float(cfg['z_max']),
        'speed_max': float(cfg.get('speed_max', 15.0)),
        'comm_range': float(cfg['comm_range']),
        'max_queue': MAX_QUEUE_REF,
        'initial_energy': INITIAL_ENERGY_REF,
        'ttl': TTL_REF,
        'lifetime_ref': LIFETIME_REF,
        'hop_cap': HOP_CAP,
        # Degree normaliser. Was max(N-1, 1) read off the live graph and never
        # persisted, so the persist-and-reuse-unchanged guarantee did not cover
        # it. Same value, now auditable. Note it makes normalised degree
        # conflate density with network size: degree 18 is 0.409 at N=45 but
        # would be 0.947 at N=20, and medium_slow (4.37/29 = 0.151) sits almost
        # on top of sparse_fast (2.51/19 = 0.132) despite a 1.7x density gap.
        # State that when reporting the generalisation result.
        'degree_ref': float(max(int(cfg['num_drones']) - 1, 1)),
        # None = whole-network aggregation (controller-assisted).
        'local_horizon': LOCAL_HORIZON,
        'schema_version': FEATURE_SCHEMA_VERSION,
    }


def extract_frame(G, nc):
    """FRAME-LEVEL features. Returns (node_ids, node_feat, edge_index, edge_feat).

    node_ids  : (N,) int   — graph node ids in the row order used by node_feat
    node_feat : (N, F_node) float32
    edge_index: (2, E) int — undirected edges stored ONCE (i<j)
    edge_feat : (E, F_edge) float32
    """
    node_ids = sorted(G.nodes())
    idx_of = {n: i for i, n in enumerate(node_ids)}
    N = len(node_ids)

    zspan = max(nc['z_max'] - nc['z_min'], 1e-6)
    smax = max(nc['speed_max'], 1e-6)

    nf = np.zeros((N, len(NODE_FEATURES)), dtype=np.float32)
    for n in node_ids:
        i = idx_of[n]
        d = G.nodes[n]
        nf[i] = (
            d['x'] / nc['area_x'],
            d['y'] / nc['area_y'],
            (d['z'] - nc['z_min']) / zspan,
            np.clip(d.get('vx', 0.0) / smax, -1.0, 1.0),
            np.clip(d.get('vy', 0.0) / smax, -1.0, 1.0),
            np.clip(d.get('vz', 0.0) / smax, -1.0, 1.0),
            d.get('queue_occupancy', 0.0),
            d.get('energy', 0.0) / nc['initial_energy'],
            G.degree(n) / nc['degree_ref'],
        )

    edges = [(u, v) for u, v in G.edges()]
    E = len(edges)
    ei = np.zeros((2, E), dtype=np.int32)
    ef = np.zeros((E, len(EDGE_FEATURES)), dtype=np.float32)
    for k, (u, v) in enumerate(edges):
        a, b = idx_of[u], idx_of[v]
        ei[0, k], ei[1, k] = a, b
        e = G.edges[u, v]
        ef[k] = (
            e.get('distance', 0.0) / nc['comm_range'],
            e.get('link_quality', 0.0),
            e.get('packet_error_rate', 0.0),
            min(e.get('estimated_link_lifetime', 0.0), nc['lifetime_ref']) / nc['lifetime_ref'],
            min(e.get('relative_velocity', 0.0) / max(2.0 * smax, 1e-6), 1.0),
        )
    return np.asarray(node_ids, dtype=np.int32), nf, ei, ef


MANIFEST_FEATURE_KEYS = (
    ('node_features', 'NODE_FEATURES'),
    ('edge_features', 'EDGE_FEATURES'),
    ('query_features', 'QUERY_FEATURES'),
    ('candidate_features', 'CANDIDATE_FEATURES'),
)


def assert_manifest_compatible(man, context=''):
    """Fail loudly if a dataset's persisted feature lists disagree with THIS module.

    Every feature lookup in both checkers is name-based
    (NODE_FEATURES.index('queue_occupancy'), enumerate(QUERY_FEATURES)), so
    nothing breaks by index arithmetic. What breaks is VERSION SKEW: a checker
    run against a dataset generated under a different features_v2.py resolves
    names against the new module while the stored array still has the old
    column layout. That misaligns silently -- no exception, plausible numbers,
    wrong conclusions. Exactly the failure mode the norm_constants warning
    describes.

    generate_dataset_v2.py has always persisted these four lists. Nothing ever
    read them back. This makes the manifest a real compatibility boundary
    rather than a documented one.

    Returns a list of human-readable mismatch strings; empty means compatible.
    """
    problems = []
    live_ver = FEATURE_SCHEMA_VERSION
    man_ver = man.get('feature_schema_version')
    if man_ver is None:
        problems.append(
            'manifest has no feature_schema_version: it predates the schema '
            'boundary and CANNOT be validated against this module. Regenerate.')
    elif man_ver != live_ver:
        problems.append(
            f'feature_schema_version mismatch: manifest={man_ver} '
            f'module={live_ver}')

    for man_key, mod_name in MANIFEST_FEATURE_KEYS:
        stored = man.get(man_key)
        live = globals()[mod_name]
        if stored is None:
            problems.append(f'manifest is missing {man_key!r}')
            continue
        if list(stored) != list(live):
            problems.append(
                f'{man_key}: manifest={list(stored)} != module={list(live)}')

    man_h = man.get('local_horizon', '<absent>')
    if man_h != LOCAL_HORIZON:
        problems.append(
            f'local_horizon mismatch: manifest={man_h} module={LOCAL_HORIZON}. '
            f'The observability scoping of the dataset differs from the code '
            f'about to read it.')

    if problems and context:
        problems = [f'({context}) {p}' for p in problems]
    return problems


def hop_distances_to(G, dst):
    """BFS hop distance from every reachable node to `dst`.

    This is the SAME quantity SP-BP uses internally. It is supplied to the model
    as an EXPLICIT feature because a k-layer GNN propagates information exactly
    k hops and therefore structurally cannot compute it for destinations further
    than k away (M4 spec section 3.3). Supplying it also de-confounds the
    GNN-depth ablation: depth then measures congestion-lookahead radius only,
    not the model's ability to approximate BFS.
    """
    try:
        return nx.single_source_shortest_path_length(G, dst)
    except nx.NodeNotFound:
        return {}


def extract_decision(G, pkt, candidates, nc, h_map, n_inflight,
                     network_mean_occ, ttl_const=None):
    """DECISION-LEVEL features.

    Returns (query_feat (F_query,), cand_feat (K, F_cand)) where K = len(candidates).
    `candidates` must already be the VALID (visited-excluded) set, in canonical
    order — see generate_dataset_v2.canonical_candidates.
    """
    ttl_ref = float(ttl_const or nc['ttl'])
    c, dst = pkt.current, pkt.dst
    cpos = np.array([G.nodes[c]['x'], G.nodes[c]['y'], G.nodes[c]['z']])
    dpos = np.array([G.nodes[dst]['x'], G.nodes[dst]['y'], G.nodes[dst]['z']])
    diag = float(np.hypot(nc['area_x'], nc['area_y']))
    dist_cd = float(np.linalg.norm(dpos - cpos))

    h_cur = h_map.get(c, nc['hop_cap'])

    # Observability scoping (see LOCAL_HORIZON). With a horizon set, both load
    # aggregates are restricted to the k-hop neighbourhood of the CURRENT node,
    # so every query feature is locally computable by the forwarding node. With
    # horizon None the caller's whole-network values are used unchanged, which
    # is the controller-assisted variant.
    k = nc.get('local_horizon')
    if k is not None:
        nbrs = nx.single_source_shortest_path_length(G, c, cutoff=int(k))
        occs = [G.nodes[u].get('queue_occupancy', 0.0) for u in nbrs]
        if occs:
            # Local restriction of exactly the same two quantities: mean queue
            # occupancy, and the packet count those occupancies represent
            # (occupancy IS length/MAX_QUEUE, so x max_queue recovers packets).
            neigh_mean_occ = float(np.mean(occs))
            neigh_buffered = float(np.sum(occs)) * nc['max_queue']
        else:
            neigh_mean_occ, neigh_buffered = 0.0, 0.0
    else:
        neigh_mean_occ = float(network_mean_occ)
        neigh_buffered = float(n_inflight)

    qf = np.array([
        max(ttl_ref - pkt.hops, 0.0) / ttl_ref,
        min(dist_cd / max(diag, 1e-6), 1.0),
        G.nodes[c].get('queue_occupancy', 0.0),
        min(neigh_buffered / 100.0, 1.0),
        neigh_mean_occ,
        min(h_cur / nc['hop_cap'], 1.0),
    ], dtype=np.float32)

    K = len(candidates)
    cf = np.zeros((K, len(CANDIDATE_FEATURES)), dtype=np.float32)
    for j, u in enumerate(candidates):
        upos = np.array([G.nodes[u]['x'], G.nodes[u]['y'], G.nodes[u]['z']])
        dist_ud = float(np.linalg.norm(dpos - upos))
        cf[j] = (
            np.clip((dist_cd - dist_ud) / max(dist_cd, 1.0), -1.0, 1.0),
            min(h_map.get(u, nc['hop_cap']) / nc['hop_cap'], 1.0),
            1.0 if u == dst else 0.0,
        )
    return qf, cf
