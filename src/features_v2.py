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
    'snr',
    'estimated_link_lifetime',
    'relative_velocity',
]

QUERY_FEATURES = [
    'ttl_left',
    'hops_so_far',
    'dist_to_dest',
    'current_queue_occupancy',
    'n_inflight',
    'network_mean_occupancy',
    'hop_distance_to_dst',
]

CANDIDATE_FEATURES = [
    'progress',            # normalised geographic progress toward destination
    'cand_hop_distance',   # candidate's BFS hop distance to destination
    'is_destination',      # candidate IS the destination
]

# Physical constants used for normalisation (mirrors simulator_v2 / models.py).
MAX_QUEUE_REF = 50.0
INITIAL_ENERGY_REF = 100.0
TTL_REF = 20.0
SNR_REF = 40.0          # dB, generous ceiling for scaling
LIFETIME_REF = 60.0     # seconds, matches estimate_link_lifetime's max
HOP_CAP = 10.0          # hop distances clipped/normalised against this


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
        'snr_ref': SNR_REF,
        'lifetime_ref': LIFETIME_REF,
        'hop_cap': HOP_CAP,
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
            G.degree(n) / max(N - 1, 1),
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
            np.clip(e.get('snr', 0.0) / nc['snr_ref'], -1.0, 2.0),
            min(e.get('estimated_link_lifetime', 0.0), nc['lifetime_ref']) / nc['lifetime_ref'],
            min(e.get('relative_velocity', 0.0) / max(2.0 * smax, 1e-6), 1.0),
        )
    return np.asarray(node_ids, dtype=np.int32), nf, ei, ef


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
    qf = np.array([
        max(ttl_ref - pkt.hops, 0.0) / ttl_ref,
        min(pkt.hops / ttl_ref, 1.0),
        min(dist_cd / max(diag, 1e-6), 1.0),
        G.nodes[c].get('queue_occupancy', 0.0),
        min(n_inflight / 100.0, 1.0),
        network_mean_occ,
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
