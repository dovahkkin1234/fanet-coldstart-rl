"""
routing_teachers_v2.py
Congestion-aware teachers for the multi-packet simulator. Same call signature as
routing_teachers.py: f(G, current, destination) -> next_hop id or None.

Graph node attributes expected: x, y, z, energy, queue_occupancy, queue_len
Graph edge attributes expected: link_quality, estimated_link_lifetime, distance

M2 only needs `queue_aware_greedy_next_hop` for the G2 gate. `backpressure_next_hop`
is included as the throughput-optimal reference (used more heavily in M3), but is
lightweight and has no persistent state, so it lives here now.
"""

import numpy as np
import networkx as nx


def _pos(G, n):
    return np.array([G.nodes[n]['x'], G.nodes[n]['y'], G.nodes[n]['z']])


def queue_aware_greedy_next_hop(G, current, destination,
                                w_progress=1.0, w_queue=1.0, w_quality=0.5):
    """Greedy geographic forwarding that steers away from congested neighbors.

    score(n) = w_progress * normalized_progress(n)
             - w_queue    * queue_occupancy(n)
             + w_quality  * link_quality(current, n)

    Picks the neighbor with the highest score. Congestion-aware: a neighbor that
    makes good geographic progress but is badly congested can lose to a slightly
    worse-positioned but empty neighbor. Falls back to pure progress if all else
    ties.
    """
    neighbors = list(G.neighbors(current))
    if not neighbors:
        return None
    dest_pos = _pos(G, destination)
    curr_pos = _pos(G, current)
    dist_cd = float(np.linalg.norm(dest_pos - curr_pos))

    best, best_score = None, -float('inf')
    for n in neighbors:
        n_pos = _pos(G, n)
        dist_nd = float(np.linalg.norm(dest_pos - n_pos))
        progress = (dist_cd - dist_nd) / max(dist_cd, 1.0)
        occ = float(G.nodes[n].get('queue_occupancy', 0.0))
        lq = float(G.edges[current, n].get('link_quality', 0.0))
        score = w_progress * progress - w_queue * occ + w_quality * lq
        if score > best_score:
            best_score, best = score, n
    return best


def backpressure_next_hop(G, current, destination,
                          w_backlog=1.0, w_quality=1.0):
    """Classical backpressure (Tassiulas-Ephremides style, single-commodity
    approximation). Forward toward the neighbor that maximizes the queue-backlog
    differential weighted by link quality:

        weight(current, n) = link_quality(current, n) * (Q_current - Q_n)

    Chooses the neighbor with the largest positive weight. If the destination is
    a direct neighbor, always deliver to it (drain toward the sink). If no
    neighbor has a positive differential, fall back to the destination-ward
    (progress) neighbor so packets don't stall in empty low-load networks.

    Note: this is the throughput-optimal reference for heavy load; under light
    load pure backpressure wanders (no destination bias), which is exactly why
    SP-BP / drift-plus-penalty exist — those come in M3.
    """
    neighbors = list(G.neighbors(current))
    if not neighbors:
        return None
    if destination in neighbors:
        return destination

    q_cur = float(G.nodes[current].get('queue_len', 0.0))
    best, best_w = None, 0.0
    for n in neighbors:
        q_n = float(G.nodes[n].get('queue_len', 0.0))
        lq = float(G.edges[current, n].get('link_quality', 0.0))
        w = lq * (q_cur - q_n)
        if w > best_w:
            best_w, best = w, n

    if best is not None:
        return best

    # fallback: destination-ward progress (avoids stalling at low load)
    dest_pos = _pos(G, destination)
    curr_pos = _pos(G, current)
    dist_cd = float(np.linalg.norm(dest_pos - curr_pos))
    best, best_prog = None, -float('inf')
    for n in neighbors:
        dist_nd = float(np.linalg.norm(dest_pos - _pos(G, n)))
        prog = dist_cd - dist_nd
        if prog > best_prog:
            best_prog, best = prog, n
    return best
