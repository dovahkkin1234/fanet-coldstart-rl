"""
routing_teachers.py
The five classical routing teachers. Each independently picks a next hop
given the current network graph, the current node, and the destination.

Graph node attributes expected: x, y, z, energy
Graph edge attributes expected: link_quality, estimated_link_lifetime
"""

import numpy as np
import networkx as nx


def _pos(G, n):
    return np.array([G.nodes[n]['x'], G.nodes[n]['y'], G.nodes[n]['z']])


def dijkstra_next_hop(G, current, destination):
    """Shortest path by hop count."""
    try:
        path = nx.shortest_path(G, current, destination)
        return path[1] if len(path) >= 2 else None
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def gpsr_next_hop(G, current, destination):
    """Greedy geographic forwarding: neighbor closest to destination."""
    neighbors = list(G.neighbors(current))
    if not neighbors:
        return None
    dest_pos = _pos(G, destination)
    best, best_d = None, float('inf')
    for n in neighbors:
        d = np.linalg.norm(dest_pos - _pos(G, n))
        if d < best_d:
            best_d, best = d, n
    return best


def stable_path_next_hop(G, current, destination):
    """Weighted score: link quality + lifetime + progress + neighbor energy."""
    neighbors = list(G.neighbors(current))
    if not neighbors:
        return None
    curr_pos = _pos(G, current)
    dest_pos = _pos(G, destination)
    dist_cd = np.linalg.norm(dest_pos - curr_pos)

    best, best_score = None, -float('inf')
    for n in neighbors:
        edge = G.edges[current, n]
        dist_nd = np.linalg.norm(dest_pos - _pos(G, n))
        progress = (dist_cd - dist_nd) / max(dist_cd, 1.0)
        lifetime_norm = min(edge.get('estimated_link_lifetime', 0.0) / 30.0, 1.0)
        energy_norm = G.nodes[n].get('energy', 100.0) / 100.0

        score = (0.35 * edge.get('link_quality', 0.0)
                 + 0.30 * lifetime_norm
                 + 0.20 * max(progress, 0.0)
                 + 0.15 * energy_norm)
        if score > best_score:
            best_score, best = score, n
    return best


class AODVLikeRouter:
    """Reactive routing with a route cache and link-break invalidation."""

    def __init__(self):
        self.route_cache = {}

    def next_hop(self, G, current, destination):
        key = (current, destination)
        if key in self.route_cache:
            path = self.route_cache[key]
            valid = (len(path) >= 2 and
                     all(G.has_edge(path[i], path[i + 1])
                         for i in range(len(path) - 1)))
            if valid:
                return path[1]
            del self.route_cache[key]

        try:
            path = nx.shortest_path(G, current, destination)
            if len(path) >= 2:
                for i in range(len(path)):
                    self.route_cache[(path[i], destination)] = path[i:]
                return path[1]
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass
        return None


def link_lifetime_next_hop(G, current, destination):
    """Dijkstra weighted by inverse link lifetime (prefer durable links)."""
    try:
        for u, v, data in G.edges(data=True):
            lt = data.get('estimated_link_lifetime', 0.1)
            data['lt_weight'] = 1.0 / max(lt, 0.1)
        path = nx.shortest_path(G, current, destination, weight='lt_weight')
        return path[1] if len(path) >= 2 else None
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None
