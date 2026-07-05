"""
rl_env.py
Gymnasium-compatible environment that wraps the FANET simulator.

Each "step" is one forwarding decision (one packet at one node choosing one neighbor).
The episode ends when the simulated time reaches the duration limit.

State (22 features): same as the MLP warm-start input
Action: index into the current neighbor list (0..MAX_NEIGHBORS-1, masked)
Reward: composite reward from simulator (delivery + progress + link quality
        - delay - drop - loop penalty, etc.)

Critical design: the environment scores each candidate neighbor independently
(same as the MLP). The agent picks the highest-scoring candidate. This makes
the MLP weights directly transferable to the Q-network.
"""

import os, sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from itertools import combinations
import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mobility import DroneRWP
from link_model import compute_link_features, estimate_link_lifetime, RSSI_SENSITIVITY
from models import EnergyModel, NodeQueue
from routing_teachers import dijkstra_next_hop

TTL            = 20
MAX_QUEUE      = 50
INITIAL_ENERGY = 100.0
HOP_DELAY_MS   = 2.0
MAX_NEIGHBORS  = 15   # bound for action space — larger than any realistic case

# Feature order must EXACTLY match train_mlp.py FEATURE_COLS
FEATURE_COLS = [
    'candidate_distance', 'candidate_relative_velocity', 'candidate_link_quality',
    'candidate_rssi', 'candidate_packet_error_rate', 'candidate_estimated_lifetime',
    'candidate_energy', 'candidate_queue_occupancy', 'candidate_dist_to_dest',
    'candidate_progress', 'current_energy', 'current_queue_occupancy',
    'current_num_neighbors', 'current_speed', 'distance_to_destination',
    'bearing_to_destination', 'destination_reachable', 'avg_degree',
    'network_density', 'num_components', 'avg_link_quality', 'topology_change_rate',
]


class Packet:
    __slots__ = ('pid','flow_id','src','dst','gen_time','current',
                 'path','hops','energy_used','delivered','dropped',
                 'drop_reason','cum_delay_ms')
    def __init__(self, pid, flow_id, src, dst, gen_time):
        self.pid=pid; self.flow_id=flow_id; self.src=src; self.dst=dst
        self.gen_time=gen_time; self.current=src; self.path=[src]
        self.hops=0; self.energy_used=0.0; self.delivered=False
        self.dropped=False; self.drop_reason=None; self.cum_delay_ms=0.0


class FANETRoutingEnv(gym.Env):
    """
    One env step = one forwarding decision for one packet.
    One episode  = one full FANET scenario simulation.
    """
    metadata = {'render_modes': []}

    def __init__(self, scenario_cfg, seed=42, max_neighbors=MAX_NEIGHBORS,
                 regime_model=None):
        super().__init__()
        self.cfg = scenario_cfg
        self.max_neighbors = max_neighbors
        self.regime_model  = regime_model    # optional: KMeans + scaler bundle

        # Observation = MAX_NEIGHBORS rows × 22 features (one per candidate)
        # plus a binary mask of length MAX_NEIGHBORS.
        self.observation_space = spaces.Dict({
            'features': spaces.Box(low=-np.inf, high=np.inf,
                                    shape=(max_neighbors, len(FEATURE_COLS)),
                                    dtype=np.float32),
            'mask':     spaces.Box(low=0, high=1, shape=(max_neighbors,),
                                    dtype=np.int8),
        })
        self.action_space = spaces.Discrete(max_neighbors)

        self.base_seed = seed
        self._episode_seed = seed
        self._init_state()

    # ─── initialisation helpers ───────────────────────────────────────────────

    def _init_state(self):
        """(re-)initialise everything at the start of an episode."""
        N        = self.cfg['num_drones']
        self.N   = N
        self.dt  = self.cfg['timestep']
        self.duration  = self.cfg['duration']
        self.comm_range = self.cfg['comm_range']
        self.t = 0.0

        self.drones  = [
            DroneRWP(i, self.cfg['area_x'], self.cfg['area_y'],
                     self.cfg['z_min'], self.cfg['z_max'],
                     self.cfg['speed_min'], self.cfg['speed_max'],
                     self.cfg['pause_max'], seed=self._episode_seed)
            for i in range(N)
        ]
        self.energy  = [INITIAL_ENERGY] * N
        self.queues  = [NodeQueue(MAX_QUEUE) for _ in range(N)]
        self.rng     = np.random.default_rng(self._episode_seed)
        self.num_flows = max(1, N // 4)
        self.flows   = []
        for fid in range(self.num_flows):
            src, dst = self.rng.choice(N, size=2, replace=False)
            self.flows.append({'flow_id': fid, 'src': int(src), 'dst': int(dst)})

        self.prev_links = set()
        self.active_packets = []
        self.finished      = []
        self.packet_counter = 0
        self.time_since_gen = {f['flow_id']: 0.0 for f in self.flows}
        self.current_packet = None
        self.cached_graph   = None
        self.cached_topo    = None
        self.cached_comp    = None
        self.cached_neighbors = []
        # per-episode metrics
        self.episode_delivered = 0
        self.episode_dropped   = 0
        self.episode_total_delay = 0.0
        self.episode_total_hops  = 0

    # ─── Gym API ──────────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._episode_seed = seed
        else:
            self._episode_seed = self.base_seed
        self._init_state()
        self._advance_until_decision()
        return self._build_observation(), self._info()

    def step(self, action):
        """Apply action (index into neighbor list), advance one forwarding step."""
        pkt = self.current_packet
        if pkt is None:
            # Episode done — no decision pending
            return self._build_observation(), 0.0, True, False, self._info()

        c, dst = pkt.current, pkt.dst
        nbrs   = self.cached_neighbors

        # Translate masked action to actual neighbor; fallback to Dijkstra if invalid
        if action < len(nbrs):
            next_hop = nbrs[action]
        else:
            next_hop = None

        # Loop guard: refuse already-visited neighbors
        if next_hop is None or next_hop in pkt.path:
            unvisited = [n for n in nbrs if n not in pkt.path]
            if unvisited:
                next_hop = unvisited[0]
            else:
                next_hop = dijkstra_next_hop(self.cached_graph, c, dst)

        reward, hop_outcome = self._forward_one_hop(pkt, next_hop)

        if hop_outcome in ('delivered', 'dropped'):
            self.finished.append(pkt)
            self.active_packets.remove(pkt)
            if pkt.delivered:
                self.episode_delivered += 1
                self.episode_total_delay += pkt.cum_delay_ms
                self.episode_total_hops  += pkt.hops
            else:
                self.episode_dropped += 1

        # Advance time + active packets until we hit the next forwarding decision
        # or the episode ends
        done = self._advance_until_decision()

        return self._build_observation(), float(reward), done, False, self._info()

    # ─── simulator core ───────────────────────────────────────────────────────

    def _build_graph(self):
        G = nx.Graph()
        for d in self.drones:
            G.add_node(d.id, x=d.x, y=d.y, z=d.z,
                       vx=d.vx, vy=d.vy, vz=d.vz, speed=d.speed,
                       energy=self.energy[d.id])
        cur_links = set()
        for i, j in combinations(range(self.N), 2):
            di, dj = self.drones[i], self.drones[j]
            dist   = float(np.linalg.norm(di.pos - dj.pos))
            if dist > self.comm_range:
                continue
            rssi, snr, lq, per = compute_link_features(dist)
            if rssi <= RSSI_SENSITIVITY:
                continue
            lt = estimate_link_lifetime(di.pos, dj.pos, di.vel, dj.vel, self.comm_range)
            rv = float(np.linalg.norm(di.vel - dj.vel))
            G.add_edge(i, j, distance=dist, relative_velocity=rv,
                       link_quality=lq, estimated_link_lifetime=lt,
                       rssi=rssi, snr=snr, packet_error_rate=per)
            cur_links.add((i, j))

        # topology change rate (fraction of links changed since last build)
        sym_diff = len(self.prev_links.symmetric_difference(cur_links))
        union    = max(len(self.prev_links) + len(cur_links), 1)
        change_rate = sym_diff / union
        self.prev_links = cur_links

        n_edges = G.number_of_edges()
        max_e   = self.N * (self.N - 1) / 2
        comps   = list(nx.connected_components(G))
        degrees = [d for _, d in G.degree()]
        topo = {
            'avg_degree':         float(np.mean(degrees)) if degrees else 0.0,
            'network_density':    n_edges / max_e if max_e > 0 else 0.0,
            'num_components':     len(comps),
            'avg_link_quality':   float(np.mean([G.edges[e]['link_quality'] for e in G.edges])) if n_edges else 0.0,
            'topology_change_rate': change_rate,
        }
        comp_labels = {n: ci for ci, c in enumerate(comps) for n in c}
        return G, topo, comp_labels

    def _advance_until_decision(self):
        """
        Move time forward, generate packets, until we find a packet that needs
        a forwarding decision. Return True if the episode is over.
        """
        while True:
            # Find packet at front of active queue that needs a decision
            pkt = self._next_packet_needing_decision()
            if pkt is not None:
                self.current_packet = pkt
                # Refresh graph snapshot and neighbor list
                self.cached_graph, self.cached_topo, self.cached_comp = self._build_graph()
                self.cached_neighbors = list(self.cached_graph.neighbors(pkt.current))
                return False

            # No pending decision — advance time
            if self.t >= self.duration:
                # Episode end: drop remaining packets
                for p in self.active_packets:
                    p.dropped = True
                    p.drop_reason = 'episode_end'
                    self.finished.append(p)
                    self.episode_dropped += 1
                self.active_packets = []
                self.current_packet = None
                return True

            # Step physics
            for d in self.drones:
                d.step(self.dt)
                self.energy[d.id] = EnergyModel.consume_idle(self.energy[d.id], self.dt)

            # Generate packets from flows
            for f in self.flows:
                self.time_since_gen[f['flow_id']] += self.dt
                interval = 1.0 / self.cfg['packet_rate']
                while self.time_since_gen[f['flow_id']] >= interval:
                    self.time_since_gen[f['flow_id']] -= interval
                    pkt = Packet(self.packet_counter, f['flow_id'],
                                 f['src'], f['dst'], self.t)
                    self.packet_counter += 1
                    self.active_packets.append(pkt)

            self.t += self.dt

    def _next_packet_needing_decision(self):
        """Returns the next active packet that has neighbors and isn't done."""
        # Rebuild graph just-in-time to check neighbor counts
        if not self.active_packets:
            return None
        G, _, _ = self._build_graph()
        for pkt in self.active_packets:
            if pkt.current == pkt.dst:
                pkt.delivered = True
                continue
            if pkt.hops >= TTL:
                pkt.dropped = True
                pkt.drop_reason = 'ttl_expired'
                continue
            nbrs = list(G.neighbors(pkt.current))
            if not nbrs:
                pkt.dropped = True
                pkt.drop_reason = 'no_route'
                continue
            return pkt
        # Cleanup: anything marked delivered/dropped above
        for p in list(self.active_packets):
            if p.delivered or p.dropped:
                self.finished.append(p)
                self.active_packets.remove(p)
                if p.delivered:
                    self.episode_delivered += 1
                    self.episode_total_delay += p.cum_delay_ms
                    self.episode_total_hops += p.hops
                else:
                    self.episode_dropped += 1
        return None

    def _forward_one_hop(self, pkt, next_hop):
        """
        Forward `pkt` one hop to `next_hop`. Return (reward, outcome).
        outcome: 'delivered' | 'dropped' | 'continuing'
        """
        G = self.cached_graph
        c, dst = pkt.current, pkt.dst

        if next_hop is None or next_hop not in G.neighbors(c):
            pkt.dropped = True; pkt.drop_reason = 'no_route'
            return self._compute_reward(pkt, None, 0.0, 0.0, dropped=True), 'dropped'

        # Distance bookkeeping for reward
        dest_pos = np.array([G.nodes[dst]['x'], G.nodes[dst]['y'], G.nodes[dst]['z']])
        d_before = float(np.linalg.norm(np.array(
            [G.nodes[c]['x'], G.nodes[c]['y'], G.nodes[c]['z']]) - dest_pos))
        d_after = float(np.linalg.norm(np.array(
            [G.nodes[next_hop]['x'], G.nodes[next_hop]['y'], G.nodes[next_hop]['z']]) - dest_pos))

        # Energy check
        if not EnergyModel.is_alive(self.energy[c]):
            pkt.dropped = True; pkt.drop_reason = 'energy_depleted'
            return self._compute_reward(pkt, next_hop, d_before, d_after, dropped=True), 'dropped'

        # Packet-error sim
        per = G.edges[c, next_hop].get('packet_error_rate', 0)
        if self.rng.random() < per:
            pkt.dropped = True; pkt.drop_reason = 'packet_error'
            self.energy[c] = EnergyModel.consume_tx(self.energy[c])
            return self._compute_reward(pkt, next_hop, d_before, d_after, dropped=True), 'dropped'

        loop = next_hop in pkt.path

        # Apply forwarding
        self.energy[c]        = EnergyModel.consume_tx(self.energy[c])
        self.energy[next_hop] = EnergyModel.consume_rx(self.energy[next_hop])
        pkt.energy_used += EnergyModel.TX_COST + EnergyModel.RX_COST
        pkt.current = next_hop
        pkt.path.append(next_hop)
        pkt.hops += 1
        pkt.cum_delay_ms += HOP_DELAY_MS

        if pkt.current == dst:
            pkt.delivered = True
            return self._compute_reward(pkt, next_hop, d_before, d_after,
                                         delivered=True, loop=loop), 'delivered'
        if pkt.hops >= TTL:
            pkt.dropped = True; pkt.drop_reason = 'ttl_expired'
            return self._compute_reward(pkt, next_hop, d_before, d_after,
                                         dropped=True, loop=loop), 'dropped'

        return self._compute_reward(pkt, next_hop, d_before, d_after, loop=loop), 'continuing'

    def _compute_reward(self, pkt, next_hop, d_before, d_after,
                        delivered=False, dropped=False, loop=False):
        G = self.cached_graph
        r_del   = 10.0 if delivered else 0.0
        r_prog  = 1.0 * (d_before - d_after) / max(d_before, 1.0) if not dropped else 0.0
        lq = lt = 0.0
        if next_hop is not None and G is not None and G.has_edge(
                pkt.path[-2] if len(pkt.path) >= 2 else pkt.current, next_hop):
            e = G.edges[pkt.path[-2] if len(pkt.path)>=2 else pkt.current, next_hop]
            lq, lt = e.get('link_quality', 0), e.get('estimated_link_lifetime', 0)
        r_lq    = 0.3 * lq
        r_lt    = 0.2 * min(lt / 30.0, 1.0)
        p_delay = -0.1 * self.dt
        p_eng   = -0.05 * (EnergyModel.TX_COST + EnergyModel.RX_COST)
        p_q     = -0.2 * (self.queues[next_hop].occupancy if next_hop is not None else 0)
        p_drop  = -10.0 if dropped else 0.0
        p_loop  = -2.0 if loop else 0.0
        return r_del + r_prog + r_lq + r_lt + p_delay + p_eng + p_q + p_drop + p_loop

    # ─── observation builder ──────────────────────────────────────────────────

    def _build_observation(self):
        feats = np.zeros((self.max_neighbors, len(FEATURE_COLS)), dtype=np.float32)
        mask  = np.zeros(self.max_neighbors, dtype=np.int8)

        pkt = self.current_packet
        if pkt is None or self.cached_graph is None:
            return {'features': feats, 'mask': mask}

        G    = self.cached_graph
        c, dst = pkt.current, pkt.dst
        nbrs = self.cached_neighbors[:self.max_neighbors]

        c_node = G.nodes[c]; d_node = G.nodes[dst]
        c_pos  = np.array([c_node['x'], c_node['y'], c_node['z']])
        d_pos  = np.array([d_node['x'], d_node['y'], d_node['z']])
        dist_to_dest = float(np.linalg.norm(d_pos - c_pos))
        bearing      = float(np.arctan2(d_pos[1]-c_pos[1], d_pos[0]-c_pos[0]))
        dest_reach   = int(self.cached_comp.get(c,-1) == self.cached_comp.get(dst,-2))
        topo = self.cached_topo

        for i, n in enumerate(nbrs):
            e = G.edges[c, n]
            n_pos = np.array([G.nodes[n]['x'], G.nodes[n]['y'], G.nodes[n]['z']])
            dist_n   = float(np.linalg.norm(d_pos - n_pos))
            progress = (dist_to_dest - dist_n) / max(dist_to_dest, 1e-6)
            feats[i] = [
                e.get('distance', 0), e.get('relative_velocity', 0),
                e.get('link_quality', 0), e.get('rssi', -90),
                e.get('packet_error_rate', 1), e.get('estimated_link_lifetime', 0),
                self.energy[n], self.queues[n].occupancy,
                dist_n, progress,
                self.energy[c], self.queues[c].occupancy, len(nbrs),
                c_node.get('speed', 0),
                dist_to_dest, bearing, dest_reach,
                topo['avg_degree'], topo['network_density'],
                topo['num_components'], topo['avg_link_quality'],
                topo['topology_change_rate'],
            ]
            mask[i] = 1

        return {'features': feats, 'mask': mask}

    def _info(self):
        delivered = self.episode_delivered
        total     = delivered + self.episode_dropped
        return {
            'pdr': delivered / max(total, 1),
            'delivered': delivered, 'dropped': self.episode_dropped,
            'mean_delay_ms': self.episode_total_delay / max(delivered, 1),
            'mean_hops': self.episode_total_hops / max(delivered, 1),
            't': self.t,
        }


# ─── Scenario configs (mirror configs/pilot_configs.yaml) ────────────────────

TRAINING_SCENARIOS = {
    'easy_dense':   dict(num_drones=10, area_x=500,  area_y=500,
                         speed_min=5,  speed_max=15, comm_range=200),
    'medium':       dict(num_drones=20, area_x=1000, area_y=1000,
                         speed_min=10, speed_max=25, comm_range=250),
    'hard_sparse':  dict(num_drones=25, area_x=1500, area_y=1500,
                         speed_min=15, speed_max=35, comm_range=250),
}

EVAL_SCENARIOS = {
    'eval_easy':   dict(num_drones=10, area_x=500,  area_y=500,
                        speed_min=5,  speed_max=15, comm_range=200),
    'eval_medium': dict(num_drones=20, area_x=1000, area_y=1000,
                        speed_min=10, speed_max=25, comm_range=250),
    'eval_hard':   dict(num_drones=30, area_x=1500, area_y=1500,
                        speed_min=20, speed_max=40, comm_range=250),
    'eval_sparse': dict(num_drones=25, area_x=2000, area_y=2000,
                        speed_min=5,  speed_max=15, comm_range=300),
}

SHARED_DEFAULTS = dict(z_min=50, z_max=150, pause_max=5.0,
                       timestep=0.5, packet_rate=1.0)


def make_env(scenario_name, scenario_dict=None, duration=60, seed=42, **kwargs):
    """Helper: build an env for a named scenario."""
    if scenario_dict is None:
        if scenario_name in TRAINING_SCENARIOS:
            scenario_dict = TRAINING_SCENARIOS[scenario_name]
        elif scenario_name in EVAL_SCENARIOS:
            scenario_dict = EVAL_SCENARIOS[scenario_name]
        else:
            raise ValueError(f"Unknown scenario: {scenario_name}")
    cfg = {**SHARED_DEFAULTS, **scenario_dict, 'duration': duration}
    return FANETRoutingEnv(cfg, seed=seed, **kwargs)
