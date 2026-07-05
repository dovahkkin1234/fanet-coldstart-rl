"""
rl_env_full.py  — Approach 1: single-packet trajectory episodes.

WHY THIS EXISTS (flaws fixed vs the multi-packet version):
  Flaw 1 — cross-packet credit assignment: old env forwarded one packet then
           returned a DIFFERENT packet's state as next_obs, so the DQN
           bootstrapped packet A's value from packet B's state. Corrupt.
  Flaw 2 — episode-level `done`: delivery/drop (the real terminals) were stored
           with done=False, so terminal rewards got a spurious gamma*maxQ(next)
           added. The agent never learned delivery is terminal.
  Flaw 3 — action/execution mismatch: when the agent picked a visited node the
           old step() silently executed a different hop but stored the agent's
           original action. The buffer lied about what was done.

APPROACH 1 FIX:
  One episode = one packet's journey (src -> delivery/drop). Only one packet
  exists at a time, so every transition (s,a,r,s',done) belongs to the same
  packet by construction. `done` is True exactly when the packet is delivered
  or dropped. Visited neighbors are masked OUT of the action space, so the
  agent can never pick them (action == execution; loops impossible). Drones
  still move dt between hops, so link quality / lifetime / topology_change_rate
  stay meaningful within a trajectory.

  TRADE-OFF (documented; Approach 2 restores it): with one packet in flight,
  queues never contend, so *_queue_occupancy features are ~0 during RL. The
  warm-start weights already encode how to use them; Approach 2 (multi-flow)
  brings back queue contention for the realistic-application results.

State:  22 base features (+ k regime one-hot when regime_model is passed).
Action: index into the current UNVISITED-neighbor list (masked).
Reward: composite — delivery + progress + link quality - delay - drop - loop.
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

TTL            = 20
MAX_QUEUE      = 50
INITIAL_ENERGY = 100.0
HOP_DELAY_MS   = 2.0
MAX_NEIGHBORS  = 15

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
    __slots__ = ('pid','src','dst','gen_time','current',
                 'path','hops','energy_used','delivered','dropped',
                 'drop_reason','cum_delay_ms')
    def __init__(self, pid, src, dst, gen_time):
        self.pid=pid; self.src=src; self.dst=dst
        self.gen_time=gen_time; self.current=src; self.path=[src]
        self.hops=0; self.energy_used=0.0; self.delivered=False
        self.dropped=False; self.drop_reason=None; self.cum_delay_ms=0.0


class FANETRoutingEnv(gym.Env):
    """One episode = one packet's journey. See module docstring."""
    metadata = {'render_modes': []}

    def __init__(self, scenario_cfg, seed=42, max_neighbors=MAX_NEIGHBORS,
                 regime_model=None):
        super().__init__()
        self.cfg = scenario_cfg
        self.max_neighbors = max_neighbors
        self.regime_model  = regime_model
        self.use_regime    = regime_model is not None
        self.n_regime      = regime_model['k'] if self.use_regime else 0
        self.feature_dim   = len(FEATURE_COLS) + self.n_regime

        self.observation_space = spaces.Dict({
            'features': spaces.Box(low=-np.inf, high=np.inf,
                                    shape=(max_neighbors, self.feature_dim),
                                    dtype=np.float32),
            'mask':     spaces.Box(low=0, high=1, shape=(max_neighbors,),
                                    dtype=np.int8),
        })
        self.action_space = spaces.Discrete(max_neighbors)

        self.base_seed = seed
        self._episode_seed = seed
        self._init_state()

    # ─── initialisation ───────────────────────────────────────────────────────

    def _init_state(self):
        N        = self.cfg['num_drones']
        self.N   = N
        self.dt  = self.cfg['timestep']
        self.max_traj_time = self.cfg.get('duration', 60)
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

        self.prev_links = set()
        self.packet = None
        self.done   = False

        self.cached_graph = None
        self.cached_topo  = None
        self.cached_comp  = None
        self.cached_valid_neighbors = []
        self.cached_full_degree = 0

        self.episode_delivered = 0
        self.episode_dropped   = 0
        self.episode_regime_counts = {}

    # ─── graph builder (correct topology_change_rate via prev_links) ───────────

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

        if self.use_regime:
            largest_cc = max((len(c) for c in comps), default=0)
            largest_component_ratio = largest_cc / self.N if self.N > 0 else 0.0
            if n_edges > 0:
                avg_link_lifetime = float(np.mean(
                    [G.edges[e]['estimated_link_lifetime'] for e in G.edges]))
                avg_relative_velocity = float(np.mean(
                    [G.edges[e]['relative_velocity'] for e in G.edges]))
            else:
                avg_link_lifetime = 0.0
                avg_relative_velocity = 0.0
            regime_features = {
                'avg_degree':               topo['avg_degree'],
                'network_density':          topo['network_density'],
                'num_connected_components': topo['num_components'],
                'largest_component_ratio':  largest_component_ratio,
                'avg_link_quality':         topo['avg_link_quality'],
                'avg_link_lifetime':        avg_link_lifetime,
                'avg_relative_velocity':    avg_relative_velocity,
                'avg_residual_energy':      float(np.mean(self.energy)),
                'min_residual_energy':      float(np.min(self.energy)),
            }
            ordered_feats = self.regime_model['features']
            # float64 required — sklearn KMeans/StandardScaler fit with float64.
            vec = np.array([[regime_features[f] for f in ordered_feats]], dtype=np.float64)
            vec_scaled = self.regime_model['scaler'].transform(vec)
            regime_id = int(self.regime_model['kmeans'].predict(vec_scaled)[0])
            topo['regime_id'] = regime_id
            self.episode_regime_counts[regime_id] = \
                self.episode_regime_counts.get(regime_id, 0) + 1
        else:
            topo['regime_id'] = -1

        comp_labels = {n: ci for ci, c in enumerate(comps) for n in c}
        return G, topo, comp_labels

    # ─── decision-state refresh ────────────────────────────────────────────────

    def _refresh_decision_state(self):
        """Rebuild graph for current time; compute UNVISITED neighbors of the
        packet's current node. Returns True if a decision can be made."""
        self.cached_graph, self.cached_topo, self.cached_comp = self._build_graph()
        pkt = self.packet
        all_nbrs = list(self.cached_graph.neighbors(pkt.current))
        self.cached_full_degree = len(all_nbrs)
        valid = [n for n in all_nbrs if n not in pkt.path]
        self.cached_valid_neighbors = valid[:self.max_neighbors]
        return len(self.cached_valid_neighbors) > 0

    def _advance_mobility(self):
        for d in self.drones:
            d.step(self.dt)
            self.energy[d.id] = EnergyModel.consume_idle(self.energy[d.id], self.dt)
        self.t += self.dt

    # ─── Gym API ───────────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        self._episode_seed = seed if seed is not None else self.base_seed
        self._init_state()

        # Pick a src that currently has a neighbor. If the snapshot is edgeless,
        # advance drones a few dt so links can form, then retry. Guarantees a
        # non-trivial starting state for the packet.
        chosen = False
        for _ in range(20):
            G, _, _ = self._build_graph()
            candidates = [n for n in G.nodes if G.degree(n) > 0]
            if candidates:
                src = int(self.rng.choice(candidates))
                dst = int(self.rng.choice([n for n in range(self.N) if n != src]))
                self.packet = Packet(0, src, dst, self.t)
                if self._refresh_decision_state():
                    chosen = True
                    break
            self._advance_mobility()
            self.prev_links = set()

        if not chosen:
            src = int(self.rng.integers(0, self.N))
            dst = int((src + 1) % self.N)
            self.packet = Packet(0, src, dst, self.t)
            self._refresh_decision_state()

        self.done = False
        return self._build_observation(), self._info()

    def step(self, action):
        pkt = self.packet
        if self.done or pkt is None:
            return self._build_observation(), 0.0, True, False, self._info()

        valid = self.cached_valid_neighbors
        if len(valid) == 0:
            pkt.dropped = True; pkt.drop_reason = 'dead_end'
            reward = self._compute_reward(pkt, None, 0.0, 0.0, dropped=True)
            return self._finish(reward)

        if action >= len(valid):
            action = 0   # defensive clamp; correct masking makes this unreachable
        next_hop = valid[action]

        reward, outcome = self._forward_one_hop(pkt, next_hop)  # executes agent's choice

        if outcome in ('delivered', 'dropped'):
            return self._finish(reward)

        # Advance mobility so topology evolves between hops.
        self._advance_mobility()

        if pkt.hops >= TTL:
            pkt.dropped = True; pkt.drop_reason = 'ttl_expired'
            return self._finish(reward)
        if self.t >= self.max_traj_time:
            pkt.dropped = True; pkt.drop_reason = 'time_expired'
            return self._finish(reward)

        if not self._refresh_decision_state():
            pkt.dropped = True; pkt.drop_reason = 'dead_end'
            return self._finish(reward)

        return self._build_observation(), float(reward), False, False, self._info()

    def _finish(self, reward):
        pkt = self.packet
        if pkt.delivered:
            self.episode_delivered = 1
        else:
            self.episode_dropped = 1
        self.done = True
        return self._build_observation(), float(reward), True, False, self._info()

    # ─── one-hop forward ───────────────────────────────────────────────────────

    def _forward_one_hop(self, pkt, next_hop):
        G = self.cached_graph
        c, dst = pkt.current, pkt.dst

        if next_hop is None or not G.has_edge(c, next_hop):
            pkt.dropped = True; pkt.drop_reason = 'no_route'
            return self._compute_reward(pkt, None, 0.0, 0.0, dropped=True), 'dropped'

        dest_pos = np.array([G.nodes[dst]['x'], G.nodes[dst]['y'], G.nodes[dst]['z']])
        d_before = float(np.linalg.norm(np.array(
            [G.nodes[c]['x'], G.nodes[c]['y'], G.nodes[c]['z']]) - dest_pos))
        d_after = float(np.linalg.norm(np.array(
            [G.nodes[next_hop]['x'], G.nodes[next_hop]['y'], G.nodes[next_hop]['z']]) - dest_pos))

        if not EnergyModel.is_alive(self.energy[c]):
            pkt.dropped = True; pkt.drop_reason = 'energy_depleted'
            return self._compute_reward(pkt, next_hop, d_before, d_after, dropped=True), 'dropped'

        per = G.edges[c, next_hop].get('packet_error_rate', 0)
        if self.rng.random() < per:
            pkt.dropped = True; pkt.drop_reason = 'packet_error'
            self.energy[c] = EnergyModel.consume_tx(self.energy[c])
            return self._compute_reward(pkt, next_hop, d_before, d_after, dropped=True), 'dropped'

        loop = next_hop in pkt.path   # impossible with masking; kept for reward parity

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
        feats = np.zeros((self.max_neighbors, self.feature_dim), dtype=np.float32)
        mask  = np.zeros(self.max_neighbors, dtype=np.int8)

        pkt = self.packet
        if pkt is None or self.cached_graph is None or self.done:
            return {'features': feats, 'mask': mask}

        G    = self.cached_graph
        c, dst = pkt.current, pkt.dst
        nbrs = self.cached_valid_neighbors

        c_node = G.nodes[c]; d_node = G.nodes[dst]
        c_pos  = np.array([c_node['x'], c_node['y'], c_node['z']])
        d_pos  = np.array([d_node['x'], d_node['y'], d_node['z']])
        dist_to_dest = float(np.linalg.norm(d_pos - c_pos))
        bearing      = float(np.arctan2(d_pos[1]-c_pos[1], d_pos[0]-c_pos[0]))
        dest_reach   = int(self.cached_comp.get(c,-1) == self.cached_comp.get(dst,-2))
        topo = self.cached_topo

        regime_onehot = None
        if self.use_regime:
            regime_onehot = np.zeros(self.n_regime, dtype=np.float32)
            rid = topo.get('regime_id', -1)
            if 0 <= rid < self.n_regime:
                regime_onehot[rid] = 1.0

        for i, n in enumerate(nbrs):
            e = G.edges[c, n]
            n_pos = np.array([G.nodes[n]['x'], G.nodes[n]['y'], G.nodes[n]['z']])
            dist_n   = float(np.linalg.norm(d_pos - n_pos))
            progress = (dist_to_dest - dist_n) / max(dist_to_dest, 1e-6)
            base = [
                e.get('distance', 0), e.get('relative_velocity', 0),
                e.get('link_quality', 0), e.get('rssi', -90),
                e.get('packet_error_rate', 1), e.get('estimated_link_lifetime', 0),
                self.energy[n], self.queues[n].occupancy,
                dist_n, progress,
                self.energy[c], self.queues[c].occupancy, self.cached_full_degree,
                c_node.get('speed', 0),
                dist_to_dest, bearing, dest_reach,
                topo['avg_degree'], topo['network_density'],
                topo['num_components'], topo['avg_link_quality'],
                topo['topology_change_rate'],
            ]
            if self.use_regime:
                feats[i] = np.concatenate([base, regime_onehot])
            else:
                feats[i] = base
            mask[i] = 1

        return {'features': feats, 'mask': mask}

    def _info(self):
        delivered = self.episode_delivered
        total     = delivered + self.episode_dropped
        pkt = self.packet
        return {
            'pdr': float(delivered) if total > 0 else 0.0,
            'delivered': delivered,
            'dropped': self.episode_dropped,
            'hops': pkt.hops if pkt else 0,
            'drop_reason': pkt.drop_reason if pkt else None,
            'mean_delay_ms': pkt.cum_delay_ms if (pkt and pkt.delivered) else 0.0,
            'mean_hops': pkt.hops if (pkt and pkt.delivered) else 0,
            't': self.t,
            'regime_counts': dict(self.episode_regime_counts),
        }


# ─── Scenario configs — FULL SCALE (matches configs/rwp_full_configs.yaml) ────

SHARED_DEFAULTS = dict(z_min=50, z_max=150, timestep=0.5, packet_rate=1.0)

TRAINING_SCENARIOS = {
    'rwp_sc01': dict(num_drones=10, area_x=500,  area_y=500,
                     speed_min=5,  speed_max=15, comm_range=180, pause_max=5.0),
    'rwp_sc02': dict(num_drones=10, area_x=500,  area_y=500,
                     speed_min=25, speed_max=40, comm_range=200, pause_max=3.0),
    'rwp_sc03': dict(num_drones=20, area_x=1000, area_y=1000,
                     speed_min=5,  speed_max=15, comm_range=250, pause_max=5.0),
    'rwp_sc04': dict(num_drones=20, area_x=1000, area_y=1000,
                     speed_min=25, speed_max=40, comm_range=250, pause_max=3.0),
    'rwp_sc07': dict(num_drones=40, area_x=1500, area_y=1500,
                     speed_min=10, speed_max=25, comm_range=250, pause_max=5.0),
    'rwp_sc08': dict(num_drones=15, area_x=1500, area_y=1500,
                     speed_min=5,  speed_max=15, comm_range=250, pause_max=5.0),
    'rwp_sc10': dict(num_drones=25, area_x=800,  area_y=800,
                     speed_min=35, speed_max=50, comm_range=200, pause_max=2.0),
    'rwp_sc12': dict(num_drones=40, area_x=2000, area_y=2000,
                     speed_min=20, speed_max=35, comm_range=300, pause_max=3.0),
    'rwp_sc15': dict(num_drones=25, area_x=1200, area_y=1200,
                     speed_min=5,  speed_max=50, comm_range=250, pause_max=5.0),
}

EVAL_SCENARIOS = {
    'rwp_sc03': dict(num_drones=20, area_x=1000, area_y=1000,
                     speed_min=5,  speed_max=15, comm_range=250, pause_max=5.0),
    'rwp_sc10': dict(num_drones=25, area_x=800,  area_y=800,
                     speed_min=35, speed_max=50, comm_range=200, pause_max=2.0),
    'rwp_sc07': dict(num_drones=40, area_x=1500, area_y=1500,
                     speed_min=10, speed_max=25, comm_range=250, pause_max=5.0),
}

GENERALIZATION_SCENARIOS = {
    'rwp_sc06': dict(num_drones=30, area_x=1500, area_y=1500,
                     speed_min=25, speed_max=40, comm_range=250, pause_max=3.0),
    'rwp_sc11': dict(num_drones=20, area_x=2000, area_y=2000,
                     speed_min=5,  speed_max=20, comm_range=300, pause_max=5.0),
    'rwp_sc14': dict(num_drones=12, area_x=2000, area_y=2000,
                     speed_min=5,  speed_max=15, comm_range=350, pause_max=8.0),
}


def make_env(scenario_name, scenario_dict=None, duration=60, seed=42, **kwargs):
    if scenario_dict is None:
        if scenario_name in TRAINING_SCENARIOS:
            scenario_dict = TRAINING_SCENARIOS[scenario_name]
        elif scenario_name in EVAL_SCENARIOS:
            scenario_dict = EVAL_SCENARIOS[scenario_name]
        elif scenario_name in GENERALIZATION_SCENARIOS:
            scenario_dict = GENERALIZATION_SCENARIOS[scenario_name]
        else:
            raise ValueError(f"Unknown scenario: {scenario_name}")
    cfg = {**SHARED_DEFAULTS, **scenario_dict, 'duration': duration}
    return FANETRoutingEnv(cfg, seed=seed, **kwargs)
