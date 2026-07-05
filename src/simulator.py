"""
simulator.py
Core FANET simulator. Produces the 7 dataset tables for one episode:

  mobility_trace.csv, link_states.csv, topology_features.csv,
  traffic_flows.csv, packet_logs.csv, routing_decisions.csv, rl_transitions.csv

Run one episode via FANETSimulator(config).run().
"""

import os
import itertools
import numpy as np
import pandas as pd
import networkx as nx

from mobility import DroneRWP
from link_model import compute_link_features, estimate_link_lifetime, RSSI_SENSITIVITY
from models import EnergyModel, NodeQueue
from routing_teachers import (
    dijkstra_next_hop, gpsr_next_hop, stable_path_next_hop,
    AODVLikeRouter, link_lifetime_next_hop,
)

TTL = 20
MAX_QUEUE = 50
INITIAL_ENERGY = 100.0
HOP_DELAY_MS = 2.0  # transmission + processing delay per hop (milliseconds)
MAX_HOPS_PER_STEP = 2   # limits multi-hop to allow queue buildup


class Packet:
    __slots__ = ('pid', 'flow_id', 'src', 'dst', 'gen_time', 'current',
                 'path', 'hops', 'energy_used', 'delivered', 'dropped',
                 'drop_reason', 'delivery_time', 'cum_delay_ms')

    def __init__(self, pid, flow_id, src, dst, gen_time):
        self.pid = pid
        self.flow_id = flow_id
        self.src = src
        self.dst = dst
        self.gen_time = gen_time
        self.current = src
        self.path = [src]
        self.hops = 0
        self.energy_used = 0.0
        self.delivered = False
        self.dropped = False
        self.drop_reason = None
        self.delivery_time = None
        self.cum_delay_ms = 0.0
        


class FANETSimulator:
    def __init__(self, config):
        self.cfg = config
        self.scenario_id = config['scenario_id']
        self.episode = config['episode']
        self.seed = config['seed']
        self.N = config['num_drones']
        self.area_x = config['area_x']
        self.area_y = config['area_y']
        self.z_min = config['z_min']
        self.z_max = config['z_max']
        self.comm_range = config['comm_range']
        self.dt = config['timestep']
        self.duration = config['duration']
        self.mobility_model = config.get('mobility_model', 'random_waypoint')
        self.packet_rate = config['packet_rate']

        self.rng = np.random.default_rng(self.seed)

        # Drones
        self.drones = [
            DroneRWP(i, self.area_x, self.area_y, self.z_min, self.z_max,
                     config['speed_min'], config['speed_max'],
                     config['pause_max'], seed=self.seed)
            for i in range(self.N)
        ]

        # Per-node state
        self.energy = [INITIAL_ENERGY] * self.N
        self.queues = [NodeQueue(MAX_QUEUE) for _ in range(self.N)]

        # Traffic flows
        self.num_flows = max(1, self.N // 4)
        self.flows = self._make_flows()

        # AODV needs persistent state across the episode
        self.aodv = AODVLikeRouter()

        # Logging buffers
        self.mobility_rows = []
        self.link_rows = []
        self.topo_rows = []
        self.packet_rows = []
        self.routing_rows = []
        self.rl_rows = []

        # Counters
        self.packet_counter = 0
        self.decision_counter = 0
        self.transition_counter = 0
        self.active_packets = []
        self.finished_outcomes = []  # rolling list of delivered(1)/dropped(0)

        # Link tracking for topology_change_rate
        self.prev_links = set()
        self._last_change_rate = 0.0

    # ---------- setup ----------

    def _make_flows(self):
        flows = []
        for fid in range(self.num_flows):
            src, dst = self.rng.choice(self.N, size=2, replace=False)
            flows.append({
                'flow_id': fid,
                'source_id': int(src),
                'destination_id': int(dst),
                'start_time': 0.0,
                'end_time': self.duration,
                'packet_rate': self.packet_rate,
                'packet_size': 1024,
            })
        return flows

    # ---------- per-step helpers ----------

    def _build_graph(self, t):
        """Build network graph for current positions; log link states."""
        G = nx.Graph()
        for d in self.drones:
            G.add_node(d.id, x=d.x, y=d.y, z=d.z,
                       vx=d.vx, vy=d.vy, vz=d.vz,
                       energy=self.energy[d.id])

        current_links = set()
        for i, j in itertools.combinations(range(self.N), 2):
            di, dj = self.drones[i], self.drones[j]
            dist = float(np.linalg.norm(di.pos - dj.pos))
            if dist > self.comm_range:
                continue
            rssi, snr, lq, per = compute_link_features(dist)
            if rssi <= RSSI_SENSITIVITY:
                continue

            lifetime = estimate_link_lifetime(di.pos, dj.pos, di.vel, dj.vel,
                                              self.comm_range)
            rel_vel = float(np.linalg.norm(di.vel - dj.vel))
            heading_diff = abs(di.heading - dj.heading)
            if heading_diff > np.pi:
                heading_diff = 2 * np.pi - heading_diff

            G.add_edge(i, j, distance=dist, relative_velocity=rel_vel,
                       link_quality=lq, estimated_link_lifetime=lifetime,
                       rssi=rssi, snr=snr, packet_error_rate=per)
            current_links.add((i, j))

            # Log link state (one row, undirected i<j, with neighbor=j context)
            self.link_rows.append({
                'scenario_id': self.scenario_id, 'episode': self.episode,
                'timestamp': t, 'node_i': i, 'node_j': j,
                'distance': dist, 'relative_velocity': rel_vel,
                'heading_difference': heading_diff,
                'rssi': rssi, 'snr': snr, 'link_quality': lq,
                'packet_error_rate': per,
                'estimated_link_lifetime': lifetime,
                'actual_link_duration': np.nan,  # filled in post-processing
                'link_status': 1,
                'node_j_energy': self.energy[j],
                'node_j_queue_occupancy': self.queues[j].occupancy,
                'node_j_x': dj.x, 'node_j_y': dj.y, 'node_j_z': dj.z,
                'mobility_model': self.mobility_model,
            })

        # Log links that broke this step (status=0)
        broken = self.prev_links - current_links
        for (i, j) in broken:
            self.link_rows.append({
                'scenario_id': self.scenario_id, 'episode': self.episode,
                'timestamp': t, 'node_i': i, 'node_j': j,
                'distance': np.nan, 'relative_velocity': np.nan,
                'heading_difference': np.nan, 'rssi': np.nan, 'snr': np.nan,
                'link_quality': 0.0, 'packet_error_rate': 1.0,
                'estimated_link_lifetime': 0.0,
                'actual_link_duration': np.nan, 'link_status': 0,
                'node_j_energy': self.energy[j],
                'node_j_queue_occupancy': self.queues[j].occupancy,
                'node_j_x': self.drones[j].x, 'node_j_y': self.drones[j].y,
                'node_j_z': self.drones[j].z,
                'mobility_model': self.mobility_model,
            })

        # Store change rate for topology logging
        if len(self.prev_links) == 0 and len(current_links) == 0:
            self._last_change_rate = 0.0
        else:
            sym_diff = self.prev_links.symmetric_difference(current_links)
            union = self.prev_links.union(current_links)
            self._last_change_rate = len(sym_diff) / max(len(union), 1)
        
        self.prev_links = current_links
        return G


    def _log_mobility(self, t):
        for d in self.drones:
            self.mobility_rows.append({
                'scenario_id': self.scenario_id, 'episode': self.episode,
                'seed': self.seed, 'timestamp': t, 'drone_id': d.id,
                'x': d.x, 'y': d.y, 'z': d.z,
                'vx': d.vx, 'vy': d.vy, 'vz': d.vz,
                'speed': d.speed, 'heading': d.heading, 'pitch': d.pitch,
                'residual_energy': self.energy[d.id],
                'queue_length': self.queues[d.id].length,
                'queue_occupancy': self.queues[d.id].occupancy,
                'mobility_model': self.mobility_model,
            })

    def _log_topology(self, G, t, active_flows, change_rate):
        degrees = [deg for _, deg in G.degree()]
        comps = list(nx.connected_components(G))
        n_edges = G.number_of_edges()
        max_edges = self.N * (self.N - 1) / 2

        if n_edges > 0:
            lq = np.mean([G.edges[e]['link_quality'] for e in G.edges])
            lt = np.mean([G.edges[e]['estimated_link_lifetime'] for e in G.edges])
            rv = np.mean([G.edges[e]['relative_velocity'] for e in G.edges])
        else:
            lq = lt = rv = 0.0

        largest = max((len(c) for c in comps), default=0)

        self.topo_rows.append({
            'scenario_id': self.scenario_id, 'episode': self.episode,
            'timestamp': t, 'num_nodes': self.N, 'num_active_links': n_edges,
            'avg_degree': float(np.mean(degrees)) if degrees else 0.0,
            'network_density': n_edges / max_edges if max_edges > 0 else 0.0,
            'num_connected_components': len(comps),
            'largest_component_size': largest,
            'largest_component_ratio': largest / self.N,
            'avg_link_quality': float(lq), 'avg_link_lifetime': float(lt),
            'avg_relative_velocity': float(rv),
            'topology_change_rate': change_rate,
            'avg_residual_energy': float(np.mean(self.energy)),
            'min_residual_energy': float(np.min(self.energy)),
            'avg_queue_occupancy': float(np.mean([q.occupancy for q in self.queues])),
            'max_queue_occupancy': float(np.max([q.occupancy for q in self.queues])),
            'active_flows': active_flows,
            'regime_id': -1,
            'mobility_model': self.mobility_model,
        })

    def _run_teachers(self, G, packet):
        """Return dict: teacher_name -> chosen neighbor id (or None)."""
        c, d = packet.current, packet.dst
        return {
            'dijkstra': dijkstra_next_hop(G, c, d),
            'gpsr': gpsr_next_hop(G, c, d),
            'stable_path': stable_path_next_hop(G, c, d),
            'aodv': self.aodv.next_hop(G, c, d),
            'link_lifetime': link_lifetime_next_hop(G, c, d),
        }

    def _log_routing_decision(self, G, t, packet, teacher_choices):
        """One row per candidate neighbor for this forwarding decision."""
        c, dst = packet.current, packet.dst
        neighbors = list(G.neighbors(c))
        if not neighbors:
            return

        did = self.decision_counter
        self.decision_counter += 1

        cd = self.drones[c]
        dest_pos = np.array([G.nodes[dst]['x'], G.nodes[dst]['y'], G.nodes[dst]['z']])
        curr_pos = cd.pos
        dist_to_dest = float(np.linalg.norm(dest_pos - curr_pos))
        bearing = float(np.arctan2(dest_pos[1] - curr_pos[1],
                                   dest_pos[0] - curr_pos[0]))
        reachable = 1 if nx.has_path(G, c, dst) else 0

        topo = self.topo_rows[-1] if self.topo_rows else {}

        # Compute consensus per candidate
        consensus = {n: sum(1 for v in teacher_choices.values() if v == n)
                     for n in neighbors}
        max_consensus = max(consensus.values()) if consensus else 0
        # best label resolves ties by link_quality
        best_candidates = [n for n in neighbors if consensus[n] == max_consensus
                           and max_consensus > 0]
        if best_candidates:
            best_label_node = max(
                best_candidates,
                key=lambda n: G.edges[c, n]['link_quality'])
        else:
            best_label_node = None

        for n in neighbors:
            edge = G.edges[c, n]
            n_pos = _node_pos(G, n)
            dist_nd = float(np.linalg.norm(dest_pos - n_pos))
            progress = (dist_to_dest - dist_nd) / max(dist_to_dest, 1.0)
            cc = consensus[n]

            self.routing_rows.append({
                'scenario_id': self.scenario_id, 'episode': self.episode,
                'timestamp': t, 'packet_id': packet.pid, 'decision_id': did,
                'current_node': c, 'destination_node': dst,
                'current_x': cd.x, 'current_y': cd.y, 'current_z': cd.z,
                'current_vx': cd.vx, 'current_vy': cd.vy, 'current_vz': cd.vz,
                'current_speed': cd.speed, 'current_energy': self.energy[c],
                'current_queue_occupancy': self.queues[c].occupancy,
                'current_num_neighbors': len(neighbors),
                'distance_to_destination': dist_to_dest,
                'bearing_to_destination': bearing,
                'destination_reachable': reachable,
                'candidate_neighbor_id': n,
                'candidate_distance': edge['distance'],
                'candidate_relative_velocity': edge['relative_velocity'],
                'candidate_link_quality': edge['link_quality'],
                'candidate_rssi': edge['rssi'],
                'candidate_packet_error_rate': edge['packet_error_rate'],
                'candidate_estimated_lifetime': edge['estimated_link_lifetime'],
                'candidate_energy': self.energy[n],
                'candidate_queue_occupancy': self.queues[n].occupancy,
                'candidate_x': G.nodes[n]['x'], 'candidate_y': G.nodes[n]['y'],
                'candidate_z': G.nodes[n]['z'],
                'candidate_dist_to_dest': dist_nd,
                'candidate_progress': progress,
                'avg_degree': topo.get('avg_degree', 0.0),
                'network_density': topo.get('network_density', 0.0),
                'num_components': topo.get('num_connected_components', 1),
                'avg_link_quality': topo.get('avg_link_quality', 0.0),
                'topology_change_rate': topo.get('topology_change_rate', 0.0),
                'regime_id': -1,
                'dijkstra_chose_this': int(teacher_choices['dijkstra'] == n),
                'gpsr_chose_this': int(teacher_choices['gpsr'] == n),
                'stable_path_chose_this': int(teacher_choices['stable_path'] == n),
                'aodv_chose_this': int(teacher_choices['aodv'] == n),
                'link_lifetime_chose_this': int(teacher_choices['link_lifetime'] == n),
                'teacher_consensus_count': cc,
                'consensus_confidence': cc / 5.0,
                'is_best_label': int(n == best_label_node),
                'packet_outcome': np.nan,  # filled in post-processing
                'mobility_model': self.mobility_model,
            })

    def _forward_one_hop(self, G, t, packet):
        """Forward a packet one hop using the stable_path teacher as the actor."""
        c, dst = packet.current, packet.dst

        if c == dst:
            packet.delivered = True
            packet.delivery_time = t
            return True

        neighbors = list(G.neighbors(c))
        if not neighbors:
            packet.dropped = True
            packet.drop_reason = 'no_route'
            return True

        # Log routing decision (all teachers) before acting
        teacher_choices = self._run_teachers(G, packet)
        self._log_routing_decision(G, t, packet, teacher_choices)

        # The acting policy for pilot data = Dijkstra (loop-free by construction).
        # All five teachers are still logged above for supervised training;
        # using Dijkstra to physically move the packet guarantees no oscillation.
        next_hop = teacher_choices['dijkstra']
        if next_hop is None:
            next_hop = teacher_choices['stable_path']
        # Avoid revisiting nodes already on this packet's path
        if next_hop is not None and next_hop in packet.path:
            unvisited = [n for n in neighbors if n not in packet.path]
            next_hop = unvisited[0] if unvisited else None
        if next_hop is None:
            packet.dropped = True
            packet.drop_reason = 'no_route'
            self._log_transition(t, packet, None, G, dropped=True)
            return True

        # Distance-based reward bookkeeping
        dest_pos = _node_pos(G, dst)
        dist_before = float(np.linalg.norm(_node_pos(G, c) - dest_pos))
        dist_after = float(np.linalg.norm(_node_pos(G, next_hop) - dest_pos))

        # Energy check
        if not EnergyModel.is_alive(self.energy[c]):
            packet.dropped = True
            packet.drop_reason = 'energy_depleted'
            self._log_transition(t, packet, next_hop, G, dropped=True,
                                 dist_before=dist_before, dist_after=dist_after)
            return True

        # Packet error check (probabilistic loss)
        per = G.edges[c, next_hop]['packet_error_rate']
        if self.rng.random() < per:
            packet.dropped = True
            packet.drop_reason = 'packet_error'
            self.energy[c] = EnergyModel.consume_tx(self.energy[c])
            self._log_transition(t, packet, next_hop, G, dropped=True,
                                 dist_before=dist_before, dist_after=dist_after)
            return True

        # Queue check at receiver
        if not self.queues[next_hop].enqueue(packet):
            packet.dropped = True
            packet.drop_reason = 'queue_overflow'
            self._log_transition(t, packet, next_hop, G, dropped=True,
                                 dist_before=dist_before, dist_after=dist_after)
            return True
        # note: removed immediate dequeue — packets now stay in queue
        # until drained at start of next timestep

        # Loop check
        loop = next_hop in packet.path

        # Apply forwarding
        e_tx = EnergyModel.TX_COST
        e_rx = EnergyModel.RX_COST
        self.energy[c] = EnergyModel.consume_tx(self.energy[c])
        self.energy[next_hop] = EnergyModel.consume_rx(self.energy[next_hop])
        packet.energy_used += (e_tx + e_rx)
        packet.current = next_hop
        packet.path.append(next_hop)
        packet.hops += 1
        packet.cum_delay_ms += HOP_DELAY_MS

        self._log_transition(t, packet, next_hop, G, dropped=False,
                             dist_before=dist_before, dist_after=dist_after,
                             loop=loop)

        if packet.current == dst:
            packet.delivered = True
            packet.delivery_time = t
            return True

        if packet.hops >= TTL:
            packet.dropped = True
            packet.drop_reason = 'ttl_expired'
            return True

        return False  # still traveling

    def _log_transition(self, t, packet, next_hop, G, dropped,
                        dist_before=0.0, dist_after=0.0, loop=False):
        delivered_now = (not dropped) and (packet.current == packet.dst)
        r_delivery = 10.0 if delivered_now else 0.0
        r_progress = (1.0 * (dist_before - dist_after) / max(dist_before, 1.0)
                      if not dropped else 0.0)
        if next_hop is not None and not dropped and G.has_edge(
                packet.path[-2] if len(packet.path) >= 2 else packet.current, next_hop):
            edge = G.edges[packet.path[-2], next_hop] if len(packet.path) >= 2 else None
        else:
            edge = None
        lq = edge['link_quality'] if edge else 0.0
        lt = edge['estimated_link_lifetime'] if edge else 0.0
        r_link_q = 0.3 * lq
        r_link_lt = 0.2 * min(lt / 30.0, 1.0)
        p_delay = -0.1 * self.dt
        p_energy = -0.05 * (EnergyModel.TX_COST + EnergyModel.RX_COST)
        p_queue = -0.2 * (self.queues[next_hop].occupancy if next_hop is not None else 0.0)
        p_drop = -10.0 if dropped else 0.0
        p_loop = -2.0 if loop else 0.0
        total = (r_delivery + r_progress + r_link_q + r_link_lt +
                 p_delay + p_energy + p_queue + p_drop + p_loop)

        # rolling PDR over last 10 finished packets
        last10 = self.finished_outcomes[-10:]
        pdr10 = float(np.mean(last10)) if last10 else 0.0

        self.rl_rows.append({
            'scenario_id': self.scenario_id, 'episode': self.episode,
            'timestamp': t, 'transition_id': self.transition_counter,
            'packet_id': packet.pid, 'current_node': packet.path[-2]
            if len(packet.path) >= 2 else packet.current,
            'destination_node': packet.dst,
            'chosen_next_hop': next_hop if next_hop is not None else -1,
            'reward_delivery': r_delivery, 'reward_progress': r_progress,
            'reward_link_quality': r_link_q, 'reward_link_lifetime': r_link_lt,
            'penalty_delay': p_delay, 'penalty_energy': p_energy,
            'penalty_queue': p_queue, 'penalty_drop': p_drop,
            'penalty_loop': p_loop, 'reward_total': total,
            'next_node': packet.current,
            'packet_delivered': int(delivered_now),
            'packet_dropped': int(dropped),
            'drop_reason': packet.drop_reason if dropped else None,
            'done': int(dropped or delivered_now),
            'pdr_window_last_10': pdr10,
            'regime_id': -1,
            'mobility_model': self.mobility_model,
        })
        self.transition_counter += 1

    def _finish_packet(self, packet):
        self.finished_outcomes.append(1 if packet.delivered else 0)
        delay = packet.cum_delay_ms if packet.delivered else np.nan
        self.packet_rows.append({
            'scenario_id': self.scenario_id, 'episode': self.episode,
            'packet_id': packet.pid, 'flow_id': packet.flow_id,
            'source_id': packet.src, 'destination_id': packet.dst,
            'generation_time': packet.gen_time,
            'delivery_time': packet.delivery_time if packet.delivered else np.nan,
            'delivered': int(packet.delivered), 'dropped': int(packet.dropped),
            'drop_reason': packet.drop_reason,
            'hop_count': packet.hops,
            'path_taken': ','.join(map(str, packet.path)),
            'end_to_end_delay': delay,
            'total_energy_used': packet.energy_used,
            'ttl_remaining': TTL - packet.hops,
        })

    # ---------- main loop ----------

    def run(self):
        n_steps = int(self.duration / self.dt)
        time_since_gen = {f['flow_id']: 0.0 for f in self.flows}

        for step in range(n_steps):
            t = step * self.dt

            # 1. Move drones + idle energy drain
            for d in self.drones:
                d.step(self.dt)
                self.energy[d.id] = EnergyModel.consume_idle(self.energy[d.id], self.dt)
            self._log_mobility(t)

            # 2. Build graph + log links
            G = self._build_graph(t)

            # 3. Topology change rate — compute from prev_links diff
            active_flows = sum(1 for f in self.flows if f['start_time'] <= t <= f['end_time'])
            self._log_topology(G, t, active_flows, self._last_change_rate)


            # 4. Generate packets
            for f in self.flows:
                if not (f['start_time'] <= t <= f['end_time']):
                    continue
                time_since_gen[f['flow_id']] += self.dt
                interval = 1.0 / f['packet_rate']
                while time_since_gen[f['flow_id']] >= interval:
                    time_since_gen[f['flow_id']] -= interval
                    pkt = Packet(self.packet_counter, f['flow_id'],
                                 f['source_id'], f['destination_id'], t)
                    self.packet_counter += 1
                    self.active_packets.append(pkt)

            # 5. Drain queues from previous timestep
            for q in self.queues:
                while q.length > 0:
                    q.dequeue()
            
            # 6. Forward active packets (limited hops per timestep)
            still_active = []
            for pkt in self.active_packets:
                done = False
                for _ in range(MAX_HOPS_PER_STEP):
                    finished = self._forward_one_hop(G, t, pkt)
                    if finished:
                        done = True
                        break
                if done:
                    self._finish_packet(pkt)
                else:
                    still_active.append(pkt)
            self.active_packets = still_active




        # Drop any packets still in flight at episode end
        for pkt in self.active_packets:
            pkt.dropped = True
            pkt.drop_reason = 'episode_end'
            self._finish_packet(pkt)
        self.active_packets = []

        return self._assemble_dataframes()

    def _assemble_dataframes(self):
        dfs = {
            'mobility_trace': pd.DataFrame(self.mobility_rows),
            'link_states': pd.DataFrame(self.link_rows),
            'topology_features': pd.DataFrame(self.topo_rows),
            'traffic_flows': pd.DataFrame([{
                'scenario_id': self.scenario_id, 'episode': self.episode, **f
            } for f in self.flows]),
            'packet_logs': pd.DataFrame(self.packet_rows),
            'routing_decisions': pd.DataFrame(self.routing_rows),
            'rl_transitions': pd.DataFrame(self.rl_rows),
        }
        dfs['link_states'] = self._fill_link_durations(dfs['link_states'])
        dfs['routing_decisions'] = self._fill_packet_outcomes(
            dfs['routing_decisions'], dfs['packet_logs'])
        return dfs

    def _fill_link_durations(self, df):
        if df.empty:
            return df
        df = df.sort_values(['node_i', 'node_j', 'timestamp']).reset_index(drop=True)
        durations = np.full(len(df), np.nan)
        for (ni, nj), idx in df.groupby(['node_i', 'node_j']).groups.items():
            idx = list(idx)
            sub = df.loc[idx]
            times = sub['timestamp'].values
            status = sub['link_status'].values
            t_end = times.max()
            for k, gi in enumerate(idx):
                if status[k] == 0:
                    durations[gi] = 0.0
                    continue
                future_breaks = [times[m] for m in range(k, len(times))
                                 if status[m] == 0]
                if future_breaks:
                    durations[gi] = max(future_breaks[0] - times[k], 0.0)
                else:
                    durations[gi] = max(t_end - times[k], 0.0)
        df['actual_link_duration'] = durations
        return df

    def _fill_packet_outcomes(self, routing_df, packet_df):
        if routing_df.empty or packet_df.empty:
            return routing_df
        outcome = dict(zip(packet_df['packet_id'], packet_df['delivered']))
        routing_df['packet_outcome'] = routing_df['packet_id'].map(outcome).fillna(0).astype(int)
        return routing_df


def _node_pos(G, n):
    return np.array([G.nodes[n]['x'], G.nodes[n]['y'], G.nodes[n]['z']])
