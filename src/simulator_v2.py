"""
simulator_v2.py  —  Multi-packet, interference-coupled FANET simulator (M2).

Implements the M2 design spec (docs/M2_SIMULATOR_V2_DESIGN.md):
  - Two-level time: frames (mobility/topology) contain slots (channel/forwarding).
  - Slotted channel: transmissions in the same slot interfere via link_model_v2.
  - Real queues with bounded per-slot service (no total-drain).
  - Node transmission activity derived from queue state -> interference (the
    load -> activity -> interference feedback loop).
  - Per-packet trajectory tracking (clean RL credit assignment across N packets).
  - Full drop taxonomy; interference-off degenerate mode for regression vs A1.

This M2 build uses a pluggable teacher as the forwarding actor (default:
queue_aware_greedy). Reward design and RL observation are deferred to M5/M4;
a light per-hop reward is recorded so trajectories are non-empty, but it is NOT
the final reward function.

Run one episode:  FANETSimulatorV2(config).run()  -> dict of metrics + logs.
"""

import itertools
import numpy as np
import networkx as nx

from mobility import DroneRWP
import link_model_v2 as lm2
from models import EnergyModel, NodeQueue
from routing_teachers_v2 import queue_aware_greedy_next_hop, backpressure_next_hop
from routing_teachers import dijkstra_next_hop

# ── Fixed simulation constants ───────────────────────────────────────────────
TTL            = 20
MAX_QUEUE      = 50
INITIAL_ENERGY = 100.0
HOP_DELAY_MS   = 2.0

# Two-level time (design spec §1)
FRAME_DT        = 0.5      # seconds per mobility/topology frame (= old dt)
SLOT_DT         = 0.01     # seconds per channel slot
SLOTS_PER_FRAME = int(round(FRAME_DT / SLOT_DT))   # 50

# Service model (design spec §2)
SERVICE_RATE = 1          # packets a node may transmit per slot (one radio)

# Load -> activity mapping (design spec §3.1); calibrated against G1's 0-0.2 band
# CALIBRATION NOTE (M2 bring-up): ACT_BETA locked to 0.0. Interference is driven
# purely by real backlog (occupancy), not by the mere presence of a packet. A node
# that receives one packet and forwards it immediately is not congested and should
# not radiate interference like a backed-up node. This (a) concentrates interference
# at genuinely congested nodes so a router can steer AROUND hot-spots rather than
# facing uniform loss, and (b) gives the clean causal story the thesis needs:
# congestion causes interference. Empirically raised PDR 0.279 -> 0.311 and cut
# link_error share at fixed load.
ACT_ALPHA = 1.0           # weight on queue occupancy
ACT_BETA  = 0.0           # LOCKED 0: interference only from real backlog
ACT_MAX   = 0.5           # ceiling on per-slot transmission activity

# ARQ retransmission limit (design spec addendum, M2 bring-up).
# Without ARQ, PER is a near-step function of SINR: one hidden terminal firing =
# guaranteed packet death, so interference acts as spatially-uniform random loss
# that no routing policy can avoid (measured actor spread was ~0.008, i.e. none).
# Real 802.11-style MACs retransmit; retry limits of 4-7 are standard. ARQ converts
# interference from a binary packet-killer into a graded delay/energy cost, which
# is the mechanism that makes congestion-aware routing pay off. This restored
# measurable actor spread (up to +0.06 PDR for queue-aware vs dijkstra).
DEFAULT_MAX_RETX = 5

TEACHERS = {
    'queue_aware_greedy': queue_aware_greedy_next_hop,
    'backpressure':       backpressure_next_hop,
    'dijkstra':           dijkstra_next_hop,
}


class PacketV2:
    __slots__ = ('pid', 'flow_id', 'src', 'dst', 'gen_time', 'current',
                 'path', 'hops', 'energy_used', 'delivered', 'dropped',
                 'drop_reason', 'delivery_time', 'cum_delay_ms',
                 'trajectory', 'pending_obs')

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
        self.trajectory = []          # per-packet (obs, action, reward, next_obs, done)
        self.pending_obs = None       # obs captured at decision time, awaiting outcome


class FANETSimulatorV2:
    def __init__(self, config):
        self.cfg = config
        self.scenario_id = config.get('scenario_id', 'v2')
        self.episode = config.get('episode', 0)
        self.seed = config['seed']
        self.N = config['num_drones']
        self.area_x = config['area_x']
        self.area_y = config['area_y']
        self.z_min = config['z_min']
        self.z_max = config['z_max']
        self.comm_range = config['comm_range']
        self.duration = config['duration']
        self.mobility_model = config.get('mobility_model', 'random_waypoint')
        self.packet_rate = config['packet_rate']

        # Toggles for the interference on/off ablation & degenerate A1 mode
        self.interference_on = bool(config.get('interference_on', True))
        self.shadowing_sigma = float(config.get('shadowing_sigma',
                                                lm2.SHADOWING_SIGMA_DB))
        self.act_max = float(config.get('act_max', ACT_MAX))
        self.max_retx = int(config.get('max_retx', DEFAULT_MAX_RETX))  # ARQ limit

        # Actor policy (teacher) that physically moves packets
        self.actor_name = config.get('actor', 'queue_aware_greedy')
        self.actor = TEACHERS[self.actor_name]

        self.rng = np.random.default_rng(self.seed)

        self.drones = [
            DroneRWP(i, self.area_x, self.area_y, self.z_min, self.z_max,
                     config['speed_min'], config['speed_max'],
                     config['pause_max'], seed=self.seed)
            for i in range(self.N)
        ]

        self.energy = [INITIAL_ENERGY] * self.N
        self.queues = [NodeQueue(MAX_QUEUE) for _ in range(self.N)]

        self.num_flows = max(1, self.N // 4)
        self.flows = self._make_flows()

        # Metrics / counters
        self.packet_counter = 0
        self.n_generated = 0
        self.n_delivered = 0
        self.n_dropped = 0
        self.drop_reasons = {}
        self.delivered_delays = []
        self.delivered_hops = []
        self.tx_attempts = []        # ARQ attempts per hop (delay-sanity diagnostics)
        self.completed_trajectories = []   # per-packet finished trajectories

        # Time-series diagnostics (per frame)
        self.ts_mean_occ = []
        self.ts_max_occ = []
        self.ts_mean_activity = []
        self.ts_inflight = []
        self.ts_mean_linkq = []

        # Per-node activity for the current frame (occupancy-derived; §3.1)
        self.activity = np.zeros(self.N)

    # ── setup ────────────────────────────────────────────────────────────────
    def _make_flows(self):
        flows = []
        for fid in range(self.num_flows):
            src, dst = self.rng.choice(self.N, size=2, replace=False)
            flows.append({'flow_id': fid, 'source_id': int(src),
                          'destination_id': int(dst),
                          'packet_rate': self.packet_rate})
        return flows

    # ── topology ─────────────────────────────────────────────────────────────
    def _build_graph(self):
        """Rebuild the decodable-link graph from current positions (per frame).
        Node attrs include live queue state so congestion-aware teachers see it.
        Edge link_quality here is the BASE (interference-free) value; the actual
        per-hop SINR under this slot's interference is computed at forward time."""
        G = nx.Graph()
        for d in self.drones:
            G.add_node(d.id, x=d.x, y=d.y, z=d.z, vx=d.vx, vy=d.vy, vz=d.vz,
                       energy=self.energy[d.id],
                       queue_occupancy=self.queues[d.id].occupancy,
                       queue_len=self.queues[d.id].length)
        for i, j in itertools.combinations(range(self.N), 2):
            di, dj = self.drones[i], self.drones[j]
            dist = float(np.linalg.norm(di.pos - dj.pos))
            if dist > self.comm_range:
                continue
            if not lm2.link_exists(dist, self.comm_range):
                continue
            # base (interference-free) features
            rssi, sinr, lq, per = lm2.compute_link_features_v2(
                dist, interference_mw=0.0, n_contenders=1, shadowing_db=0.0)
            lifetime = lm2.estimate_link_lifetime(di.pos, dj.pos, di.vel, dj.vel,
                                                  self.comm_range)
            rel_vel = float(np.linalg.norm(di.vel - dj.vel))
            G.add_edge(i, j, distance=dist, relative_velocity=rel_vel,
                       link_quality=lq, base_link_quality=lq,
                       estimated_link_lifetime=lifetime,
                       rssi=rssi, snr=sinr, packet_error_rate=per)
        return G

    # ── load -> activity coupling (§3.1) ─────────────────────────────────────
    def _update_activity(self):
        """Per-slot transmission activity of each node from its queue state.
        Idle nodes ~0 activity (no interference); congested nodes near act_max."""
        if not self.interference_on:
            self.activity[:] = 0.0
            return
        for k in range(self.N):
            occ = self.queues[k].occupancy
            nonempty = 1.0 if self.queues[k].length > 0 else 0.0
            self.activity[k] = min(ACT_ALPHA * occ + ACT_BETA * nonempty,
                                   self.act_max)

    # ── per-slot interference at a receiver (§1.2) ───────────────────────────
    def _interference_at(self, G, tx, rx, tx_positions):
        """Summed interferer power (mW) at rx, from hidden-terminal-band nodes
        that are 'transmitting' this slot (activity as Bernoulli), plus the
        Bianchi contender count in carrier-sense range of tx."""
        if not self.interference_on:
            return 0.0, 1
        cs_range = lm2.CARRIER_SENSE_MULT * self.comm_range
        if_range = lm2.INTERFERENCE_RANGE_MULT * self.comm_range
        rx_pos = self.drones[rx].pos
        tx_pos = self.drones[tx].pos

        interf_mw = 0.0
        n_contenders = 1
        for k in range(self.N):
            if k == tx or k == rx:
                continue
            kp = self.drones[k].pos
            dk_tx = float(np.linalg.norm(kp - tx_pos))
            dk_rx = float(np.linalg.norm(kp - rx_pos))
            # carrier-sense contenders (for Bianchi collision)
            if dk_tx <= cs_range and self.rng.random() < self.activity[k]:
                n_contenders += 1
            # hidden-terminal interferers (for SINR)
            if dk_tx > cs_range and dk_rx <= if_range:
                if self.rng.random() < self.activity[k]:
                    sh = self.rng.normal(0.0, self.shadowing_sigma)
                    interf_mw += lm2.rx_power_mw(dk_rx, sh)
        return interf_mw, n_contenders

    # ── decision + forwarding for one packet (one slot) ──────────────────────
    def _try_forward(self, G, t, pkt):
        """Attempt to forward pkt one hop this slot. Returns True if the packet
        finished (delivered/dropped) this slot, False if it stays active."""
        c, dst = pkt.current, pkt.dst

        if c == dst:
            pkt.delivered = True
            pkt.delivery_time = t
            self._record_transition(pkt, action=None, reward=10.0,
                                    done=True, delivered=True)
            return True

        if c not in G or dst not in G:
            # node isolated this frame (no edges) -> cannot route this slot
            if pkt.hops >= TTL:
                pkt.dropped = True; pkt.drop_reason = 'ttl_expired'
                self._record_transition(pkt, None, -10.0, True, dropped=True)
                return True
            return False

        neighbors = list(G.neighbors(c))
        if not neighbors:
            if pkt.hops >= TTL:
                pkt.dropped = True; pkt.drop_reason = 'ttl_expired'
                self._record_transition(pkt, None, -10.0, True, dropped=True)
                return True
            return False

        # capture obs at decision time (§5) — placeholder obs for M2
        pkt.pending_obs = self._make_obs(G, pkt, neighbors)

        next_hop = self.actor(G, c, dst)
        # avoid revisiting nodes already on this packet's path (loop-free actor)
        if next_hop is not None and next_hop in pkt.path:
            unvisited = [n for n in neighbors if n not in pkt.path]
            next_hop = unvisited[0] if unvisited else None
        if next_hop is None:
            if pkt.hops >= TTL:
                pkt.dropped = True; pkt.drop_reason = 'ttl_expired'
            else:
                pkt.dropped = True; pkt.drop_reason = 'no_route'
            self._record_transition(pkt, None, -10.0, True, dropped=True)
            return True

        # energy check at forwarding node
        if not EnergyModel.is_alive(self.energy[c]):
            pkt.dropped = True; pkt.drop_reason = 'energy_depleted'
            self._record_transition(pkt, next_hop, -10.0, True, dropped=True)
            return True

        # per-hop transmission with ARQ: retransmit on link error, re-sampling
        # the (bursty) interference each attempt. Interference becomes a graded
        # delay/energy cost rather than a binary packet-killer.
        dist = G.edges[c, next_hop]['distance']
        delivered_hop = False
        attempts = 0
        lq_eff = 0.0
        for _ in range(self.max_retx + 1):
            interf_mw, n_cont = self._interference_at(G, c, next_hop, None)
            sh = self.rng.normal(0.0, self.shadowing_sigma) if self.interference_on else 0.0
            _, _, lq_eff, per_eff = lm2.compute_link_features_v2(
                dist, interference_mw=interf_mw, n_contenders=n_cont, shadowing_db=sh)
            attempts += 1
            self.energy[c] = EnergyModel.consume_tx(self.energy[c])  # each attempt costs TX
            if self.rng.random() >= per_eff:
                delivered_hop = True
                break
        pkt.cum_delay_ms += HOP_DELAY_MS * attempts   # retries add latency
        self.tx_attempts.append(attempts)

        if not delivered_hop:
            pkt.dropped = True; pkt.drop_reason = 'link_error'
            self._record_transition(pkt, next_hop, -10.0, True, dropped=True,
                                    lq=lq_eff)
            return True

        # receiver queue admission (tail-drop)
        if not self.queues[next_hop].enqueue(pkt):
            pkt.dropped = True; pkt.drop_reason = 'queue_overflow'
            self._record_transition(pkt, next_hop, -10.0, True, dropped=True,
                                    lq=lq_eff)
            return True

        # commit the hop (TX energy already charged per attempt in the ARQ loop)
        self.energy[next_hop] = EnergyModel.consume_rx(self.energy[next_hop])
        pkt.energy_used += (EnergyModel.TX_COST * attempts + EnergyModel.RX_COST)
        pkt.current = next_hop
        pkt.path.append(next_hop)
        pkt.hops += 1
        pkt.cum_delay_ms += HOP_DELAY_MS

        delivered_now = (next_hop == dst)
        if delivered_now:
            pkt.delivered = True
            pkt.delivery_time = t
        r = 10.0 if delivered_now else 0.3 * lq_eff  # placeholder reward (M5 replaces)
        self._record_transition(pkt, next_hop, r, done=delivered_now,
                                delivered=delivered_now, lq=lq_eff)
        return delivered_now

    def _make_obs(self, G, pkt, neighbors):
        """Placeholder observation (M4 replaces with GNN+attention features).
        Kept minimal but real so trajectories are structurally valid now."""
        c, dst = pkt.current, pkt.dst
        dpos = np.array([G.nodes[dst]['x'], G.nodes[dst]['y'], G.nodes[dst]['z']])
        cpos = self.drones[c].pos
        return {
            'current': c, 'dst': dst, 'ttl_left': TTL - pkt.hops,
            'dist_to_dest': float(np.linalg.norm(dpos - cpos)),
            'num_neighbors': len(neighbors),
            'cur_queue_occ': self.queues[c].occupancy,
        }

    def _record_transition(self, pkt, action, reward, done,
                           delivered=False, dropped=False, lq=0.0):
        obs = pkt.pending_obs
        pkt.trajectory.append({
            'obs': obs, 'action': action, 'reward': float(reward),
            'done': bool(done), 'delivered': bool(delivered),
            'dropped': bool(dropped), 'link_quality': float(lq),
            'hop': pkt.hops,
        })
        pkt.pending_obs = None

    def _finish_packet(self, pkt):
        if pkt.delivered:
            self.n_delivered += 1
            self.delivered_delays.append(pkt.cum_delay_ms)
            self.delivered_hops.append(pkt.hops)
        else:
            self.n_dropped += 1
            self.drop_reasons[pkt.drop_reason] = \
                self.drop_reasons.get(pkt.drop_reason, 0) + 1
        self.completed_trajectories.append({
            'pid': pkt.pid, 'delivered': pkt.delivered,
            'drop_reason': pkt.drop_reason, 'hops': pkt.hops,
            'trajectory': pkt.trajectory,
        })

    # ── main loop ────────────────────────────────────────────────────────────
    def run(self):
        n_frames = int(self.duration / FRAME_DT)
        active = []
        time_since_gen = {f['flow_id']: 0.0 for f in self.flows}

        for frame in range(n_frames):
            t_frame = frame * FRAME_DT

            # 1. mobility + idle energy (per frame)
            for d in self.drones:
                d.step(FRAME_DT)
                self.energy[d.id] = EnergyModel.consume_idle(self.energy[d.id], FRAME_DT)

            # 2. rebuild topology (per frame)
            G = self._build_graph()

            # 3. generate packets for this frame
            for f in self.flows:
                time_since_gen[f['flow_id']] += FRAME_DT
                interval = 1.0 / f['packet_rate']
                while time_since_gen[f['flow_id']] >= interval:
                    time_since_gen[f['flow_id']] -= interval
                    pkt = PacketV2(self.packet_counter, f['flow_id'],
                                   f['source_id'], f['destination_id'], t_frame)
                    self.packet_counter += 1
                    self.n_generated += 1
                    # admit to source queue (tail-drop if full)
                    if self.queues[f['source_id']].enqueue(pkt):
                        active.append(pkt)
                    else:
                        pkt.dropped = True; pkt.drop_reason = 'queue_overflow'
                        self._finish_packet(pkt)

            # 4. run slots within this frame
            for slot in range(SLOTS_PER_FRAME):
                t = t_frame + slot * SLOT_DT
                self._update_activity()   # activity from current queue state (§3.1)

                # each node serves up to SERVICE_RATE head-of-line packets/slot
                served_this_slot = {k: 0 for k in range(self.N)}
                still_active = []
                # iterate a snapshot; packets may finish or move queues
                for pkt in active:
                    node = pkt.current
                    # service limit: only SERVICE_RATE forwards per node per slot
                    if served_this_slot[node] >= SERVICE_RATE:
                        still_active.append(pkt)
                        continue
                    # only the head-of-line packet of that node's queue transmits
                    q = self.queues[node].buffer
                    if q and q[0] is not pkt and pkt in q:
                        # not head-of-line; wait
                        still_active.append(pkt)
                        continue
                    served_this_slot[node] += 1
                    # remove pkt from its current node's queue (it's being serviced)
                    if pkt in self.queues[node].buffer:
                        self.queues[node].buffer.remove(pkt)
                    finished = self._try_forward(G, t, pkt)
                    if finished:
                        self._finish_packet(pkt)
                    else:
                        still_active.append(pkt)
                active = still_active

                # TTL sweep for anything stuck too long
                survivors = []
                for pkt in active:
                    if pkt.hops >= TTL and not (pkt.delivered or pkt.dropped):
                        pkt.dropped = True; pkt.drop_reason = 'ttl_expired'
                        self._finish_packet(pkt)
                    else:
                        survivors.append(pkt)
                active = survivors

            # 5. per-frame diagnostics
            occs = [self.queues[k].occupancy for k in range(self.N)]
            self.ts_mean_occ.append(float(np.mean(occs)))
            self.ts_max_occ.append(float(np.max(occs)))
            self.ts_mean_activity.append(float(np.mean(self.activity)))
            self.ts_inflight.append(len(active))
            if G.number_of_edges() > 0:
                self.ts_mean_linkq.append(float(np.mean(
                    [G.edges[e]['link_quality'] for e in G.edges])))
            else:
                self.ts_mean_linkq.append(0.0)

        # drain: anything still active at episode end is dropped
        for pkt in active:
            pkt.dropped = True; pkt.drop_reason = 'episode_end'
            self._finish_packet(pkt)

        return self._metrics()

    def _metrics(self):
        pdr = self.n_delivered / max(self.n_generated, 1)
        return {
            'scenario_id': self.scenario_id, 'seed': self.seed,
            'actor': self.actor_name, 'interference_on': self.interference_on,
            'packet_rate': self.packet_rate,
            'n_generated': self.n_generated, 'n_delivered': self.n_delivered,
            'n_dropped': self.n_dropped, 'network_pdr': pdr,
            'drop_reasons': dict(self.drop_reasons),
            'mean_delay_ms': float(np.mean(self.delivered_delays)) if self.delivered_delays else float('nan'),
            'mean_hops': float(np.mean(self.delivered_hops)) if self.delivered_hops else float('nan'),
            'mean_queue_occ': float(np.mean(self.ts_mean_occ)) if self.ts_mean_occ else 0.0,
            'max_queue_occ': float(np.max(self.ts_max_occ)) if self.ts_max_occ else 0.0,
            'mean_activity': float(np.mean(self.ts_mean_activity)) if self.ts_mean_activity else 0.0,
            'peak_inflight': int(np.max(self.ts_inflight)) if self.ts_inflight else 0,
            'mean_link_quality': float(np.mean(self.ts_mean_linkq)) if self.ts_mean_linkq else 0.0,
            'n_completed_trajectories': len(self.completed_trajectories),
            'mean_tx_attempts': float(np.mean(self.tx_attempts)) if self.tx_attempts else 0.0,
            'max_tx_attempts': int(np.max(self.tx_attempts)) if self.tx_attempts else 0,
            'mean_delay_per_hop_ms': (float(np.mean(self.delivered_delays)) /
                                      max(float(np.mean(self.delivered_hops)), 1e-9))
                                     if self.delivered_hops else float('nan'),
        }


def make_config(scenario, **overrides):
    """Convenience: build a config dict from a scenario preset + overrides."""
    cfg = dict(scenario)
    cfg.update(overrides)
    return cfg
