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
import routing_teachers_v2 as _rt2
from routing_teachers_v2 import (queue_aware_greedy_next_hop, backpressure_next_hop,
                                 spbp_next_hop, da_gpsr_next_hop,
                                 etx_dijkstra_next_hop, lq_dijkstra_next_hop,
                                 arq_etx_next_hop, dpp_next_hop, car_next_hop,
                                 spbp_lookahead_next_hop, random_next_hop)
from routing_teachers import dijkstra_next_hop, gpsr_next_hop

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

# SINR below which a packet is effectively undecodable. PER is a near-step
# function of SINR under the BER/PER model in link_model_v2 (PER crosses from
# ~0 to ~1 within a few dB), so a single interferer that pushes SINR under this
# knee is treated as fatal when estimating measured link statistics. 10 dB.
LETHAL_SINR_LINEAR = 10.0 ** (10.0 / 10.0)

# M3 panel (docs/M3_TEACHERS_ORACLE_DESIGN.md §2): 2 congestion-blind
# references + 4 congestion-aware. 'random' is the G3 sanity floor, not a panel
# member. 'queue_aware_greedy' is retained as an alias of da_gpsr for M2 compat.
TEACHERS = {
    # congestion-blind references
    'dijkstra':           dijkstra_next_hop,
    'gpsr':               gpsr_next_hop,
    # congestion-aware
    'backpressure':       backpressure_next_hop,
    'spbp':               spbp_next_hop,
    'da_gpsr':            da_gpsr_next_hop,
    'dpp':                dpp_next_hop,
    'car':                car_next_hop,
    'spbp_lookahead':     spbp_lookahead_next_hop,
    'lq_dijkstra':        lq_dijkstra_next_hop,
    # candidates / retained for reporting
    'arq_etx':            arq_etx_next_hop,
    'etx_dijkstra':       etx_dijkstra_next_hop,
    # aliases / baselines
    'queue_aware_greedy': queue_aware_greedy_next_hop,   # == da_gpsr (M2 compat)
    'random':             random_next_hop,
}

# M3 panel (docs/M3_TEACHERS_ORACLE_DESIGN.md §2), REVISED after the dynamic-
# metric head-to-head: lq_dijkstra lost 10/12 cells to plain Dijkstra, and
# arq_etx (bounded, physically exact for this simulator's ARQ mechanism) lost
# all 12. Neither global path-optimization variant beat greedy per-hop use of
# the same link_quality signal (SP-BP, DA-GPSR already use it locally). The
# negative result is reported directly rather than carried in the panel: under
# bursty per-slot interference, frame-stale link-quality estimates compound
# error along a multi-hop path faster than they help avoid it, so path-level
# optimization on this signal loses to one-hop-ahead greedy use of it.
#
# lq_dijkstra, etx_dijkstra, arq_etx remain callable via TEACHERS for the
# methodology section's head-to-head comparison; they are NOT panel members.
#
# EXPANDED after the first 30-seed run showed SP-BP winning all 12 oracle
# cells (10/12 by a robust margin). Three additional hybrid teachers were
# added, each designed to isolate a different question about WHY:
#   dpp            - same hybrid-backpressure family as SP-BP, but penalizes
#                    retransmission cost (not hop distance) and does not
#                    weight the penalty by link_quality. Tests whether the
#                    backpressure-hybrid IDEA wins broadly, or specifically
#                    SP-BP's own formula.
#   car            - a genuinely different lineage: pure geometric progress
#                    steered by a NEIGHBOURHOOD congestion field rather than
#                    any backpressure-style differential. Tests whether non-
#                    backpressure congestion-awareness can compete at all.
#   spbp_lookahead - SP-BP extended with second-hop backlog visibility. Tests
#                    the staleness-vs-information tradeoff directly: does
#                    looking further ahead help (more info) or hurt (staler,
#                    second-hand data), the same question the M4 GNN-depth
#                    ablation asks, answered here with a hand-built teacher
#                    before any learned model exists.
PANEL = ['dijkstra', 'gpsr', 'backpressure', 'spbp', 'da_gpsr',
        'dpp', 'car', 'spbp_lookahead']


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

        # Drain phase (design spec M3 §6): stop generating new packets this many
        # seconds before episode end, but keep simulating so in-flight packets
        # terminate naturally instead of being counted as 'episode_end' drops.
        # Default 0.0 => no drain => bit-identical to M2 behaviour (G2 still valid).
        self.drain_time = float(config.get('drain_time', 0.0))
        self.gen_cutoff_t = max(self.duration - self.drain_time, 0.0)
        self.n_generated_predrain = 0
        self.n_delivered_predrain = 0

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
        # M3-audit diagnostics, measured DURING routing (the previous versions
        # were sampled post-hoc on a frozen graph and were therefore meaningless).
        self.n_decisions = 0         # routing decisions taken
        self.n_overrides = 0         # teacher choice overridden by loop-avoidance
        self.n_bp_zerodiff = 0       # backpressure decisions with no queue gradient
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

    # ── measured channel conditions (what nodes OBSERVE) ─────────────────────
    def _channel_state(self):
        """Precompute the per-frame quantities needed to estimate what each node
        MEASURES about its links under current load.

        WHY THIS EXISTS: real protocols do not compute interference from theory —
        they MEASURE it (observed PER, ETX probing, delivery statistics). The
        graph handed to teachers must reflect current channel conditions, or the
        congestion-aware teachers are blind to congestion. Building the graph with
        interference_mw=0 made packet_error_rate identically 0, which silently
        collapsed ETX-Dijkstra into plain hop-count Dijkstra and stripped the
        congestion signal out of link_quality entirely.

        WHY PROBABILITY, NOT EXPECTED POWER: link quality is convex in
        interference power, so evaluating quality at the MEAN interference badly
        underestimates the mean quality (Jensen). Physically, interference here is
        bursty and near-binary — G1 showed a single hidden terminal is usually
        enough to kill a packet — so the meaningful measured quantity is
        P(at least one lethal interferer fires this slot), which is exactly what a
        node's observed loss rate reports.

        Returns (dist_matrix (N,N), rx_power_matrix (N,N)).
        """
        pos = np.array([d.pos for d in self.drones])             # (N,3)
        diff = pos[:, None, :] - pos[None, :, :]
        dist = np.linalg.norm(diff, axis=2)                      # (N,N)
        rxp = lm2.rx_power_mw_array(dist)                        # (N,N)
        return dist, rxp

    def _measured_link(self, i, j, dist_ij, dist_m, rxp_m, cs_range, if_range):
        """Estimate the link features node j measures for link (i->j) under
        current load: probability a lethal interferer fires, folded into PER and
        link_quality alongside Bianchi contention."""
        signal_mw = lm2.rx_power_mw(dist_ij)
        # clean-channel reference
        _, sinr_clean, lq_clean, per_clean = lm2.compute_link_features_v2(
            dist_ij, interference_mw=0.0, n_contenders=1, shadowing_db=0.0)

        # Candidate interferers at receiver j: in interference range of j, and
        # outside carrier-sense range of transmitter i (hidden terminals — nodes
        # near i defer via CSMA and so do not transmit concurrently).
        d_to_j = dist_m[j]
        d_to_i = dist_m[i]
        cand = (d_to_j <= if_range) & (d_to_i > cs_range)
        cand[i] = False
        cand[j] = False
        if not cand.any():
            p_clear = 1.0
        else:
            # Which candidates are individually lethal (drive SINR below the
            # decode knee)? PER is a near-step function of SINR, so treat any
            # interferer that pushes SINR under the knee as fatal.
            interf_pw = rxp_m[j][cand]
            sinr_lin = signal_mw / (lm2._NOISE_MW + interf_pw)
            lethal = sinr_lin < LETHAL_SINR_LINEAR
            acts = self.activity[cand]
            p_clear = float(np.prod(1.0 - acts[lethal])) if lethal.any() else 1.0

        p_interf = 1.0 - p_clear

        # Bianchi contention from carrier-sense neighbours of the transmitter
        n_cont = 1 + int(round(float(self.activity[(d_to_i <= cs_range)].sum()
                                     - self.activity[i])))
        p_coll = lm2.bianchi_collision_prob(max(n_cont, 1))

        per = 1.0 - (1.0 - per_clean) * (1.0 - p_interf) * (1.0 - p_coll)
        # FIX (M3 audit): lq must fold in MAC contention too. Previously lq was
        # lq_clean * p_clear only -- it carried the hidden-terminal term but NOT
        # the Bianchi collision term, while per carried both. That left every
        # teacher scoring on link_quality (da_gpsr, backpressure, spbp) blind to
        # MAC contention, and made lq and per describe different physics. The
        # omission is material: p_coll is 0.10 at 2 contenders and 0.58 at 8.
        lq = lq_clean * p_clear * (1.0 - p_coll)
        return float(np.clip(lq, 0.0, 1.0)), float(np.clip(per, 0.0, 1.0)), sinr_clean

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
        # Per-frame channel state used to estimate measured link statistics.
        dist_m, rxp_m = self._channel_state()
        cs_range = lm2.CARRIER_SENSE_MULT * self.comm_range
        if_range = lm2.INTERFERENCE_RANGE_MULT * self.comm_range

        for i, j in itertools.combinations(range(self.N), 2):
            di, dj = self.drones[i], self.drones[j]
            dist = float(np.linalg.norm(di.pos - dj.pos))
            if dist > self.comm_range:
                continue
            if not lm2.link_exists(dist, self.comm_range):
                continue

            # Base (interference-free) features — retained for diagnostics and
            # for the interference-off ablation.
            _, _, lq_base, per_base = lm2.compute_link_features_v2(
                dist, interference_mw=0.0, n_contenders=1, shadowing_db=0.0)

            # Measured features under current load. Interference at the receiver
            # excludes the transmitter's own contribution (a node does not
            # interfere with the link it is transmitting on). Undirected graph:
            # use the mean of both directions so one edge attribute is a fair
            # summary of a bidirectional link.
            if self.interference_on:
                # Undirected edge: average both directions so one attribute is a
                # fair summary of the bidirectional link.
                lq_ij, per_ij, s_ij = self._measured_link(
                    i, j, dist, dist_m, rxp_m, cs_range, if_range)
                lq_ji, per_ji, s_ji = self._measured_link(
                    j, i, dist, dist_m, rxp_m, cs_range, if_range)
                rssi = lm2.rx_power_dbm(dist)
                sinr = 0.5 * (s_ij + s_ji)
                lq = 0.5 * (lq_ij + lq_ji)
                per = 0.5 * (per_ij + per_ji)
            else:
                rssi = lm2.rx_power_dbm(dist)
                _, sinr, lq, per = lm2.compute_link_features_v2(
                    dist, interference_mw=0.0, n_contenders=1, shadowing_db=0.0)
            lifetime = lm2.estimate_link_lifetime(di.pos, dj.pos, di.vel, dj.vel,
                                                  self.comm_range)
            rel_vel = float(np.linalg.norm(di.vel - dj.vel))
            G.add_edge(i, j, distance=dist, relative_velocity=rel_vel,
                       link_quality=lq,               # measured, load-dependent
                       base_link_quality=lq_base,     # interference-free reference
                       base_packet_error_rate=per_base,
                       estimated_link_lifetime=lifetime,
                       rssi=rssi, snr=sinr,
                       packet_error_rate=per)         # measured, load-dependent
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

        self.n_decisions += 1
        if self.actor_name == 'random':
            next_hop = self.actor(G, c, dst, rng=self.rng)   # seeded per episode
        elif self.actor_name == 'backpressure':
            next_hop, zerodiff = self.actor(G, c, dst, return_zerodiff_flag=True)
            if zerodiff:
                self.n_bp_zerodiff += 1
        else:
            next_hop = self.actor(G, c, dst)
        # avoid revisiting nodes already on this packet's path (loop-free actor).
        # NOTE: the override picks the first unvisited neighbour rather than
        # re-scoring with the teacher's own rule, so it is arbitrary. It does not
        # affect teachers equally -- wandering-prone policies (pure backpressure)
        # trigger it far more than inherently loop-free shortest-path policies.
        # Instrumented here so the effect is visible rather than silently folded
        # into PDR; a full fix (re-score on a visited-excluded subgraph) is M4.
        if next_hop is not None and next_hop in pkt.path:
            self.n_overrides += 1
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
            if pkt.gen_time < self.gen_cutoff_t:
                self.n_delivered_predrain += 1
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
        # Per-teacher degeneracy counters are module-level in routing_teachers_v2,
        # so they must be reset per episode or they accumulate across runs (and,
        # under ProcessPoolExecutor, would silently mix runs sharing a worker).
        _rt2.reset_teacher_stats()
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

            # 3. generate packets for this frame (suppressed during drain phase)
            for f in self.flows:
                time_since_gen[f['flow_id']] += FRAME_DT
                interval = 1.0 / f['packet_rate']
                while time_since_gen[f['flow_id']] >= interval:
                    time_since_gen[f['flow_id']] -= interval
                    if t_frame >= self.gen_cutoff_t:
                        continue          # drain phase: no new packets
                    pkt = PacketV2(self.packet_counter, f['flow_id'],
                                   f['source_id'], f['destination_id'], t_frame)
                    self.n_generated_predrain += 1
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
            'drain_time': self.drain_time,
            'n_generated_predrain': self.n_generated_predrain,
            'n_delivered_predrain': self.n_delivered_predrain,
            # Headline metric when drain_time > 0: PDR over packets generated
            # before the cutoff, so packets that simply ran out of episode are
            # not counted as failures (design spec M3 §6).
            'pdr_predrain': (self.n_delivered_predrain /
                             max(self.n_generated_predrain, 1)),
            'n_decisions': self.n_decisions,
            'override_rate': self.n_overrides / max(self.n_decisions, 1),
            'bp_zerodiff_rate': self.n_bp_zerodiff / max(self.n_decisions, 1),
            # Per-teacher degeneracy for the ACTOR of this episode: did it ever
            # abandon its own rule (fallback), or run its rule with no
            # discriminating signal (flat)? See routing_teachers_v2._TEACHER_STATS.
            'teacher_stats': _rt2.get_teacher_stats(),
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
