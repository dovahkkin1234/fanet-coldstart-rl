"""
eval_routers.py
Analysis 3: XGBoost warm-start model vs all classical baselines as routing policies.

All routers run on IDENTICAL episodes (same seed = same drone positions, same
traffic flows). PDR differences are purely due to routing logic, not randomness.

Routers compared:
  1. Random          — lower bound sanity check
  2. Dijkstra        — shortest hop-count path
  3. GPSR            — greedy geographic forwarding
  4. Stable-Path     — weighted quality + lifetime + progress
  5. AODV-like       — reactive with route cache
  6. Link-Lifetime   — prefers long-lived links
  7. XGBoost         — our supervised warm-start model

Eval scenarios (4 levels of difficulty):
  pilot_01 — easy    (10 drones, dense, slow)
  pilot_02 — medium  (20 drones, balanced)  [trained on]
  pilot_03 — hard    (30 drones, fast, sparse) [TEST — unseen]
  pilot_05 — hard    (25 drones, very sparse)  [TEST — unseen]

Usage:
    python src\\eval_routers.py
    python src\\eval_routers.py --episodes 5 --duration 120   # quick test
    python src\\eval_routers.py --model models\\xgboost_warmstart.pkl
"""

import os, sys, argparse, pickle, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
from itertools import combinations
from pathlib import Path

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mobility import DroneRWP
from link_model import compute_link_features, estimate_link_lifetime, RSSI_SENSITIVITY
from models import EnergyModel, NodeQueue
from routing_teachers import (dijkstra_next_hop, gpsr_next_hop,
                               stable_path_next_hop, AODVLikeRouter,
                               link_lifetime_next_hop)

TTL           = 20
MAX_QUEUE     = 50
INITIAL_ENERGY = 100.0
HOP_DELAY_MS  = 2.0

# Must exactly match FEATURE_COLS in train_warmstart.py
FEATURE_COLS = [
    'candidate_distance', 'candidate_relative_velocity', 'candidate_link_quality',
    'candidate_rssi', 'candidate_packet_error_rate', 'candidate_estimated_lifetime',
    'candidate_energy', 'candidate_queue_occupancy', 'candidate_dist_to_dest',
    'candidate_progress', 'current_energy', 'current_queue_occupancy',
    'current_num_neighbors', 'current_speed', 'distance_to_destination',
    'bearing_to_destination', 'destination_reachable', 'avg_degree',
    'network_density', 'num_components', 'avg_link_quality', 'topology_change_rate',
]

# Eval scenarios: (name, config_dict)
EVAL_SCENARIOS = {
    'pilot_01_easy':   dict(num_drones=10, area_x=500,  area_y=500,  speed_min=5,  speed_max=15, comm_range=200),
    'pilot_02_medium': dict(num_drones=20, area_x=1000, area_y=1000, speed_min=10, speed_max=25, comm_range=250),
    'pilot_03_hard':   dict(num_drones=30, area_x=1500, area_y=1500, speed_min=20, speed_max=40, comm_range=250),
    'pilot_05_sparse': dict(num_drones=25, area_x=2000, area_y=2000, speed_min=5,  speed_max=15, comm_range=300),
}

SHARED_DEFAULTS = dict(z_min=50, z_max=150, pause_max=5.0,
                       timestep=0.5, packet_rate=1.0)

# ─── Packet ──────────────────────────────────────────────────────────────────

class Packet:
    __slots__ = ('pid','flow_id','src','dst','gen_time','current',
                 'path','hops','energy_used','delivered','dropped',
                 'drop_reason','cum_delay_ms')
    def __init__(self, pid, flow_id, src, dst, gen_time):
        self.pid=pid; self.flow_id=flow_id; self.src=src; self.dst=dst
        self.gen_time=gen_time; self.current=src; self.path=[src]
        self.hops=0; self.energy_used=0.0; self.delivered=False
        self.dropped=False; self.drop_reason=None; self.cum_delay_ms=0.0

# ─── Router wrappers ─────────────────────────────────────────────────────────

class RandomRouter:
    name = 'Random'
    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)
    def reset(self): pass
    def next_hop(self, G, c, dst, energy, queues, topo, comp):
        nbrs = list(G.neighbors(c))
        return int(self.rng.choice(nbrs)) if nbrs else None

class DijkstraRouter:
    name = 'Dijkstra'
    def reset(self): pass
    def next_hop(self, G, c, dst, energy, queues, topo, comp):
        return dijkstra_next_hop(G, c, dst)

class GPSRRouter:
    name = 'GPSR'
    def reset(self): pass
    def next_hop(self, G, c, dst, energy, queues, topo, comp):
        return gpsr_next_hop(G, c, dst)

class StablePathRouter:
    name = 'Stable-Path'
    def reset(self): pass
    def next_hop(self, G, c, dst, energy, queues, topo, comp):
        return stable_path_next_hop(G, c, dst)

class AODVRouter:
    name = 'AODV-like'
    def __init__(self): self._r = AODVLikeRouter()
    def reset(self): self._r = AODVLikeRouter()   # fresh cache each episode
    def next_hop(self, G, c, dst, energy, queues, topo, comp):
        return self._r.next_hop(G, c, dst)

class LinkLifetimeRouter:
    name = 'Link-Lifetime'
    def reset(self): pass
    def next_hop(self, G, c, dst, energy, queues, topo, comp):
        return link_lifetime_next_hop(G, c, dst)

class XGBoostRouter:
    name = 'XGBoost'
    def __init__(self, model):
        self.model = model

    def reset(self): pass

    def next_hop(self, G, c, dst, energy, queues, topo, comp):
        nbrs = list(G.neighbors(c))
        if not nbrs:
            return None

        c_node = G.nodes[c]
        d_node = G.nodes[dst]
        c_pos  = np.array([c_node['x'], c_node['y'], c_node['z']])
        d_pos  = np.array([d_node['x'], d_node['y'], d_node['z']])

        dist_to_dest   = float(np.linalg.norm(d_pos - c_pos))
        bearing        = float(np.arctan2(d_pos[1]-c_pos[1], d_pos[0]-c_pos[0]))
        dest_reachable = int(comp.get(c, -1) == comp.get(dst, -2))

        rows = []
        for n in nbrs:
            e    = G.edges[c, n]
            n_pos = np.array([G.nodes[n]['x'], G.nodes[n]['y'], G.nodes[n]['z']])
            dist_n = float(np.linalg.norm(d_pos - n_pos))
            progress = (dist_to_dest - dist_n) / max(dist_to_dest, 1e-6)
            rows.append([
                e.get('distance', 0),
                e.get('relative_velocity', 0),
                e.get('link_quality', 0),
                e.get('rssi', -90),
                e.get('packet_error_rate', 1),
                e.get('estimated_link_lifetime', 0),
                energy[n], queues[n].occupancy,
                dist_n, progress,
                energy[c], queues[c].occupancy, len(nbrs),
                c_node.get('speed', 0),
                dist_to_dest, bearing, dest_reachable,
                topo['avg_degree'], topo['network_density'],
                topo['num_components'], topo['avg_link_quality'],
                topo['topology_change_rate'],
            ])

        X      = np.array(rows, dtype=np.float32)
        scores = self.model.predict_proba(X)[:, 1]
        return nbrs[int(np.argmax(scores))]

# ─── Lightweight evaluation simulator ────────────────────────────────────────

class EvalSimulator:
    """
    Stripped-down FANET simulator: no CSV logging.
    Runs one episode with a given router and returns PDR / delay / hops.
    """
    def __init__(self, cfg, seed):
        self.N         = cfg['num_drones']
        self.area_x    = cfg['area_x']
        self.area_y    = cfg['area_y']
        self.z_min     = cfg['z_min']
        self.z_max     = cfg['z_max']
        self.comm_range= cfg['comm_range']
        self.dt        = cfg['timestep']
        self.duration  = cfg['duration']
        self.pkt_rate  = cfg['packet_rate']
        self.seed      = seed

        self.rng = np.random.default_rng(seed)

        self.drones = [
            DroneRWP(i, self.area_x, self.area_y, self.z_min, self.z_max,
                     cfg['speed_min'], cfg['speed_max'], cfg['pause_max'],
                     seed=seed)
            for i in range(self.N)
        ]
        self.energy    = [INITIAL_ENERGY] * self.N
        self.queues    = [NodeQueue(MAX_QUEUE) for _ in range(self.N)]
        self.num_flows = max(1, self.N // 4)
        self.flows     = self._make_flows()
        self.prev_links = set()

    def _make_flows(self):
        flows = []
        for fid in range(self.num_flows):
            src, dst = self.rng.choice(self.N, size=2, replace=False)
            flows.append({'flow_id': fid, 'src': int(src), 'dst': int(dst)})
        return flows

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
            lt     = estimate_link_lifetime(di.pos, dj.pos, di.vel, dj.vel, self.comm_range)
            rv     = float(np.linalg.norm(di.vel - dj.vel))
            G.add_edge(i, j, distance=dist, relative_velocity=rv,
                       link_quality=lq, estimated_link_lifetime=lt,
                       rssi=rssi, snr=snr, packet_error_rate=per)
            cur_links.add((i, j))

        sym_diff   = len(self.prev_links.symmetric_difference(cur_links))
        union_size = max(len(self.prev_links) + len(cur_links), 1)
        change_rate = sym_diff / union_size
        self.prev_links = cur_links

        n_edges  = G.number_of_edges()
        max_e    = self.N * (self.N - 1) / 2
        comps    = list(nx.connected_components(G))
        degrees  = [d for _, d in G.degree()]

        topo = {
            'avg_degree':         float(np.mean(degrees)) if degrees else 0.0,
            'network_density':    n_edges / max_e if max_e > 0 else 0.0,
            'num_components':     len(comps),
            'avg_link_quality':   float(np.mean([G.edges[e]['link_quality'] for e in G.edges])) if n_edges else 0.0,
            'topology_change_rate': change_rate,
        }

        comp_labels = {}
        for ci, comp in enumerate(comps):
            for node in comp:
                comp_labels[node] = ci

        return G, topo, comp_labels

    def run(self, router):
        router.reset()
        n_steps          = int(self.duration / self.dt)
        time_since_gen   = {f['flow_id']: 0.0 for f in self.flows}
        active_packets   = []
        finished_packets = []
        pkt_counter      = 0

        for step in range(n_steps):
            t = step * self.dt

            for d in self.drones:
                d.step(self.dt)
                self.energy[d.id] = EnergyModel.consume_idle(self.energy[d.id], self.dt)

            G, topo, comp = self._build_graph()

            # Generate packets
            for f in self.flows:
                time_since_gen[f['flow_id']] += self.dt
                interval = 1.0 / self.pkt_rate
                while time_since_gen[f['flow_id']] >= interval:
                    time_since_gen[f['flow_id']] -= interval
                    pkt = Packet(pkt_counter, f['flow_id'], f['src'], f['dst'], t)
                    pkt_counter += 1
                    active_packets.append(pkt)

            # Forward packets (multi-hop within timestep)
            still_active = []
            for pkt in active_packets:
                done = False
                for _ in range(TTL):
                    c, dst = pkt.current, pkt.dst

                    if c == dst:
                        pkt.delivered = True
                        done = True
                        break

                    nbrs = list(G.neighbors(c))
                    if not nbrs:
                        pkt.dropped = True; pkt.drop_reason = 'no_route'
                        done = True; break

                    # Loop guard: exclude already-visited nodes
                    unvisited = [n for n in nbrs if n not in pkt.path]
                    if not unvisited:
                        pkt.dropped = True; pkt.drop_reason = 'no_route'
                        done = True; break

                    # Ask the router
                    next_hop = router.next_hop(G, c, dst, self.energy, self.queues, topo, comp)

                    # Fall back if router picked a visited node or None
                    if next_hop is None or next_hop in pkt.path:
                        next_hop = dijkstra_next_hop(G, c, dst)
                    if next_hop is None:
                        pkt.dropped = True; pkt.drop_reason = 'no_route'
                        done = True; break

                    # Energy check
                    if not EnergyModel.is_alive(self.energy[c]):
                        pkt.dropped = True; pkt.drop_reason = 'energy_depleted'
                        done = True; break

                    # Packet error (probabilistic loss on this link)
                    per = G.edges[c, next_hop].get('packet_error_rate', 0)
                    if self.rng.random() < per:
                        pkt.dropped = True; pkt.drop_reason = 'packet_error'
                        done = True; break

                    # Queue check at receiver
                    if not self.queues[next_hop].enqueue(pkt):
                        pkt.dropped = True; pkt.drop_reason = 'queue_overflow'
                        done = True; break
                    self.queues[next_hop].dequeue()

                    # Forward
                    self.energy[c]        = EnergyModel.consume_tx(self.energy[c])
                    self.energy[next_hop] = EnergyModel.consume_rx(self.energy[next_hop])
                    pkt.energy_used += EnergyModel.TX_COST + EnergyModel.RX_COST
                    pkt.current = next_hop
                    pkt.path.append(next_hop)
                    pkt.hops += 1
                    pkt.cum_delay_ms += HOP_DELAY_MS

                    if pkt.current == dst:
                        pkt.delivered = True; done = True; break
                    if pkt.hops >= TTL:
                        pkt.dropped = True; pkt.drop_reason = 'ttl_expired'
                        done = True; break

                if done:
                    finished_packets.append(pkt)
                else:
                    still_active.append(pkt)
            active_packets = still_active

        # Drop remaining packets
        for pkt in active_packets:
            pkt.dropped = True; pkt.drop_reason = 'episode_end'
            finished_packets.append(pkt)

        # Metrics
        if not finished_packets:
            return {'pdr': 0.0, 'delay_ms': 0.0, 'hop_count': 0.0,
                    'n_packets': 0, 'drop_reasons': {}}

        delivered  = [p for p in finished_packets if p.delivered]
        n_total    = len(finished_packets)
        n_del      = len(delivered)
        pdr        = n_del / n_total if n_total else 0.0
        delay_ms   = float(np.mean([p.cum_delay_ms for p in delivered])) if delivered else 0.0
        hop_count  = float(np.mean([p.hops for p in delivered])) if delivered else 0.0
        drop_reasons = {}
        for p in finished_packets:
            if p.dropped and p.drop_reason:
                drop_reasons[p.drop_reason] = drop_reasons.get(p.drop_reason, 0) + 1

        return {'pdr': pdr, 'delay_ms': delay_ms, 'hop_count': hop_count,
                'n_packets': n_total, 'drop_reasons': drop_reasons}

# ─── Plotting ─────────────────────────────────────────────────────────────────

ROUTER_COLORS = {
    'Random':       '#adb5bd',
    'Dijkstra':     '#457b9d',
    'GPSR':         '#1d3557',
    'Stable-Path':  '#2d6a4f',
    'AODV-like':    '#52b788',
    'Link-Lifetime':'#e9c46a',
    'XGBoost':      '#e63946',
}

def plot_pdr_comparison(results_df, out_path):
    """Grouped bar chart: PDR per router per scenario."""
    scenarios = results_df['scenario'].unique()
    routers   = results_df['router'].unique()
    x         = np.arange(len(scenarios))
    width     = 0.8 / len(routers)

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, router in enumerate(routers):
        vals = []
        errs = []
        for sc in scenarios:
            sub = results_df[(results_df['scenario']==sc) & (results_df['router']==router)]
            vals.append(sub['pdr'].mean())
            errs.append(sub['pdr'].std())
        offset = (i - len(routers)/2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width*0.9, yerr=errs, capsize=3,
                      label=router, color=ROUTER_COLORS.get(router, '#888888'),
                      alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels([s.replace('_', '\n') for s in scenarios], fontsize=9)
    ax.set_ylabel('Packet Delivery Ratio (PDR)', fontsize=11)
    ax.set_title('Routing Protocol Comparison — PDR across Scenarios\n'
                 '(XGBoost = supervised warm-start model)', fontweight='bold')
    ax.legend(loc='upper right', ncol=2, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_pdr_boxplot(results_df, out_path):
    """Box plot: PDR distribution per router (all scenarios combined)."""
    routers = list(results_df['router'].unique())
    data    = [results_df[results_df['router']==r]['pdr'].values for r in routers]
    colors  = [ROUTER_COLORS.get(r, '#888888') for r in routers]

    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops={'color': 'black', 'linewidth': 2})
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
    ax.set_xticklabels(routers, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('PDR', fontsize=11)
    ax.set_title('PDR Distribution per Routing Protocol (all scenarios)',
                 fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_delay_hops(results_df, out_path):
    """Side-by-side: mean delay and mean hop count per router."""
    routers  = list(results_df['router'].unique())
    mean_del = [results_df[results_df['router']==r]['delay_ms'].mean() for r in routers]
    mean_hop = [results_df[results_df['router']==r]['hop_count'].mean() for r in routers]
    colors   = [ROUTER_COLORS.get(r, '#888888') for r in routers]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, vals, title, unit in zip(
            axes,
            [mean_del, mean_hop],
            ['Mean End-to-End Delay', 'Mean Hop Count'],
            ['ms', 'hops']):
        bars = ax.bar(routers, vals, color=colors, alpha=0.9)
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel(unit)
        ax.set_xticklabels(routers, rotation=20, ha='right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02*max(vals),
                    f'{v:.1f}', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model',    default='models/xgboost_warmstart.pkl')
    ap.add_argument('--out_dir',  default='models')
    ap.add_argument('--episodes', type=int, default=10,
                    help='Episodes per (scenario × router). Default: 10')
    ap.add_argument('--duration', type=int, default=180,
                    help='Episode duration in seconds. Default: 180')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load XGBoost model
    with open(args.model, 'rb') as f:
        bundle = pickle.load(f)
    xgb_model = bundle['model']
    print(f"\nLoaded XGBoost model from {args.model}")

    # Build router list
    routers = [
        RandomRouter(seed=42),
        DijkstraRouter(),
        GPSRRouter(),
        StablePathRouter(),
        AODVRouter(),
        LinkLifetimeRouter(),
        XGBoostRouter(xgb_model),
    ]

    # Seeds for eval episodes — completely separate from training seeds
    # Training used: config_index*1000 + episode (0–9999 range)
    # Eval uses: 90000 + scenario_offset*100 + episode
    eval_seeds = [90000 + ep for ep in range(args.episodes)]

    print(f"\nEval config: {args.episodes} episodes × "
          f"{len(EVAL_SCENARIOS)} scenarios × {len(routers)} routers = "
          f"{args.episodes*len(EVAL_SCENARIOS)*len(routers)} runs")
    print(f"Episode duration: {args.duration}s\n")

    all_rows  = []
    t0 = time.time()

    for sc_name, sc_cfg in EVAL_SCENARIOS.items():
        cfg = {**SHARED_DEFAULTS, **sc_cfg, 'duration': args.duration}
        print(f"\n{'═'*60}")
        print(f"  Scenario: {sc_name}")
        print(f"  {sc_cfg['num_drones']} drones | "
              f"{sc_cfg['area_x']}×{sc_cfg['area_y']}m | "
              f"speed {sc_cfg['speed_min']}–{sc_cfg['speed_max']} m/s | "
              f"range {sc_cfg['comm_range']}m")
        print(f"{'─'*60}")
        print(f"  {'Router':<14} {'PDR':>6}  {'±std':>5}  "
              f"{'Delay(ms)':>9}  {'Hops':>5}  {'Pkts':>6}")
        print(f"  {'─'*55}")

        for router in routers:
            ep_results = []
            for seed in eval_seeds:
                sim    = EvalSimulator(cfg, seed=seed)
                result = sim.run(router)
                ep_results.append(result)
                all_rows.append({
                    'scenario': sc_name,
                    'router':   router.name,
                    'seed':     seed,
                    **result,
                })

            mean_pdr   = np.mean([r['pdr']       for r in ep_results])
            std_pdr    = np.std( [r['pdr']       for r in ep_results])
            mean_delay = np.mean([r['delay_ms']  for r in ep_results])
            mean_hops  = np.mean([r['hop_count'] for r in ep_results])
            mean_pkts  = np.mean([r['n_packets'] for r in ep_results])

            marker = '  ◀ our model' if router.name == 'XGBoost' else ''
            print(f"  {router.name:<14} {mean_pdr:>6.3f}  "
                  f"{std_pdr:>5.3f}  {mean_delay:>9.1f}  "
                  f"{mean_hops:>5.1f}  {mean_pkts:>6.0f}{marker}")

    total_time = time.time() - t0
    print(f"\nTotal evaluation time: {total_time:.1f}s")

    # ── Build results table ───────────────────────────────────────────────────
    df = pd.DataFrame(all_rows)

    # Summary table: mean ± std across all scenarios
    summary = (df.groupby('router')
                 .agg(mean_pdr=('pdr','mean'), std_pdr=('pdr','std'),
                      mean_delay=('delay_ms','mean'),
                      mean_hops=('hop_count','mean'))
                 .round(4))
    # Force router order
    router_order = [r.name for r in routers]
    summary = summary.reindex([r for r in router_order if r in summary.index])

    print(f"\n{'═'*60}")
    print("  OVERALL SUMMARY (all scenarios combined)")
    print(f"{'═'*60}")
    print(f"  {'Router':<14} {'PDR':>6}  {'±std':>5}  "
          f"{'Delay(ms)':>9}  {'Hops':>5}")
    print(f"  {'─'*50}")
    for router_name, row in summary.iterrows():
        marker = '  ◀ our model' if router_name == 'XGBoost' else ''
        print(f"  {router_name:<14} {row['mean_pdr']:>6.3f}  "
              f"{row['std_pdr']:>5.3f}  {row['mean_delay']:>9.1f}  "
              f"{row['mean_hops']:>5.1f}{marker}")

    # XGBoost vs best classical baseline comparison
    classical = summary.drop('Random', errors='ignore').drop('XGBoost', errors='ignore')
    if not classical.empty:
        best_classical_name = classical['mean_pdr'].idxmax()
        best_pdr  = classical.loc[best_classical_name, 'mean_pdr']
        xgb_pdr   = summary.loc['XGBoost', 'mean_pdr'] if 'XGBoost' in summary.index else None
        if xgb_pdr is not None:
            delta = (xgb_pdr - best_pdr) * 100
            print(f"\n  XGBoost vs best classical ({best_classical_name}): "
                  f"{'+'if delta>=0 else ''}{delta:.1f}% PDR")

    # Save CSVs
    csv_path = os.path.join(args.out_dir, 'router_comparison.csv')
    df.to_csv(csv_path, index=False)
    summary_path = os.path.join(args.out_dir, 'router_comparison_summary.csv')
    summary.to_csv(summary_path)
    print(f"\n  Raw results  → {csv_path}")
    print(f"  Summary      → {summary_path}")

    # Plots
    plot_pdr_comparison(df, os.path.join(args.out_dir, 'router_pdr_comparison.png'))
    plot_pdr_boxplot(df,    os.path.join(args.out_dir, 'router_pdr_boxplot.png'))
    plot_delay_hops(df,     os.path.join(args.out_dir, 'router_delay_hops.png'))
    print(f"  PDR bar chart→ {os.path.join(args.out_dir, 'router_pdr_comparison.png')}")
    print(f"  PDR boxplot  → {os.path.join(args.out_dir, 'router_pdr_boxplot.png')}")
    print(f"  Delay/hops   → {os.path.join(args.out_dir, 'router_delay_hops.png')}")
    print(f"\n  Done.")


if __name__ == '__main__':
    main()
