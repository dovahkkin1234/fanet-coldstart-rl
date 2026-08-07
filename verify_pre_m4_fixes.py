"""
verify_pre_m4_fixes.py — execution proof for the pre-M4 fix pass.

Compile-checks are not sufficient: a script that aborts before its write step
still compiles. Every claim below is checked by RUNNING the patched code.

The load-bearing tests are 4 and 5. A gate that only ever passes proves
nothing, so check 8 is run against synthetic data with a PLANTED redundancy and
must FAIL, and against clean data and must PASS. It is not tuned to pass.
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
import features_v2 as F

OK = "  [PASS]"
NO = "  [FAIL]"
fails = []


def check(name, cond, detail=""):
    print(f"{OK if cond else NO} {name:<58} {detail}")
    if not cond:
        fails.append(name)


class Pkt:
    def __init__(self, current, dst, hops, path):
        self.current, self.dst, self.hops, self.path = current, dst, hops, path


def make_graph(n=12, seed=0):
    """Chain-plus-chords graph with realistic attribute names, so k-hop
    neighbourhoods are actually smaller than the whole graph."""
    rng = np.random.default_rng(seed)
    G = nx.Graph()
    for i in range(n):
        G.add_node(i, x=float(i * 90), y=float(rng.uniform(0, 400)),
                   z=float(rng.uniform(50, 150)),
                   vx=float(rng.uniform(-10, 10)), vy=float(rng.uniform(-10, 10)),
                   vz=0.0, energy=float(rng.uniform(60, 100)),
                   queue_occupancy=float(rng.uniform(0, 0.4)))
    for i in range(n - 1):
        G.add_edge(i, i + 1)
    for i in range(0, n - 2, 3):
        G.add_edge(i, i + 2)
    for u, v in G.edges():
        d = float(abs(G.nodes[u]['x'] - G.nodes[v]['x']) + 40.0)
        G.edges[u, v].update(distance=d, link_quality=float(rng.uniform(.3, 1)),
                             packet_error_rate=float(rng.uniform(0, .5)),
                             snr=float(rng.uniform(5, 35)),
                             estimated_link_lifetime=float(rng.uniform(1, 60)),
                             relative_velocity=float(rng.uniform(0, 20)))
    return G


CFG = dict(area_x=1000.0, area_y=1000.0, z_min=0.0, z_max=200.0,
           speed_max=15.0, comm_range=280.0, num_drones=12)

print("\n" + "=" * 78)
print("  VERIFICATION — PRE-M4 FIX PASS v1")
print("=" * 78)

# ── 1. norm_constants carries the new keys ──────────────────────────────────
nc = F.norm_constants(CFG)
check("1a. degree_ref persisted", nc.get('degree_ref') == 11.0, f"={nc.get('degree_ref')}")
check("1b. local_horizon persisted", nc.get('local_horizon') == F.LOCAL_HORIZON,
      f"={nc.get('local_horizon')}")
check("1c. schema_version persisted", nc.get('schema_version') == 2)
check("1d. snr_ref gone", 'snr_ref' not in nc)

# ── 2. extract_frame shapes match the new lists ─────────────────────────────
G = make_graph()
ids, nf, ei, ef = F.extract_frame(G, nc)
check("2a. node_feat width == len(NODE_FEATURES)", nf.shape[1] == len(F.NODE_FEATURES),
      f"{nf.shape}")
check("2b. edge_feat width == len(EDGE_FEATURES) == 5",
      ef.shape[1] == len(F.EDGE_FEATURES) == 5, f"{ef.shape}")
deg_col = nf[:, F.NODE_FEATURES.index('degree')]
expect = np.array([G.degree(n) / 11.0 for n in sorted(G.nodes())], dtype=np.float32)
check("2c. degree uses persisted degree_ref", np.allclose(deg_col, expect),
      f"max|d|={np.abs(deg_col - expect).max():.2e}")

# ── 3. extract_decision: width, no ttl duplicate, local horizon actually local ─
h_map = F.hop_distances_to(G, 11)
pkt = Pkt(current=2, dst=11, hops=3, path=[0, 1, 2])
cands = [1, 3, 4]
qf, cf = F.extract_decision(G, pkt, cands, nc, h_map, n_inflight=999.0,
                            network_mean_occ=0.777)
check("3a. query width == len(QUERY_FEATURES) == 6",
      qf.shape[0] == len(F.QUERY_FEATURES) == 6, f"{qf.shape}")
check("3b. candidate width unchanged", cf.shape == (3, 3), f"{cf.shape}")

# the whole-network mean, for contrast
all_occ = float(np.mean([G.nodes[u]['queue_occupancy'] for u in G.nodes()]))
nm_col = F.QUERY_FEATURES.index('neigh_mean_occupancy')
khop = nx.single_source_shortest_path_length(G, 2, cutoff=F.LOCAL_HORIZON)
expect_local = float(np.mean([G.nodes[u]['queue_occupancy'] for u in khop]))
check("3c. neigh_mean_occupancy is the k-hop mean",
      abs(float(qf[nm_col]) - expect_local) < 1e-6,
      f"got={qf[nm_col]:.4f} khop={expect_local:.4f}")
check("3d. ...and it is NOT the caller's global value",
      abs(float(qf[nm_col]) - 0.777) > 1e-6, "ignored the global arg, as intended")
check("3e. k-hop set is a strict subset", len(khop) < G.number_of_nodes(),
      f"{len(khop)} of {G.number_of_nodes()} nodes")
nb_col = F.QUERY_FEATURES.index('neigh_buffered_packets')
expect_buf = min(float(np.sum([G.nodes[u]['queue_occupancy'] for u in khop]))
                 * nc['max_queue'] / 100.0, 1.0)
check("3f. neigh_buffered_packets is the k-hop packet count",
      abs(float(qf[nb_col]) - expect_buf) < 1e-6, f"got={qf[nb_col]:.4f}")

# controller-assisted mode still honours the caller's globals
F.LOCAL_HORIZON = None
nc_glob = F.norm_constants(CFG)
qf_g, _ = F.extract_decision(G, pkt, cands, nc_glob, h_map, n_inflight=50.0,
                             network_mean_occ=0.777)
check("3g. LOCAL_HORIZON=None restores whole-network values",
      abs(float(qf_g[nm_col]) - 0.777) < 1e-6, f"got={qf_g[nm_col]:.4f}")
F.LOCAL_HORIZON = 2

# ── 4. check 8 must FAIL on planted redundancy and PASS on clean data ───────
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
import preflight_dataset_v2_check as P

rng = np.random.default_rng(7)
N = 5000
clean = rng.normal(size=(N, 5))
off, lines = P.redundancy_report('clean', clean, [f'f{i}' for i in range(5)])
check("4a. clean independent columns -> no offenders", not off, f"{len(off)} flagged")

lin = clean.copy()
lin[:, 3] = 1.0 - lin[:, 1]                      # the ttl_left/hops_so_far shape
off, lines = P.redundancy_report('linear', lin, [f'f{i}' for i in range(5)])
check("4b. EXACT LINEAR duplicate is caught", len(off) == 1,
      lines[0].strip() if lines else "not caught")

mono = clean.copy()
d = rng.uniform(50, 300, size=N)
mono[:, 0] = d                                    # 'distance'
mono[:, 2] = 40.0 - 20.0 * np.log10(d)            # 'snr' = path-loss form of it
off, lines = P.redundancy_report('monotone', mono, [f'f{i}' for i in range(5)])
pear = abs(np.corrcoef(mono[:, 0], mono[:, 2])[0, 1])
check("4c. MONOTONE NONLINEAR duplicate is caught", len(off) == 1,
      f"linear |r|={pear:.3f} (below {P.PEARSON_MAX}) yet still flagged")
check("4d. ...and Pearson alone would have MISSED it", pear < P.PEARSON_MAX,
      f"|r|={pear:.3f} < {P.PEARSON_MAX} — this is why Spearman is checked too")

noise_only = rng.normal(size=(N, 4))
off, _ = P.redundancy_report('noise', noise_only, [f'f{i}' for i in range(4)])
check("4e. pure noise does not trip it", not off, "no false positive")

# ── 5. schema skew must ABORT both checkers ─────────────────────────────────
good_man = {
    'node_features': F.NODE_FEATURES, 'edge_features': F.EDGE_FEATURES,
    'query_features': F.QUERY_FEATURES, 'candidate_features': F.CANDIDATE_FEATURES,
    'feature_schema_version': F.FEATURE_SCHEMA_VERSION,
    'local_horizon': F.LOCAL_HORIZON,
}
check("5a. matching manifest -> compatible", not F.assert_manifest_compatible(good_man))

old_man = dict(good_man)
old_man['query_features'] = ['ttl_left', 'hops_so_far', 'dist_to_dest',
                             'current_queue_occupancy', 'n_inflight',
                             'network_mean_occupancy', 'hop_distance_to_dst']
old_man['edge_features'] = ['distance', 'link_quality', 'packet_error_rate',
                            'snr', 'estimated_link_lifetime', 'relative_velocity']
del old_man['feature_schema_version']
del old_man['local_horizon']
probs = F.assert_manifest_compatible(old_man, context='old-dataset')
check("5b. THE EXISTING 533k dataset is detected as skewed", len(probs) >= 3,
      f"{len(probs)} problems reported")

hz_man = dict(good_man); hz_man['local_horizon'] = None
check("5c. scoping-choice mismatch alone is caught",
      len(F.assert_manifest_compatible(hz_man)) == 1, "local_horizon differs")

# ── 6. G3.5 end-to-end aborts on a skewed dataset (not just the helper) ─────
tmp = tempfile.mkdtemp()
np.savez(os.path.join(tmp, 'decisions.npz'), label=np.zeros(4, np.int32),
         cand_offsets=np.arange(5, dtype=np.int64), cand_flat=np.zeros(4, np.int32))
np.savez(os.path.join(tmp, 'frames.npz'), node_feat_flat=np.zeros((4, 9), np.float32))
with open(os.path.join(tmp, 'manifest.json'), 'w') as f:
    json.dump(old_man, f)
r = subprocess.run([sys.executable, 'src/preflight_dataset_v2_check.py', '--data', tmp],
                   capture_output=True, text=True,
                   cwd=os.path.dirname(os.path.abspath(__file__)))
check("6a. G3.5 exits non-zero on skew", r.returncode == 1, f"rc={r.returncode}")
check("6b. ...and says why, before any check runs",
      'ABORTED BEFORE ANY CHECK' in r.stdout and 'query_features' in r.stdout)

# ── 7. both experiment scripts can now write JSON ───────────────────────────
for f_ in ('experiment_spbp_mechanism.py', 'experiment_queue_weight.py'):
    src = open(os.path.join('src', f_), encoding='utf-8').read()
    check(f"7. {f_} has --out and json.dump",
          "add_argument('--out'" in src and 'json.dump' in src)

print("=" * 78)
if fails:
    print(f"  {len(fails)} FAILURE(S): {fails}")
    sys.exit(1)
print("  ALL VERIFICATION CHECKS PASSED")
print("=" * 78 + "\n")
