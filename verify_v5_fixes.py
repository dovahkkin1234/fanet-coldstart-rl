"""
verify_v5_fixes.py — execution proof for the full v1-v5 stack.

    python verify_v5_fixes.py --src src

SUPERSEDES verify_v4_fixes.py. Three of that file's assertions encode v4-era
state that v5 deliberately changed, so it now reports 3 FAILURES on a correct
tree:
    3b  asserted FEATURE_SCHEMA_VERSION == 3   -> v5 J1 bumped it to 4
    4a  asserted the cand_reachable exemption exists
    4b  asserted it is reported                -> v5 J2 removed it, because
        measurement (|r|=0.8391, |rho|=0.2597) showed the pair is NOT
        redundant and a standing exemption for a non-redundant pair would
        silence the gate if it ever became one.
Delete verify_v4_fixes.py once this passes.

Tests 1b and 4b are the load-bearing ones. The strengthened kinf control is
re-run against a rebuilt copy of the OLD proxy logic and must reject it; the
redundancy exemption is checked to be NARROW, i.e. it must still fail on a
pair that is not on the list.
"""
import argparse
import os
import sys

import numpy as np
import networkx as nx

ap = argparse.ArgumentParser()
ap.add_argument('--src', default='src')
args = ap.parse_args()
SRC = os.path.abspath(args.src)
sys.path.insert(0, SRC)

OK, NO = "  [PASS]", "  [FAIL]"
fails = []


def check(name, cond, detail=""):
    print(f"{OK if cond else NO} {name:<58} {detail}")
    if not cond:
        fails.append(name)


def sparse_graph(rng):
    n = int(rng.integers(5, 14))
    p = 0.12 + 0.23 * rng.random()
    G = nx.Graph(); G.graph['comm_range'] = 250.0
    for i in range(n):
        G.add_node(i, x=float(rng.integers(0, 900)), y=float(rng.integers(0, 900)),
                   z=100.0, energy=90.0, queue_occupancy=float(rng.random()),
                   queue_len=float(rng.integers(0, 6)))
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                G.add_edge(i, j, distance=float(rng.integers(50, 250)),
                           link_quality=float(rng.random()),
                           packet_error_rate=float(rng.random()) * 0.3)
    return G, n


print("\n" + "=" * 78)
print("  VERIFICATION — FULL v1-v5 STACK")
print("=" * 78)

# ── 1. H1/H2: kinf now matches panel SP-BP ─────────────────────────────────
import routing_teachers_v3_local as L
from routing_teachers_v2 import spbp_next_hop

n_part = getattr(L.assert_kinf_matches_global, 'n_partitioned', 0)
check("1a. strengthened control exercised partitions", n_part >= 20,
      f"{n_part} partitioned cases")

rng = np.random.default_rng(0)
agree = tot = part = 0
for _ in range(400):
    G, n = sparse_graph(rng)
    s, d = 0, n - 1
    if not list(G.neighbors(s)) or d not in G:
        continue
    try:
        reach = set(nx.single_source_shortest_path_length(G, d))
    except nx.NodeNotFound:
        continue
    if any(nb not in reach for nb in G.neighbors(s)) or s not in reach:
        part += 1
    tot += 1
    if L.spbp_kinf_next_hop(G, s, d) == spbp_next_hop(G, s, d):
        agree += 1
check("1b. spbp_khop(k=inf) == panel SP-BP", agree == tot,
      f"{agree}/{tot} (was 279/345), {part} partitioned")

# finite k must be UNCHANGED — the proxy is the experiment, not a bug
rng = np.random.default_rng(1)
diff_k1 = tot_k1 = 0
for _ in range(300):
    G, n = sparse_graph(rng)
    s, d = 0, n - 1
    if not list(G.neighbors(s)) or d not in G:
        continue
    tot_k1 += 1
    if L.spbp_k1_next_hop(G, s, d) != spbp_next_hop(G, s, d):
        diff_k1 += 1
check("1c. finite k still DIFFERS from panel (proxy retained)",
      diff_k1 > 0, f"k=1 differs on {diff_k1}/{tot_k1} — the horizon is real")

# ── 2. the control must reject the old logic ───────────────────────────────
_orig = L.spbp_kinf_next_hop


def _old_kinf(G, current, destination, v_bias=L.SPBP_V_BIAS):
    """Pre-patch: proxy for every node missing from h_known, even at k=inf."""
    neighbors = list(G.neighbors(current))
    if not neighbors:
        return None
    if destination in neighbors:
        return destination
    try:
        h_known = nx.single_source_shortest_path_length(G, destination)
    except nx.NodeNotFound:
        return None
    comm_range = L._comm_range_of(G)
    dpos = L._pos(G, destination)

    def h_of(nd):
        if nd in h_known:
            return float(h_known[nd])
        return float(np.linalg.norm(dpos - L._pos(G, nd))) / max(comm_range, 1e-6)

    q_cur = float(G.nodes[current].get('queue_len', 0.0))
    h_cur = h_of(current)
    best, best_score = None, -float('inf')
    for nd in neighbors:
        q_n = float(G.nodes[nd].get('queue_len', 0.0))
        lq = float(G.edges[current, nd].get('link_quality', 0.0))
        score = lq * ((q_cur - q_n) + v_bias * (h_cur - h_of(nd)))
        if score > best_score:
            best_score, best = score, nd
    return best if best is not None else L._progress_fallback(G, current, destination)


L.spbp_kinf_next_hop = _old_kinf
try:
    L.assert_kinf_matches_global()
    caught = False
except AssertionError:
    caught = True
finally:
    L.spbp_kinf_next_hop = _orig
check("2. control REJECTS the pre-patch k=inf", caught,
      "can actually fail" if caught else "control is decorative")

# ── 3. H3: cand_reachable ──────────────────────────────────────────────────
import features_v2 as F

check("3a. CANDIDATE_FEATURES is 4 wide", len(F.CANDIDATE_FEATURES) == 4,
      str(F.CANDIDATE_FEATURES))
check("3b. schema bumped to 4 (v5)", F.FEATURE_SCHEMA_VERSION == 4,
      f"={F.FEATURE_SCHEMA_VERSION}")
check("3g. packet_error_rate dropped from EDGE_FEATURES",
      'packet_error_rate' not in F.EDGE_FEATURES and len(F.EDGE_FEATURES) == 4,
      f"{F.EDGE_FEATURES}")
check("3h. snr still gone, hops_so_far still gone",
      'snr' not in F.EDGE_FEATURES and 'hops_so_far' not in F.QUERY_FEATURES,
      "earlier removals survived the later patches")

G = nx.Graph(); G.graph['comm_range'] = 250.0
for i in range(6):
    G.add_node(i, x=float(i * 100), y=0.0, z=100.0, vx=0.0, vy=0.0, vz=0.0,
               energy=90.0, queue_occupancy=0.1)
for i in range(3):
    G.add_edge(i, i + 1, distance=100.0, link_quality=0.8,
               packet_error_rate=0.1, estimated_link_lifetime=30.0,
               relative_velocity=1.0)
# node 4 and 5 are deliberately left with NO edges -> outside the
# destination's component. The first draft of this test wired 0-4, which made
# node 4 reachable and let a wrong assertion pass for the wrong reason.
nc = F.norm_constants(dict(area_x=1000.0, area_y=1000.0, z_min=0.0, z_max=200.0,
                           speed_max=15.0, comm_range=280.0, num_drones=6))
h_map = F.hop_distances_to(G, 3)


class P:
    current, dst, hops, path = 0, 3, 0, [0]


qf, cf = F.extract_decision(G, P(), [1, 4], nc, h_map, 0.0, 0.0)
id_ = F.CANDIDATE_FEATURES.index('is_destination')
rc = F.CANDIDATE_FEATURES.index('cand_reachable')
check("3c. candidate block is 4 wide", cf.shape == (2, 4), str(cf.shape))
check("3d. reachable candidate -> 1.0", cf[0, rc] == 1.0)
check("3e. unreachable candidate -> 0.0", cf[1, rc] == 0.0,
      "node 4 has no edges at all")
check("3f. is_destination NOT displaced by the new column",
      cf[0, id_] == 0.0 and cf[1, id_] == 0.0,
      "column order matches CANDIDATE_FEATURES")

# ── 4. H4: the exemption is narrow ─────────────────────────────────────────
import preflight_dataset_v2_check as PF

check("4a. exemption list is EMPTY (v5 J2)",
      PF.ALLOWED_REDUNDANT_PAIRS == {},
      "nothing exempted on a prediction — only on a measurement")

x = np.random.default_rng(3).normal(size=(3000, 4))
x[:, 3] = 1.0 - x[:, 1]        # cand_hop_distance <-> cand_reachable, planted
off, lines = PF.redundancy_report(
    'candidate', x, ['progress', 'cand_hop_distance', 'is_destination',
                     'cand_reachable'])
check("4b. the formerly-exempt pair NOW fails if it is redundant",
      len(off) == 1,
      "no longer silenced — measured |r|=0.8391 |rho|=0.2597 in reality")

y = np.random.default_rng(4).normal(size=(3000, 4))
y[:, 0] = 1.0 - y[:, 2]                   # progress <-> is_destination: NOT exempt
off2, _ = PF.redundancy_report(
    'candidate', y, ['progress', 'cand_hop_distance', 'is_destination',
                     'cand_reachable'])
check("4c. a NON-exempt pair still FAILS", len(off2) == 1,
      "the exemption is narrow, not a blanket loosening")

# the machinery itself must still work, so a future genuine case can use it
PF.ALLOWED_REDUNDANT_PAIRS[('candidate', 'progress', 'is_destination')] = "test"
w = np.random.default_rng(9).normal(size=(3000, 4))
w[:, 0] = 1.0 - w[:, 2]
off_w, lines_w = PF.redundancy_report(
    'candidate', w, ['progress', 'cand_hop_distance', 'is_destination',
                     'cand_reachable'])
PF.ALLOWED_REDUNDANT_PAIRS.clear()
check("4e. exemption MACHINERY still functions when used", not off_w
      and any('[exempt]' in l for l in lines_w),
      "empty list, working mechanism")

z = np.random.default_rng(5).normal(size=(3000, 4))
z[:, 3] = 1.0 - z[:, 1]
off3, _ = PF.redundancy_report(
    'node', z, ['progress', 'cand_hop_distance', 'is_destination',
                'cand_reachable'])
check("4d. the same pair in a DIFFERENT block still fails", len(off3) == 1,
      "exemptions are keyed by block, not by name alone")

# ── 5. H6: the geo-progress variant ────────────────────────────────────────
import experiment_spbp_mechanism as M

check("5a. spbp_ab_geohop registered", 'spbp_ab_geohop' in M.ABLATIONS)
rng = np.random.default_rng(6)
differs = tot_g = 0
for _ in range(300):
    G2, n = sparse_graph(rng)
    s, d = 0, n - 1
    if not list(G2.neighbors(s)) or d not in G2:
        continue
    a, b = M.spbp_ab_geohop(G2, s, d), M.spbp_ab_full(G2, s, d)
    tot_g += 1
    if a != b:
        differs += 1
check("5b. geohop makes DIFFERENT decisions from full", differs > 0,
      f"differs on {differs}/{tot_g} — the metric swap is live")
check("5c. ...but not on everything", differs < tot_g,
      "geo and hop distance correlate 0.89-0.93, so most agree")

# ── 6. J3: the split assertion exists and names the seed range ────────────
import inspect
import generate_dataset_v2 as GD
src_split = inspect.getsource(GD._split_sizes)
check("6a. split completeness assertion present",
      'belong to NO split' in src_split)
check("6b. ...and it names the valid seed ranges",
      '101-135' in src_split and '143-150' in src_split,
      "failure message is actionable, not just a raise")


class _D(dict):
    pass


fake = [{'scenario': 'very_dense', 'seed': 1} for _ in range(10)]
try:
    GD._split_sizes(fake)
    fired = False
except AssertionError:
    fired = True
check("6c. it FIRES on out-of-range seeds", fired,
      "seeds outside 101-150 are rejected at generation time")
ok_dec = ([{'scenario': 'very_dense', 'seed': 101} for _ in range(10)]
          + [{'scenario': 'medium_slow', 'seed': 7} for _ in range(5)])
try:
    got = GD._split_sizes(ok_dec)
    passed = (got['train'] == 10 and got['generalisation'] == 5)
except AssertionError:
    passed = False
check("6d. ...and does NOT fire on a valid split", passed,
      "held-out scenario is exempt from the seed range, as designed")

print("=" * 78)
if fails:
    print(f"  {len(fails)} FAILURE(S): {fails}")
    sys.exit(1)
print("  ALL VERIFICATION CHECKS PASSED")
print("=" * 78 + "\n")
