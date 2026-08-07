"""
verify_mechanism_fixes_v3.py — execution proof for apply_mechanism_fixes_v3.py.

    python verify_mechanism_fixes_v3.py --src src

Test 2 is the load-bearing one. A control that only ever passes proves nothing,
so the ported control is re-run against a deliberately reverted copy of the old
h.get(..., 999.0) logic and MUST raise. Test 3 compares the ablation to panel
SP-BP decision-by-decision on the real partitioned scenarios where the numbers
actually diverged, rather than on synthetic graphs.
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


print("\n" + "=" * 78)
print("  VERIFICATION — MECHANISM + GRID FIXES v3")
print("=" * 78)

import experiment_spbp_mechanism as M
from routing_teachers_v2 import spbp_next_hop

# ── 1. the control ran at import and exercised the branch that broke ────────
n_part = getattr(M.assert_controls, 'n_partitioned', 0)
check("1a. assert_controls ran at import", n_part > 0)
check("1b. ...and exercised partitioned graphs", n_part >= 20,
      f"{n_part} partitioned cases (>=20 required)")

# ── 2. the control MUST fail on the old logic ───────────────────────────────
# Rebuild the pre-patch scorer exactly and point spbp_ab_full at it, then
# re-run the control. If it still passes, the control is decorative.
_orig = M.spbp_ab_full


def _old_spbp_full(G, current, destination, v_bias=M.SPBP_V_BIAS):
    """The pre-patch _spbp_generic(True, True, True), verbatim on the two
    lines that mattered: h.get(current, 999.0) and h.get(n, 999.0)."""
    neighbors = list(G.neighbors(current))
    if not neighbors:
        return None
    if destination in neighbors:
        return destination
    try:
        h = nx.single_source_shortest_path_length(G, destination)
    except nx.NodeNotFound:
        return None
    h_cur = float(h.get(current, 999.0))
    q_cur = float(G.nodes[current].get('queue_len', 0.0))
    best, best_score = None, -float('inf')
    for n in neighbors:
        q_n = float(G.nodes[n].get('queue_len', 0.0))
        lq = float(G.edges[current, n].get('link_quality', 0.0))
        hop = v_bias * (h_cur - float(h.get(n, 999.0)))
        score = lq * ((q_cur - q_n) + hop)
        if score > best_score:
            best_score, best = score, n
    return best if best is not None else M._progress_fallback(G, current, destination)


M.spbp_ab_full = _old_spbp_full
try:
    M.assert_controls()
    caught = False
except AssertionError:
    caught = True
finally:
    M.spbp_ab_full = _orig
check("2. control REJECTS the pre-patch implementation", caught,
      "the control can actually fail" if caught else "control is decorative")

# ── 3. agreement with panel SP-BP on the real partitioned scenarios ─────────
from simulator_v2 import FANETSimulatorV2

BASE = dict(z_min=50, z_max=150, duration=40.0, drain_time=10.0,
            interference_on=True)


def sample_graphs(scen_cfg, seed, n_frames=6):
    """Pull real mid-rollout graphs out of the simulator, not synthetic ones."""
    sim = FANETSimulatorV2({**BASE, **scen_cfg, 'packet_rate': 2.0,
                            'seed': seed, 'actor': 'spbp'})
    sim.run()
    out = []
    try:
        for _ in range(n_frames):
            sim._update_positions(1.0) if hasattr(sim, '_update_positions') else None
            out.append(sim._build_graph())
    except Exception:
        pass
    return out


agree_new = agree_old = total = 0
reach_frac = []
for scen in ('medium_slow', 'sparse_fast'):
    cfg = M.SCENARIOS[scen]
    for seed in (1, 2, 3):
        for G in sample_graphs(cfg, seed):
            nodes = sorted(G.nodes())
            if len(nodes) < 3:
                continue
            rng = np.random.default_rng(seed)
            for _ in range(60):
                c, d = rng.choice(nodes, size=2, replace=False)
                c, d = int(c), int(d)
                if not list(G.neighbors(c)) or d in G.neighbors(c):
                    continue
                ref = spbp_next_hop(G, c, d)
                total += 1
                if M.spbp_ab_full(G, c, d) == ref:
                    agree_new += 1
                if _old_spbp_full(G, c, d) == ref:
                    agree_old += 1
            try:
                reach = nx.single_source_shortest_path_length(G, nodes[-1])
                reach_frac.append(len(reach) / len(nodes))
            except Exception:
                pass

if total >= 100:
    check("3a. patched spbp_ab_full matches panel SP-BP everywhere",
          agree_new == total, f"{agree_new}/{total} decisions")
    check("3b. pre-patch version did NOT",
          agree_old < total,
          f"{agree_old}/{total} ({100*(total-agree_old)/total:.1f}% divergent)")
    check("3c. ...on genuinely partitioned graphs", np.mean(reach_frac) < 0.95,
          f"mean reachable fraction {np.mean(reach_frac):.2f}")
else:
    check("3. decision-level comparison collected enough cases", False,
          f"only {total}")

# ── 4. grids are mirrored and bracket each reference ───────────────────────
import experiment_queue_weight as Q

check("4a. UP sweep now samples BELOW the reference",
      min(Q.DA_WEIGHTS) < 1.0, f"min w={min(Q.DA_WEIGHTS):g}")
check("4b. UP sweep brackets the reference on both sides",
      min(Q.DA_WEIGHTS) < 1.0 < max(Q.DA_WEIGHTS),
      f"{min(Q.DA_WEIGHTS):g} .. {max(Q.DA_WEIGHTS):g}")
check("4c. DOWN sweep extends past its apparent optimum",
      max(Q.SP_SCALES) > 3.0, f"max q={max(Q.SP_SCALES):g} (was 3)")
check("4d. reference values still present in both",
      1.0 in Q.DA_WEIGHTS and 1.0 in Q.SP_SCALES)
check("4e. q=0.02 retained (down_given depends on it)", 0.02 in Q.SP_SCALES)
check("4f. w=15 retained so the collapse stays in the table",
      15.0 in Q.DA_WEIGHTS)
runs = (len(Q.DA_WEIGHTS) + len(Q.SP_SCALES) + 2) * 12 * 30
check("4g. runtime increase is bounded", runs < 9000,
      f"{runs} runs (was 4320, ~{runs/4320:.2f}x)")

print("=" * 78)
if fails:
    print(f"  {len(fails)} FAILURE(S): {fails}")
    sys.exit(1)
print("  ALL VERIFICATION CHECKS PASSED")
print("=" * 78 + "\n")
