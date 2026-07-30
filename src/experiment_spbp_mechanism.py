"""
experiment_spbp_mechanism.py  —  What is SP-BP's advantage ACTUALLY made of?

WHY THIS EXISTS
===============
The locality experiment returned mean gap(k=3) = +0.0005 PDR, i.e. restricting
SP-BP's hop-distance knowledge from global BFS to a 3-hop horizon costs
essentially nothing. Worse for the original story, spbp_k1 WON several cells
outright -- and more information should never hurt.

Working out what k=1 computes explains it. At k=1 the current node and its
candidates are almost never within 1 hop of the destination, so both fall to
the geographic proxy h_of(n) = k + geo(n,dst)/R, and the hop term collapses:

    h_cur - h_n = [1 + geo_cur/R] - [1 + geo_n/R] = (geo_cur - geo_n)/R

The horizon constant cancels. At k=1, SP-BP's "hop-distance" term IS plain
geographic progress, with zero topology knowledge -- and it matches exact BFS.

THIS FALSIFIES THE MECHANISM M3 CLAIMED
---------------------------------------
M3 explained SP-BP's 12/12 dominance as an INFORMATION SUPERSET: it uniquely
combines Dijkstra's global hop-distance with backpressure's local queue state.
But if global hop-distance is worth ~0.0005 PDR, that cannot be the operative
mechanism. Something else is driving the win, and the paper currently states
the wrong reason.

This also dissolves reviewer finding M-3 (the superset story contradicting
Lookahead-SPBP's failure) -- not by reconciling the two, but by showing the
superset story was wrong to begin with.

WHAT REMAINS AS CANDIDATE MECHANISMS
------------------------------------
Comparing SP-BP against DA-GPSR (the consistent runner-up) once hop-distance
is removed as a differentiator, three structural differences survive:

  SP-BP    : lq * [ (Q_v - Q_u) + V*(h_v - h_u) ]      MULTIPLICATIVE in lq,
                                                        queue DIFFERENTIAL
  DA-GPSR  : w_p*progress - w_q*occ_u + w_l*lq          ADDITIVE in lq,
                                                        CANDIDATE queue only

  (1) link quality MULTIPLIES the whole score vs being an additive term
  (2) queue DIFFERENTIAL (Q_v - Q_u) vs candidate-only queue (Q_u)
  (3) the progress term's scaling

THREE PARTS
-----------
  A. Confirm the geographic-proxy hypothesis: measure the correlation between
     geo_distance/comm_range and true BFS hop count in these topologies. If it
     is high, the locality result is explained and is a property of the
     SCENARIO GENERATOR (uniform random placement, no obstacles), not a
     general claim about FANET routing.
  B. Decision-level agreement between k=1 and k=inf. If they pick the same
     next hop almost always, identical PDR is trivial. If they diverge but
     still tie on PDR, the topology has many equally good paths -- a different
     and important finding for how much routing quality can matter at all.
  C. Component ablation of SP-BP itself, to identify which structural feature
     actually produces the advantage. This is what gives reviewer finding M-3
     a falsifiable mechanism stated in terms that predict BOTH SP-BP's success
     AND Lookahead-SPBP's failure.

Usage:
    python src\\experiment_spbp_mechanism.py --max_workers 16
    python src\\experiment_spbp_mechanism.py --part A --quick
"""

import os, sys, argparse
import numpy as np
import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulator_v2 import FANETSimulatorV2, TEACHERS
from routing_teachers_v2 import SPBP_V_BIAS, _pos, _progress_fallback
from routing_teachers_v3_local import spbp_khop_next_hop
from teacher_panel import build_oracle_table, paired_ttest

SCENARIOS = {
    'very_dense':  dict(num_drones=45, area_x=700,  area_y=700,  comm_range=250,
                        speed_min=5,  speed_max=15, pause_max=5.0),
    'dense_slow':  dict(num_drones=30, area_x=800,  area_y=800,  comm_range=250,
                        speed_min=5,  speed_max=15, pause_max=5.0),
    'medium_slow': dict(num_drones=30, area_x=1300, area_y=1300, comm_range=280,
                        speed_min=5,  speed_max=15, pause_max=5.0),
    'sparse_fast': dict(num_drones=20, area_x=1500, area_y=1500, comm_range=300,
                        speed_min=35, speed_max=50, pause_max=2.0),
}
RATES = [0.5, 2.0, 4.0]
BASE = dict(z_min=50, z_max=150, duration=40.0, drain_time=10.0,
            interference_on=True)


# ---------------------------------------------------------------- part C ----
# SP-BP component ablations. Each isolates ONE structural feature so the
# mechanism can be attributed rather than asserted.

def _spbp_generic(G, current, destination, use_lq_mult, use_queue_diff,
                  use_queue, v_bias=SPBP_V_BIAS):
    """Parameterised SP-BP. Hop term uses full-graph BFS throughout, so the
    ablation isolates STRUCTURE, not information -- part A/B already showed
    the information horizon is not the operative variable."""
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
        if not use_queue:
            qterm = 0.0
        elif use_queue_diff:
            qterm = q_cur - q_n
        else:
            qterm = -q_n                      # candidate-only, DA-GPSR style
        score = lq * (qterm + hop) if use_lq_mult else (qterm + hop + lq)
        if score > best_score:
            best_score, best = score, n
    return best if best is not None else _progress_fallback(G, current, destination)


def spbp_ab_full(G, c, d):
    """Baseline: the real SP-BP (multiplicative lq, queue differential)."""
    return _spbp_generic(G, c, d, True, True, True)


def spbp_ab_noqueue(G, c, d):
    """Hop term only -- isolates how much the queue term contributes at all."""
    return _spbp_generic(G, c, d, True, True, False)


def spbp_ab_candqueue(G, c, d):
    """Candidate queue only (DA-GPSR style) instead of the differential.
    Isolates whether the DIFFERENTIAL specifically matters."""
    return _spbp_generic(G, c, d, True, False, True)


def spbp_ab_additive(G, c, d):
    """Link quality as an additive term instead of a multiplier.
    Isolates whether the MULTIPLICATIVE structure is what matters."""
    return _spbp_generic(G, c, d, False, True, True)


ABLATIONS = {
    'spbp_ab_full':      spbp_ab_full,
    'spbp_ab_noqueue':   spbp_ab_noqueue,
    'spbp_ab_candqueue': spbp_ab_candqueue,
    'spbp_ab_additive':  spbp_ab_additive,
}
for _n, _f in ABLATIONS.items():
    TEACHERS[_n] = _f


# ---------------------------------------------------------------- part A ----
def part_a(seeds, verbose=True):
    """Is geographic distance a sufficient proxy for hop distance here?"""
    print("\n" + "-" * 78)
    print("  A. IS GEOGRAPHIC DISTANCE A PROXY FOR HOP DISTANCE?")
    print("-" * 78)
    print(f"  {'scenario':<14}{'pearson r':>12}{'rank r':>10}"
          f"{'mean hops':>11}{'reachable':>11}{'n pairs':>10}")
    out = {}
    for sc, cfg in SCENARIOS.items():
        geos, hops = [], []
        n_possible = 0        # all ordered pairs, reachable or not
        for sd in seeds[:8]:
            sim = FANETSimulatorV2({**BASE, **cfg, 'packet_rate': 1.0,
                                    'seed': sd, 'actor': 'spbp'})
            for d in sim.drones:
                d.step(0.5)
            G = sim._build_graph()
            R = float(cfg['comm_range'])
            nodes = list(G.nodes())
            n_possible += len(nodes) * (len(nodes) - 1)
            for src in nodes:
                try:
                    h = nx.single_source_shortest_path_length(G, src)
                except nx.NodeNotFound:
                    continue
                for tgt, hd in h.items():
                    if tgt == src:
                        continue
                    geo = float(np.linalg.norm(_pos(G, src) - _pos(G, tgt)))
                    geos.append(geo / R)
                    hops.append(float(hd))
        geos, hops = np.array(geos), np.array(hops)
        if len(geos) < 10:
            continue
        r = float(np.corrcoef(geos, hops)[0, 1])
        rank = float(np.corrcoef(np.argsort(np.argsort(geos)),
                                 np.argsort(np.argsort(hops)))[0, 1])
        # REACHABILITY, added after the first run exposed the problem:
        # nx.single_source_shortest_path_length returns ONLY reachable nodes, so
        # unreachable pairs are silently dropped from the correlation. In a
        # partitioned topology that is most of the network, and it also makes
        # 'mean hops' misleadingly LOW (only nearby pairs survive) -- which is
        # why sparse_fast first reported 1.73 mean hops, BELOW dense_slow's 2.65,
        # an otherwise nonsensical ordering.
        reach = len(geos) / max(n_possible, 1)
        out[sc] = {'pearson': r, 'rank': rank, 'mean_hops': float(hops.mean()),
                   'n': int(len(geos)), 'reachable_frac': float(reach)}
        print(f"  {sc:<14}{r:>12.4f}{rank:>10.4f}{hops.mean():>11.2f}"
              f"{100*reach:>10.1f}%{len(geos):>10}")

    rs = [v['pearson'] for v in out.values()]
    low_reach = {k: v['reachable_frac'] for k, v in out.items()
                 if v['reachable_frac'] < 0.5}
    if low_reach:
        print()
        print("  ** REACHABILITY CAVEAT: in "
              f"{', '.join(low_reach)} most node pairs are NOT connected")
        print("     (" + ", ".join(f"{k} {100*v:.0f}% reachable"
                                   for k, v in low_reach.items()) + ").")
        print("     The correlation above is therefore measured on a BIASED subset --")
        print("     only pairs that happen to be connected, which skews toward")
        print("     nearby ones. It also explains why 'mean hops' can read LOWER in")
        print("     a sparse scenario than a dense one. The proxy claim holds WHERE")
        print("     IT IS MEASURABLE; in a heavily partitioned regime most of the")
        print("     network is not.")
    print()
    if rs and min(rs) > 0.85:
        print("  -> STRONG correlation in every scenario. This EXPLAINS the locality")
        print("     result: the global BFS carries almost no information that")
        print("     geographic distance (already in every packet header) does not.")
        print()
        print("     IMPORTANT SCOPE LIMIT, and it must be stated in the paper:")
        print("     this is a property of the SCENARIO GENERATOR -- nodes placed")
        print("     uniformly at random in an open 2D area with no obstacles, where")
        print("     hop_count ~= geo_distance / comm_range almost by construction.")
        print("     In deployments with voids, obstacles, or concave boundaries the")
        print("     proxy breaks down -- which is precisely why GPSR needs perimeter")
        print("     routing at all. The conclusion 'global topology knowledge is")
        print("     worth ~nothing' is therefore SCENARIO-SPECIFIC, not general.")
    else:
        print("  -> Correlation is NOT uniformly strong; the geographic-proxy")
        print("     hypothesis does not by itself explain the locality result.")
        print("     Investigate before drawing conclusions.")
    return out


# ---------------------------------------------------------------- part B ----
def part_b(seeds, verbose=True):
    """Do k=1 and k=inf actually make different decisions, or identical ones?"""
    print("\n" + "-" * 78)
    print("  B. DECISION-LEVEL AGREEMENT BETWEEN k=1 AND k=inf")
    print("-" * 78)
    print(f"  {'scenario':<14}{'agree rate':>12}{'decisions':>12}")
    out = {}
    for sc, cfg in SCENARIOS.items():
        agree = tot = 0
        for sd in seeds[:6]:
            sim = FANETSimulatorV2({**BASE, **cfg, 'packet_rate': 2.0,
                                    'seed': sd, 'actor': 'spbp'})
            sim.run()
            G = sim._build_graph()
            rng = np.random.default_rng(sd)
            nodes = [n for n in G.nodes() if list(G.neighbors(n))]
            for _ in range(250):
                if len(nodes) < 2:
                    break
                c, d = rng.choice(nodes, size=2, replace=False)
                a = spbp_khop_next_hop(G, int(c), int(d), k=1)
                b = spbp_khop_next_hop(G, int(c), int(d), k=None)
                if a is None or b is None:
                    continue
                tot += 1
                agree += (a == b)
        rate = agree / max(tot, 1)
        out[sc] = {'agree_rate': rate, 'n': tot}
        print(f"  {sc:<14}{rate:>12.4f}{tot:>12}")
    rates = [v['agree_rate'] for v in out.values()]
    print()
    if rates and min(rates) > 0.95:
        print("  -> k=1 and k=inf pick the SAME next hop almost always, so identical")
        print("     PDR is trivial rather than informative: the two policies are")
        print("     nearly the same policy in this environment.")
    else:
        print("  -> The horizons DO make materially different decisions, yet PDR is")
        print("     unchanged. That is the more interesting reading: many distinct")
        print("     routes are near-equally good here, so per-decision routing")
        print("     quality matters less than the topology allows. Worth stating --")
        print("     it bounds how much ANY routing policy, learned or classical,")
        print("     can achieve in these scenarios.")
    return out


# ---------------------------------------------------------------- part C ----
def part_c(seeds, rates, max_workers, quick=False):
    """Which STRUCTURAL feature of SP-BP produces its advantage?"""
    print("\n" + "-" * 78)
    print("  C. SP-BP COMPONENT ABLATION")
    print("-" * 78)
    scen = {'medium_slow': SCENARIOS['medium_slow']} if quick else SCENARIOS
    panel = list(ABLATIONS) + ['da_gpsr']
    table, raw, cong, diag, stats = build_oracle_table(
        scen, rates, seeds, panel=panel, base_cfg=BASE,
        verbose=False, max_workers=max_workers)

    print(f"  {'class':<13}{'load':<8}" + "".join(f"{t.replace('spbp_ab_',''):>13}" for t in panel))
    for key in sorted(table):
        bt = {t: v for t, v in table[key]}
        print(f"  {key[0]:<13}{key[1]:<8}" +
              "".join(f"{bt.get(t, float('nan')):>13.3f}" for t in panel))

    means = {t: float(np.mean([dict(table[k]).get(t, np.nan) for k in table]))
             for t in panel}
    print("\n  mean PDR across all cells:")
    for t, v in sorted(means.items(), key=lambda kv: -kv[1]):
        print(f"    {t:<20} {v:.4f}")

    full = means.get('spbp_ab_full', np.nan)
    print("\n  cost of removing each structural feature (vs full SP-BP):")
    for t, label in [('spbp_ab_noqueue', 'queue term entirely'),
                     ('spbp_ab_candqueue', 'queue DIFFERENTIAL -> candidate-only'),
                     ('spbp_ab_additive', 'lq MULTIPLICATIVE -> additive')]:
        print(f"    {label:<42} {full - means.get(t, np.nan):+.4f} PDR")
    print()
    print("  The largest drop identifies the mechanism actually responsible for")
    print("  SP-BP's dominance -- replacing M3's 'information superset' story,")
    print("  which the locality experiment has already falsified.")
    return means


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=list(range(1, 31)))
    ap.add_argument('--rates', type=float, nargs='+', default=RATES)
    ap.add_argument('--max_workers', type=int, default=None)
    ap.add_argument('--part', choices=['A', 'B', 'C', 'all'], default='all')
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()

    print("\n" + "=" * 78)
    print("  SP-BP MECHANISM EXPERIMENT")
    print("  (follows up the locality result; addresses reviewer finding M-3)")
    print("=" * 78)

    if args.part in ('A', 'all'):
        part_a(args.seeds)
    if args.part in ('B', 'all'):
        part_b(args.seeds)
    if args.part in ('C', 'all'):
        part_c(args.seeds, args.rates, args.max_workers, args.quick)

    print("\n" + "=" * 78 + "\n")
    return 0


if __name__ == '__main__':
    sys.exit(main())
