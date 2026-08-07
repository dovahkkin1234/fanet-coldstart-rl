"""
experiment_queue_weight.py  —  Is SP-BP's residual advantage just queue-weight scaling?

WHERE THIS COMES FROM
=====================
Two experiments have now stripped away the explanations M3 originally gave for
SP-BP winning all 12 oracle cells:

  locality experiment : global BFS hop-distance is worth +0.0005 PDR. At k=1
                        the hop term algebraically collapses to plain
                        geographic progress and matches exact BFS. So the
                        "global information superset" story is FALSIFIED.

  mechanism ablation  : removing the queue term costs -0.0615 PDR (the real
                        driver), but changing its FORM costs nothing --
                        differential -> candidate-only = 0.0000 exactly,
                        multiplicative -> additive = 0.0005.

The exact zero is explained: `full` and `candqueue` differ by lq*Q_v, which
vanishes iff Q_v = 0, and M3.5 measured Q_v ~ 0 at essentially every decision
(exactly 0.000% nonzero in very_dense) because the packet is dequeued before
the decision is recorded. So backpressure's defining queue DIFFERENTIAL is
inoperative here -- the same fact behind backpressure's measured 69%
zero-gradient rate in M3.

WHAT REMAINS UNEXPLAINED
------------------------
SP-BP still beats DA-GPSR by ~0.029 PDR, and none of the three structural
ablations account for it. The surviving difference is SCALE, not structure:

    SP-BP   : lq * ( -Q_u + 1.0*hop_progress )    Q_u = queue_len,  range 0..50
    DA-GPSR : 1.0*geo_prog - 1.0*occ_u + 0.5*lq   occ_u = occupancy, range 0..1

queue_len = 50 * occupancy -- the SAME signal at 50x the weight, both sitting
against a progress term of order 1. SP-BP therefore penalises a queued
candidate far more aggressively:

    candidate with 3 packets queued
        SP-BP inner term  = -3.00 + 1.00 = -2.00   (strongly avoid)
        DA-GPSR inner     = -0.06 + 0.30 = +0.24   (barely notices)

HYPOTHESIS: SP-BP's residual advantage is queue-avoidance AGGRESSIVENESS -- a
weight-scaling choice -- not algorithmic structure.

This matters for the paper. "Our best classical teacher wins because of a
hyperparameter scale" is a much weaker claim than "because of its algorithmic
design", and it is far better to establish it ourselves than to have a
reviewer notice that queue_len and queue_occupancy are the same signal.

TESTED IN BOTH DIRECTIONS (one direction alone would be weak evidence)
----------------------------------------------------------------------
  UP   : DA-GPSR with its queue weight swept 1 -> 50. If it climbs to meet
         SP-BP, aggressiveness is sufficient to explain the gap.
  DOWN : SP-BP with queue_len replaced by occupancy (i.e. 50x weaker queue
         term, everything else identical). If it falls to DA-GPSR's level,
         aggressiveness is necessary.
  Sufficient AND necessary => the mechanism is scaling, not structure.
  If only one direction moves, something structural remains and we keep looking.

ALSO SWEEPS PAST SP-BP's WEIGHT: if PDR keeps rising beyond the equivalent of
50, then SP-BP is not even at the optimum and the "best classical teacher"
framing is itself an artifact of an untuned constant -- which would need
saying plainly.

Usage:
    python src\\experiment_queue_weight.py --max_workers 16
    python src\\experiment_queue_weight.py --quick
"""

import os, sys, argparse
import json
import numpy as np
import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulator_v2 import TEACHERS
from routing_teachers_v2 import (SPBP_V_BIAS, DAGPSR_W_PROGRESS,
                                 DAGPSR_W_QUALITY, _pos, _progress_fallback)
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

MAX_QUEUE = 50.0     # matches simulator_v2.MAX_QUEUE; occupancy = len / MAX_QUEUE


# ---- direction UP: DA-GPSR with a swept queue weight ------------------------
def _da_gpsr_w(G, current, destination, w_queue):
    """DA-GPSR with an explicit queue weight. w_queue=1.0 is the panel default;
    w_queue=50 makes its queue penalty scale-equivalent to SP-BP's use of raw
    queue_len (since occupancy = queue_len / 50)."""
    neighbors = list(G.neighbors(current))
    if not neighbors:
        return None
    dest_pos = _pos(G, destination)
    dist_cd = float(np.linalg.norm(dest_pos - _pos(G, current)))
    best, best_score = None, -float('inf')
    for n in neighbors:
        dist_nd = float(np.linalg.norm(dest_pos - _pos(G, n)))
        progress = (dist_cd - dist_nd) / max(dist_cd, 1.0)
        occ = float(G.nodes[n].get('queue_occupancy', 0.0))
        lq = float(G.edges[current, n].get('link_quality', 0.0))
        score = DAGPSR_W_PROGRESS * progress - w_queue * occ + DAGPSR_W_QUALITY * lq
        if score > best_score:
            best_score, best = score, n
    return best


# ---- direction DOWN: SP-BP with a weakened queue term -----------------------
def _spbp_qscale(G, current, destination, qscale, v_bias=SPBP_V_BIAS):
    """SP-BP with its queue term multiplied by `qscale`.

    qscale=1.0   -> the real SP-BP (raw queue_len)
    qscale=0.02  -> queue term equivalent to occupancy (1/50), i.e. DA-GPSR's
                    scale, with SP-BP's structure otherwise untouched.
    Isolates SCALE from STRUCTURE in the opposite direction to _da_gpsr_w."""
    neighbors = list(G.neighbors(current))
    if not neighbors:
        return None
    if destination in neighbors:
        return destination
    try:
        h = nx.single_source_shortest_path_length(G, destination)
    except nx.NodeNotFound:
        return None
    # MUST mirror spbp_next_hop's unreachability handling EXACTLY, or qscale=1
    # is not the real SP-BP and the whole DOWN sweep is measured against the
    # wrong baseline. The first version used h.get(n, 999.0) and kept going when
    # `current` was unreachable; panel SP-BP instead RETURNS None for an
    # unreachable current node and SKIPS unreachable candidates. That divergence
    # cost 0.0067 PDR (0.4056 vs 0.4123) and slipped past a control that only
    # checked src->dst reachability, never candidate reachability.
    if current not in h:
        return None
    h_cur = float(h[current])
    q_cur = float(G.nodes[current].get('queue_len', 0.0))
    best, best_score = None, -float('inf')
    for n in neighbors:
        if n not in h:
            continue
        q_n = float(G.nodes[n].get('queue_len', 0.0))
        lq = float(G.edges[current, n].get('link_quality', 0.0))
        score = lq * (qscale * (q_cur - q_n) + v_bias * (h_cur - float(h[n])))
        if score > best_score:
            best_score, best = score, n
    return best if best is not None else _progress_fallback(G, current, destination)


# Register sweep variants. DA-GPSR weights chosen to bracket SP-BP's effective
# 50x; SP-BP scales chosen to bracket DA-GPSR's effective 1/50.
# MIRRORED GRIDS.
# The previous pair was not comparable: DOWN sampled 0.02-3x (dense below and
# around the reference) while UP sampled 1-150x (nothing below the reference,
# smallest step 5x). Since DA-GPSR is already at -216% of the gap by w=5, that
# grid could not distinguish "queue weight cannot help DA-GPSR" from "DA-GPSR's
# queue weight is already at or above its optimum" -- opposite conclusions.
# UP now has sub-reference resolution; DOWN now extends past its own apparent
# optimum (q=3 scored +0.0010 ABOVE q=1, so the old grid stopped too early).
# w=15 kept so the collapse stays visible in the same table; 50 and 150 dropped
# as pure re-demonstration -- they remain in the previously committed JSON.
# Configs 12 -> 21, so runs 4320 -> 7560.
DA_WEIGHTS = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 15.0]
SP_SCALES = [0.02, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 3.0, 10.0]

for _w in DA_WEIGHTS:
    TEACHERS[f'dagpsr_w{_w:g}'] = (lambda w: (lambda G, c, d: _da_gpsr_w(G, c, d, w)))(_w)
for _s in SP_SCALES:
    TEACHERS[f'spbp_q{_s:g}'] = (lambda s: (lambda G, c, d: _spbp_qscale(G, c, d, s)))(_s)


def assert_controls():
    """dagpsr_w1 must reproduce panel DA-GPSR, and spbp_q1 panel SP-BP, EXACTLY.

    STRENGTHENED after the first version passed while spbp_q1 was in fact
    wrong. That control only ran when nx.has_path(src, dst) held, so it never
    exercised the case where some CANDIDATE is unreachable from the
    destination -- which is where the two implementations actually diverged.
    Part A then measured that unreachability is the COMMON case in this
    environment (medium_slow 55% reachable, sparse_fast 21%), so the untested
    branch was being hit constantly.

    This version deliberately uses SPARSE connection probabilities to force
    partitioned graphs, counts how many test cases actually had an unreachable
    candidate, and REQUIRES that count to be non-trivial -- so the control
    cannot pass by only ever exercising the easy path again.
    """
    from routing_teachers_v2 import da_gpsr_next_hop, spbp_next_hop
    rng = np.random.default_rng(0)
    n_da = n_sp = n_partitioned = 0
    for trial in range(300):
        n = int(rng.integers(5, 14))
        # sparse on purpose: p in [0.12, 0.35] reliably produces partitions
        p_edge = 0.12 + 0.23 * rng.random()
        G = nx.Graph(); G.graph['comm_range'] = 250.0
        for i in range(n):
            G.add_node(i, x=float(rng.integers(0, 900)), y=float(rng.integers(0, 900)),
                       z=100.0, energy=90.0,
                       queue_occupancy=float(rng.random()),
                       queue_len=float(rng.integers(0, 6)))
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < p_edge:
                    G.add_edge(i, j, distance=float(rng.integers(50, 250)),
                               link_quality=float(rng.random()),
                               packet_error_rate=float(rng.random()) * 0.3)
        src, dst = 0, n - 1
        if not list(G.neighbors(src)) or dst not in G:
            continue
        if _da_gpsr_w(G, src, dst, 1.0) != da_gpsr_next_hop(G, src, dst):
            raise AssertionError(f"dagpsr_w1 diverged from panel da_gpsr (trial {trial})")
        n_da += 1
        # track whether this case actually exercises the unreachable branch
        try:
            reach = set(nx.single_source_shortest_path_length(G, dst))
        except nx.NodeNotFound:
            continue
        if any(nb not in reach for nb in G.neighbors(src)) or src not in reach:
            n_partitioned += 1
        if _spbp_qscale(G, src, dst, 1.0) != spbp_next_hop(G, src, dst):
            raise AssertionError(
                f"spbp_q1 diverged from panel spbp (trial {trial}) -- this is the "
                f"exact failure the first control missed")
        n_sp += 1
    if n_da < 50 or n_sp < 50:
        raise AssertionError(f"controls checked too few cases: da={n_da} sp={n_sp}")
    if n_partitioned < 20:
        raise AssertionError(
            f"control only exercised {n_partitioned} partitioned cases -- too few to "
            f"trust; it would pass without testing the branch that actually broke")
    assert_controls.n_partitioned = n_partitioned


assert_controls()


def _jsonable(o):
    """numpy scalars/arrays -> plain Python, recursively. json.dump chokes on
    np.float32 and on tuple dict keys, both of which the oracle tables use."""
    import numpy as _np
    if isinstance(o, dict):
        return {(str(k) if not isinstance(k, str) else k): _jsonable(v)
                for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, _np.generic):
        return o.item()
    if isinstance(o, _np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, float) and (o != o):
        return None
    return o

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=list(range(1, 31)))
    ap.add_argument('--rates', type=float, nargs='+', default=RATES)
    ap.add_argument('--max_workers', type=int, default=None)
    ap.add_argument('--quick', action='store_true')
    # See the note in experiment_spbp_mechanism.main: this script could not
    # write its results either, and Finding 4's refutation depends on them.
    ap.add_argument('--out', default='results/queue_weight.json',
                    help='write results here; "" disables')
    args = ap.parse_args()

    scen = {'medium_slow': SCENARIOS['medium_slow']} if args.quick else SCENARIOS
    panel = ([f'dagpsr_w{w:g}' for w in DA_WEIGHTS] +
             [f'spbp_q{s:g}' for s in SP_SCALES] + ['spbp', 'da_gpsr'])

    print("\n" + "=" * 78)
    print("  QUEUE-WEIGHT EXPERIMENT")
    print("  Is SP-BP's residual advantage over DA-GPSR just queue-weight scaling?")
    print("=" * 78)
    print("  controls passed at import: dagpsr_w1 == panel da_gpsr,"
          " spbp_q1 == panel spbp")
    print(f"  (control exercised {getattr(assert_controls, 'n_partitioned', 0)} "
          f"PARTITIONED graphs -- the branch the first control missed)")
    print(f"  runs: {len(scen)*len(args.rates)*len(args.seeds)*len(panel)}")

    table, raw, cong, diag, stats = build_oracle_table(
        scen, args.rates, args.seeds, panel=panel, base_cfg=BASE,
        verbose=False, max_workers=args.max_workers)

    means = {t: float(np.mean([dict(table[k]).get(t, np.nan) for k in table]))
             for t in panel}
    sp_ref, da_ref = means['spbp'], means['da_gpsr']
    gap = sp_ref - da_ref

    print("\n" + "-" * 78)
    print("  DIRECTION UP — DA-GPSR with increasing queue weight")
    print("-" * 78)
    print(f"  reference: da_gpsr (w=1) = {da_ref:.4f}   spbp = {sp_ref:.4f}"
          f"   gap = {gap:+.4f}")
    print(f"  {'w_queue':>10}{'mean PDR':>12}{'vs da_gpsr':>13}{'% of gap closed':>18}")
    for w in DA_WEIGHTS:
        v = means[f'dagpsr_w{w:g}']
        closed = (v - da_ref) / gap * 100 if abs(gap) > 1e-9 else float('nan')
        print(f"  {w:>10g}{v:>12.4f}{v-da_ref:>+13.4f}{closed:>17.1f}%")

    print("\n" + "-" * 78)
    print("  DIRECTION DOWN — SP-BP with weakened queue term")
    print("-" * 78)
    print(f"  {'q scale':>10}{'mean PDR':>12}{'vs spbp':>13}{'% of gap given up':>19}")
    for s in SP_SCALES:
        v = means[f'spbp_q{s:g}']
        given = (sp_ref - v) / gap * 100 if abs(gap) > 1e-9 else float('nan')
        print(f"  {s:>10g}{v:>12.4f}{v-sp_ref:>+13.4f}{given:>18.1f}%")

    # ---- verdict ----
    best_da = max(means[f'dagpsr_w{w:g}'] for w in DA_WEIGHTS)
    up_closed = (best_da - da_ref) / gap if abs(gap) > 1e-9 else 0.0
    sp_weak = means[f'spbp_q{0.02:g}']
    down_given = (sp_ref - sp_weak) / gap if abs(gap) > 1e-9 else 0.0

    print("\n" + "=" * 78)
    print("  VERDICT")
    print("=" * 78)
    print(f"  UP  : best DA-GPSR weight closes {100*up_closed:.0f}% of the gap")
    print(f"  DOWN: weakening SP-BP's queue term gives up {100*down_given:.0f}% of it")
    print()
    if up_closed > 0.7 and down_given > 0.7:
        print("  -> MECHANISM IS SCALE, NOT STRUCTURE (sufficient AND necessary).")
        print("     SP-BP's advantage over DA-GPSR is queue-avoidance")
        print("     aggressiveness -- a weight choice. This MUST be stated in the")
        print("     paper: the 'best classical teacher' is best because of a")
        print("     hyperparameter scale, not an algorithmic insight. It also means")
        print("     the panel comparison is partly a comparison of tuning, and any")
        print("     teacher could be re-tuned. Report the swept curve, not a single")
        print("     point, so the comparison is honest.")
    elif up_closed > 0.7 or down_given > 0.7:
        print("  -> ONLY ONE DIRECTION MOVES. Scaling is part of the story but not")
        print("     all of it; something structural remains. Do not claim the")
        print("     mechanism is settled -- investigate the asymmetry.")
    else:
        print("  -> NEITHER DIRECTION EXPLAINS THE GAP. The queue-weight hypothesis")
        print("     is REFUTED; SP-BP's advantage is structural after all, and the")
        print("     structural ablation must be extended to find where.")

    best_w = max(DA_WEIGHTS, key=lambda w: means[f'dagpsr_w{w:g}'])
    best_q = max(SP_SCALES, key=lambda q: means[f'spbp_q{q:g}'])

    # WHERE DOES EACH REFERENCE SIT ON ITS OWN CURVE? The old grid could not
    # answer this for DA-GPSR at all: w=1 was its smallest value, so an
    # optimum below the reference was unobservable by construction.
    print()
    print(f"  sweep optima:  DA-GPSR best at w={best_w:g} ({best_da:.4f});  "
          f"SP-BP best at q={best_q:g} ({means[f'spbp_q{best_q:g}']:.4f})")
    if best_w < 1.0:
        print("  ** DA-GPSR's reference weight is ABOVE its optimum -- the panel")
        print("     default is over-weighting the queue term, so the old UP sweep")
        print("     was climbing away from the peak from its first step. Any")
        print("     'queue weight cannot help DA-GPSR' claim is REFUTED.")
    elif best_w > 1.0:
        print("  ** DA-GPSR improves with MORE queue weight -- scaling is part of")
        print("     the gap after all. Report the swept curve, not a single point.")
    else:
        print("  ** DA-GPSR's reference weight IS its optimum on this grid, now")
        print("     bracketed on both sides. The gap is not a weight choice.")
    if best_q != 1.0:
        print(f"  ** SP-BP's reference is NOT its optimum either (best q={best_q:g}).")
        print("     The panel comparison is partly a comparison of tuning; say so.")

    if best_da > sp_ref + 0.002:
        print()
        print(f"  ** ALSO: DA-GPSR at w={best_w:g} ({best_da:.4f}) EXCEEDS SP-BP")
        print(f"     ({sp_ref:.4f}). SP-BP is then not the best classical policy --")
        print("     it merely had a better-scaled queue term than the panel default.")
        print("     The M3 oracle table would need re-running with tuned weights,")
        print("     and 'SP-BP dominates' restated as 'aggressive queue avoidance")
        print("     dominates, and SP-BP happened to implement it'.")
    if args.out:
        out = {
            'seeds': list(args.seeds), 'rates': list(args.rates),
            'da_weights': list(DA_WEIGHTS), 'means': _jsonable(means),
            'spbp_reference': float(sp_ref), 'dagpsr_reference': float(da_ref),
            'gap': float(gap),
            'up_fraction_closed': float(up_closed),
            'down_fraction_given': float(down_given),
            'best_da_weight': float(best_w), 'best_da_pdr': float(best_da),
            'verdict': ('scale_not_structure' if (up_closed > 0.7 and down_given > 0.7)
                        else 'one_direction_only' if (up_closed > 0.7 or down_given > 0.7)
                        else 'refuted_structural'),
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"  wrote {args.out}")
    print("=" * 78 + "\n")
    return 0


if __name__ == '__main__':
    sys.exit(main())
