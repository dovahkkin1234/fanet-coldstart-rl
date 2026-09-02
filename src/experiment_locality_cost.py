"""
experiment_locality_cost.py  —  What is SP-BP's global knowledge actually worth?

ADDRESSES REVIEWER FINDINGS M-1 and M-2, which are the two most attackable
points in the work so far:

  M-1  Every teacher queries a graph with complete, instantaneous, zero-cost
       global state -- including exact BFS hop-distance from any node to any
       destination. No distributed FANET router has this without continuous
       link-state flooding, and this project charges nothing for the control
       traffic that would buy it, while modelling data-traffic interference
       rigorously. Presenting SP-BP as a deployable classical baseline is
       therefore not defensible as written.

  M-2  The M4 design feeds that same global hop-distance to the GNN as an
       explicit input feature. If it is doing the heavy lifting, then the
       planned GNN-depth ablation is confounded: the model already has the
       hardest signal for free regardless of depth, so "shallow message
       passing suffices" would be an artifact, not a finding.

WHAT THIS MEASURES
------------------
Network PDR for SP-BP under a k-hop information horizon, k in {1,2,3,4,inf},
across the full scenario x load grid with the established 30-seed protocol.
Beyond k hops the router falls back on a locally-computable geographic proxy
(destination position from the packet header -- GPSR's standard assumption).

HOW THE RESULT SHOULD BE READ (decision rule fixed BEFORE running)
-------------------------------------------------------------------
Let gap(k) = PDR(k=inf) - PDR(k).

  If gap(3) is small (< ~0.02 PDR) in most cells:
      global knowledge is nearly free at realistic horizons. A 3-layer GNN
      could plausibly derive it, so the M4 explicit hop-distance feature is a
      convenience rather than a crutch, and SP-BP is defensible as a strong
      (if idealised) baseline with a stated, quantified caveat.

  If gap(3) is large:
      the global BFS is doing the real work. Then, honestly:
        * SP-BP must be relabelled a GENIE-AIDED UPPER BOUND, not a
          deployable baseline, with spbp_k3 reported alongside as the
          realistic comparison point;
        * the M4 hop-distance feature must be declared a pretraining-only
          auxiliary signal, and the GNN-depth ablation must be re-scoped or
          re-run without it, or its conclusions are confounded.

Either outcome is publishable. The first is convenient; the SECOND IS MORE
INTERESTING, because "quantifying the cost of locality in FANET routing" is a
contribution in its own right and pre-empts the reviewer question rather than
leaving it open.

PRE-REGISTERED PREDICTION (recorded before the run, so this is a test)
-----------------------------------------------------------------------
The M3.5 audit measured mean hop-distance-to-destination per scenario:
    very_dense 1.64   dense_slow 2.09   medium_slow 5.32   sparse_fast 6.74
A k=3 horizon should therefore COVER the typical destination in the dense
regimes and MISS it in the sparse ones, so the locality cost should be small
in very_dense/dense_slow and large in medium_slow/sparse_fast. If the cost is
instead uniform across regimes, this model of locality is wrong and the
conclusion must not be trusted.

Usage:
    python src\\experiment_locality_cost.py --max_workers 16
    python src\\experiment_locality_cost.py --seeds 1 2 3 --quick
"""

import os, sys, json, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from teacher_panel import build_oracle_table, paired_ttest, pearson_r

# Config now lives in config_v2.py.
from config_v2 import SCENARIOS, RATES, BASE  # noqa: F401
HORIZONS = ['spbp_k1', 'spbp_k2', 'spbp_k3', 'spbp_k4', 'spbp_kinf']

# Mean hop-distance to destination per scenario, measured in the M3.5 audit.
# Used only to check the pre-registered prediction, never to alter the result.
MEAN_HOPS = {'very_dense': 1.64, 'dense_slow': 2.09,
             'medium_slow': 5.32, 'sparse_fast': 6.74}

SMALL_GAP = 0.02      # PDR gap below which locality is considered ~free


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=list(range(1, 31)))
    ap.add_argument('--rates', type=float, nargs='+', default=RATES)
    ap.add_argument('--duration', type=float, default=BASE['duration'])
    ap.add_argument('--drain_time', type=float, default=BASE['drain_time'])
    ap.add_argument('--max_workers', type=int, default=None)
    ap.add_argument('--quick', action='store_true',
                    help='single scenario, for a fast smoke run')
    ap.add_argument('--out', default='results/locality_cost.json')
    args = ap.parse_args()

    scenarios = ({'medium_slow': SCENARIOS['medium_slow']} if args.quick
                 else SCENARIOS)
    base = dict(z_min=BASE['z_min'], z_max=BASE['z_max'], duration=args.duration,
                interference_on=BASE['interference_on'], drain_time=args.drain_time)

    n_runs = len(scenarios) * len(args.rates) * len(args.seeds) * len(HORIZONS)
    print("\n" + "=" * 78)
    print("  LOCALITY-COST EXPERIMENT  (reviewer findings M-1 / M-2)")
    print("=" * 78)
    print(f"  horizons: {', '.join(HORIZONS)}")
    print(f"  scenarios: {', '.join(scenarios)}")
    print(f"  rates: {args.rates}   seeds: {len(args.seeds)}   runs: {n_runs}")
    print(f"  decision rule (fixed before running): gap(k=3) < {SMALL_GAP} PDR in")
    print(f"  most cells => locality is ~free; otherwise SP-BP must be relabelled")
    print(f"  a genie-aided upper bound and the M4 hop-distance feature declared")
    print(f"  a pretraining-only auxiliary signal.")

    table, raw, congestion, diagnostics, table_stats = build_oracle_table(
        scenarios, args.rates, args.seeds, panel=HORIZONS,
        base_cfg=base, verbose=True, max_workers=args.max_workers)

    # ---- per-cell gaps, with paired significance ----
    print("\n" + "-" * 78)
    print("  PDR BY INFORMATION HORIZON  (mean over seeds)")
    print("-" * 78)
    hdr = f"  {'scenario_class':<14}{'load':<8}" + "".join(f"{h.replace('spbp_',''):>9}" for h in HORIZONS)
    print(hdr + f"{'gap(k3)':>10}{'p(k3 vs inf)':>14}")

    rows = []
    for key in sorted(table):
        by_teacher = {t: v for t, v in table[key]}
        line = f"  {key[0]:<14}{key[1]:<8}"
        for h in HORIZONS:
            line += f"{by_teacher.get(h, float('nan')):>9.3f}"
        gap3 = by_teacher.get('spbp_kinf', np.nan) - by_teacher.get('spbp_k3', np.nan)

        st = table_stats.get(key, {})
        ps3, psinf = st.get('spbp_k3', {}), st.get('spbp_kinf', {})
        common = sorted(set(ps3.get('per_seed', {})) & set(psinf.get('per_seed', {})))
        if len(common) >= 2:
            x = [psinf['per_seed'][s] for s in common]
            y = [ps3['per_seed'][s] for s in common]
            _, _, p = paired_ttest(x, y)
        else:
            p = float('nan')
        line += f"{gap3:>10.3f}{p:>14.4f}"
        print(line)
        rows.append({'scenario_class': key[0], 'load_bucket': key[1],
                     'pdr': by_teacher, 'gap_k3': float(gap3), 'p_k3_vs_inf': float(p)})

    # ---- check the pre-registered prediction ----
    print("\n" + "-" * 78)
    print("  PRE-REGISTERED PREDICTION CHECK")
    print("-" * 78)
    print("  Predicted: locality cost small where destinations are near (dense),")
    print("  large where they are far (sparse). Measured mean hops per scenario:")
    for sc, mh in MEAN_HOPS.items():
        print(f"    {sc:<13} {mh:.2f} hops   (k=3 horizon "
              f"{'COVERS' if mh <= 3 else 'MISSES'} the typical destination)")

    by_class = {}
    for r in rows:
        by_class.setdefault(r['scenario_class'], []).append(r['gap_k3'])
    print("\n  Measured mean gap(k=3) by scenario class:")
    for c, g in sorted(by_class.items()):
        print(f"    {c:<14} {np.mean(g):+.4f}")

    dense_like = [np.mean(g) for c, g in by_class.items()
                  if c in ('dense', 'very-dense')]
    sparse_like = [np.mean(g) for c, g in by_class.items()
                   if c in ('medium', 'sparse-fast')]
    if dense_like and sparse_like:
        pred_holds = np.mean(sparse_like) > np.mean(dense_like)
        print(f"\n  dense-like mean gap  = {np.mean(dense_like):+.4f}")
        print(f"  sparse-like mean gap = {np.mean(sparse_like):+.4f}")
        print(f"  prediction (sparse cost > dense cost): "
              f"{'HELD' if pred_holds else 'FAILED'}")
        if not pred_holds:
            print("    ** The locality model does not behave as predicted. Do NOT")
            print("       draw conclusions from this experiment until the reason is")
            print("       understood -- an unpredicted uniform cost suggests the")
            print("       horizon is not the operative variable.")

    # ---- verdict against the pre-fixed decision rule ----
    gaps3 = [r['gap_k3'] for r in rows]
    frac_small = float(np.mean([g < SMALL_GAP for g in gaps3]))
    print("\n" + "=" * 78)
    print("  VERDICT")
    print("=" * 78)
    print(f"  mean gap(k=3)   = {np.mean(gaps3):+.4f} PDR")
    print(f"  median gap(k=3) = {np.median(gaps3):+.4f} PDR")
    print(f"  cells with gap < {SMALL_GAP}: {frac_small:.0%}")
    print()
    if frac_small >= 0.5 and np.mean(gaps3) < SMALL_GAP:
        print("  -> LOCALITY IS APPROXIMATELY FREE at a 3-hop horizon.")
        print("     SP-BP remains defensible as a strong baseline, with the")
        print("     idealised-information caveat stated and quantified by this")
        print("     experiment. The M4 explicit hop-distance feature is a")
        print("     convenience, not a crutch: a 3-layer GNN could derive")
        print("     comparable information. The depth ablation stays valid.")
    else:
        print("  -> GLOBAL KNOWLEDGE IS DOING REAL WORK. Required consequences:")
        print("     1. SP-BP must be presented as a GENIE-AIDED UPPER BOUND, with")
        print("        spbp_k3 reported alongside as the realistic comparison.")
        print("     2. The M4 `hop_distance_to_dst` input must be declared a")
        print("        pretraining-only auxiliary signal, and the GNN-depth")
        print("        ablation re-scoped or re-run without it -- otherwise its")
        print("        conclusions are confounded by information the message")
        print("        passing never had to earn.")
        print("     3. This gap, reported honestly, becomes a CONTRIBUTION:")
        print("        the cost of locality in FANET routing, quantified.")
    print("=" * 78 + "\n")

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({'rows': rows, 'mean_gap_k3': float(np.mean(gaps3)),
                   'frac_cells_small_gap': frac_small,
                   'seeds': list(args.seeds), 'rates': list(args.rates),
                   'horizons': HORIZONS, 'small_gap_threshold': SMALL_GAP,
                   'mean_hops_reference': MEAN_HOPS}, f, indent=2)
    print(f"  saved to {args.out}\n")
    return 0


if __name__ == '__main__':
    sys.exit(main())
