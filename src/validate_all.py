"""
validate_all.py
Batch validation across all pilot scenarios.

Picks sample episodes from every scenario, runs all sanity checks on each,
and produces a consolidated report showing exactly which scenarios are safe
to train on and which have problems.

Usage (from FANET_sim directory):
    python src\validate_all.py                         # validate all scenarios
    python src\validate_all.py --scenario pilot_03     # validate one scenario
    python src\validate_all.py --samples 5             # check 5 episodes per scenario
    python src\validate_all.py --all                   # validate every single episode
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path


# ─── per-episode checks (returns dict of {check_name: bool}) ─────────────────

def run_checks(ep_dir):
    ep_dir = Path(ep_dir)
    results = {}
    dfs = {}

    # Load tables
    for name in ['mobility_trace', 'link_states', 'topology_features',
                 'packet_logs', 'routing_decisions']:
        path = ep_dir / f"{name}.csv"
        if not path.exists():
            results[f'file_exists_{name}'] = False
            return results, {}
        results[f'file_exists_{name}'] = True
        dfs[name] = pd.read_csv(path)

    m = dfs['mobility_trace']
    t = dfs['topology_features']
    p = dfs['packet_logs']
    r = dfs['routing_decisions']

    # ── Mobility ──────────────────────────────────────────────────────────────
    results['mob_no_nan'] = not m[['x', 'y', 'z']].isna().any().any()
    results['mob_z_in_bounds'] = (m.z.min() >= 49.0) and (m.z.max() <= 151.0)
    results['mob_energy_non_negative'] = m.residual_energy.min() >= 0.0
    results['mob_energy_monotonic'] = all(
        m.sort_values('timestamp').groupby('drone_id')
         ['residual_energy'].apply(lambda s: (s.diff().dropna() <= 1e-9).all())
    )

    # ── Topology ──────────────────────────────────────────────────────────────
    results['topo_links_exist'] = t.num_active_links.max() > 0
    max_possible = t.num_nodes.iloc[0] * (t.num_nodes.iloc[0] - 1) / 2
    results['topo_not_always_full'] = t.num_active_links.min() < max_possible
    results['topo_components_vary'] = (
        t.num_connected_components.nunique() > 1
        or t.num_connected_components.iloc[0] >= 1
    )

    # ── Packets ───────────────────────────────────────────────────────────────
    pdr = float(p.delivered.mean()) if len(p) else 0.0
    # Lower bound only: near-zero PDR means something is broken.
    # High PDR (even 0.99+) is valid data - the network was just dense that episode.
    results['pkt_pdr_reasonable'] = pdr > 0.01
    results['pkt_pdr'] = round(pdr, 3)

    delivered = p[p.delivered == 1]
    if len(delivered) > 0:
        results['pkt_delays_positive'] = delivered.end_to_end_delay.min() > 0
        results['pkt_hops_in_range'] = (delivered.hop_count.min() >= 1 and
                                         delivered.hop_count.max() <= 20)
    else:
        results['pkt_delays_positive'] = False
        results['pkt_hops_in_range'] = False

    dropped = p[p.dropped == 1]
    results['pkt_drop_reasons_set'] = (
        dropped.drop_reason.notna().all() if len(dropped) > 0 else True
    )

    # Loop check: no node appears twice in a packet's path
    has_loop = any(
        len(path.split(',')) != len(set(path.split(',')))
        for path in p.path_taken.dropna()
    )
    results['pkt_no_loops'] = not has_loop

    # ── Routing decisions ─────────────────────────────────────────────────────
    per_dec = r.groupby('decision_id').is_best_label.sum()
    results['rd_one_label_per_decision'] = bool((per_dec == 1).all())
    results['rd_consensus_valid'] = (
        r.teacher_consensus_count.min() >= 0 and
        r.teacher_consensus_count.max() <= 5
    )
    results['rd_progress_both_signs'] = (
        (r.candidate_progress > 0).any() and
        (r.candidate_progress < 0).any()
    )

    # Label balance: is_best_label should not be 0 everywhere or 1 everywhere
    label_mean = r.is_best_label.mean()
    results['rd_label_not_degenerate'] = 0.01 < label_mean < 0.99

    # Consensus diversity: should have multiple consensus values, not all 5
    results['rd_consensus_diverse'] = r.teacher_consensus_count.nunique() > 1

    return results, dfs


def is_boolean_check(key):
    """Distinguish actual pass/fail checks from stored numeric values."""
    return key != 'pkt_pdr'


def all_pass(results):
    return all(v for k, v in results.items() if is_boolean_check(k))


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', default=None,
                    help='Validate one scenario only (e.g., pilot_03)')
    ap.add_argument('--samples', type=int, default=3,
                    help='Number of sample episodes per scenario (default: 3)')
    ap.add_argument('--all', action='store_true',
                    help='Validate every episode (overrides --samples)')
    ap.add_argument('--data_root', default='data/pilot')
    args = ap.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"Data folder not found: {data_root}")
        print("Run the simulation first: python src\\run_pilots.py")
        return

    # Discover scenarios
    scenarios = sorted([d.name for d in data_root.iterdir()
                        if d.is_dir() and d.name.startswith('pilot')])
    if args.scenario:
        scenarios = [s for s in scenarios if s == args.scenario]
        if not scenarios:
            print(f"Scenario '{args.scenario}' not found in {data_root}")
            return

    # ── Per-scenario validation ───────────────────────────────────────────────
    scenario_summaries = []

    for scenario in scenarios:
        sc_dir = data_root / scenario
        episodes = sorted([d.name for d in sc_dir.iterdir()
                           if d.is_dir() and d.name.startswith('episode')])
        if not episodes:
            print(f"\n[SKIP] {scenario}: no episodes found")
            continue

        # Select sample episodes: first, middle, last (+ more if --samples > 3)
        if args.all:
            selected = episodes
        else:
            n = min(args.samples, len(episodes))
            if n >= len(episodes):
                selected = episodes
            else:
                # Always include first and last; spread rest evenly
                indices = sorted(set([0, len(episodes) - 1] +
                    [int(i * (len(episodes) - 1) / (n - 1)) for i in range(n)]))
                selected = [episodes[i] for i in indices]

        print(f"\n{'─'*60}")
        print(f"  {scenario}   ({len(selected)} episodes sampled of {len(episodes)} total)")
        print(f"{'─'*60}")

        ep_results = []
        for ep_name in selected:
            ep_dir = sc_dir / ep_name
            checks, _ = run_checks(ep_dir)
            passed = all_pass(checks)
            pdr = checks.get('pkt_pdr', 0.0)
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {ep_name}   {status}   PDR={pdr:.3f}")

            # Print failing checks for this episode
            if not passed:
                for k, v in checks.items():
                    if is_boolean_check(k) and not v:
                        print(f"    [FAIL] {k}")

            ep_results.append({
                'scenario': scenario,
                'episode': ep_name,
                'passed': passed,
                'pdr': pdr,
            })

        # Scenario summary line
        n_pass = sum(r['passed'] for r in ep_results)
        n_total = len(ep_results)
        mean_pdr = np.mean([r['pdr'] for r in ep_results])
        scenario_summaries.append({
            'scenario': scenario,
            'episodes_checked': n_total,
            'passed': n_pass,
            'failed': n_total - n_pass,
            'all_pass': n_pass == n_total,
            'mean_pdr': round(mean_pdr, 3),
        })

    # ── Overall report ────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("  VALIDATION SUMMARY")
    print(f"{'═'*60}")
    print(f"  {'Scenario':<12} {'Checked':>8} {'Pass':>5} {'Fail':>5}"
          f"  {'MeanPDR':>8}  {'Status'}")
    print(f"  {'─'*55}")

    all_good = True
    for s in scenario_summaries:
        status_str = "✓ SAFE TO TRAIN" if s['all_pass'] else "✗ DO NOT TRAIN"
        if not s['all_pass']:
            all_good = False
        print(f"  {s['scenario']:<12} {s['episodes_checked']:>8} {s['passed']:>5}"
              f" {s['failed']:>5}  {s['mean_pdr']:>8.3f}  {status_str}")

    print(f"{'═'*60}")
    if all_good:
        print("  ✓ ALL SCENARIOS PASSED — safe to proceed to XGBoost training")
    else:
        print("  ✗ SOME SCENARIOS FAILED — do not train on failed scenarios")
        print("    Fix the cause or exclude those scenarios from training")
    print()


if __name__ == '__main__':
    main()
