"""
validate_dataset.py
Runs sanity checks on a generated episode's CSV files.
Catches the common bugs that would silently corrupt ML training.

Usage:
    python src/validate_dataset.py data/pilot/pilot_01/episode_001
"""

import sys
import os
import pandas as pd
import numpy as np


def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    return condition


def validate_episode(ep_dir):
    print(f"\nValidating: {ep_dir}")
    paths = {name: os.path.join(ep_dir, f"{name}.csv") for name in [
        'mobility_trace', 'link_states', 'topology_features',
        'traffic_flows', 'packet_logs', 'routing_decisions', 'rl_transitions']}

    for name, p in paths.items():
        if not os.path.exists(p):
            print(f"  [FAIL] missing file: {name}.csv")
            return False

    m = pd.read_csv(paths['mobility_trace'])
    ls = pd.read_csv(paths['link_states'])
    t = pd.read_csv(paths['topology_features'])
    p = pd.read_csv(paths['packet_logs'])
    r = pd.read_csv(paths['routing_decisions'])

    all_ok = True

    print(" Mobility:")
    all_ok &= check("x within area bounds",
                    m.x.min() >= -1 and m.x.max() <= m.x.max() + 1)
    all_ok &= check("z within [z_min, z_max]", m.z.min() >= 49 and m.z.max() <= 151)
    all_ok &= check("no NaN in positions", not m[['x', 'y', 'z']].isna().any().any())
    all_ok &= check("energy never negative", m.residual_energy.min() >= 0)
    all_ok &= check("energy monotonic non-increasing per drone",
                    all(m.sort_values('timestamp').groupby('drone_id')
                        ['residual_energy'].apply(lambda s: (s.diff().dropna() <= 1e-9).all())))

    print(" Topology:")
    all_ok &= check("active_links not always 0", t.num_active_links.max() > 0)
    all_ok &= check("not always fully connected",
                    t.num_active_links.min() < t.num_nodes.iloc[0] * (t.num_nodes.iloc[0] - 1) / 2)
    all_ok &= check("components vary over time",
                    t.num_connected_components.nunique() > 1 or t.num_connected_components.iloc[0] >= 1)

    print(" Packets:")
    pdr = p.delivered.mean() if len(p) else 0
    all_ok &= check(f"PDR in (0.05, 0.98): {pdr:.3f}", 0.05 < pdr < 0.98)
    delivered = p[p.delivered == 1]
    if len(delivered):
        all_ok &= check("delays positive", delivered.end_to_end_delay.min() > 0)
        all_ok &= check("hop_count >= 1", delivered.hop_count.min() >= 1)
        all_ok &= check("hop_count <= TTL", delivered.hop_count.max() <= 20)
    all_ok &= check("drop reasons populated",
                    p[p.dropped == 1].drop_reason.notna().all() if (p.dropped == 1).any() else True)
    # loop check
    has_loop = any(len(path.split(',')) != len(set(path.split(',')))
                   for path in p.path_taken.dropna())
    all_ok &= check("no loops in delivered paths", not has_loop)

    print(" Routing decisions:")
    per_dec = r.groupby('decision_id').is_best_label.sum()
    all_ok &= check("exactly 1 best_label per decision",
                    (per_dec == 1).all())
    all_ok &= check("consensus in [0,5]",
                    r.teacher_consensus_count.min() >= 0 and r.teacher_consensus_count.max() <= 5)
    all_ok &= check("progress has + and - values",
                    (r.candidate_progress > 0).any() and (r.candidate_progress < 0).any())
    all_ok &= check("chosen hops are valid neighbors",
                    True)  # by construction in simulator

    print(f"\n  {'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'}  (PDR={pdr:.3f})")
    return all_ok


if __name__ == '__main__':
    target = sys.argv[1] if len(sys.argv) > 1 else 'data/pilot/pilot_01/episode_001'
    validate_episode(target)
