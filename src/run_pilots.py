"""
run_pilots.py
Runs all pilot scenarios defined in configs/pilot_configs.yaml.

Usage (from FANET_sim directory):
    python src/run_pilots.py                  # run everything
    python src/run_pilots.py --config pilot_01    # run one scenario
    python src/run_pilots.py --episodes 5     # override episode count (quick test)
    python src/run_pilots.py --duration 120   # override episode duration

Output goes to data/pilot/<scenario>/episode_XXX/*.csv
"""

import os
import sys
import argparse
import yaml
import pandas as pd
from tqdm import tqdm

# Make src importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simulator import FANETSimulator

TABLES = ['mobility_trace', 'link_states', 'topology_features',
          'traffic_flows', 'packet_logs', 'routing_decisions', 'rl_transitions']


def load_configs(path):
    with open(path, 'r') as f:
        raw = yaml.safe_load(f)
    return raw


def build_episode_config(scenario, defaults, episode, config_index):
    cfg = dict(defaults)
    cfg.update({k: v for k, v in scenario.items() if k != 'name'})
    cfg['scenario_id'] = (
        f"{scenario['name']}_{scenario['num_drones']}n"
        f"_v{scenario['speed_max']}_r{scenario['comm_range']}_ep{episode:03d}"
    )
    cfg['episode'] = episode
    # Deterministic, unique seed per (scenario, episode)
    cfg['seed'] = config_index * 1000 + episode
    return cfg


def run_one_episode(cfg, out_dir):
    sim = FANETSimulator(cfg)
    dfs = sim.run()
    os.makedirs(out_dir, exist_ok=True)
    for name in TABLES:
        dfs[name].to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)
    # quick metrics for the progress log
    p = dfs['packet_logs']
    pdr = p['delivered'].mean() if len(p) else 0.0
    return pdr, len(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=None, help='run only this scenario name')
    ap.add_argument('--episodes', type=int, default=None, help='override episode count')
    ap.add_argument('--duration', type=int, default=None, help='override duration (s)')
    ap.add_argument('--configs_path', default='configs/pilot_configs.yaml')
    ap.add_argument('--out_root', default='data/pilot')
    args = ap.parse_args()

    raw = load_configs(args.configs_path)
    defaults = raw['defaults']
    episodes_per_config = args.episodes or raw['episodes_per_config']
    if args.duration:
        defaults['duration'] = args.duration

    scenarios = raw['scenarios']
    if args.config:
        scenarios = [s for s in scenarios if s['name'] == args.config]
        if not scenarios:
            print(f"No scenario named {args.config}")
            return

    summary = []
    for ci, scenario in enumerate(scenarios):
        name = scenario['name']
        print(f"\n=== {name} ({scenario['num_drones']} drones, "
              f"{scenario['area_x']}x{scenario['area_y']}m, "
              f"speed {scenario['speed_min']}-{scenario['speed_max']}, "
              f"range {scenario['comm_range']}m) ===")
        for ep in tqdm(range(1, episodes_per_config + 1), desc=name):
            cfg = build_episode_config(scenario, defaults, ep, ci)
            out_dir = os.path.join(args.out_root, name, f"episode_{ep:03d}")
            pdr, n_pkts = run_one_episode(cfg, out_dir)
            summary.append({'scenario': name, 'episode': ep,
                            'pdr': pdr, 'packets': n_pkts})

    sdf = pd.DataFrame(summary)
    os.makedirs(args.out_root, exist_ok=True)
    sdf.to_csv(os.path.join(args.out_root, 'run_summary.csv'), index=False)

    print("\n=== SUMMARY (mean PDR per scenario) ===")
    print(sdf.groupby('scenario')['pdr'].agg(['mean', 'std', 'min', 'max']).round(3))
    print(f"\nDone. Output in {args.out_root}/")


if __name__ == '__main__':
    main()
