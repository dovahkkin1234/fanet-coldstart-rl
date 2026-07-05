"""
run_full_rwp.py
Full-scale RWP dataset generator — 15 scenarios, 390 episodes.

Reads configs/rwp_full_configs.yaml and generates the 7 CSV files per episode.
Supports running specific scenarios (for parallelization across terminals).

Usage:
    # Full dataset (all 15 scenarios — ~7-12 hours)
    python src\run_full_rwp.py

    # Specific scenarios (for parallel runs in separate terminals)
    python src\run_full_rwp.py --scenarios rwp_sc01 rwp_sc02 rwp_sc03

    # Dry-run: 1 episode of easiest + hardest to verify PDR boundaries
    python src\run_full_rwp.py --dry_run

    # Override episodes/duration
    python src\run_full_rwp.py --scenarios rwp_sc01 --episodes 5 --duration 120

    # Resume: skip episodes that already exist on disk
    python src\run_full_rwp.py --resume
"""

import os, sys, argparse, time, yaml, traceback
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simulator import FANETSimulator


def load_configs(yaml_path):
    """Load rwp_full_configs.yaml and merge defaults into each scenario."""
    with open(yaml_path, 'r') as f:
        raw = yaml.safe_load(f)

    defaults = raw.get('defaults', {})
    scenarios = []
    for sc in raw.get('scenarios', []):
        cfg = {**defaults, **sc}
        scenarios.append(cfg)
    return scenarios


def build_episode_config(scenario, episode_num):
    """Build the config dict that FANETSimulator expects."""
    sc_num = int(scenario['name'].replace('rwp_sc', ''))
    seed = sc_num * 10000 + episode_num

    return {
        'scenario_id':   scenario['name'],
        'episode':       episode_num,
        'seed':          seed,
        'num_drones':    scenario['num_drones'],
        'area_x':        scenario['area_x'],
        'area_y':        scenario['area_y'],
        'z_min':         scenario.get('z_min', 50),
        'z_max':         scenario.get('z_max', 150),
        'speed_min':     scenario['speed_min'],
        'speed_max':     scenario['speed_max'],
        'comm_range':    scenario['comm_range'],
        'pause_max':     scenario.get('pause_max', 5.0),
        'timestep':      scenario.get('timestep', 0.5),
        'duration':      scenario.get('duration', 300),
        'packet_rate':   scenario.get('packet_rate', 1.0),
        'mobility_model': scenario.get('mobility_model', 'random_waypoint'),
    }


def save_episode(dfs, out_dir):
    """Save the 7 DataFrames as CSVs into an episode directory."""
    os.makedirs(out_dir, exist_ok=True)
    for name, df in dfs.items():
        df.to_csv(os.path.join(out_dir, f'{name}.csv'), index=False)


def run_one_episode(scenario, episode_num, out_root, duration_override=None):
    """Run a single episode and save to disk. Returns (pdr, elapsed_sec)."""
    cfg = build_episode_config(scenario, episode_num)
    if duration_override:
        cfg['duration'] = duration_override

    ep_dir = os.path.join(out_root, scenario['name'],
                          f'episode_{episode_num:03d}')

    t0 = time.time()
    sim = FANETSimulator(cfg)
    dfs = sim.run()
    elapsed = time.time() - t0

    save_episode(dfs, ep_dir)

    # Compute PDR
    pk = dfs.get('packet_logs', pd.DataFrame())
    if len(pk) > 0 and 'delivered' in pk.columns:
        pdr = float(pk['delivered'].mean())
    else:
        pdr = 0.0

    # Quick sanity: check topology_change_rate
    tf = dfs.get('topology_features', pd.DataFrame())
    tcr_mean = float(tf['topology_change_rate'].mean()) if 'topology_change_rate' in tf.columns else -1
    qocc_mean = float(tf['avg_queue_occupancy'].mean()) if 'avg_queue_occupancy' in tf.columns else -1

    return pdr, elapsed, tcr_mean, qocc_mean, len(pk)


def run_dry_run(scenarios, out_root):
    """Run 1 episode each of the easiest and hardest scenario to verify PDR."""
    # Find sc01 (easiest) and sc14 (hardest)
    easy = next((s for s in scenarios if s['name'] == 'rwp_sc01'), scenarios[0])
    hard = next((s for s in scenarios if s['name'] == 'rwp_sc14'), scenarios[-1])

    print(f"\n{'═'*65}")
    print(f"  DRY RUN — PDR boundary check")
    print(f"{'═'*65}\n")

    for label, sc in [('EASIEST', easy), ('HARDEST', hard)]:
        print(f"  {label}: {sc['name']}")
        print(f"    {sc['num_drones']} drones | "
              f"{sc['area_x']}×{sc['area_y']}m | "
              f"speed {sc['speed_min']}–{sc['speed_max']} m/s | "
              f"range {sc['comm_range']}m")

        pdr, elapsed, tcr, qocc, n_pkts = run_one_episode(
            sc, episode_num=9999, out_root=out_root, duration_override=300)

        print(f"    PDR:                    {pdr:.3f}  {'✓' if 0.05 < pdr < 0.98 else '✗ OUT OF BOUNDS'}")
        print(f"    topology_change_rate:   {tcr:.4f}  {'✓' if tcr > 0.001 else '✗ STILL ZERO — Bug 1 not fixed!'}")
        print(f"    avg_queue_occupancy:    {qocc:.4f}  {'✓' if qocc > 0.0001 else '⚠ Very low or zero — check Bug 2'}")
        print(f"    Packets:                {n_pkts}")
        print(f"    Time:                   {elapsed:.1f}s\n")

    print(f"  If PDR is out of bounds or topology_change_rate is 0.0,")
    print(f"  fix the simulator bugs before running the full generation.\n")


def main():
    ap = argparse.ArgumentParser(description='Generate full-scale RWP dataset')
    ap.add_argument('--configs', default='configs/rwp_full_configs.yaml',
                    help='Path to YAML config file')
    ap.add_argument('--out_root', default='data/rwp_full',
                    help='Output root directory')
    ap.add_argument('--scenarios', nargs='+', default=None,
                    help='Run only these scenarios (e.g. rwp_sc01 rwp_sc02)')
    ap.add_argument('--episodes', type=int, default=None,
                    help='Override episodes per scenario')
    ap.add_argument('--duration', type=int, default=None,
                    help='Override episode duration (seconds)')
    ap.add_argument('--dry_run', action='store_true',
                    help='Run 1 episode of easiest + hardest to check PDR')
    ap.add_argument('--resume', action='store_true',
                    help='Skip episodes that already exist on disk')
    args = ap.parse_args()

    # Load configs
    if not os.path.exists(args.configs):
        print(f"  [FATAL] Config file not found: {args.configs}")
        print(f"  Copy rwp_full_configs.yaml to configs/ first.")
        sys.exit(1)

    scenarios = load_configs(args.configs)
    print(f"\n{'═'*65}")
    print(f"  RWP FULL-SCALE DATASET GENERATOR")
    print(f"{'═'*65}")
    print(f"  Config:    {args.configs}")
    print(f"  Output:    {args.out_root}")
    print(f"  Scenarios: {len(scenarios)} loaded")

    # Filter scenarios if specified
    if args.scenarios:
        scenarios = [s for s in scenarios if s['name'] in args.scenarios]
        print(f"  Filtered:  {[s['name'] for s in scenarios]}")
        if not scenarios:
            print(f"  [FATAL] No matching scenarios found.")
            sys.exit(1)

    # Dry run mode
    if args.dry_run:
        all_scenarios = load_configs(args.configs)  # need full list for easy/hard
        run_dry_run(all_scenarios, args.out_root)
        return

    # Calculate total episodes
    total_episodes = 0
    for sc in scenarios:
        n_ep = args.episodes or sc.get('episodes', 30)
        total_episodes += n_ep

    print(f"  Total:     {total_episodes} episodes to generate")
    if args.resume:
        print(f"  Mode:      RESUME (skipping existing)")
    print()

    # ── Main generation loop ──────────────────────────────────────────────────
    t0_global = time.time()
    completed = 0
    skipped = 0
    failed = 0
    pdr_records = []

    for sc in scenarios:
        sc_name = sc['name']
        n_episodes = args.episodes or sc.get('episodes', 30)
        duration = args.duration or sc.get('duration', 300)

        print(f"{'─'*65}")
        print(f"  Scenario: {sc_name}  ({sc['num_drones']} drones, "
              f"{sc['area_x']}×{sc['area_y']}m, "
              f"v={sc['speed_min']}–{sc['speed_max']}, "
              f"r={sc['comm_range']}m)")
        print(f"  Episodes: {n_episodes} × {duration}s")
        print(f"  {'Ep':>4}  {'Seed':>7}  {'PDR':>6}  {'TCR':>6}  {'QOcc':>6}  "
              f"{'Pkts':>5}  {'Time':>6}  Status")
        print(f"  {'─'*60}")

        sc_pdrs = []

        for ep_num in range(1, n_episodes + 1):
            ep_dir = os.path.join(args.out_root, sc_name,
                                  f'episode_{ep_num:03d}')

            # Resume check
            if args.resume and os.path.exists(ep_dir):
                rd_path = os.path.join(ep_dir, 'routing_decisions.csv')
                if os.path.exists(rd_path):
                    skipped += 1
                    continue

            sc_num = int(sc_name.replace('rwp_sc', ''))
            seed = sc_num * 10000 + ep_num

            try:
                pdr, elapsed, tcr, qocc, n_pkts = run_one_episode(
                    sc, ep_num, args.out_root, duration_override=duration)

                status = '✓'
                if pdr < 0.05:
                    status = '⚠ LOW PDR'
                elif pdr > 0.98:
                    status = '⚠ HIGH PDR'
                if tcr < 0.001:
                    status += ' ⚠TCR=0'

                print(f"  {ep_num:>4}  {seed:>7}  {pdr:>6.3f}  {tcr:>6.3f}  "
                      f"{qocc:>6.4f}  {n_pkts:>5}  {elapsed:>5.1f}s  {status}")

                sc_pdrs.append(pdr)
                pdr_records.append({
                    'scenario': sc_name, 'episode': ep_num, 'seed': seed,
                    'pdr': pdr, 'topology_change_rate_mean': tcr,
                    'queue_occupancy_mean': qocc, 'n_packets': n_pkts,
                    'time_sec': elapsed,
                })
                completed += 1

            except Exception as e:
                print(f"  {ep_num:>4}  {seed:>7}  {'FAILED':>6}  "
                      f"{'':>6}  {'':>6}  {'':>5}  {'':>6}  ✗ {str(e)[:40]}")
                traceback.print_exc()
                failed += 1

        if sc_pdrs:
            mean_pdr = np.mean(sc_pdrs)
            std_pdr = np.std(sc_pdrs)
            print(f"\n  {sc_name} summary: PDR = {mean_pdr:.3f} ± {std_pdr:.3f}  "
                  f"({len(sc_pdrs)} episodes)")

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed_total = time.time() - t0_global
    hours = elapsed_total / 3600

    print(f"\n{'═'*65}")
    print(f"  GENERATION COMPLETE")
    print(f"{'═'*65}")
    print(f"  Completed: {completed}")
    print(f"  Skipped:   {skipped} (resume)")
    print(f"  Failed:    {failed}")
    print(f"  Total time: {elapsed_total:.0f}s ({hours:.1f} hours)")
    print(f"  Output:    {args.out_root}")

    # Save summary CSV
    if pdr_records:
        summary_df = pd.DataFrame(pdr_records)
        summary_path = os.path.join(args.out_root, 'run_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"  Summary:   {summary_path}")

        # Per-scenario summary
        print(f"\n  {'Scenario':<12} {'Episodes':>8} {'PDR_mean':>9} {'PDR_std':>8} "
              f"{'TCR_mean':>9} {'QOcc_mean':>10}")
        print(f"  {'─'*62}")
        for sc_name, grp in summary_df.groupby('scenario'):
            print(f"  {sc_name:<12} {len(grp):>8} {grp['pdr'].mean():>9.3f} "
                  f"{grp['pdr'].std():>8.3f} "
                  f"{grp['topology_change_rate_mean'].mean():>9.4f} "
                  f"{grp['queue_occupancy_mean'].mean():>10.5f}")

        # Flag problematic scenarios
        print()
        for sc_name, grp in summary_df.groupby('scenario'):
            mean_pdr = grp['pdr'].mean()
            mean_tcr = grp['topology_change_rate_mean'].mean()
            if mean_pdr < 0.05:
                print(f"  ⚠ {sc_name}: PDR too low ({mean_pdr:.3f}) — "
                      f"increase comm_range or reduce area")
            if mean_pdr > 0.98:
                print(f"  ⚠ {sc_name}: PDR too high ({mean_pdr:.3f}) — "
                      f"reduce comm_range or increase area")
            if mean_tcr < 0.001:
                print(f"  ⚠ {sc_name}: topology_change_rate ≈ 0 — "
                      f"Bug 1 not fixed!")

    print(f"\n{'═'*65}")
    print(f"  Next: python src\\train_warmstart.py --data_root data\\rwp_full")
    print(f"{'═'*65}\n")


if __name__ == '__main__':
    main()
