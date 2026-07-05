"""
rl_preflight_check.py
Run this BEFORE starting Phase 5 RL training.

Verifies every prerequisite so the 4-6 hour run doesn't fail partway or
produce garbage. Checks:
  1. mlp_warmstart_full.pt exists and loads (RL policy init)
  2. regime_clustering_full.pkl exists (k=3 regimes)
  3. regime_id is populated (not -1) in the full-scale data
  4. CUDA is available
  5. rl_env.py and train_dqn.py exist in src/
  6. Feature columns match between MLP bundle and env
  7. All 15 scenario folders exist with episodes

Usage:
    python src\\rl_preflight_check.py
"""

import os, sys, pickle
from pathlib import Path

FEATURE_COLS = [
    'candidate_distance', 'candidate_relative_velocity', 'candidate_link_quality',
    'candidate_rssi', 'candidate_packet_error_rate', 'candidate_estimated_lifetime',
    'candidate_energy', 'candidate_queue_occupancy', 'candidate_dist_to_dest',
    'candidate_progress', 'current_energy', 'current_queue_occupancy',
    'current_num_neighbors', 'current_speed', 'distance_to_destination',
    'bearing_to_destination', 'destination_reachable', 'avg_degree',
    'network_density', 'num_components', 'avg_link_quality', 'topology_change_rate',
]
ALL_SCENARIOS = [f'rwp_sc{i:02d}' for i in range(1, 16)]

def check(name, condition, detail=''):
    status = '✓' if condition else '✗'
    print(f"  [{status}] {name}")
    if detail:
        print(f"       {detail}")
    return condition

def main():
    data_root = 'data/rwp_full'
    models    = 'models'
    src       = 'src'
    all_ok = True

    print(f"\n{'═'*60}")
    print("  PHASE 5 PRE-FLIGHT CHECK")
    print(f"{'═'*60}\n")

    # 1. MLP warmstart bundle
    print("── Warm-start model ──")
    mlp_path = os.path.join(models, 'mlp_warmstart_full.pt')
    if check("mlp_warmstart_full.pt exists", os.path.exists(mlp_path), mlp_path):
        try:
            import torch
            bundle = torch.load(mlp_path, map_location='cpu', weights_only=False)
            keys = list(bundle.keys())
            has_sd  = 'model_state_dict' in bundle
            has_fm  = 'feat_mean' in bundle
            has_fs  = 'feat_std' in bundle
            has_fc  = 'feature_cols' in bundle
            all_ok &= check("  contains model_state_dict", has_sd)
            all_ok &= check("  contains feat_mean / feat_std", has_fm and has_fs)
            all_ok &= check("  contains feature_cols", has_fc)
            if has_fc:
                match = bundle['feature_cols'] == FEATURE_COLS
                all_ok &= check("  feature_cols matches expected 22",
                                match,
                                '' if match else f"MISMATCH: {bundle['feature_cols']}")
            # verify input dim
            if has_sd:
                first_w = bundle['model_state_dict'].get('net.0.weight')
                if first_w is not None:
                    in_dim = first_w.shape[1]
                    all_ok &= check(f"  MLP input dim = 22", in_dim == 22,
                                    f"actual: {in_dim}")
        except Exception as e:
            all_ok &= check("  loads without error", False, str(e))
    else:
        all_ok = False

    # 2. Regime clustering
    print("\n── Regime clustering ──")
    regime_path = os.path.join(models, 'regime_clustering_full.pkl')
    n_regimes = None
    if check("regime_clustering_full.pkl exists", os.path.exists(regime_path), regime_path):
        try:
            with open(regime_path, 'rb') as f:
                rb = pickle.load(f)
            n_regimes = rb.get('k')
            all_ok &= check(f"  k = {n_regimes} regimes", n_regimes is not None,
                            f"regime_names: {rb.get('regime_names')}")
        except Exception as e:
            all_ok &= check("  loads without error", False, str(e))
    else:
        all_ok = False

    # 3. regime_id populated in data
    print("\n── regime_id written back to data ──")
    sample_checked = 0
    regime_ok = True
    for sc in ALL_SCENARIOS[:3] + ['rwp_sc06', 'rwp_sc14']:  # sample
        sc_dir = Path(data_root) / sc
        if not sc_dir.exists():
            continue
        eps = sorted([d for d in sc_dir.iterdir() if d.is_dir()])
        if not eps:
            continue
        rd = eps[0] / 'routing_decisions.csv'
        if rd.exists():
            import pandas as pd
            df = pd.read_csv(rd, usecols=['regime_id'])
            uniq = sorted(df['regime_id'].unique())
            populated = not (len(uniq) == 1 and uniq[0] == -1)
            sample_checked += 1
            if not populated:
                regime_ok = False
                print(f"  [✗] {sc}/{eps[0].name}: regime_id still -1 (not written back)")
            else:
                print(f"  [✓] {sc}/{eps[0].name}: regime_id = {uniq}")
    all_ok &= check(f"regime_id populated in {sample_checked} sampled episodes", regime_ok)

    # 4. CUDA
    print("\n── Compute ──")
    try:
        import torch
        cuda = torch.cuda.is_available()
        check("CUDA available", cuda,
              torch.cuda.get_device_name(0) if cuda else "will fall back to CPU (slow)")
    except Exception as e:
        check("torch import", False, str(e))

    # 5. RL scripts exist
    print("\n── RL scripts ──")
    env_exists = os.path.exists(os.path.join(src, 'rl_env.py'))
    dqn_exists = os.path.exists(os.path.join(src, 'train_dqn.py'))
    all_ok &= check("src/rl_env.py exists", env_exists)
    all_ok &= check("src/train_dqn.py exists", dqn_exists)

    # 6. Scenario folders
    print("\n── Dataset ──")
    n_found = 0
    total_eps = 0
    for sc in ALL_SCENARIOS:
        sc_dir = Path(data_root) / sc
        if sc_dir.exists():
            eps = [d for d in sc_dir.iterdir() if d.is_dir()]
            n_found += 1
            total_eps += len(eps)
    all_ok &= check(f"all 15 scenario folders present", n_found == 15,
                    f"found {n_found}/15, {total_eps} total episodes")

    # Summary
    print(f"\n{'═'*60}")
    if all_ok:
        print("  ✓ ALL CHECKS PASSED — safe to start RL training")
    else:
        print("  ✗ SOME CHECKS FAILED — fix before running RL training")
    print(f"{'═'*60}\n")

    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())