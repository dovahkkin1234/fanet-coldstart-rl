"""
cluster_regimes.py
Phase 4 — Topology Regime Detection via Unsupervised Clustering.

Loads topology_features.csv from all pilot episodes, clusters them into
network "regimes" (e.g., dense-stable, sparse-fast, partition-prone), and
writes regime_id back into every routing_decisions.csv and rl_transitions.csv
for use as a feature in the RL agent's state vector.

Usage:
    python src\\cluster_regimes.py
    python src\\cluster_regimes.py --k 6
    python src\\cluster_regimes.py --k 5 --no_plot

Why this matters:
- Single MDP assumption is wrong for FANETs. The optimal routing policy
  in a dense-stable network is different from one in a sparse-partitioned
  network. RL agents trained without regime awareness average across these
  policies and underperform in both extremes.
- By conditioning the policy on regime_id, the agent can implicitly learn
  different sub-policies for different topology regimes.
"""

import os, sys, argparse, time, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Features used for clustering. Excluded:
#   - identifiers (scenario_id, timestamp, etc.)
#   - degenerate features (topology_change_rate, queue_occupancy: always 0)
#   - dependent features (largest_component_size duplicates ratio)
CLUSTER_FEATURES = [
    'avg_degree',
    'network_density',
    'num_connected_components',
    'largest_component_ratio',
    'avg_link_quality',
    'avg_link_lifetime',
    'avg_relative_velocity',
    'avg_residual_energy',
    'min_residual_energy',
]


def load_all_topology_data(data_root):
    """Load topology_features.csv from every episode of every pilot."""
    parts = []
    print("  Loading topology features...")
    for sc_dir in sorted(Path(data_root).iterdir()):
        if not sc_dir.is_dir() or not sc_dir.name.startswith('pilot'):
            continue
        for ep_dir in sorted(sc_dir.iterdir()):
            if not ep_dir.is_dir():
                continue
            f = ep_dir / 'topology_features.csv'
            if f.exists():
                df = pd.read_csv(f)
                df['_scenario'] = sc_dir.name
                df['_episode']  = ep_dir.name
                df['_csv_path'] = str(f)
                parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    print(f"  Loaded {len(out):,} topology snapshots from "
          f"{out['_scenario'].nunique()} scenarios.")
    return out


def fit_kmeans(X, k, seed=42):
    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    labels = km.fit_predict(X)
    return km, labels


def evaluate_k(X, k_range, seed=42):
    """Sweep k from k_range[0] to k_range[1] and return inertia + silhouette."""
    results = []
    print(f"  Sweeping k = {k_range[0]}..{k_range[1]}:")
    print(f"  {'k':>3}  {'inertia':>11}  {'silhouette':>11}")
    print(f"  {'─'*30}")
    sample_idx = np.random.default_rng(seed).choice(
        len(X), size=min(20000, len(X)), replace=False)  # silhouette is O(n^2)
    for k in range(k_range[0], k_range[1] + 1):
        km, labels = fit_kmeans(X, k, seed=seed)
        sil = silhouette_score(X[sample_idx], labels[sample_idx]) if k > 1 else 0.0
        results.append({'k': k, 'inertia': km.inertia_, 'silhouette': sil})
        print(f"  {k:>3}  {km.inertia_:>11.1f}  {sil:>11.4f}")
    return pd.DataFrame(results)


def interpret_centroids(centroids_orig, feature_names):
    """Assign a human-readable name to each cluster based on its centroid."""
    df = pd.DataFrame(centroids_orig, columns=feature_names)
    df.index.name = 'regime_id'
    K = len(df)

    # Z-score each feature across clusters for relative comparison
    z = (df - df.mean()) / (df.std() + 1e-8)

    names = []
    for i in range(K):
        deg_z   = z.loc[i, 'avg_degree']
        comps_z = z.loc[i, 'num_connected_components']
        vel_z   = z.loc[i, 'avg_relative_velocity']
        lc_z    = z.loc[i, 'largest_component_ratio']
        den_z   = z.loc[i, 'network_density']

        # Build a descriptive 2-word label from the strongest two axes
        parts = []
        if den_z > 0.8 or deg_z > 0.8:
            parts.append("Dense")
        elif den_z < -0.5 or deg_z < -0.5:
            parts.append("Sparse")
        else:
            parts.append("Medium")

        if comps_z > 0.8 or lc_z < -0.5:
            parts.append("Partitioned")
        elif vel_z > 0.5:
            parts.append("Fast")
        elif vel_z < -0.5:
            parts.append("Slow")
        else:
            parts.append("Stable")

        names.append("-".join(parts))

    # If still duplicates, disambiguate by appending an index
    seen = {}
    for i, n in enumerate(names):
        if n in seen:
            seen[n] += 1
            names[i] = f"{n}-{seen[n]}"
        else:
            seen[n] = 1

    df['regime_name'] = names
    return df


def write_regime_back(data_root, topo_df, label_col='regime_id_new'):
    """
    Write regime_id back into:
      1. topology_features.csv  — one row per timestamp
      2. routing_decisions.csv  — joined by (scenario, episode, timestamp)
      3. rl_transitions.csv     — joined by (scenario, episode, timestamp)
    """
    print("\n  Writing regime_id back into all CSV tables...")
    n_updated = {'topology_features': 0, 'routing_decisions': 0, 'rl_transitions': 0}

    by_episode = topo_df.groupby(['_scenario', '_episode'])
    total_eps = len(by_episode)

    for i, ((sc, ep), grp) in enumerate(by_episode):
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{total_eps} episodes updated")

        ep_dir = Path(data_root) / sc / ep

        # Build a (timestamp -> regime_id) lookup for this episode
        lookup = dict(zip(grp['timestamp'].round(6), grp[label_col]))

        # 1. topology_features.csv (overwrite regime_id column)
        f = ep_dir / 'topology_features.csv'
        if f.exists():
            df = pd.read_csv(f)
            df['regime_id'] = df['timestamp'].round(6).map(lookup).fillna(-1).astype(int)
            df.to_csv(f, index=False)
            n_updated['topology_features'] += 1

        # 2. routing_decisions.csv
        f = ep_dir / 'routing_decisions.csv'
        if f.exists():
            df = pd.read_csv(f)
            df['regime_id'] = df['timestamp'].round(6).map(lookup).fillna(-1).astype(int)
            df.to_csv(f, index=False)
            n_updated['routing_decisions'] += 1

        # 3. rl_transitions.csv
        f = ep_dir / 'rl_transitions.csv'
        if f.exists():
            df = pd.read_csv(f)
            df['regime_id'] = df['timestamp'].round(6).map(lookup).fillna(-1).astype(int)
            df.to_csv(f, index=False)
            n_updated['rl_transitions'] += 1

    print(f"\n  Updated tables:")
    for tbl, n in n_updated.items():
        print(f"    {tbl}: {n} episodes")


# ─── Plots ─────────────────────────────────────────────────────────────────────

def plot_k_sweep(sweep_df, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(sweep_df['k'], sweep_df['inertia'], 'o-', color='#457b9d')
    axes[0].set_xlabel('k (number of regimes)')
    axes[0].set_ylabel('Inertia (within-cluster SSE)')
    axes[0].set_title('Elbow Method', fontweight='bold')
    axes[0].grid(alpha=0.3)

    axes[1].plot(sweep_df['k'], sweep_df['silhouette'], 'o-', color='#e63946')
    axes[1].set_xlabel('k (number of regimes)')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].set_title('Silhouette Score (higher is better)', fontweight='bold')
    axes[1].grid(alpha=0.3)

    best_k = sweep_df.loc[sweep_df['silhouette'].idxmax(), 'k']
    axes[1].axvline(best_k, color='gray', ls='--', alpha=0.5,
                    label=f'Best k = {int(best_k)}')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_regime_distribution_per_scenario(topo_df, out_path, regime_names):
    """Stacked bar chart: how much time each scenario spends in each regime."""
    K = len(regime_names)
    scenarios = sorted(topo_df['_scenario'].unique())
    matrix = np.zeros((len(scenarios), K))
    for i, sc in enumerate(scenarios):
        sub = topo_df[topo_df['_scenario'] == sc]
        for r in range(K):
            matrix[i, r] = (sub['regime_id_new'] == r).mean()

    fig, ax = plt.subplots(figsize=(11, 5))
    cmap = plt.get_cmap('tab10')
    bottom = np.zeros(len(scenarios))
    for r in range(K):
        ax.bar(scenarios, matrix[:, r], bottom=bottom,
               label=f'{r}: {regime_names[r]}', color=cmap(r), alpha=0.85)
        bottom += matrix[:, r]
    ax.set_ylabel('Fraction of time')
    ax.set_title('Regime Distribution per Scenario\n'
                 '(each bar shows fraction of time spent in each regime)',
                 fontweight='bold')
    ax.legend(bbox_to_anchor=(1.01, 1.0), loc='upper left', fontsize=9)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_centroids(centroid_df, out_path):
    """Heatmap of centroids (z-scored across features for visibility)."""
    feature_cols = CLUSTER_FEATURES
    M = centroid_df[feature_cols].values
    # Per-feature z-score for visibility
    Mz = (M - M.mean(axis=0)) / (M.std(axis=0) + 1e-8)

    fig, ax = plt.subplots(figsize=(11, max(4, 0.6 * len(M))))
    im = ax.imshow(Mz, aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2)
    ax.set_xticks(range(len(feature_cols)))
    ax.set_xticklabels(feature_cols, rotation=35, ha='right', fontsize=9)
    ax.set_yticks(range(len(M)))
    ax.set_yticklabels([f"{i}: {centroid_df['regime_name'].iloc[i]}"
                        for i in range(len(M))], fontsize=10)
    plt.colorbar(im, ax=ax, label='Z-score (red = high, blue = low)')
    ax.set_title('Regime Centroids (each row = one regime)', fontweight='bold')

    # annotate raw values
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i,j]:.1f}", ha='center', va='center',
                    color='black', fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_pca_scatter(X_scaled, labels, regime_names, out_path):
    """2D PCA projection colored by regime."""
    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X_scaled)
    # Subsample for plotting clarity
    if len(X2) > 10000:
        idx = np.random.default_rng(0).choice(len(X2), 10000, replace=False)
        X2 = X2[idx]; labels = labels[idx]

    fig, ax = plt.subplots(figsize=(10, 7))
    cmap = plt.get_cmap('tab10')
    for r in sorted(set(labels)):
        m = labels == r
        ax.scatter(X2[m, 0], X2[m, 1], s=8, alpha=0.5, color=cmap(r),
                   label=f'{r}: {regime_names[r]}')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)')
    ax.set_title('Topology Regimes — 2D PCA Projection', fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', default='data/pilot')
    ap.add_argument('--out_dir',   default='models')
    ap.add_argument('--k', type=int, default=None,
                    help='Number of regimes. If None, picks best via silhouette.')
    ap.add_argument('--k_min', type=int, default=3)
    ap.add_argument('--k_max', type=int, default=8)
    ap.add_argument('--no_plot', action='store_true')
    ap.add_argument('--no_writeback', action='store_true',
                    help="Don't write regime_id back into CSVs (dry run).")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    # ── Load ───────────────────────────────────────────────────────────────────
    print(f"\n{'═'*60}\n  PHASE 4 — REGIME CLUSTERING\n{'═'*60}\n")
    topo_df = load_all_topology_data(args.data_root)

    # ── Feature matrix ────────────────────────────────────────────────────────
    missing = [c for c in CLUSTER_FEATURES if c not in topo_df.columns]
    if missing:
        print(f"  [FATAL] Missing columns: {missing}")
        return

    X_raw = topo_df[CLUSTER_FEATURES].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    print(f"  Feature matrix: {X_scaled.shape}\n")

    # ── k sweep ───────────────────────────────────────────────────────────────
    if args.k is None:
        print(f"{'─'*60}")
        print("  Sweeping k to find the best number of regimes")
        print(f"{'─'*60}")
        sweep = evaluate_k(X_scaled, (args.k_min, args.k_max))
        best_k = int(sweep.loc[sweep['silhouette'].idxmax(), 'k'])
        print(f"\n  Best k by silhouette: {best_k}")
        if not args.no_plot:
            plot_k_sweep(sweep,
                         os.path.join(args.out_dir, 'regime_k_sweep.png'))
    else:
        best_k = args.k
        sweep = None
        print(f"  Using k = {best_k} (user-specified)")

    # ── Fit final model ───────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Fitting KMeans with k = {best_k}")
    print(f"{'─'*60}")
    km, labels = fit_kmeans(X_scaled, best_k)

    # Centroids in ORIGINAL feature scale (interpretable)
    centroids_orig = scaler.inverse_transform(km.cluster_centers_)
    centroid_df = interpret_centroids(centroids_orig, CLUSTER_FEATURES)

    print(f"\n  Regime centroids (original scale):")
    print(f"  {centroid_df.round(2).to_string()}")

    # ── Attach labels & analyze distribution ───────────────────────────────────
    topo_df['regime_id_new'] = labels

    print(f"\n  Overall regime distribution:")
    overall = topo_df['regime_id_new'].value_counts(normalize=True).sort_index()
    for r, frac in overall.items():
        name = centroid_df['regime_name'].iloc[r]
        print(f"    {r}: {name:<20s} {frac:>6.1%}  ({int(frac*len(topo_df)):,} snapshots)")

    print(f"\n  Regime distribution per scenario:")
    by_sc = (topo_df.groupby('_scenario')['regime_id_new']
                    .value_counts(normalize=True)
                    .unstack(fill_value=0)
                    .round(3))
    print(f"  {by_sc.to_string()}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    if not args.no_plot:
        plot_regime_distribution_per_scenario(
            topo_df, os.path.join(args.out_dir, 'regime_distribution.png'),
            centroid_df['regime_name'].tolist())
        plot_centroids(centroid_df,
                       os.path.join(args.out_dir, 'regime_centroids.png'))
        plot_pca_scatter(X_scaled, labels, centroid_df['regime_name'].tolist(),
                         os.path.join(args.out_dir, 'regime_pca.png'))

    # ── Save model bundle ─────────────────────────────────────────────────────
    bundle = {
        'kmeans':           km,
        'scaler':           scaler,
        'features':         CLUSTER_FEATURES,
        'k':                best_k,
        'centroids_orig':   centroids_orig,
        'regime_names':     centroid_df['regime_name'].tolist(),
    }
    model_path = os.path.join(args.out_dir, 'regime_clustering.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(bundle, f)
    print(f"\n  Model bundle saved → {model_path}")

    centroid_df.to_csv(os.path.join(args.out_dir, 'regime_centroids.csv'))
    if sweep is not None:
        sweep.to_csv(os.path.join(args.out_dir, 'regime_k_sweep.csv'), index=False)

    # ── Write back to CSVs ────────────────────────────────────────────────────
    if not args.no_writeback:
        write_regime_back(args.data_root, topo_df, label_col='regime_id_new')
    else:
        print("\n  --no_writeback set: skipping CSV updates (dry run)")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"\n{'═'*60}")
    print(f"  DONE — regime_clustering.pkl ready")
    print(f"  RL state vector now includes regime_id (Phase 5)")
    print(f"{'═'*60}\n")


if __name__ == '__main__':
    main()
