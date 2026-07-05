"""
cluster_regimes_full.py
Phase 4 (full-scale) — Topology Regime Clustering for the 15-scenario RWP dataset.

Same clustering approach as the pilot (KMeans on 9 topology features,
k selected by silhouette sweep), adapted for:
  - Nested layout: data/rwp_full/rwp_scXX/episode_NNN/topology_features.csv
  - 15 scenarios instead of 10 — wider k sweep range (3-10) since more
    genuine topological diversity is expected
  - Runs across ALL scenarios (train+val+test) since regime_id needs to be
    available network-wide for RL state representation, not just train

Usage (from FANET_sim directory):
    python src\\cluster_regimes_full.py
    python src\\cluster_regimes_full.py --k 6
    python src\\cluster_regimes_full.py --no_writeback   (dry run)
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

ALL_SCENARIOS = [f'rwp_sc{i:02d}' for i in range(1, 16)]


# ── Data loading ──────────────────────────────────────────────────────────────

def discover_scenarios(data_root):
    root = Path(data_root)
    found = sorted([d for d in root.iterdir() if d.is_dir() and d.name in ALL_SCENARIOS])
    return found


def discover_episodes(scenario_dir):
    return sorted([d for d in Path(scenario_dir).iterdir() if d.is_dir()])


def load_topology(scenarios):
    parts = []
    print("  Loading topology_features.csv from all scenarios/episodes...")
    for sc_dir in scenarios:
        sc_name = sc_dir.name
        episodes = discover_episodes(sc_dir)
        n_loaded = 0
        for ep in episodes:
            f = ep / 'topology_features.csv'
            if not f.exists():
                continue
            df = pd.read_csv(f)
            df['_scenario'] = sc_name
            df['_episode']  = ep.name
            df['_csv_path'] = str(f)
            parts.append(df)
            n_loaded += 1
        print(f"    {sc_name}: {n_loaded} episodes")
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    print(f"\n  Loaded {len(out):,} topology snapshots from "
          f"{out['_scenario'].nunique()} scenarios, "
          f"{out.groupby('_scenario')['_episode'].nunique().sum()} episodes")
    return out


# ── Clustering ────────────────────────────────────────────────────────────────

def fit_kmeans(X, k, seed=42):
    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    labels = km.fit_predict(X)
    return km, labels


def sweep_k(X, k_min, k_max, seed=42, sample_size=20000):
    results = []
    sample = np.random.default_rng(seed).choice(
        len(X), size=min(sample_size, len(X)), replace=False)
    print(f"  {'k':>3}  {'inertia':>13}  {'silhouette':>11}")
    print(f"  {'─'*32}")
    for k in range(k_min, k_max + 1):
        km, labels = fit_kmeans(X, k, seed)
        sil = silhouette_score(X[sample], labels[sample])
        results.append({'k': k, 'inertia': km.inertia_, 'silhouette': sil})
        print(f"  {k:>3}  {km.inertia_:>13.1f}  {sil:>11.4f}")
    return pd.DataFrame(results)


def interpret_centroids(centroids_orig, feature_names):
    df = pd.DataFrame(centroids_orig, columns=feature_names)
    df.index.name = 'regime_id'
    z = (df - df.mean()) / (df.std() + 1e-8)
    names = []
    for i in range(len(df)):
        deg_z   = z.loc[i, 'avg_degree']
        comps_z = z.loc[i, 'num_connected_components']
        vel_z   = z.loc[i, 'avg_relative_velocity']
        lc_z    = z.loc[i, 'largest_component_ratio']
        den_z   = z.loc[i, 'network_density']
        parts = []
        if den_z > 0.8 or deg_z > 0.8:      parts.append("Dense")
        elif den_z < -0.5 or deg_z < -0.5:  parts.append("Sparse")
        else:                                parts.append("Medium")
        if comps_z > 0.8 or lc_z < -0.5:    parts.append("Partitioned")
        elif vel_z > 0.5:                    parts.append("Fast")
        elif vel_z < -0.5:                   parts.append("Slow")
        else:                                parts.append("Stable")
        names.append("-".join(parts))
    seen = {}
    for i, n in enumerate(names):
        if n in seen:
            seen[n] += 1; names[i] = f"{n}-{seen[n]}"
        else:
            seen[n] = 1
    df['regime_name'] = names
    return df


# ── Write-back ────────────────────────────────────────────────────────────────

def write_regime_back(topo_df, label_col='regime_id_new'):
    print("\n  Writing regime_id back into CSVs...")
    updated = {'topology_features': 0, 'routing_decisions': 0, 'rl_transitions': 0}
    grouped = topo_df.groupby(['_scenario', '_episode'])
    for (sc_name, ep_name), grp in grouped:
        ep_dir = Path(grp['_csv_path'].iloc[0]).parent
        lookup = dict(zip(grp['timestamp'].round(6), grp[label_col]))

        for tbl in ['topology_features', 'routing_decisions', 'rl_transitions']:
            f = ep_dir / f'{tbl}.csv'
            if f.exists():
                df = pd.read_csv(f)
                df['regime_id'] = df['timestamp'].round(6).map(lookup).fillna(-1).astype(int)
                df.to_csv(f, index=False)
                updated[tbl] += 1

    for tbl, n in updated.items():
        print(f"    {tbl}: {n} episodes updated")


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_k_sweep(sweep_df, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(sweep_df['k'], sweep_df['inertia'], 'o-', color='#457b9d')
    axes[0].set_xlabel('k'); axes[0].set_ylabel('Inertia')
    axes[0].set_title('Elbow Method — Full-Scale RWP', fontweight='bold')
    axes[0].grid(alpha=0.3)
    axes[1].plot(sweep_df['k'], sweep_df['silhouette'], 'o-', color='#e63946')
    best_k = int(sweep_df.loc[sweep_df['silhouette'].idxmax(), 'k'])
    axes[1].axvline(best_k, color='gray', ls='--', alpha=0.6, label=f'Best k={best_k}')
    axes[1].set_xlabel('k'); axes[1].set_ylabel('Silhouette')
    axes[1].set_title('Silhouette Score — Full-Scale RWP', fontweight='bold')
    axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight'); plt.close()


def plot_centroids(centroid_df, out_path):
    feats = CLUSTER_FEATURES
    M = centroid_df[feats].values
    Mz = (M - M.mean(axis=0)) / (M.std(axis=0) + 1e-8)
    fig, ax = plt.subplots(figsize=(12, max(3, 0.65 * len(M))))
    im = ax.imshow(Mz, aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2)
    ax.set_xticks(range(len(feats)))
    ax.set_xticklabels(feats, rotation=35, ha='right', fontsize=9)
    ax.set_yticks(range(len(M)))
    ax.set_yticklabels([f"{i}: {centroid_df['regime_name'].iloc[i]}" for i in range(len(M))],
                       fontsize=10)
    plt.colorbar(im, ax=ax, label='Z-score')
    ax.set_title('Regime Centroids — Full-Scale RWP', fontweight='bold')
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i,j]:.1f}", ha='center', va='center',
                    color='black', fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight'); plt.close()


def plot_distribution(topo_df, regime_names, out_path):
    K = len(regime_names)
    scenarios = sorted(topo_df['_scenario'].unique())
    matrix = np.zeros((len(scenarios), K))
    for i, sc in enumerate(scenarios):
        sub = topo_df[topo_df['_scenario'] == sc]
        for r in range(K):
            matrix[i, r] = (sub['regime_id_new'] == r).mean()
    fig, ax = plt.subplots(figsize=(13, 5))
    cmap = plt.get_cmap('tab10')
    bottom = np.zeros(len(scenarios))
    for r in range(K):
        ax.bar(scenarios, matrix[:, r], bottom=bottom,
               label=f'{r}: {regime_names[r]}', color=cmap(r), alpha=0.85)
        bottom += matrix[:, r]
    ax.set_ylabel('Fraction of time')
    ax.set_title('Regime Distribution per Scenario — Full-Scale RWP', fontweight='bold')
    ax.legend(bbox_to_anchor=(1.01, 1.0), loc='upper left', fontsize=9)
    plt.xticks(rotation=30, ha='right', fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight'); plt.close()


def plot_pca(X_scaled, labels, regime_names, out_path, sample_size=15000):
    idx = np.random.default_rng(42).choice(
        len(X_scaled), size=min(sample_size, len(X_scaled)), replace=False)
    pca = PCA(n_components=2, random_state=42)
    X2  = pca.fit_transform(X_scaled[idx])
    labels_s = labels[idx]
    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.get_cmap('tab10')
    for r in sorted(set(labels_s)):
        m = labels_s == r
        ax.scatter(X2[m, 0], X2[m, 1], s=8, alpha=0.4, color=cmap(r),
                   label=f'{r}: {regime_names[r]}')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)')
    ax.set_title('Topology Regimes — PCA — Full-Scale RWP (15k sample)', fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight'); plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root',    default='data/rwp_full')
    ap.add_argument('--out_dir',      default='models')
    ap.add_argument('--k',            type=int, default=None,
                    help='Force k. If None, sweeps k_min..k_max and picks best.')
    ap.add_argument('--k_min',        type=int, default=3)
    ap.add_argument('--k_max',        type=int, default=10)
    ap.add_argument('--no_plot',      action='store_true')
    ap.add_argument('--no_writeback', action='store_true')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f"\n{'═'*62}")
    print("  PHASE 4 — REGIME CLUSTERING (FULL-SCALE RWP)")
    print(f"{'═'*62}\n")

    # ── Load ─────────────────────────────────────────────────────────────────
    scenarios = discover_scenarios(args.data_root)
    if not scenarios:
        print(f"  [FATAL] No rwp_scXX folders found in {args.data_root}")
        sys.exit(1)
    print(f"  Found {len(scenarios)} scenarios: {[s.name for s in scenarios]}\n")

    topo_df = load_topology(scenarios)
    if len(topo_df) == 0:
        print("  [FATAL] No topology data loaded."); sys.exit(1)

    missing = [c for c in CLUSTER_FEATURES if c not in topo_df.columns]
    if missing:
        print(f"  [FATAL] Missing columns: {missing}"); sys.exit(1)

    X_raw    = topo_df[CLUSTER_FEATURES].fillna(0).values
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    print(f"\n  Feature matrix: {X_scaled.shape}")

    # ── k sweep ──────────────────────────────────────────────────────────────
    if args.k is None:
        print(f"\n  Sweeping k = {args.k_min}..{args.k_max} "
              f"(silhouette computed on 20k-row sample per k)")
        sweep = sweep_k(X_scaled, args.k_min, args.k_max)
        best_k = int(sweep.loc[sweep['silhouette'].idxmax(), 'k'])
        print(f"\n  Best k by silhouette: {best_k}")
        if not args.no_plot:
            plot_k_sweep(sweep, out_dir / 'regime_k_sweep_full.png')
        sweep.to_csv(out_dir / 'regime_k_sweep_full.csv', index=False)
    else:
        best_k = args.k
        print(f"  Using k = {best_k} (user-specified)")

    # ── Fit final model ───────────────────────────────────────────────────────
    print(f"\n  Fitting KMeans k={best_k} on {len(X_scaled):,} rows...")
    km, labels = fit_kmeans(X_scaled, best_k)
    centroids_orig = scaler.inverse_transform(km.cluster_centers_)
    centroid_df    = interpret_centroids(centroids_orig, CLUSTER_FEATURES)

    print(f"\n  Regime centroids (original scale):")
    print(f"  {centroid_df.round(2).to_string()}\n")

    topo_df['regime_id_new'] = labels

    print("  Overall regime distribution:")
    overall = topo_df['regime_id_new'].value_counts(normalize=True).sort_index()
    for r, frac in overall.items():
        name = centroid_df['regime_name'].iloc[r]
        print(f"    {r}: {name:<24s} {frac:>6.1%}  ({int(frac*len(topo_df)):,} snapshots)")

    print("\n  Per-scenario dominant regime:")
    by_sc = (topo_df.groupby('_scenario')['regime_id_new']
                    .value_counts(normalize=True)
                    .unstack(fill_value=0).round(3))
    print(f"  {by_sc.to_string()}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    if not args.no_plot:
        plot_centroids(centroid_df, out_dir / 'regime_centroids_full.png')
        plot_distribution(topo_df, centroid_df['regime_name'].tolist(),
                          out_dir / 'regime_distribution_full.png')
        plot_pca(X_scaled, labels, centroid_df['regime_name'].tolist(),
                 out_dir / 'regime_pca_full.png')

    # ── Save model ────────────────────────────────────────────────────────────
    bundle = {
        'kmeans':         km,
        'scaler':         scaler,
        'features':       CLUSTER_FEATURES,
        'k':              best_k,
        'centroids_orig': centroids_orig,
        'regime_names':   centroid_df['regime_name'].tolist(),
    }
    pkl_path = out_dir / 'regime_clustering_full.pkl'
    with open(pkl_path, 'wb') as f:
        pickle.dump(bundle, f)
    centroid_df.to_csv(out_dir / 'regime_centroids_full.csv')
    print(f"\n  Model saved → {pkl_path}")

    # ── Write back ────────────────────────────────────────────────────────────
    if not args.no_writeback:
        write_regime_back(topo_df)
    else:
        print("\n  --no_writeback set: skipping CSV updates (dry run)")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"\n{'═'*62}")
    print(f"  DONE — outputs in {out_dir}")
    print(f"  Next: Phase 5 — DQN RL training with regime-conditioned state")
    print(f"{'═'*62}\n")


if __name__ == '__main__':
    main()
