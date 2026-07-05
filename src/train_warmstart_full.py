"""
train_warmstart_full.py
Phase 3 (full-scale) — Supervised Warm-Start: XGBoost next-hop classifier.

Same logic as the original train_warmstart.py, but updated for the
full-scale RWP dataset: 15 scenarios (rwp_sc01-rwp_sc15), 9/3/3 split.

Split strategy: BY SCENARIO (not by row) — avoids data leakage.
    Train : rwp_sc01, rwp_sc02, rwp_sc03, rwp_sc04, rwp_sc07,
            rwp_sc08, rwp_sc10, rwp_sc12, rwp_sc15    (9 scenarios, 270 episodes)
    Val   : rwp_sc05, rwp_sc09, rwp_sc13                (3 scenarios, 60 episodes)
    Test  : rwp_sc06, rwp_sc11, rwp_sc14                (3 scenarios, 60 episodes)

Sample weight: consensus_confidence — labels agreed on by more teachers
               get more training weight.

Usage:
    python src\\train_warmstart_full.py
    python src\\train_warmstart_full.py --data_root data/rwp_full
    python src\\train_warmstart_full.py --no_plot
"""

import os
import sys
import argparse
import time
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score

warnings.filterwarnings('ignore')

# ─── Scenario split (full-scale RWP: 15 scenarios) ────────────────────────────
TRAIN_SCENARIOS = ['rwp_sc01', 'rwp_sc02', 'rwp_sc03', 'rwp_sc04', 'rwp_sc07',
                   'rwp_sc08', 'rwp_sc10', 'rwp_sc12', 'rwp_sc15']
VAL_SCENARIOS   = ['rwp_sc05', 'rwp_sc09', 'rwp_sc13']
TEST_SCENARIOS  = ['rwp_sc06', 'rwp_sc11', 'rwp_sc14']

# ─── Feature columns (unchanged from pilot) ────────────────────────────────────
FEATURE_COLS = [
    'candidate_distance',
    'candidate_relative_velocity',
    'candidate_link_quality',
    'candidate_rssi',
    'candidate_packet_error_rate',
    'candidate_estimated_lifetime',
    'candidate_energy',
    'candidate_queue_occupancy',
    'candidate_dist_to_dest',
    'candidate_progress',
    'current_energy',
    'current_queue_occupancy',
    'current_num_neighbors',
    'current_speed',
    'distance_to_destination',
    'bearing_to_destination',
    'destination_reachable',
    'avg_degree',
    'network_density',
    'num_components',
    'avg_link_quality',
    'topology_change_rate',
]

LABEL_COL  = 'is_best_label'
WEIGHT_COL = 'consensus_confidence'


# ─── Data loading ───────────────────────────────────────────────────────────────

def load_scenario(sc_dir, scenario_name):
    """Load and concatenate all routing_decisions.csv files for one scenario."""
    rows = []
    sc_path = Path(sc_dir)
    for ep_dir in sorted(sc_path.iterdir()):
        if not ep_dir.is_dir():
            continue
        f = ep_dir / 'routing_decisions.csv'
        if f.exists():
            df = pd.read_csv(f)
            df['_scenario'] = scenario_name
            df['_episode']  = ep_dir.name
            rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def load_split(data_root, scenario_list):
    """Load and concatenate data for a list of scenarios."""
    parts = []
    for sc in scenario_list:
        sc_dir = Path(data_root) / sc
        if not sc_dir.exists():
            print(f"  [WARN] Scenario not found, skipping: {sc}")
            continue
        df = load_scenario(sc_dir, sc)
        if len(df) == 0:
            print(f"  [WARN] No routing_decisions found in {sc}")
            continue
        parts.append(df)
        print(f"  Loaded {sc}: {len(df):>9,} rows")
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def prepare_XYW(df):
    """Extract feature matrix X, labels y, and sample weights w."""
    df = df.dropna(subset=FEATURE_COLS + [LABEL_COL])
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[LABEL_COL].values.astype(np.int32)
    w = df[WEIGHT_COL].fillna(0.2).clip(0.01, 1.0).values.astype(np.float32)
    return X, y, w, df


# ─── Per-decision top-k accuracy ────────────────────────────────────────────────

def top_k_accuracy(df, scores, k=1):
    df = df.copy()
    df['_score'] = scores
    correct = 0
    total = 0
    for _, group in df.groupby(['_scenario', '_episode', 'decision_id']):
        if group[LABEL_COL].sum() == 0:
            continue
        ranked = group.sort_values('_score', ascending=False)
        top_k_ids = set(ranked.head(k).index)
        true_best = set(group[group[LABEL_COL] == 1].index)
        if top_k_ids & true_best:
            correct += 1
        total += 1
    return correct / max(total, 1), total


# ─── Evaluation ──────────────────────────────────────────────────────────────────

def evaluate(model, X, y, w, df, split_name):
    scores = model.predict_proba(X)[:, 1]
    preds  = (scores >= 0.5).astype(int)

    auc    = roc_auc_score(y, scores, sample_weight=w)
    acc    = accuracy_score(y, preds, sample_weight=w)
    top1, n_dec = top_k_accuracy(df, scores, k=1)
    top2, _     = top_k_accuracy(df, scores, k=2)

    print(f"\n  {'─'*50}")
    print(f"  {split_name.upper()} RESULTS  ({len(y):,} rows, {n_dec:,} decisions)")
    print(f"  {'─'*50}")
    print(f"  AUC-ROC           : {auc:.4f}")
    print(f"  Accuracy (thresh) : {acc:.4f}")
    print(f"  Top-1 accuracy    : {top1:.4f}  ({int(top1*n_dec)}/{n_dec} decisions)")
    print(f"  Top-2 accuracy    : {top2:.4f}  ({int(top2*n_dec)}/{n_dec} decisions)")

    if '_scenario' in df.columns:
        print(f"\n  Per-scenario breakdown:")
        print(f"  {'Scenario':<12} {'Rows':>9} {'AUC':>6} {'Top-1':>7} {'Top-2':>7}")
        print(f"  {'─'*44}")
        for sc in sorted(df['_scenario'].unique()):
            mask = df['_scenario'] == sc
            if mask.sum() < 10:
                continue
            sc_scores = scores[mask]
            sc_y      = y[mask]
            sc_w      = w[mask]
            sc_df     = df[mask].copy()
            sc_df['_score'] = sc_scores
            try:
                sc_auc = roc_auc_score(sc_y, sc_scores, sample_weight=sc_w)
            except ValueError:
                sc_auc = float('nan')
            sc_t1, sc_nd = top_k_accuracy(sc_df, sc_scores, k=1)
            sc_t2, _     = top_k_accuracy(sc_df, sc_scores, k=2)
            print(f"  {sc:<12} {mask.sum():>9,} {sc_auc:>6.3f} {sc_t1:>7.3f} {sc_t2:>7.3f}")

    return {'auc': auc, 'acc': acc, 'top1': top1, 'top2': top2, 'n_decisions': n_dec}


# ─── Plots ────────────────────────────────────────────────────────────────────────

def plot_feature_importance(model, feature_names, out_path):
    importance = model.feature_importances_
    idx = np.argsort(importance)[::-1]

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ['#e63946' if importance[i] > np.percentile(importance, 75)
              else '#457b9d' for i in idx]
    ax.barh(range(len(idx)), importance[idx][::-1], color=colors[::-1])
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feature_names[i] for i in idx][::-1], fontsize=9)
    ax.set_xlabel('Feature Importance (gain)')
    ax.set_title('XGBoost Feature Importance — Full-Scale RWP', fontweight='bold')
    ax.axvline(np.mean(importance), color='gray', linestyle='--', alpha=0.5,
               label='Mean importance')
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Feature importance plot saved: {out_path}")


def plot_learning_curve(eval_results, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for metric, ax in zip(['logloss', 'auc'], axes):
        if metric not in eval_results.get('train', {}):
            continue
        ax.plot(eval_results['train'][metric], label='Train', color='#457b9d')
        if 'val' in eval_results:
            ax.plot(eval_results['val'][metric], label='Val', color='#e63946')
        ax.set_xlabel('Boosting Round')
        ax.set_ylabel(metric)
        ax.set_title(f'XGBoost — {metric} (Full-Scale RWP)', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Learning curve saved: {out_path}")


def plot_scenario_pdr_vs_accuracy(df_all_test, scores, out_path):
    """Bar chart: per-scenario Top-1 accuracy, to visualize hard-scenario performance."""
    df = df_all_test.copy()
    df['_score'] = scores
    rows = []
    for sc in sorted(df['_scenario'].unique()):
        sub = df[df['_scenario'] == sc]
        t1, n = top_k_accuracy(sub, sub['_score'].values, k=1)
        rows.append({'scenario': sc, 'top1': t1, 'n_decisions': n})
    plot_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(plot_df['scenario'], plot_df['top1'], color='#e63946', alpha=0.85)
    ax.set_ylabel('Top-1 Accuracy')
    ax.set_title('Test Scenario Top-1 Accuracy — Full-Scale RWP', fontweight='bold')
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Test scenario breakdown plot saved: {out_path}")


# ─── Main ────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', default='data/rwp_full')
    ap.add_argument('--out_dir',   default='models')
    ap.add_argument('--n_estimators', type=int, default=300)
    ap.add_argument('--max_depth',    type=int, default=6)
    ap.add_argument('--lr',           type=float, default=0.1)
    ap.add_argument('--no_plot', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    print("\n" + "═"*60)
    print("  LOADING DATA — FULL-SCALE RWP")
    print("═"*60)

    print(f"\n  Train scenarios ({len(TRAIN_SCENARIOS)}): {TRAIN_SCENARIOS}")
    df_train = load_split(args.data_root, TRAIN_SCENARIOS)

    print(f"\n  Validation scenarios ({len(VAL_SCENARIOS)}): {VAL_SCENARIOS}")
    df_val = load_split(args.data_root, VAL_SCENARIOS)

    print(f"\n  Test scenarios ({len(TEST_SCENARIOS)}, hardest — unseen): {TEST_SCENARIOS}")
    df_test = load_split(args.data_root, TEST_SCENARIOS)

    X_train, y_train, w_train, df_train = prepare_XYW(df_train)
    X_val,   y_val,   w_val,   df_val   = prepare_XYW(df_val)
    X_test,  y_test,  w_test,  df_test  = prepare_XYW(df_test)

    print(f"\n  Dataset sizes:")
    print(f"    Train : {len(X_train):>10,} rows  ({y_train.sum():,} positive, "
          f"{(y_train==0).sum():,} negative)")
    print(f"    Val   : {len(X_val):>10,} rows  ({y_val.sum():,} positive)")
    print(f"    Test  : {len(X_test):>10,} rows  ({y_test.sum():,} positive)")

    if len(X_train) == 0:
        print("\n  [FATAL] No training rows found. Check --data_root path.")
        sys.exit(1)

    # ── Class imbalance ─────────────────────────────────────────────────────────
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    spw   = round(n_neg / max(n_pos, 1), 2)
    print(f"\n  Class imbalance  →  scale_pos_weight = {spw}")

    # ── Train ────────────────────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  TRAINING XGBoost")
    print("═"*60)
    print(f"  n_estimators={args.n_estimators}, max_depth={args.max_depth}, "
          f"lr={args.lr}, scale_pos_weight={spw}")

    model = xgb.XGBClassifier(
        n_estimators       = args.n_estimators,
        max_depth          = args.max_depth,
        learning_rate      = args.lr,
        subsample          = 0.8,
        colsample_bytree   = 0.8,
        scale_pos_weight   = spw,
        objective          = 'binary:logistic',
        eval_metric        = ['logloss', 'auc'],
        random_state       = 42,
        n_jobs             = -1,
        verbosity          = 0,
        early_stopping_rounds = 30,
    )

    model.fit(
        X_train, y_train,
        sample_weight      = w_train,
        eval_set           = [(X_train, y_train), (X_val, y_val)],
        sample_weight_eval_set = [w_train, w_val],
        verbose            = False,
    )

    raw_results = model.evals_result()
    eval_results = {
        'train': raw_results.get('validation_0', {}),
        'val':   raw_results.get('validation_1', {}),
    }
    best_round = model.best_iteration
    print(f"\n  Training complete. Best round: {best_round}")

    # ── Evaluate ─────────────────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  EVALUATION")
    print("═"*60)

    train_metrics = evaluate(model, X_train, y_train, w_train, df_train, 'train')
    val_metrics   = evaluate(model, X_val,   y_val,   w_val,   df_val,   'val')
    test_metrics  = evaluate(model, X_test,  y_test,  w_test,  df_test,  'test')

    fi   = model.feature_importances_
    top3 = sorted(zip(FEATURE_COLS, fi), key=lambda x: x[1], reverse=True)[:3]
    print(f"\n  Top 3 features: {[(n, round(v,4)) for n,v in top3]}")

    # ── Save model ───────────────────────────────────────────────────────────────
    model_path = os.path.join(args.out_dir, 'xgboost_warmstart_full.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump({'model': model, 'feature_cols': FEATURE_COLS,
                     'train_scenarios': TRAIN_SCENARIOS,
                     'val_scenarios': VAL_SCENARIOS,
                     'test_scenarios': TEST_SCENARIOS}, f)
    print(f"\n  Model saved → {model_path}")

    # ── Write report ─────────────────────────────────────────────────────────────
    report_path = os.path.join(args.out_dir, 'warmstart_report_full.txt')
    with open(report_path, 'w') as f:
        f.write("XGBoost Warm-Start Model — Full-Scale RWP Training Report\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Train scenarios : {TRAIN_SCENARIOS}\n")
        f.write(f"Val scenarios   : {VAL_SCENARIOS}\n")
        f.write(f"Test scenarios  : {TEST_SCENARIOS}\n\n")
        f.write(f"Training rows   : {len(X_train):,}\n")
        f.write(f"Val rows        : {len(X_val):,}\n")
        f.write(f"Test rows       : {len(X_test):,}\n\n")
        f.write(f"scale_pos_weight: {spw}\n")
        f.write(f"Best round      : {best_round}\n\n")
        f.write("METRICS\n" + "-"*30 + "\n")
        for split, m in [('Train', train_metrics), ('Val', val_metrics), ('Test', test_metrics)]:
            f.write(f"{split:6} | AUC={m['auc']:.4f} | "
                    f"Top-1={m['top1']:.4f} | Top-2={m['top2']:.4f} | "
                    f"Decisions={m['n_decisions']}\n")
        f.write("\nFEATURE IMPORTANCE (top 10)\n" + "-"*30 + "\n")
        ranked = sorted(zip(FEATURE_COLS, fi), key=lambda x: x[1], reverse=True)
        for name, val in ranked[:10]:
            f.write(f"  {name:<35} {val:.5f}\n")
    print(f"  Report saved   → {report_path}")

    # ── Plots ────────────────────────────────────────────────────────────────────
    if not args.no_plot:
        plot_feature_importance(
            model, FEATURE_COLS,
            os.path.join(args.out_dir, 'feature_importance_full.png'))
        plot_learning_curve(
            eval_results,
            os.path.join(args.out_dir, 'learning_curve_full.png'))
        test_scores = model.predict_proba(X_test)[:, 1]
        plot_scenario_pdr_vs_accuracy(
            df_test, test_scores,
            os.path.join(args.out_dir, 'test_scenario_breakdown.png'))

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")
    print("\n" + "═"*60)
    print("  DONE — Next: python src\\train_mlp_full.py")
    print("═"*60 + "\n")


if __name__ == '__main__':
    main()
