"""
train_warmstart.py
Phase 3 — Supervised Warm-Start: XGBoost next-hop classifier.

Loads routing_decisions.csv from all pilot episodes, trains an XGBoost
binary classifier that predicts which candidate neighbor is the best
next hop (is_best_label = 1 vs 0).

Split strategy: BY SCENARIO (not by row) — avoids data leakage.
    Train : pilot_01, pilot_02, pilot_04, pilot_06, pilot_07, pilot_09
    Val   : pilot_08, pilot_10
    Test  : pilot_03, pilot_05   (hardest/most sparse — unseen difficulty)

Sample weight: consensus_confidence — labels agreed on by more teachers
               get more training weight. Low-PDR episodes are NOT
               down-weighted (that would bias against hard scenarios).

Class imbalance: ~26% positive, 74% negative → scale_pos_weight = 2.84.

Usage:
    python src\train_warmstart.py                        # train + evaluate
    python src\train_warmstart.py --data_root data/pilot
    python src\train_warmstart.py --no_plot              # skip figures
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
from sklearn.metrics import (roc_auc_score, accuracy_score,
                              classification_report, confusion_matrix)

warnings.filterwarnings('ignore')

# ─── Scenario split ───────────────────────────────────────────────────────────
TRAIN_SCENARIOS = ['pilot_01', 'pilot_02', 'pilot_04',
                   'pilot_06', 'pilot_07', 'pilot_09']
VAL_SCENARIOS   = ['pilot_08', 'pilot_10']
TEST_SCENARIOS  = ['pilot_03', 'pilot_05']

# ─── Feature columns ──────────────────────────────────────────────────────────
# Everything the model can observe at the moment of a forwarding decision.
# Excludes: identifiers, labels, teacher choices, raw positions (x/y/z).
FEATURE_COLS = [
    # Candidate link features (the most important group)
    'candidate_distance',
    'candidate_relative_velocity',
    'candidate_link_quality',
    'candidate_rssi',
    'candidate_packet_error_rate',
    'candidate_estimated_lifetime',
    'candidate_energy',
    'candidate_queue_occupancy',
    'candidate_dist_to_dest',
    'candidate_progress',           # how much closer to dest vs current node
    # Current node context
    'current_energy',
    'current_queue_occupancy',
    'current_num_neighbors',
    'current_speed',
    # Destination context
    'distance_to_destination',
    'bearing_to_destination',
    'destination_reachable',
    # Network-wide topology context
    'avg_degree',
    'network_density',
    'num_components',
    'avg_link_quality',
    'topology_change_rate',
]

LABEL_COL  = 'is_best_label'
WEIGHT_COL = 'consensus_confidence'

# ─── Data loading ─────────────────────────────────────────────────────────────

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
            df['_episode'] = ep_dir.name
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
        print(f"  Loaded {sc}: {len(df):>8,} rows")
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def prepare_XYW(df):
    """Extract feature matrix X, labels y, and sample weights w."""
    # Drop any rows where key features are missing
    df = df.dropna(subset=FEATURE_COLS + [LABEL_COL])
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[LABEL_COL].values.astype(np.int32)
    w = df[WEIGHT_COL].fillna(0.2).clip(0.01, 1.0).values.astype(np.float32)
    return X, y, w, df


# ─── Per-decision top-k accuracy ──────────────────────────────────────────────

def top_k_accuracy(df, scores, k=1):
    """
    For each unique decision_id, check whether the true best candidate
    (is_best_label=1) is within the top-k scored candidates.

    This is the metric that matters for routing: does the model rank the
    correct next hop in position 1 (top-1) or at least position 2 (top-2)?
    """
    df = df.copy()
    df['_score'] = scores
    correct = 0
    total = 0
    for _, group in df.groupby(['_scenario', '_episode', 'decision_id']):
        if group[LABEL_COL].sum() == 0:
            continue  # no positive label in this decision (shouldn't happen)
        ranked = group.sort_values('_score', ascending=False)
        top_k_ids = set(ranked.head(k).index)
        true_best = set(group[group[LABEL_COL] == 1].index)
        if top_k_ids & true_best:
            correct += 1
        total += 1
    return correct / max(total, 1), total


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(model, X, y, w, df, split_name, feature_names):
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

    # Per-scenario breakdown (if multiple scenarios present)
    if '_scenario' in df.columns:
        print(f"\n  Per-scenario breakdown:")
        print(f"  {'Scenario':<14} {'Rows':>8} {'AUC':>6} {'Top-1':>7} {'Top-2':>7}")
        print(f"  {'─'*46}")
        for sc in df['_scenario'].unique():
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
            print(f"  {sc:<14} {mask.sum():>8,} {sc_auc:>6.3f} {sc_t1:>7.3f} {sc_t2:>7.3f}")

    return {'auc': auc, 'acc': acc, 'top1': top1, 'top2': top2, 'n_decisions': n_dec}


# ─── Feature importance plot ───────────────────────────────────────────────────

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
    ax.set_title('XGBoost Feature Importance — Next-Hop Selection', fontweight='bold')
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
        ax.set_title(f'XGBoost — {metric}', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Learning curve saved: {out_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', default='data/pilot')
    ap.add_argument('--out_dir',   default='models')
    ap.add_argument('--n_estimators', type=int, default=300)
    ap.add_argument('--max_depth',    type=int, default=6)
    ap.add_argument('--lr',           type=float, default=0.1)
    ap.add_argument('--no_plot', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  LOADING DATA")
    print("═"*60)

    print(f"\n  Train scenarios: {TRAIN_SCENARIOS}")
    df_train = load_split(args.data_root, TRAIN_SCENARIOS)

    print(f"\n  Validation scenarios: {VAL_SCENARIOS}")
    df_val = load_split(args.data_root, VAL_SCENARIOS)

    print(f"\n  Test scenarios (hardest — unseen): {TEST_SCENARIOS}")
    df_test = load_split(args.data_root, TEST_SCENARIOS)

    X_train, y_train, w_train, df_train = prepare_XYW(df_train)
    X_val,   y_val,   w_val,   df_val   = prepare_XYW(df_val)
    X_test,  y_test,  w_test,  df_test  = prepare_XYW(df_test)

    print(f"\n  Dataset sizes:")
    print(f"    Train : {len(X_train):>8,} rows  ({y_train.sum():,} positive, "
          f"{(y_train==0).sum():,} negative)")
    print(f"    Val   : {len(X_val):>8,} rows  ({y_val.sum():,} positive)")
    print(f"    Test  : {len(X_test):>8,} rows  ({y_test.sum():,} positive)")

    # ── Class imbalance ───────────────────────────────────────────────────────
    n_neg   = int((y_train == 0).sum())
    n_pos   = int((y_train == 1).sum())
    spw     = round(n_neg / max(n_pos, 1), 2)
    print(f"\n  Class imbalance  →  scale_pos_weight = {spw}")
    print(f"  (This corrects within-decision imbalance: {n_pos} positive rows "
          f"vs {n_neg} negative rows in training data)")
    print(f"  Note: low-PDR episodes are NOT down-weighted — they represent")
    print(f"        real hard scenarios and must stay at full weight.")

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  TRAINING XGBoost")
    print("═"*60)
    print(f"  n_estimators={args.n_estimators}, max_depth={args.max_depth}, "
          f"lr={args.lr}, scale_pos_weight={spw}")

    eval_results = {}
    model = xgb.XGBClassifier(
        n_estimators       = args.n_estimators,
        max_depth          = args.max_depth,
        learning_rate      = args.lr,
        subsample          = 0.8,
        colsample_bytree   = 0.8,
        scale_pos_weight   = spw,
        objective          = 'binary:logistic',
        eval_metric        = ['logloss', 'auc'],
        use_label_encoder  = False,
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

    # Retrieve eval results
    raw_results = model.evals_result()
    eval_results['train'] = raw_results.get('validation_0', {})
    eval_results['val']   = raw_results.get('validation_1', {})
    best_round = model.best_iteration
    print(f"\n  Training complete. Best round: {best_round}")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  EVALUATION")
    print("═"*60)

    train_metrics = evaluate(model, X_train, y_train, w_train, df_train, 'train', FEATURE_COLS)
    val_metrics   = evaluate(model, X_val,   y_val,   w_val,   df_val,   'val',   FEATURE_COLS)
    test_metrics  = evaluate(model, X_test,  y_test,  w_test,  df_test,  'test',  FEATURE_COLS)

    # Top feature
    fi   = model.feature_importances_
    top3 = sorted(zip(FEATURE_COLS, fi), key=lambda x: x[1], reverse=True)[:3]
    print(f"\n  Top 3 features: {[(n, round(v,4)) for n,v in top3]}")

    # ── Save model ────────────────────────────────────────────────────────────
    model_path = os.path.join(args.out_dir, 'xgboost_warmstart.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump({'model': model, 'feature_cols': FEATURE_COLS,
                     'train_scenarios': TRAIN_SCENARIOS,
                     'val_scenarios': VAL_SCENARIOS,
                     'test_scenarios': TEST_SCENARIOS}, f)
    print(f"\n  Model saved → {model_path}")

    # ── Write report ──────────────────────────────────────────────────────────
    report_path = os.path.join(args.out_dir, 'warmstart_report.txt')
    with open(report_path, 'w') as f:
        f.write("XGBoost Warm-Start Model — Training Report\n")
        f.write("=" * 50 + "\n\n")
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

    # ── Plots ─────────────────────────────────────────────────────────────────
    if not args.no_plot:
        plot_feature_importance(
            model, FEATURE_COLS,
            os.path.join(args.out_dir, 'feature_importance.png'))
        plot_learning_curve(
            eval_results,
            os.path.join(args.out_dir, 'learning_curve.png'))

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")
    print("\n" + "═"*60)
    print("  DONE — Next step: use xgboost_warmstart.pkl to initialize RL policy")
    print("═"*60 + "\n")


if __name__ == '__main__':
    main()
