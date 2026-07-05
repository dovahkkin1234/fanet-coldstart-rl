"""
rdm_warmstart.py
Phases 1-3 for the Random Direction Model (RDM) dataset.

Differences from the pilot (RWP) pipeline:
  - Data layout: each rdm_* folder IS an episode (no sub-folders)
  - 5 episodes total → episode-level split: 3 train / 1 val / 1 test
  - All outputs go to  models/RDM/
  - Validation runs on each episode individually before training

Usage (from FANET_sim directory):
    python src\\rdm_warmstart.py
    python src\\rdm_warmstart.py --data_root data/random_direction
    python src\\rdm_warmstart.py --no_plot
"""

import os, sys, argparse, time, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import xgboost as xgb
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score

warnings.filterwarnings('ignore')

# ─── Feature / label config (identical to RWP pipeline) ──────────────────────
FEATURE_COLS = [
    'candidate_distance', 'candidate_relative_velocity', 'candidate_link_quality',
    'candidate_rssi', 'candidate_packet_error_rate', 'candidate_estimated_lifetime',
    'candidate_energy', 'candidate_queue_occupancy', 'candidate_dist_to_dest',
    'candidate_progress', 'current_energy', 'current_queue_occupancy',
    'current_num_neighbors', 'current_speed', 'distance_to_destination',
    'bearing_to_destination', 'destination_reachable', 'avg_degree',
    'network_density', 'num_components', 'avg_link_quality', 'topology_change_rate',
]
LABEL_COL  = 'is_best_label'
WEIGHT_COL = 'consensus_confidence'


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def discover_episodes(data_root):
    """
    RDM layout: each rdm_* folder IS an episode (routing_decisions.csv lives directly inside).
    Returns sorted list of Path objects.
    """
    root = Path(data_root)
    eps = sorted([
        d for d in root.iterdir()
        if d.is_dir() and (d.name.startswith('rdm') or d.name.startswith('RDM'))
    ])
    return eps


def load_episode(ep_dir, tag=None):
    """Load routing_decisions.csv from one episode folder."""
    f = Path(ep_dir) / 'routing_decisions.csv'
    if not f.exists():
        print(f"  [WARN] routing_decisions.csv not found in {ep_dir}")
        return pd.DataFrame()
    df = pd.read_csv(f)
    df['_episode'] = tag or Path(ep_dir).name
    df['_scenario'] = 'rdm'
    return df


def load_split(episodes):
    """Load and concatenate routing_decisions from a list of episode dirs."""
    parts = []
    for ep in episodes:
        df = load_episode(ep)
        if len(df) == 0:
            continue
        print(f"  Loaded {ep.name}: {len(df):>8,} rows")
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def prepare_XYW(df):
    df = df.dropna(subset=FEATURE_COLS + [LABEL_COL])
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[LABEL_COL].values.astype(np.int32)
    w = df[WEIGHT_COL].fillna(0.2).clip(0.01, 1.0).values.astype(np.float32)
    return X, y, w, df


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION (Phase 1 equivalent)
# ═══════════════════════════════════════════════════════════════════════════════

def validate_episode(ep_dir):
    """Run basic sanity checks on one RDM episode. Returns (pass_count, fail_list)."""
    ep = Path(ep_dir)
    checks = {}

    for tbl in ['routing_decisions', 'packet_logs', 'mobility_trace',
                'topology_features', 'link_states']:
        checks[f'file_{tbl}'] = (ep / f'{tbl}.csv').exists()

    rd_path = ep / 'routing_decisions.csv'
    if rd_path.exists():
        rd = pd.read_csv(rd_path)
        checks['rd_has_rows']         = len(rd) > 0
        checks['rd_has_best_label']   = LABEL_COL in rd.columns
        checks['rd_has_features']     = all(c in rd.columns for c in FEATURE_COLS[:5])
        checks['rd_label_binary']     = set(rd[LABEL_COL].dropna().unique()).issubset({0, 1})
        checks['rd_positive_exist']   = rd[LABEL_COL].sum() > 0
        checks['rd_has_weight_col']   = WEIGHT_COL in rd.columns
        checks['rd_no_inf']           = not np.isinf(
            rd[FEATURE_COLS].select_dtypes(include=np.number).values).any()

    pk_path = ep / 'packet_logs.csv'
    if pk_path.exists():
        pk = pd.read_csv(pk_path)
        if len(pk) > 0:
            pdr = float(pk['delivered'].mean()) if 'delivered' in pk.columns else 0
            checks['pkt_pdr_nonzero'] = pdr > 0.01
            checks['pkt_pdr_value']   = round(pdr, 3)

    passed = [k for k, v in checks.items() if v is True]
    failed = [k for k, v in checks.items() if v is False]
    return checks, passed, failed


def run_validation(episodes, out_dir):
    print(f"\n{'═'*60}")
    print("  PHASE 1 — VALIDATION")
    print(f"{'═'*60}\n")

    report_lines = []
    all_passed = True
    for ep in episodes:
        checks, passed, failed = validate_episode(ep)
        pdr_val = checks.get('pkt_pdr_value', 'N/A')
        status  = "✓ PASS" if not failed else f"✗ FAIL ({len(failed)} checks)"
        print(f"  {ep.name:<40s}  PDR={pdr_val}  {status}")
        if failed:
            all_passed = False
            for f in failed:
                print(f"      ✗ {f}")
        report_lines.append(f"{ep.name}: {status}  PDR={pdr_val}")

    # Save report
    rpt = out_dir / 'validation_report.txt'
    with open(rpt, 'w') as f:
        f.write("RDM Validation Report\n")
        f.write("=" * 50 + "\n")
        for line in report_lines:
            f.write(line + "\n")
        f.write(f"\nOverall: {'ALL PASSED' if all_passed else 'SOME FAILURES'}\n")
    print(f"\n  Report → {rpt}")
    return all_passed


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def top_k_accuracy(df, scores, k=1):
    """Per-decision top-k accuracy: is the true best in top-k scored candidates?"""
    correct = 0
    total   = 0
    df = df.copy()
    df['_score'] = scores
    for _, grp in df.groupby(['_episode', 'decision_id']):
        if LABEL_COL not in grp.columns or grp[LABEL_COL].sum() == 0:
            continue
        top_k  = grp.nlargest(k, '_score').index
        best   = grp[grp[LABEL_COL] == 1].index
        correct += int(len(set(top_k) & set(best)) > 0)
        total   += 1
    return correct / total if total > 0 else 0.0, correct, total


def evaluate(tag, scores, X, y, df, label_col=LABEL_COL):
    auc  = roc_auc_score(y, scores) if len(np.unique(y)) > 1 else float('nan')
    acc  = accuracy_score(y, (scores >= 0.5).astype(int))
    t1, c1, n = top_k_accuracy(df, scores, k=1)
    t2, c2, _ = top_k_accuracy(df, scores, k=2)
    print(f"\n  {tag} ({len(df):,} rows, {n:,} decisions)")
    print(f"    AUC-ROC  : {auc:.4f}")
    print(f"    Accuracy : {acc:.4f}")
    print(f"    Top-1    : {t1:.4f}  ({c1}/{n})")
    print(f"    Top-2    : {t2:.4f}  ({c2}/{n})")
    return {'auc': auc, 'acc': acc, 'top1': t1, 'top2': t2, 'n_decisions': n}


# ═══════════════════════════════════════════════════════════════════════════════
# XGBOOST (Phase 2)
# ═══════════════════════════════════════════════════════════════════════════════

def train_xgboost(X_tr, y_tr, w_tr, X_val, y_val, w_val, out_dir, no_plot):
    print(f"\n{'═'*60}")
    print("  PHASE 2 — XGBOOST WARM-START")
    print(f"{'═'*60}\n")

    pos = int(y_tr.sum())
    neg = int((y_tr == 0).sum())
    spw = round(neg / max(pos, 1), 2)
    print(f"  Train: {len(y_tr):,} rows  (+{pos} / -{neg})  pos_weight={spw}")

    dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr)
    dval   = xgb.DMatrix(X_val, label=y_val, weight=w_val)

    params = {
        'objective': 'binary:logistic', 'eval_metric': 'auc',
        'max_depth': 6, 'learning_rate': 0.05, 'n_estimators': 300,
        'subsample': 0.8, 'colsample_bytree': 0.8,
        'scale_pos_weight': spw, 'tree_method': 'hist',
        'seed': 42, 'verbosity': 0,
    }
    evals_result = {}
    model = xgb.train(
        params, dtrain, num_boost_round=300,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=20,
        evals_result=evals_result,
        verbose_eval=False,
    )
    print(f"  Best round: {model.best_iteration}  Best val AUC: {model.best_score:.4f}")

    path = out_dir / 'xgboost_warmstart_rdm.pkl'
    with open(path, 'wb') as f:
        pickle.dump({'model': model, 'features': FEATURE_COLS}, f)
    print(f"  Saved → {path}")

    if not no_plot:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(evals_result['train']['auc'], label='Train AUC', color='#457b9d')
        ax.plot(evals_result['val']['auc'],   label='Val AUC',   color='#e63946')
        ax.axvline(model.best_iteration, color='gray', ls='--', alpha=0.6,
                   label=f'Best (round {model.best_iteration})')
        ax.set_xlabel('Boosting Round'); ax.set_ylabel('AUC')
        ax.set_title('XGBoost Learning Curve — RDM', fontweight='bold')
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        p = out_dir / 'xgboost_learning_curve.png'
        plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
        print(f"  Plot → {p}")

        # Feature importance
        imp = model.get_score(importance_type='gain')
        imp_s = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:15]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh([i[0] for i in imp_s][::-1],
                [i[1] for i in imp_s][::-1], color='#457b9d')
        ax.set_xlabel('Gain'); ax.set_title('Feature Importance (XGBoost) — RDM', fontweight='bold')
        plt.tight_layout()
        p = out_dir / 'feature_importance.png'
        plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
        print(f"  Plot → {p}")

    return model


# ═══════════════════════════════════════════════════════════════════════════════
# MLP (Phase 3)
# ═══════════════════════════════════════════════════════════════════════════════

class NextHopMLP(nn.Module):
    def __init__(self, input_dim=22):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),        nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1),          nn.Sigmoid(),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


class RoutingDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, w):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.w = torch.tensor(w, dtype=torch.float32)
    def __len__(self):  return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i], self.w[i]


def train_mlp(X_tr, y_tr, w_tr, df_tr,
              X_val, y_val, w_val, df_val,
              out_dir, no_plot,
              epochs=40, batch_size=4096, lr=1e-3, patience=8):
    print(f"\n{'═'*60}")
    print("  PHASE 3 — MLP WARM-START")
    print(f"{'═'*60}\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    # Normalise using training stats
    feat_mean = X_tr.mean(axis=0).astype(np.float32)
    feat_std  = X_tr.std(axis=0).astype(np.float32) + 1e-8

    X_tr_n  = (X_tr  - feat_mean) / feat_std
    X_val_n = (X_val - feat_mean) / feat_std

    pos_weight = torch.tensor([(y_tr == 0).sum() / max(y_tr.sum(), 1)],
                               dtype=torch.float32).to(device)

    model     = NextHopMLP(input_dim=len(FEATURE_COLS)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss(reduction='none')

    tr_loader = torch.utils.data.DataLoader(
        RoutingDataset(X_tr_n, y_tr, w_tr),
        batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(
        RoutingDataset(X_val_n, y_val, w_val),
        batch_size=batch_size * 4, shuffle=False, num_workers=0)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params:,} parameters  ({n_params*4/1024:.1f} KB at float32)")
    print(f"  Architecture: {len(FEATURE_COLS)} → 128 → 64 → 1")
    print(f"\n  epochs={epochs}, batch={batch_size}, lr={lr}, patience={patience}")
    print(f"  {'Epoch':>6}  {'TrainLoss':>11}  {'ValLoss':>9}  {'ValAcc':>8}  {'LR':>9}  Note")
    print(f"  {'─'*62}")

    best_val_acc = 0.0
    best_state   = None
    patience_ctr = 0
    scheduler    = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=4)
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        for Xb, yb, wb in tr_loader:
            Xb, yb, wb = Xb.to(device), yb.to(device), wb.to(device)
            optimizer.zero_grad()
            pred = model(Xb)
            loss = (criterion(pred, yb) * wb * torch.where(yb == 1, pos_weight, torch.ones_like(wb))).mean()
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * len(yb)
        tr_loss /= len(tr_loader.dataset)

        model.eval()
        vl_loss = 0.0; vl_correct = 0
        with torch.no_grad():
            for Xb, yb, wb in val_loader:
                Xb, yb, wb = Xb.to(device), yb.to(device), wb.to(device)
                pred = model(Xb)
                loss = (criterion(pred, yb) * wb).mean()
                vl_loss += loss.item() * len(yb)
                vl_correct += ((pred >= 0.5) == yb).sum().item()
        vl_loss /= len(val_loader.dataset)
        vl_acc   = vl_correct / len(val_loader.dataset)
        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['val_acc'].append(vl_acc)
        scheduler.step(vl_loss)
        cur_lr = optimizer.param_groups[0]['lr']

        note = ''
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
            note = '★ best'
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                note = '✖ early stop'
                print(f"  {ep:>6}  {tr_loss:>11.5f}  {vl_loss:>9.5f}  {vl_acc:>8.4f}  {cur_lr:>9.6f}  {note}")
                break
        print(f"  {ep:>6}  {tr_loss:>11.5f}  {vl_loss:>9.5f}  {vl_acc:>8.4f}  {cur_lr:>9.6f}  {note}")

    model.load_state_dict(best_state)
    print(f"\n  Restored best model (val_acc={best_val_acc:.4f})")

    # Save bundle (same format as RWP mlp_warmstart.pt)
    bundle = {
        'model_state_dict': best_state,
        'feat_mean':        feat_mean,
        'feat_std':         feat_std,
        'feature_cols':     FEATURE_COLS,
        'architecture':     f'{len(FEATURE_COLS)}-128-64-1',
        'input_dim':        len(FEATURE_COLS),
    }
    path = out_dir / 'mlp_warmstart_rdm.pt'
    torch.save(bundle, path)
    print(f"  Saved → {path}")

    if not no_plot:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        xs = range(1, len(history['train_loss']) + 1)
        axes[0].plot(xs, history['train_loss'], label='Train', color='#457b9d')
        axes[0].plot(xs, history['val_loss'],   label='Val',   color='#e63946')
        axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
        axes[0].set_title('MLP Loss — RDM', fontweight='bold')
        axes[0].legend(); axes[0].grid(alpha=0.3)
        axes[1].plot(xs, history['val_acc'], color='#e63946')
        axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Val Accuracy')
        axes[1].set_title('MLP Val Accuracy — RDM', fontweight='bold')
        axes[1].grid(alpha=0.3)
        plt.tight_layout()
        p = out_dir / 'mlp_learning_curve.png'
        plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
        print(f"  Plot → {p}")

    return model, feat_mean, feat_std


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTER COMPARISON (Phase 2 baseline equivalent)
# ═══════════════════════════════════════════════════════════════════════════════

def router_comparison(episodes, out_dir, no_plot):
    """
    Load packet_logs from all episodes and compare PDR/delay/hops
    across the teacher routing protocols stored in routing_decisions.
    """
    print(f"\n{'═'*60}")
    print("  BASELINE — ROUTING PROTOCOL COMPARISON")
    print(f"{'═'*60}\n")

    records = []
    for ep in episodes:
        pk = ep / 'packet_logs.csv'
        rd = ep / 'routing_decisions.csv'
        if not pk.exists() or not rd.exists():
            continue
        pk_df = pd.read_csv(pk)
        rd_df = pd.read_csv(rd)
        ep_name = ep.name

        # Overall PDR from packet_logs
        pdr   = float(pk_df['delivered'].mean()) if 'delivered' in pk_df.columns else float('nan')
        delay = float(pk_df[pk_df['delivered'] == 1]['end_to_end_delay'].mean()) \
                if 'end_to_end_delay' in pk_df.columns else float('nan')
        hops  = float(pk_df[pk_df['delivered'] == 1]['hop_count'].mean()) \
                if 'hop_count' in pk_df.columns else float('nan')
        records.append({'episode': ep_name, 'protocol': 'consensus',
                        'pdr': pdr, 'delay_ms': delay, 'hops': hops})

        # Per-teacher columns if present (teacher_0 … teacher_4)
        teacher_cols = [c for c in rd_df.columns if c.startswith('teacher_')]
        for tc in teacher_cols:
            records.append({'episode': ep_name, 'protocol': tc,
                            'pdr': pdr, 'delay_ms': delay, 'hops': hops})

    if not records:
        print("  [WARN] No packet_logs found, skipping comparison")
        return

    df = pd.DataFrame(records)
    summary = df[df['protocol'] == 'consensus'].agg(
        {'pdr': ['mean', 'std'], 'delay_ms': ['mean', 'std'], 'hops': ['mean', 'std']}
    ).round(4)
    print("  PDR / Delay / Hops across all 5 RDM episodes (consensus router):")
    print(f"  {summary.to_string()}\n")

    per_ep = df[df['protocol'] == 'consensus'][['episode', 'pdr', 'delay_ms', 'hops']]
    print("  Per-episode:")
    print(f"  {per_ep.to_string(index=False)}\n")

    df.to_csv(out_dir / 'router_comparison_rdm.csv', index=False)
    print(f"  Saved → {out_dir / 'router_comparison_rdm.csv'}")

    if not no_plot:
        ep_names = per_ep['episode'].tolist()
        pdrs     = per_ep['pdr'].tolist()
        fig, ax  = plt.subplots(figsize=(10, 4))
        colors   = ['#457b9d'] * len(ep_names)
        ax.bar(range(len(ep_names)), pdrs, color=colors)
        ax.set_xticks(range(len(ep_names)))
        ax.set_xticklabels([e.replace('rdm_', '') for e in ep_names],
                            rotation=20, ha='right', fontsize=9)
        ax.set_ylabel('PDR'); ax.set_ylim(0, 1)
        ax.set_title('Per-Episode PDR — RDM Dataset', fontweight='bold')
        ax.grid(alpha=0.3, axis='y')
        plt.tight_layout()
        p = out_dir / 'pdr_per_episode.png'
        plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
        print(f"  Plot → {p}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', default='data/random_direction')
    ap.add_argument('--out_dir',   default='models/RDM')
    ap.add_argument('--epochs',    type=int,   default=40)
    ap.add_argument('--batch_size',type=int,   default=4096)
    ap.add_argument('--lr',        type=float, default=1e-3)
    ap.add_argument('--no_plot',   action='store_true')
    ap.add_argument('--seed',      type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f"\n{'═'*60}")
    print("  RDM PIPELINE  —  Phases 1 → 3")
    print(f"{'═'*60}")
    print(f"  data_root : {args.data_root}")
    print(f"  out_dir   : {out_dir}")

    # ── Discover episodes ─────────────────────────────────────────────────────
    episodes = discover_episodes(args.data_root)
    if not episodes:
        print(f"\n  [FATAL] No rdm_* folders found in {args.data_root}")
        sys.exit(1)

    print(f"\n  Found {len(episodes)} episodes:")
    for ep in episodes:
        print(f"    {ep.name}")

    # Episode split: deterministic by sorted order
    # 5 episodes → seed10=train, seed20=train, seed30=train, seed40=val, seed50=test
    train_eps = episodes[:3]
    val_eps   = episodes[3:4]
    test_eps  = episodes[4:5]
    print(f"\n  Split  Train : {[e.name for e in train_eps]}")
    print(f"         Val   : {[e.name for e in val_eps]}")
    print(f"         Test  : {[e.name for e in test_eps]}")

    # ── Phase 1: Validation ───────────────────────────────────────────────────
    run_validation(episodes, out_dir)

    # ── Baseline: Router comparison ───────────────────────────────────────────
    router_comparison(episodes, out_dir, args.no_plot)

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("  LOADING ROUTING DECISIONS")
    print(f"{'═'*60}\n")

    print("  Train:")
    df_tr  = load_split(train_eps)
    print("  Val:")
    df_val = load_split(val_eps)
    print("  Test:")
    df_te  = load_split(test_eps)

    X_tr,  y_tr,  w_tr,  df_tr  = prepare_XYW(df_tr)
    X_val, y_val, w_val, df_val = prepare_XYW(df_val)
    X_te,  y_te,  w_te,  df_te  = prepare_XYW(df_te)

    pos = int(y_tr.sum()); neg = int((y_tr == 0).sum())
    print(f"\n  Train  : {len(y_tr):>8,} rows  (+{pos} / -{neg})")
    print(f"  Val    : {len(y_val):>8,} rows")
    print(f"  Test   : {len(y_te):>8,} rows")

    if len(y_tr) == 0:
        print("\n  [FATAL] No training rows found. Check data_root path.")
        sys.exit(1)

    # ── Phase 2: XGBoost ──────────────────────────────────────────────────────
    xgb_model = train_xgboost(X_tr, y_tr, w_tr, X_val, y_val, w_val,
                               out_dir, args.no_plot)

    print(f"\n{'─'*60}\n  XGBOOST EVALUATION\n{'─'*60}")
    dtr = xgb.DMatrix(X_tr);  dval = xgb.DMatrix(X_val);  dte = xgb.DMatrix(X_te)
    xgb_results = {
        'train': evaluate('TRAIN', xgb_model.predict(dtr),  X_tr,  y_tr,  df_tr),
        'val':   evaluate('VAL',   xgb_model.predict(dval), X_val, y_val, df_val),
        'test':  evaluate('TEST',  xgb_model.predict(dte),  X_te,  y_te,  df_te),
    }

    # ── Phase 3: MLP ──────────────────────────────────────────────────────────
    mlp_model, feat_mean, feat_std = train_mlp(
        X_tr, y_tr, w_tr, df_tr,
        X_val, y_val, w_val, df_val,
        out_dir, args.no_plot,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
    )

    print(f"\n{'─'*60}\n  MLP EVALUATION\n{'─'*60}")
    device = next(mlp_model.parameters()).device
    def mlp_score(X):
        mlp_model.eval()
        Xn = torch.tensor((X - feat_mean) / feat_std, dtype=torch.float32).to(device)
        with torch.no_grad():
            return mlp_model(Xn).cpu().numpy()

    mlp_results = {
        'train': evaluate('TRAIN', mlp_score(X_tr),  X_tr,  y_tr,  df_tr),
        'val':   evaluate('VAL',   mlp_score(X_val), X_val, y_val, df_val),
        'test':  evaluate('TEST',  mlp_score(X_te),  X_te,  y_te,  df_te),
    }

    # ── Summary report ────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    report = [
        "=" * 60,
        "RDM WARM-START TRAINING REPORT",
        "=" * 60,
        f"Data root   : {args.data_root}",
        f"Episodes    : {len(episodes)} (train={len(train_eps)}, val={len(val_eps)}, test={len(test_eps)})",
        f"Train rows  : {len(y_tr):,}",
        f"Val rows    : {len(y_val):,}",
        f"Test rows   : {len(y_te):,}",
        f"Total time  : {elapsed:.1f}s",
        "",
        "── XGBoost ──",
        f"  Train  AUC={xgb_results['train']['auc']:.4f}  Top-1={xgb_results['train']['top1']:.4f}  Top-2={xgb_results['train']['top2']:.4f}",
        f"  Val    AUC={xgb_results['val']['auc']:.4f}  Top-1={xgb_results['val']['top1']:.4f}  Top-2={xgb_results['val']['top2']:.4f}",
        f"  Test   AUC={xgb_results['test']['auc']:.4f}  Top-1={xgb_results['test']['top1']:.4f}  Top-2={xgb_results['test']['top2']:.4f}",
        "",
        "── MLP ──",
        f"  Train  AUC={mlp_results['train']['auc']:.4f}  Top-1={mlp_results['train']['top1']:.4f}  Top-2={mlp_results['train']['top2']:.4f}",
        f"  Val    AUC={mlp_results['val']['auc']:.4f}  Top-1={mlp_results['val']['top1']:.4f}  Top-2={mlp_results['val']['top2']:.4f}",
        f"  Test   AUC={mlp_results['test']['auc']:.4f}  Top-1={mlp_results['test']['top1']:.4f}  Top-2={mlp_results['test']['top2']:.4f}",
        "",
        "Artifacts:",
        f"  xgboost_warmstart_rdm.pkl",
        f"  mlp_warmstart_rdm.pt  (use feat_mean/feat_std for normalisation)",
    ]
    rpt_text = "\n".join(report)
    print(f"\n{rpt_text}")
    with open(out_dir / 'warmstart_report.txt', 'w') as f:
        f.write(rpt_text)

    print(f"\n{'═'*60}")
    print(f"  DONE — all outputs in {out_dir}")
    print(f"{'═'*60}\n")


if __name__ == '__main__':
    main()
