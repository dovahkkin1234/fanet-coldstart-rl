"""
train_mlp_full_accstop.py
Phase 3b (full-scale, v2) — MLP warm-start, early-stopping on VAL ACCURACY
instead of VAL LOSS.

Rationale: the loss-based checkpoint (train_mlp_full.py) restored the model
from epoch 3, but val_acc kept climbing through epoch 10-11 while val_loss
climbed too (pos_weight reweighting distorts BCE loss away from the
ranking-relevant decision boundary). Top-1/Top-2 — the metrics that matter
for next-hop selection — track val_acc, not val_loss. This version tracks
val_acc for checkpointing and early stopping so the RL policy gets
initialized from the best *ranking* model, not the best *loss* model.

Same data, same split, same architecture as train_mlp_full.py — only the
checkpoint/early-stop criterion changes. Also raises max epochs and patience
since val_acc plateaus more slowly than val_loss.

Usage:
    python src\\train_mlp_full_accstop.py
    python src\\train_mlp_full_accstop.py --epochs 30 --patience 10
"""

import os, sys, argparse, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings('ignore')

TRAIN_SCENARIOS = ['rwp_sc01', 'rwp_sc02', 'rwp_sc03', 'rwp_sc04', 'rwp_sc07',
                   'rwp_sc08', 'rwp_sc10', 'rwp_sc12', 'rwp_sc15']
VAL_SCENARIOS   = ['rwp_sc05', 'rwp_sc09', 'rwp_sc13']
TEST_SCENARIOS  = ['rwp_sc06', 'rwp_sc11', 'rwp_sc14']

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


class NextHopMLP(nn.Module):
    def __init__(self, input_dim=22):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),        nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_split(data_root, scenario_list):
    parts = []
    for sc in scenario_list:
        sc_dir = Path(data_root) / sc
        if not sc_dir.exists():
            print(f"  [WARN] Scenario not found, skipping: {sc}")
            continue
        n_before = len(parts)
        for ep_dir in sorted(sc_dir.iterdir()):
            if not ep_dir.is_dir():
                continue
            f = ep_dir / 'routing_decisions.csv'
            if f.exists():
                df = pd.read_csv(f)
                df['_scenario'] = sc
                df['_episode']  = ep_dir.name
                parts.append(df)
        print(f"  Loaded {sc}: {len(parts) - n_before} episodes")
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def prepare_tensors(df, device, feat_mean=None, feat_std=None):
    df = df.dropna(subset=FEATURE_COLS + [LABEL_COL])
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[LABEL_COL].values.astype(np.float32)
    w = df[WEIGHT_COL].fillna(0.2).clip(0.01, 1.0).values.astype(np.float32)

    if feat_mean is None:
        feat_mean = X.mean(axis=0)
        feat_std  = X.std(axis=0) + 1e-8
    X = (X - feat_mean) / feat_std

    X = torch.tensor(X, dtype=torch.float32, device=device)
    y = torch.tensor(y, dtype=torch.float32, device=device)
    w = torch.tensor(w, dtype=torch.float32, device=device)
    return X, y, w, df, feat_mean, feat_std


def top_k_accuracy(df, scores_np, k=1):
    df = df.copy()
    df['_score'] = scores_np
    correct = total = 0
    for _, grp in df.groupby(['_scenario', '_episode', 'decision_id']):
        if grp[LABEL_COL].sum() == 0:
            continue
        top_ids  = set(grp.nlargest(k, '_score').index)
        true_ids = set(grp[grp[LABEL_COL] == 1].index)
        if top_ids & true_ids:
            correct += 1
        total += 1
    return correct / max(total, 1), total


def train_epoch(model, loader, optimizer, pos_weight_t):
    model.train()
    total_loss = 0.0
    n = 0
    for X_b, y_b, w_b in loader:
        optimizer.zero_grad()
        logits = model(X_b)
        loss_unreduced = nn.functional.binary_cross_entropy_with_logits(
            logits, y_b, reduction='none', pos_weight=pos_weight_t)
        loss = (loss_unreduced * w_b).mean()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_b)
        n += len(y_b)
    return total_loss / n


@torch.no_grad()
def eval_metrics(model, X, y, w, pos_weight_t):
    model.eval()
    logits = model(X)
    loss_unreduced = nn.functional.binary_cross_entropy_with_logits(
        logits, y, reduction='none', pos_weight=pos_weight_t)
    loss  = (loss_unreduced * w).mean().item()
    probs = torch.sigmoid(logits).cpu().numpy()
    preds = (probs >= 0.5).astype(int)
    y_np  = y.cpu().numpy()
    acc   = float((preds == y_np).mean())
    return loss, acc, probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root',  default='data/rwp_full')
    ap.add_argument('--out_dir',    default='models')
    ap.add_argument('--epochs',     type=int,   default=30)
    ap.add_argument('--batch_size', type=int,   default=8192)
    ap.add_argument('--lr',         type=float, default=0.001)
    ap.add_argument('--patience',   type=int,   default=10)
    ap.add_argument('--checkpoint_metric', choices=['acc', 'top1'], default='acc',
                    help="'acc' = thresholded val accuracy (cheap, every epoch). "
                         "'top1' = val Top-1 ranking accuracy (expensive, every "
                         "epoch group-by — slower but is the metric that matters most)")
    ap.add_argument('--no_plot', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Device: {device}")
    print(f"  Checkpoint metric: val_{args.checkpoint_metric} (was val_loss in v1)")
    t0 = time.time()

    print(f"\n{'═'*60}\n  LOADING DATA — FULL-SCALE RWP\n{'═'*60}")
    print(f"\n  Train: {TRAIN_SCENARIOS}")
    df_train = load_split(args.data_root, TRAIN_SCENARIOS)
    print(f"  Val:   {VAL_SCENARIOS}")
    df_val   = load_split(args.data_root, VAL_SCENARIOS)
    print(f"  Test:  {TEST_SCENARIOS}")
    df_test  = load_split(args.data_root, TEST_SCENARIOS)

    if len(df_train) == 0:
        print("\n  [FATAL] No training data loaded. Check --data_root.")
        sys.exit(1)

    X_tr, y_tr, w_tr, df_train, feat_mean, feat_std = prepare_tensors(df_train, device)
    X_va, y_va, w_va, df_val, _, _ = prepare_tensors(df_val, device, feat_mean, feat_std)
    X_te, y_te, w_te, df_test, _, _ = prepare_tensors(df_test, device, feat_mean, feat_std)

    n_pos = int(y_tr.sum().item())
    n_neg = len(y_tr) - n_pos
    pos_weight = n_neg / max(n_pos, 1)
    pos_weight_t = torch.tensor([pos_weight], device=device)

    print(f"\n  Train: {len(X_tr):>11,} rows  ({n_pos:,}+ / {n_neg:,}-)")
    print(f"  Val:   {len(X_va):>11,} rows")
    print(f"  Test:  {len(X_te):>11,} rows")
    print(f"  pos_weight: {pos_weight:.2f}")

    train_ds = TensorDataset(X_tr, y_tr, w_tr)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, drop_last=False)

    model = NextHopMLP(input_dim=len(FEATURE_COLS)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # NOTE: scheduler still watches val_loss for LR decay purposes —
    # that's fine, LR decay and checkpoint selection are separate concerns.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model: {n_params:,} parameters ({n_params*4/1024:.1f} KB)")
    print(f"  Architecture: {len(FEATURE_COLS)} -> 128 -> 64 -> 1")

    print(f"\n{'═'*60}\n  TRAINING MLP (checkpoint on val_{args.checkpoint_metric})\n{'═'*60}")
    print(f"  epochs={args.epochs}, batch={args.batch_size}, lr={args.lr}, "
          f"patience={args.patience}")
    header_extra = 'ValTop1' if args.checkpoint_metric == 'top1' else ''
    print(f"\n  {'Epoch':>5}  {'TrainLoss':>10}  {'ValLoss':>10}  "
          f"{'ValAcc':>7}  {header_extra:>7}  {'LR':>8}  Note")
    print(f"  {'─'*66}")

    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_top1': []}
    best_metric   = -1.0
    best_epoch    = 0
    patience_cnt  = 0
    best_state    = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, pos_weight_t)
        val_loss, val_acc, val_probs = eval_metrics(model, X_va, y_va, w_va, pos_weight_t)
        scheduler.step(val_loss)
        lr_now = optimizer.param_groups[0]['lr']

        val_top1 = None
        if args.checkpoint_metric == 'top1':
            val_top1, _ = top_k_accuracy(df_val, val_probs, k=1)
            current_metric = val_top1
        else:
            current_metric = val_acc

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_top1'].append(val_top1 if val_top1 is not None else np.nan)

        note = ''
        if current_metric > best_metric:
            best_metric  = current_metric
            best_epoch   = epoch
            patience_cnt = 0
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            note = 'best'
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                note = 'early stop'

        extra_str = f"{val_top1:>7.4f}" if val_top1 is not None else f"{'':>7}"
        print(f"  {epoch:>5}  {train_loss:>10.5f}  {val_loss:>10.5f}  "
              f"{val_acc:>7.4f}  {extra_str}  {lr_now:>8.6f}  {note}")

        if patience_cnt >= args.patience:
            break

    model.load_state_dict(best_state)
    model.to(device)
    print(f"\n  Restored best model from epoch {best_epoch} "
          f"(val_{args.checkpoint_metric}={best_metric:.4f})")

    print(f"\n{'═'*60}\n  EVALUATION\n{'═'*60}")

    from sklearn.metrics import roc_auc_score
    results = {}
    for split_name, X, y, w, df in [
            ('TRAIN', X_tr, y_tr, w_tr, df_train),
            ('VAL',   X_va, y_va, w_va, df_val),
            ('TEST',  X_te, y_te, w_te, df_test)]:
        _, acc, probs = eval_metrics(model, X, y, w, pos_weight_t)
        auc = roc_auc_score(y.cpu().numpy(), probs, sample_weight=w.cpu().numpy())
        t1, nd = top_k_accuracy(df, probs, k=1)
        t2, _  = top_k_accuracy(df, probs, k=2)
        results[split_name.lower()] = {'auc': auc, 'acc': acc, 'top1': t1,
                                        'top2': t2, 'n_decisions': nd}

        print(f"\n  {split_name} ({len(y):,} rows, {nd:,} decisions)")
        print(f"    AUC-ROC  : {auc:.4f}")
        print(f"    Accuracy : {acc:.4f}")
        print(f"    Top-1    : {t1:.4f}  ({int(t1*nd)}/{nd})")
        print(f"    Top-2    : {t2:.4f}  ({int(t2*nd)}/{nd})")

        if split_name == 'TEST':
            for sc in sorted(df['_scenario'].unique()):
                mask = df['_scenario'] == sc
                sc_probs = probs[mask.values]
                sc_y     = y.cpu().numpy()[mask.values]
                sc_w     = w.cpu().numpy()[mask.values]
                sc_df    = df[mask].copy()
                try:
                    sc_auc = roc_auc_score(sc_y, sc_probs, sample_weight=sc_w)
                except ValueError:
                    sc_auc = float('nan')
                sc_t1, sc_nd = top_k_accuracy(sc_df, sc_probs, k=1)
                sc_t2, _     = top_k_accuracy(sc_df, sc_probs, k=2)
                print(f"    {sc}: AUC={sc_auc:.3f} Top-1={sc_t1:.3f} Top-2={sc_t2:.3f}")

    save_bundle = {
        'model_state_dict': best_state,
        'feature_cols':     FEATURE_COLS,
        'feat_mean':        feat_mean,
        'feat_std':         feat_std,
        'input_dim':        len(FEATURE_COLS),
        'architecture':     '22-128-64-1',
        'best_epoch':       best_epoch,
        'checkpoint_metric': args.checkpoint_metric,
        'pos_weight':       pos_weight,
        'train_scenarios':  TRAIN_SCENARIOS,
        'val_scenarios':    VAL_SCENARIOS,
        'test_scenarios':   TEST_SCENARIOS,
    }
    model_path = os.path.join(args.out_dir, 'mlp_warmstart_full_accstop.pt')
    torch.save(save_bundle, model_path)
    print(f"\n  Model saved → {model_path}")

    # Comparison note against the v1 (loss-checkpointed) model, if present
    v1_path = os.path.join(args.out_dir, 'mlp_warmstart_full.pt')
    if os.path.exists(v1_path):
        print(f"\n  Reference: v1 (loss-checkpointed) model at {v1_path}")
        print(f"  v1 test Top-1 was 0.8558 (epoch 3) — compare against this run's "
              f"test Top-1 = {results['test']['top1']:.4f} (epoch {best_epoch})")

    if not args.no_plot:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(history['train_loss'], label='Train', color='#457b9d')
        axes[0].plot(history['val_loss'],   label='Val',   color='#e63946')
        axes[0].axvline(best_epoch-1, color='gray', ls='--', alpha=0.5,
                        label=f'Best (ep {best_epoch})')
        axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
        axes[0].set_title('MLP — Loss (accstop version)', fontweight='bold')
        axes[0].legend(); axes[0].grid(alpha=0.3)

        axes[1].plot(history['val_acc'], color='#e63946', label='Val Acc')
        if args.checkpoint_metric == 'top1':
            axes[1].plot(history['val_top1'], color='#2a9d8f', label='Val Top-1')
        axes[1].axvline(best_epoch-1, color='gray', ls='--', alpha=0.5,
                        label=f'Best (ep {best_epoch})')
        axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Metric')
        axes[1].set_title(f'MLP — Checkpoint metric (val_{args.checkpoint_metric})',
                          fontweight='bold')
        axes[1].legend(); axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(args.out_dir, 'mlp_learning_curve_full_accstop.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Learning curve → {plot_path}")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"\n{'═'*60}")
    print(f"  DONE — mlp_warmstart_full_accstop.pt ready for RL policy initialization")
    print(f"{'═'*60}\n")


if __name__ == '__main__':
    main()
