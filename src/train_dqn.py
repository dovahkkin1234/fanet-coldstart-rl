"""
train_dqn.py
Phase 5 — Deep Q-Learning with MLP warm-start initialization.

Trains 3 RL variants for comparison:
  1. scratch   — DQN from random init (the cold-start baseline)
  2. warmstart — DQN initialized from mlp_warmstart.pt
  3. regime    — warmstart + regime_id as one-hot feature in state

Each variant trains on the same scenarios, with the same seeds, for the same
number of episodes. Per-episode PDR is logged so cold-start metrics
(T_50, T_90, AULC) can be computed and plotted.

Usage:
    python src\\train_dqn.py                           # all 3 variants
    python src\\train_dqn.py --variant warmstart       # just one
    python src\\train_dqn.py --episodes 100            # quick test
"""

import os, sys, argparse, time, pickle, json
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl_env import (FANETRoutingEnv, FEATURE_COLS, MAX_NEIGHBORS,
                     TRAINING_SCENARIOS, SHARED_DEFAULTS, make_env)
from train_mlp import NextHopMLP    # exact same architecture

# ─── Q-Network: scores each candidate independently ──────────────────────────
# The Q-network reuses the MLP architecture. It takes a (N_neighbors, 22)
# tensor and outputs (N_neighbors,) Q-values — one per candidate hop.
# When warmstarting, the trained MLP weights load DIRECTLY into this network.

class QNetwork(nn.Module):
    def __init__(self, input_dim=22):
        super().__init__()
        # Same shape as NextHopMLP so weights transfer 1-to-1
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x: (batch, N_neighbors, features) or (N_neighbors, features)
        return self.net(x).squeeze(-1)


# ─── Replay Buffer ────────────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buf = deque(maxlen=capacity)
    def push(self, obs, action, reward, next_obs, done):
        self.buf.append((obs, action, reward, next_obs, done))
    def sample(self, batch_size):
        idx = np.random.choice(len(self.buf), batch_size, replace=False)
        return [self.buf[i] for i in idx]
    def __len__(self):
        return len(self.buf)


# ─── Agent ────────────────────────────────────────────────────────────────────
class DQNAgent:
    def __init__(self, input_dim, lr=1e-3, gamma=0.95,
                 eps_start=1.0, eps_end=0.05, eps_decay_episodes=200,
                 device='cpu'):
        self.device = device
        self.input_dim = input_dim
        self.q_net      = QNetwork(input_dim).to(device)
        self.target_net = QNetwork(input_dim).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer  = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma      = gamma
        self.eps_start  = eps_start
        self.eps_end    = eps_end
        self.eps_decay_episodes = eps_decay_episodes
        self.epsilon    = eps_start

    def load_warmstart(self, mlp_state_dict):
        """Load MLP weights into Q-network. The MLP wraps layers in `net = Sequential(...)`,
        so its state dict keys are prefixed with 'net.'. Our QNetwork also wraps
        layers in self.net, so we load into self.q_net (whole module) directly."""
        self.q_net.load_state_dict(mlp_state_dict)
        self.target_net.load_state_dict(mlp_state_dict)
        print(f"  Warmstart weights loaded ({sum(p.numel() for p in self.q_net.parameters()):,} params)")

    def update_epsilon(self, episode):
        frac = min(1.0, episode / self.eps_decay_episodes)
        self.epsilon = self.eps_start + frac * (self.eps_end - self.eps_start)

    def select_action(self, obs, training=True):
        """Return action index. Masked epsilon-greedy."""
        mask = obs['mask']
        valid_idx = np.where(mask == 1)[0]
        if len(valid_idx) == 0:
            return 0

        if training and np.random.rand() < self.epsilon:
            return int(np.random.choice(valid_idx))

        with torch.no_grad():
            x = torch.tensor(obs['features'], dtype=torch.float32, device=self.device)
            q = self.q_net(x).cpu().numpy()
            # Mask invalid actions
            q[mask == 0] = -np.inf
            return int(np.argmax(q))

    def update(self, batch):
        if len(batch) == 0: return 0.0

        feats     = torch.tensor(np.stack([b[0]['features'] for b in batch]),
                                  dtype=torch.float32, device=self.device)
        masks     = torch.tensor(np.stack([b[0]['mask']     for b in batch]),
                                  dtype=torch.float32, device=self.device)
        actions   = torch.tensor([b[1] for b in batch], dtype=torch.long, device=self.device)
        rewards   = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=self.device)
        nxt_feats = torch.tensor(np.stack([b[3]['features'] for b in batch]),
                                  dtype=torch.float32, device=self.device)
        nxt_masks = torch.tensor(np.stack([b[3]['mask']     for b in batch]),
                                  dtype=torch.float32, device=self.device)
        dones     = torch.tensor([float(b[4]) for b in batch],
                                  dtype=torch.float32, device=self.device)

        # Current Q-values for the actions taken
        q_all = self.q_net(feats)                                  # (B, N_neighbors)
        q_taken = q_all.gather(1, actions.unsqueeze(1)).squeeze(1) # (B,)

        # Target: r + gamma * max_a' Q_target(s', a') with masking
        with torch.no_grad():
            q_next = self.target_net(nxt_feats)                    # (B, N_neighbors)
            q_next = q_next.masked_fill(nxt_masks == 0, -1e9)
            q_next_max = q_next.max(dim=1).values
            target = rewards + self.gamma * q_next_max * (1.0 - dones)

        loss = nn.functional.smooth_l1_loss(q_taken, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()

    def sync_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())


# ─── Training loop ────────────────────────────────────────────────────────────

def evaluate_greedy(agent, scenario_name, seed, duration, norm_obs, max_steps=2000):
    """Run one greedy (epsilon=0) episode for evaluation. Returns metrics."""
    env = make_env(scenario_name, duration=duration, seed=seed,
                   max_neighbors=MAX_NEIGHBORS)
    obs, info = env.reset()
    obs = norm_obs(obs)
    done = False
    steps = 0
    ep_reward = 0.0
    while not done and steps < max_steps:
        # epsilon=0 forced
        action = agent.select_action(obs, training=False)
        next_obs, reward, done, _, info = env.step(action)
        obs = norm_obs(next_obs)
        ep_reward += reward
        steps += 1
    return {
        'pdr': info['pdr'],
        'mean_delay_ms': info['mean_delay_ms'],
        'mean_hops': info['mean_hops'],
        'reward': ep_reward,
        'steps': steps,
    }


def train_variant(variant_name, mlp_bundle, args, device):
    print(f"\n{'='*60}\n  Training: {variant_name}\n{'='*60}")

    # Different epsilon schedule for warmstart vs scratch
    if variant_name == 'scratch':
        eps_start = args.eps_start          # 1.0
    else:
        eps_start = args.eps_start_warm     # 0.3 — warmstart already competent

    agent = DQNAgent(input_dim=len(FEATURE_COLS),
                     lr=args.lr, gamma=args.gamma,
                     eps_start=eps_start, eps_end=args.eps_end,
                     eps_decay_episodes=args.eps_decay,
                     device=device)

    if variant_name in ('warmstart', 'regime'):
        agent.load_warmstart(mlp_bundle['model_state_dict'])
        feat_mean = mlp_bundle['feat_mean']
        feat_std  = mlp_bundle['feat_std']
    else:
        feat_mean = np.zeros(len(FEATURE_COLS), dtype=np.float32)
        feat_std  = np.ones(len(FEATURE_COLS), dtype=np.float32)

    def norm_obs(obs):
        obs = dict(obs)
        f = obs['features']
        f = (f - feat_mean) / (feat_std + 1e-8)
        obs['features'] = f.astype(np.float32)
        return obs

    buffer = ReplayBuffer(args.buffer_size)

    # Per-scenario history: dict[scenario_name] -> list of (episode_idx_within_scenario, pdr, ...)
    scenario_names = list(TRAINING_SCENARIOS.keys())
    per_scn_train_history = {sc: {'episode': [], 'pdr': [], 'reward': []}
                              for sc in scenario_names}
    per_scn_eval_history  = {sc: {'episode': [], 'pdr': [], 'mean_delay_ms': [],
                                    'mean_hops': [], 'reward': []}
                              for sc in scenario_names}

    # Global flat history (for diagnostics)
    flat_history = {'episode': [], 'pdr': [], 'reward': [], 'epsilon': [],
                    'scenario': [], 'loss': []}

    episode_seeds  = np.random.default_rng(args.seed).integers(0, 1_000_000,
                                                                size=args.episodes)
    # Fixed seeds for greedy evaluation (same per scenario across variants)
    eval_seeds = {sc: 900000 + 1000*i for i, sc in enumerate(scenario_names)}

    t0 = time.time()
    for ep in range(args.episodes):
        sc = scenario_names[ep % len(scenario_names)]
        env = make_env(sc, duration=args.duration, seed=int(episode_seeds[ep]),
                       max_neighbors=MAX_NEIGHBORS)

        obs, info = env.reset()
        obs = norm_obs(obs)
        agent.update_epsilon(ep)

        ep_reward = 0.0
        ep_loss   = []
        steps     = 0
        done      = False

        while not done and steps < args.max_steps_per_episode:
            action = agent.select_action(obs, training=True)
            next_obs, reward, done, _, info = env.step(action)
            next_obs = norm_obs(next_obs)
            buffer.push(obs, action, reward, next_obs, done)
            obs = next_obs
            ep_reward += reward
            steps += 1

            if len(buffer) >= args.batch_size and steps % args.update_every == 0:
                batch = buffer.sample(args.batch_size)
                loss = agent.update(batch)
                ep_loss.append(loss)

        if (ep+1) % args.target_sync_every == 0:
            agent.sync_target()

        # Log training episode
        flat_history['episode'].append(ep+1)
        flat_history['pdr'].append(info['pdr'])
        flat_history['reward'].append(ep_reward)
        flat_history['epsilon'].append(agent.epsilon)
        flat_history['scenario'].append(sc)
        flat_history['loss'].append(float(np.mean(ep_loss)) if ep_loss else 0.0)

        per_scn_train_history[sc]['episode'].append(ep+1)
        per_scn_train_history[sc]['pdr'].append(info['pdr'])
        per_scn_train_history[sc]['reward'].append(ep_reward)

        # PERIODIC GREEDY EVALUATION (every eval_every episodes, one per scenario)
        if (ep+1) % args.eval_every == 0:
            for eval_sc in scenario_names:
                m = evaluate_greedy(agent, eval_sc, eval_seeds[eval_sc],
                                     args.duration, norm_obs,
                                     max_steps=args.max_steps_per_episode)
                per_scn_eval_history[eval_sc]['episode'].append(ep+1)
                per_scn_eval_history[eval_sc]['pdr'].append(m['pdr'])
                per_scn_eval_history[eval_sc]['mean_delay_ms'].append(m['mean_delay_ms'])
                per_scn_eval_history[eval_sc]['mean_hops'].append(m['mean_hops'])
                per_scn_eval_history[eval_sc]['reward'].append(m['reward'])

        if (ep+1) % args.print_every == 0:
            recent_train = flat_history['pdr'][-args.print_every:]
            # If recent eval exists, show it
            eval_str = ""
            for eval_sc in scenario_names:
                if per_scn_eval_history[eval_sc]['pdr']:
                    last_eval = per_scn_eval_history[eval_sc]['pdr'][-1]
                    eval_str += f" {eval_sc[:6]}_eval={last_eval:.3f}"
            print(f"  Ep {ep+1:4d}/{args.episodes} | scn={sc:12s} | "
                  f"train_PDR={info['pdr']:.3f} | recent={np.mean(recent_train):.3f} | "
                  f"eps={agent.epsilon:.3f} |{eval_str}")

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s ({elapsed/args.episodes:.2f}s/ep)")

    history = {
        'flat':       flat_history,
        'train':      per_scn_train_history,
        'eval':       per_scn_eval_history,
    }
    return agent, history


# ─── Cold-start metrics ───────────────────────────────────────────────────────

def cold_start_metrics_per_scenario(per_scn_eval_history):
    """Compute T_50, T_90, AULC, converged_PDR per scenario from EVAL curve."""
    results = {}
    for sc, hist in per_scn_eval_history.items():
        pdr = np.array(hist['pdr'])
        if len(pdr) < 3:
            results[sc] = {}
            continue
        # Converged PDR = average of last 30% of evaluations
        n_final = max(len(pdr) // 3, 1)
        conv_pdr = float(np.mean(pdr[-n_final:]))
        t50_target = 0.5 * conv_pdr
        t90_target = 0.9 * conv_pdr
        episodes = np.array(hist['episode'])
        # T_X = episode at which PDR first reaches X% of converged value
        idx50 = np.argmax(pdr >= t50_target) if (pdr >= t50_target).any() else len(pdr)-1
        idx90 = np.argmax(pdr >= t90_target) if (pdr >= t90_target).any() else len(pdr)-1
        T_50 = int(episodes[idx50])
        T_90 = int(episodes[idx90])
        # AULC over all evaluations
        aulc = float(np.trapezoid(pdr, x=episodes))
        results[sc] = {
            'converged_pdr':    round(conv_pdr, 4),
            'T_50':             T_50,
            'T_90':             T_90,
            'AULC':             round(aulc, 2),
            'final_pdr':        round(float(pdr[-1]), 4),
            'initial_pdr':      round(float(pdr[0]), 4),
            'eval_episodes':    len(pdr),
        }
    return results


def plot_eval_curves_per_scenario(all_histories, out_path, smooth=2):
    """One subplot per scenario, showing eval PDR curves of all variants."""
    scenarios = list(TRAINING_SCENARIOS.keys())
    colors = {'scratch': '#adb5bd', 'warmstart': '#457b9d', 'regime': '#e63946'}

    fig, axes = plt.subplots(1, len(scenarios), figsize=(5*len(scenarios), 4.5),
                              sharey=True)
    if len(scenarios) == 1: axes = [axes]
    for ax, sc in zip(axes, scenarios):
        for name, h in all_histories.items():
            eh = h['eval'].get(sc, {})
            if not eh.get('pdr'): continue
            x = np.array(eh['episode'])
            y = np.array(eh['pdr'])
            ax.plot(x, y, 'o-', color=colors.get(name,'#333'), label=name,
                     linewidth=2, markersize=4, alpha=0.85)
        ax.set_xlabel('Training Episode')
        ax.set_ylabel('Greedy Eval PDR')
        ax.set_title(f'Scenario: {sc}', fontweight='bold')
        ax.set_ylim(0, 1.0)
        ax.grid(alpha=0.3)
        ax.legend(loc='lower right', fontsize=9)
    plt.suptitle('Per-Scenario Learning Curves (greedy evaluation, ε=0)',
                  fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_cold_start_summary(all_metrics, out_path):
    """Bar chart: T_50 / T_90 / AULC per variant per scenario."""
    scenarios = list(TRAINING_SCENARIOS.keys())
    variants  = list(all_metrics.keys())
    colors = {'scratch': '#adb5bd', 'warmstart': '#457b9d', 'regime': '#e63946'}
    metrics_to_plot = [
        ('initial_pdr',   'Initial PDR (ep 1)'),
        ('T_90',          'Episodes to 90% (lower=better)'),
        ('converged_pdr', 'Converged PDR'),
    ]

    fig, axes = plt.subplots(1, len(metrics_to_plot),
                              figsize=(5*len(metrics_to_plot), 4.5))
    for ax, (metric, title) in zip(axes, metrics_to_plot):
        x = np.arange(len(scenarios))
        width = 0.8 / len(variants)
        for i, v in enumerate(variants):
            vals = [all_metrics[v].get(sc, {}).get(metric, 0) for sc in scenarios]
            ax.bar(x + i*width - 0.4 + width/2, vals, width*0.9,
                    label=v, color=colors.get(v, '#333'), alpha=0.9)
        ax.set_xticks(x); ax.set_xticklabels(scenarios, rotation=10, ha='right',
                                              fontsize=9)
        ax.set_title(title, fontweight='bold')
        ax.grid(alpha=0.3, axis='y')
        ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_rollup(all_histories, out_path, smooth=10):
    """Big-picture training-PDR curves (with epsilon-greedy exploration)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    colors = {'scratch': '#adb5bd', 'warmstart': '#457b9d', 'regime': '#e63946'}
    for name, h in all_histories.items():
        flat = h['flat']
        pdr = np.array(flat['pdr'])
        x = np.arange(1, len(pdr)+1)
        # Lightly: raw
        axes[0].plot(x, pdr, color=colors.get(name,'#333'), alpha=0.15)
        # Bold: smoothed
        if len(pdr) >= smooth:
            sm = np.convolve(pdr, np.ones(smooth)/smooth, mode='valid')
            axes[0].plot(x[smooth-1:], sm, color=colors.get(name,'#333'),
                          label=name, linewidth=2)

        # Epsilon over time
        eps = np.array(flat['epsilon'])
        axes[1].plot(x, eps, color=colors.get(name,'#333'), label=name, linewidth=2)

    axes[0].set_xlabel('Episode'); axes[0].set_ylabel('Training PDR')
    axes[0].set_title('Training PDR (with exploration noise)', fontweight='bold')
    axes[0].set_ylim(0, 1.0); axes[0].grid(alpha=0.3); axes[0].legend()

    axes[1].set_xlabel('Episode'); axes[1].set_ylabel('Epsilon')
    axes[1].set_title('Exploration Schedule', fontweight='bold')
    axes[1].grid(alpha=0.3); axes[1].legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mlp',       default='models/mlp_warmstart.pt')
    ap.add_argument('--out_dir',   default='models/rl')
    ap.add_argument('--variant',   default='all',
                    choices=['all', 'scratch', 'warmstart', 'regime'])
    ap.add_argument('--episodes',  type=int, default=300)
    ap.add_argument('--duration',  type=int, default=30)
    ap.add_argument('--max_steps_per_episode', type=int, default=2000)
    ap.add_argument('--batch_size',type=int, default=64)
    ap.add_argument('--buffer_size',type=int, default=50000)
    ap.add_argument('--lr',        type=float, default=1e-4)
    ap.add_argument('--gamma',     type=float, default=0.95)
    ap.add_argument('--eps_start',     type=float, default=1.0,
                    help='Starting epsilon for SCRATCH variant')
    ap.add_argument('--eps_start_warm',type=float, default=0.3,
                    help='Starting epsilon for WARMSTART/REGIME variants')
    ap.add_argument('--eps_end',   type=float, default=0.05)
    ap.add_argument('--eps_decay', type=int,   default=150,
                    help='Episodes over which epsilon decays from start to end')
    ap.add_argument('--target_sync_every', type=int, default=10)
    ap.add_argument('--update_every',      type=int, default=4)
    ap.add_argument('--eval_every',        type=int, default=15,
                    help='Greedy evaluation every N training episodes')
    ap.add_argument('--print_every',       type=int, default=15)
    ap.add_argument('--seed',      type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Device: {device}")

    mlp_bundle = torch.load(args.mlp, weights_only=False)
    print(f"  MLP warmstart bundle: {len(mlp_bundle['feature_cols'])} features")

    variants = ['scratch', 'warmstart', 'regime'] if args.variant == 'all' else [args.variant]
    all_histories = {}
    all_metrics   = {}

    for v in variants:
        agent, hist = train_variant(v, mlp_bundle, args, device)
        all_histories[v] = hist

        # Save model + history
        ckpt_path = os.path.join(args.out_dir, f'dqn_{v}.pt')
        torch.save({'q_net': agent.q_net.state_dict(),
                    'flat_history': hist['flat']}, ckpt_path)
        hist_path = os.path.join(args.out_dir, f'history_{v}.json')
        with open(hist_path, 'w') as f:
            json.dump(hist, f, indent=2,
                      default=lambda o: float(o) if isinstance(o,np.floating)
                                                  else int(o))

        # Per-scenario cold-start metrics
        metrics = cold_start_metrics_per_scenario(hist['eval'])
        all_metrics[v] = metrics
        print(f"\n  {v} per-scenario cold-start metrics:")
        for sc, m in metrics.items():
            if m:
                print(f"    {sc:<14s} init={m['initial_pdr']:.3f}  "
                      f"converged={m['converged_pdr']:.3f}  "
                      f"T_50={m['T_50']}  T_90={m['T_90']}  AULC={m['AULC']}")

    # ── Summary tables ────────────────────────────────────────────────────────
    if len(all_metrics) > 1:
        print(f"\n{'='*70}")
        print("  COLD-START METRICS COMPARISON (greedy eval, per scenario)")
        print(f"{'='*70}")
        for sc in TRAINING_SCENARIOS.keys():
            print(f"\n  {sc}:")
            print(f"    {'Variant':<12} {'Init PDR':>9} {'Conv PDR':>9} "
                  f"{'T_50':>6} {'T_90':>6} {'AULC':>8}")
            print(f"    {'-'*55}")
            for v in variants:
                m = all_metrics[v].get(sc, {})
                if not m: continue
                print(f"    {v:<12} {m['initial_pdr']:>9.3f} {m['converged_pdr']:>9.3f} "
                      f"{m['T_50']:>6} {m['T_90']:>6} {m['AULC']:>8.2f}")

    # Save combined metrics
    with open(os.path.join(args.out_dir, 'cold_start_metrics.json'), 'w') as f:
        json.dump(all_metrics, f, indent=2,
                   default=lambda o: float(o) if isinstance(o,np.floating)
                                                else int(o))

    if len(all_histories) > 1:
        plot_eval_curves_per_scenario(all_histories,
            os.path.join(args.out_dir, 'rl_eval_curves.png'))
        plot_cold_start_summary(all_metrics,
            os.path.join(args.out_dir, 'rl_cold_start_summary.png'))
        plot_training_rollup(all_histories,
            os.path.join(args.out_dir, 'rl_training_rollup.png'))
        print(f"\n  Plots saved to {args.out_dir}")

    print(f"\n{'='*70}")
    print("  PHASE 5 DONE")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
