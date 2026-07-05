"""
train_dqn_full.py  — v2: continual-learning-protected warmstart.

WHY THIS VERSION EXISTS — two flaws found in the v1 results:

FLAW A (catastrophic forgetting): warmstart/regime started with a strong
  cold-start advantage (e.g. sc10: 0.720 PDR at episode 1, vs scratch's 0.46)
  but by the end of 5000 episodes had COLLAPSED below scratch (0.19 vs 0.52).
  Root cause: eps_start_warm=0.30 injected ~30% random actions into the
  replay buffer for the first ~2500 episodes; those noisy transitions,
  combined with lr=1e-4 applied uniformly to the pretrained backbone,
  overwrote the MLP's routing knowledge before the agent had a chance to
  benefit from it. This is the standard failure mode fine-tuning research
  calls "feature distortion" (see Kumar et al. 2022, "Fine-Tuning can
  Distort Pretrained Features") — the fix is the standard remedy: freeze
  the pretrained backbone, let only the new/output parameters adapt first,
  then unfreeze at a much lower learning rate.

FLAW B (silent dropout-at-eval-time bug): QNetwork has Dropout(0.2) but the
  code never called .eval() anywhere — including during greedy evaluation
  and during the target network's bootstrap forward pass. This means
  "greedy" action selection was never actually deterministic, and the TD
  target used for every single gradient update was computed through a
  randomly-masked network. Both networks are now permanently set to .eval()
  mode (dropout becomes a no-op) since a target network must be
  deterministic for stable bootstrapping, and this Q-network doesn't need
  dropout regularisation for its role in DQN.

FIXES APPLIED:
  1. Dropout disabled everywhere (q_net and target_net both .eval() always).
  2. Freeze-then-finetune for warmstart/regime:
       - warmstart: freeze net.0 + net.3 (entire backbone) for
         `freeze_episodes`; only net.6 (output head) trains. After that,
         backbone unfreezes at a 10x lower LR than the head.
       - regime: net.0.weight is column-masked — the 22 pretrained columns
         are protected (snapshot/restore each step) while the 3 new regime
         columns train freely from the start. net.0.bias and net.3 are
         frozen like the warmstart case. At the unfreeze transition, Adam's
         momentum buffers for the previously-protected columns are reset to
         zero so no artificial "catch-up" jump occurs.
  3. eps_start_warm lowered 0.30 -> 0.15 (less buffer contamination).
  4. Separate, shorter eps_decay_warm=1200 (vs scratch's eps_decay=2500) —
     warmstart/regime need less exploration and should reach low-noise
     fine-tuning sooner.
  5. buffer_size lowered 50000 -> 15000. Runs generate ~15-20k transitions
     over 5000 single-packet episodes, so the old 50k buffer never evicted
     anything — every gradient step through the whole run sampled uniformly
     from a mix that was ~50% early-random-exploration noise. A smaller
     buffer gives natural recency weighting.
  6. target_sync_every raised 20 -> 40 for a more stable bootstrap target.

Usage:
    python src\\train_dqn_full.py --variant scratch   --seed 42
    python src\\train_dqn_full.py --variant scratch   --seed 123
    python src\\train_dqn_full.py --variant warmstart --seed 42
    python src\\train_dqn_full.py --variant regime    --seed 42
"""

import os, sys, argparse, time, pickle, json
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl_env_full import (FANETRoutingEnv, FEATURE_COLS, MAX_NEIGHBORS,
                          TRAINING_SCENARIOS, EVAL_SCENARIOS,
                          GENERALIZATION_SCENARIOS, make_env)


class QNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # net[0]=Linear(in,128) net[1]=ReLU net[2]=Dropout
        # net[3]=Linear(128,64) net[4]=ReLU net[5]=Dropout
        # net[6]=Linear(64,1)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),        nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


class ReplayBuffer:
    def __init__(self, capacity=15000):
        self.buf = deque(maxlen=capacity)
    def push(self, obs, action, reward, next_obs, done):
        self.buf.append((obs, action, reward, next_obs, done))
    def sample(self, batch_size):
        idx = np.random.choice(len(self.buf), batch_size, replace=False)
        return [self.buf[i] for i in idx]
    def __len__(self):
        return len(self.buf)


class DQNAgent:
    def __init__(self, input_dim, lr_head=1e-4, lr_backbone=1e-4, gamma=0.95,
                 eps_start=1.0, eps_end=0.05, eps_decay_episodes=2500,
                 device='cpu'):
        self.device = device
        self.input_dim = input_dim
        self.q_net      = QNetwork(input_dim).to(device)
        self.target_net = QNetwork(input_dim).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        # FLAW B fix: dropout must never be active for a DQN target network,
        # and greedy eval must be deterministic. Neither network is ever
        # trained via .train() mode toggling — Dropout(p) is a no-op in
        # .eval() mode, so gradients still flow normally through those
        # layers during backward(); only the random masking is disabled.
        self.q_net.eval()
        self.target_net.eval()

        # Two param groups so warmstart/regime can protect the pretrained
        # backbone with a much lower LR once unfrozen. For scratch,
        # lr_backbone == lr_head (no protection needed — nothing pretrained).
        backbone_params = list(self.q_net.net[0].parameters()) + \
                          list(self.q_net.net[3].parameters())
        head_params     = list(self.q_net.net[6].parameters())
        self.optimizer = torch.optim.Adam([
            {'params': backbone_params, 'lr': lr_backbone},
            {'params': head_params,     'lr': lr_head},
        ])

        self.gamma      = gamma
        self.eps_start  = eps_start
        self.eps_end    = eps_end
        self.eps_decay_episodes = eps_decay_episodes
        self.epsilon    = eps_start

        # Continual-learning protection state (set via protect())
        self.freeze_until_episode = 0     # 0 = never freeze (scratch default)
        self.protect_input_cols   = None  # None = whole-layer freeze; int = column-masked
        self._unfroze_already     = False

    def load_warmstart(self, mlp_state_dict):
        """Direct load if dims match; pad first layer if agent input is larger
        (regime variant: 22 pretrained cols copied, extra regime cols keep
        random init)."""
        src_w0 = mlp_state_dict['net.0.weight']
        mlp_in = src_w0.shape[1]
        if mlp_in == self.input_dim:
            self.q_net.load_state_dict(mlp_state_dict)
            self.target_net.load_state_dict(mlp_state_dict)
            self.q_net.eval(); self.target_net.eval()
            print(f"  Warmstart loaded EXACTLY (input_dim={self.input_dim}, "
                  f"{sum(p.numel() for p in self.q_net.parameters()):,} params)")
            return
        if mlp_in > self.input_dim:
            raise ValueError(f"MLP input_dim ({mlp_in}) > agent ({self.input_dim})")
        own_sd = self.q_net.state_dict()
        tgt_w0 = own_sd['net.0.weight'].clone()
        tgt_w0[:, :mlp_in] = src_w0
        new_sd = dict(mlp_state_dict)
        new_sd['net.0.weight'] = tgt_w0
        self.q_net.load_state_dict(new_sd)
        self.target_net.load_state_dict(new_sd)
        self.q_net.eval(); self.target_net.eval()
        print(f"  Warmstart loaded WITH PADDING: {mlp_in} -> {self.input_dim} "
              f"({self.input_dim - mlp_in} regime dims random-init, "
              f"{mlp_in} pretrained dims copied)")

    def protect(self, freeze_episodes, protect_input_cols=None):
        """
        Configure continual-learning protection. Call once after
        load_warmstart(), before training starts.

        freeze_episodes:    train only the output head for this many episodes.
                             0 disables protection entirely (used for scratch).
        protect_input_cols: None -> freeze net.0 + net.3 entirely (warmstart).
                             int  -> freeze net.0.bias + net.3 entirely, but
                                     keep net.0.weight trainable EXCEPT the
                                     first `protect_input_cols` columns, which
                                     are protected via snapshot/restore each
                                     step (regime variant: protects the 22
                                     pretrained columns, frees the 3 new
                                     regime columns to learn immediately).
        """
        self.freeze_until_episode = freeze_episodes
        self.protect_input_cols   = protect_input_cols
        self._unfroze_already = (freeze_episodes == 0)
        if freeze_episodes == 0:
            return  # scratch path — nothing to freeze

        self.q_net.net[3].weight.requires_grad = False
        self.q_net.net[3].bias.requires_grad   = False
        self.q_net.net[0].bias.requires_grad   = False
        if protect_input_cols is None:
            self.q_net.net[0].weight.requires_grad = False
        # else: net.0.weight stays trainable; column protection happens
        # inside update() via snapshot/restore.

        mode = "whole-layer freeze (net.0+net.3)" if protect_input_cols is None \
               else f"column-masked freeze (net.0.weight[:, :{protect_input_cols}] protected)"
        print(f"  Continual-learning protection ON for {freeze_episodes} episodes: {mode}")

    def _maybe_unfreeze(self, episode):
        if self._unfroze_already or self.freeze_until_episode == 0:
            return
        if episode < self.freeze_until_episode:
            return
        # Transition: unfreeze everything.
        self.q_net.net[3].weight.requires_grad = True
        self.q_net.net[3].bias.requires_grad   = True
        self.q_net.net[0].bias.requires_grad   = True
        if self.protect_input_cols is not None:
            self.q_net.net[0].weight.requires_grad = True
            # Reset Adam momentum for the columns that were being
            # snapshot-restored, so there's no artificial "catch-up" jump
            # from momentum that accumulated while the values were being
            # forcibly held constant.
            p = self.q_net.net[0].weight
            state = self.optimizer.state.get(p, None)
            if state and 'exp_avg' in state:
                state['exp_avg'][:, :self.protect_input_cols] = 0
                state['exp_avg_sq'][:, :self.protect_input_cols] = 0
        else:
            self.q_net.net[0].weight.requires_grad = True
        self._unfroze_already = True
        print(f"  [ep {episode}] Continual-learning protection OFF — "
              f"full backbone now trainable at reduced LR.")

    def update_epsilon(self, episode):
        frac = min(1.0, episode / self.eps_decay_episodes)
        self.epsilon = self.eps_start + frac * (self.eps_end - self.eps_start)

    def select_action(self, obs, training=True):
        mask = obs['mask']
        valid_idx = np.where(mask == 1)[0]
        if len(valid_idx) == 0:
            return 0
        if training and np.random.rand() < self.epsilon:
            return int(np.random.choice(valid_idx))
        with torch.no_grad():
            x = torch.tensor(obs['features'], dtype=torch.float32, device=self.device)
            q = self.q_net(x).cpu().numpy()   # q_net is always .eval() — deterministic
            q[mask == 0] = -np.inf
            return int(np.argmax(q))

    def update(self, batch, episode):
        if len(batch) == 0:
            return 0.0
        self._maybe_unfreeze(episode)

        feats     = torch.tensor(np.stack([b[0]['features'] for b in batch]),
                                  dtype=torch.float32, device=self.device)
        actions   = torch.tensor([b[1] for b in batch], dtype=torch.long, device=self.device)
        rewards   = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=self.device)
        nxt_feats = torch.tensor(np.stack([b[3]['features'] for b in batch]),
                                  dtype=torch.float32, device=self.device)
        nxt_masks = torch.tensor(np.stack([b[3]['mask']     for b in batch]),
                                  dtype=torch.float32, device=self.device)
        dones     = torch.tensor([float(b[4]) for b in batch],
                                  dtype=torch.float32, device=self.device)

        # Column-mask snapshot (regime, still-frozen phase only)
        snapshot = None
        freezing_now = (episode < self.freeze_until_episode) and (self.protect_input_cols is not None)
        if freezing_now:
            snapshot = self.q_net.net[0].weight.data[:, :self.protect_input_cols].clone()

        q_all   = self.q_net(feats)
        q_taken = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            q_next = self.target_net(nxt_feats)   # target_net is always .eval()
            q_next = q_next.masked_fill(nxt_masks == 0, -1e9)
            q_next_max = q_next.max(dim=1).values
            target = rewards + self.gamma * q_next_max * (1.0 - dones)

        loss = nn.functional.smooth_l1_loss(q_taken, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            [p for p in self.q_net.parameters() if p.requires_grad], 1.0)
        self.optimizer.step()

        if freezing_now:
            self.q_net.net[0].weight.data[:, :self.protect_input_cols] = snapshot

        return loss.item()

    def sync_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()


# ─── Batch greedy evaluation (single-packet PDR is 0/1, so average many) ──────

def evaluate_greedy_batch(agent, scenario_name, base_seed, duration, norm_obs,
                          regime_model, n_packets, max_steps):
    delivered = 0
    total_hops = 0
    total_delay = 0.0
    n_delivered_for_stats = 0
    for k in range(n_packets):
        env = make_env(scenario_name, duration=duration, seed=base_seed + k,
                       max_neighbors=MAX_NEIGHBORS, regime_model=regime_model)
        obs, info = env.reset()
        obs = norm_obs(obs)
        done = False
        steps = 0
        while not done and steps < max_steps:
            action = agent.select_action(obs, training=False)
            next_obs, reward, done, _, info = env.step(action)
            obs = norm_obs(next_obs)
            steps += 1
        if info['delivered']:
            delivered += 1
            n_delivered_for_stats += 1
            total_hops  += info['mean_hops']
            total_delay += info['mean_delay_ms']
    pdr = delivered / max(n_packets, 1)
    return {
        'pdr': pdr,
        'mean_hops': total_hops / max(n_delivered_for_stats, 1),
        'mean_delay_ms': total_delay / max(n_delivered_for_stats, 1),
        'n_packets': n_packets,
    }


def train_variant(variant_name, mlp_bundle, regime_bundle, args, device):
    print(f"\n{'='*60}\n  Training: {variant_name}  (seed={args.seed})\n{'='*60}")

    use_regime = (variant_name == 'regime')
    regime_model_for_env = regime_bundle if use_regime else None
    input_dim = len(FEATURE_COLS) + (regime_bundle['k'] if use_regime else 0)

    is_pretrained = variant_name in ('warmstart', 'regime')
    eps_start = args.eps_start_warm if is_pretrained else args.eps_start
    eps_decay = args.eps_decay_warm if is_pretrained else args.eps_decay
    lr_head     = args.lr_head
    lr_backbone = args.lr_backbone if is_pretrained else args.lr_head  # scratch: uniform LR

    agent = DQNAgent(input_dim=input_dim, lr_head=lr_head, lr_backbone=lr_backbone,
                     gamma=args.gamma, eps_start=eps_start, eps_end=args.eps_end,
                     eps_decay_episodes=eps_decay, device=device)

    if is_pretrained:
        agent.load_warmstart(mlp_bundle['model_state_dict'])
        feat_mean = mlp_bundle['feat_mean']
        feat_std  = mlp_bundle['feat_std']
        if use_regime:
            feat_mean = np.concatenate([feat_mean, np.zeros(regime_bundle['k'], dtype=np.float32)])
            feat_std  = np.concatenate([feat_std,  np.ones(regime_bundle['k'],  dtype=np.float32)])
            agent.protect(args.freeze_episodes, protect_input_cols=len(FEATURE_COLS))
        else:
            agent.protect(args.freeze_episodes, protect_input_cols=None)
    else:
        feat_mean = np.zeros(input_dim, dtype=np.float32)
        feat_std  = np.ones(input_dim, dtype=np.float32)
        agent.protect(0)  # scratch: no freezing

    def norm_obs(obs):
        obs = dict(obs)
        obs['features'] = ((obs['features'] - feat_mean) / (feat_std + 1e-8)).astype(np.float32)
        return obs

    buffer = ReplayBuffer(args.buffer_size)
    scenario_names = list(TRAINING_SCENARIOS.keys())
    eval_scenario_names = list(EVAL_SCENARIOS.keys())

    per_scn_eval_history = {sc: {'episode': [], 'pdr': [], 'mean_delay_ms': [],
                                  'mean_hops': []} for sc in eval_scenario_names}
    flat_history = {'episode': [], 'pdr': [], 'reward': [], 'epsilon': [],
                    'scenario': [], 'loss': []}

    ep_seed_rng  = np.random.default_rng(args.seed)
    eval_seed_rng = {sc: np.random.default_rng(900000 + 5000*i)
                     for i, sc in enumerate(eval_scenario_names)}

    t0 = time.time()
    for ep in range(args.episodes):
        sc = scenario_names[ep % len(scenario_names)]
        ep_seed = int(ep_seed_rng.integers(0, 1_000_000))
        env = make_env(sc, duration=args.duration, seed=ep_seed,
                       max_neighbors=MAX_NEIGHBORS, regime_model=regime_model_for_env)

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
                ep_loss.append(agent.update(buffer.sample(args.batch_size), ep))

        if (ep+1) % args.target_sync_every == 0:
            agent.sync_target()

        flat_history['episode'].append(ep+1)
        flat_history['pdr'].append(info['pdr'])
        flat_history['reward'].append(ep_reward)
        flat_history['epsilon'].append(agent.epsilon)
        flat_history['scenario'].append(sc)
        flat_history['loss'].append(float(np.mean(ep_loss)) if ep_loss else 0.0)

        if (ep+1) % args.eval_every == 0:
            for esc in eval_scenario_names:
                eval_base = int(eval_seed_rng[esc].integers(0, 500_000))
                m = evaluate_greedy_batch(agent, esc, eval_base,
                                          args.eval_duration, norm_obs,
                                          regime_model_for_env,
                                          n_packets=args.eval_packets,
                                          max_steps=args.max_steps_per_episode)
                per_scn_eval_history[esc]['episode'].append(ep+1)
                per_scn_eval_history[esc]['pdr'].append(m['pdr'])
                per_scn_eval_history[esc]['mean_delay_ms'].append(m['mean_delay_ms'])
                per_scn_eval_history[esc]['mean_hops'].append(m['mean_hops'])

        if (ep+1) % args.print_every == 0:
            recent = flat_history['pdr'][-args.print_every:]
            eval_str = ""
            for esc in eval_scenario_names:
                if per_scn_eval_history[esc]['pdr']:
                    eval_str += f" {esc}={per_scn_eval_history[esc]['pdr'][-1]:.3f}"
            mins = (time.time() - t0) / 60
            frozen_tag = ""
            if is_pretrained and not agent._unfroze_already:
                frozen_tag = " [FROZEN]"
            print(f"  Ep {ep+1:5d}/{args.episodes} | scn={sc:9s} | "
                  f"deliv_rate(last{args.print_every})={np.mean(recent):.3f} | "
                  f"eps={agent.epsilon:.3f} | eval:{eval_str} | {mins:.1f}min{frozen_tag}")

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s ({elapsed/60:.1f}min, "
          f"{1000*elapsed/args.episodes:.1f}ms/ep)")

    print(f"\n  Generalisation eval (held-out, {args.eval_packets} packets each)...")
    gen_results = {}
    for gsc in GENERALIZATION_SCENARIOS:
        m = evaluate_greedy_batch(agent, gsc, 700000, args.eval_duration, norm_obs,
                                  regime_model_for_env, n_packets=args.eval_packets,
                                  max_steps=args.max_steps_per_episode)
        gen_results[gsc] = {'pdr': round(m['pdr'], 4),
                            'mean_hops': round(m['mean_hops'], 2)}
        print(f"    {gsc}: PDR = {m['pdr']:.3f}")

    return agent, {'flat': flat_history, 'eval': per_scn_eval_history,
                   'generalization': gen_results}


# ─── Cold-start metrics ───────────────────────────────────────────────────────

def cold_start_metrics(per_scn_eval_history):
    results = {}
    for sc, hist in per_scn_eval_history.items():
        pdr = np.array(hist['pdr'])
        if len(pdr) < 3:
            results[sc] = {}
            continue
        n_final = max(len(pdr)//3, 1)
        conv = float(np.mean(pdr[-n_final:]))
        eps_ax = np.array(hist['episode'])
        idx50 = np.argmax(pdr >= 0.5*conv) if (pdr >= 0.5*conv).any() else len(pdr)-1
        idx90 = np.argmax(pdr >= 0.9*conv) if (pdr >= 0.9*conv).any() else len(pdr)-1
        results[sc] = {
            'converged_pdr': round(conv, 4),
            'initial_pdr':   round(float(pdr[0]), 4),
            'final_pdr':     round(float(pdr[-1]), 4),
            'T_50':          int(eps_ax[idx50]),
            'T_90':          int(eps_ax[idx90]),
            'AULC':          round(float(np.trapezoid(pdr, x=eps_ax)), 2),
            'eval_points':   len(pdr),
        }
    return results


def plot_eval_curves(all_histories, out_path, scenario_names):
    colors = {'scratch': '#adb5bd', 'warmstart': '#457b9d', 'regime': '#e63946'}
    fig, axes = plt.subplots(1, len(scenario_names),
                              figsize=(5*len(scenario_names), 4.5), sharey=True)
    if len(scenario_names) == 1: axes = [axes]
    for ax, sc in zip(axes, scenario_names):
        for name, h in all_histories.items():
            eh = h['eval'].get(sc, {})
            if not eh.get('pdr'): continue
            ax.plot(eh['episode'], eh['pdr'], 'o-',
                    color=colors.get(name.split('_seed')[0], '#333'),
                    label=name, linewidth=2, markersize=3, alpha=0.85)
        ax.set_xlabel('Training Episode (packets)')
        ax.set_ylabel('Greedy Eval PDR')
        ax.set_title(f'{sc}', fontweight='bold')
        ax.set_ylim(0, 1.0); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc='lower right')
    plt.suptitle('Cold-Start Learning Curves (batch greedy eval, ε=0, dropout disabled)',
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight'); plt.close()


def plot_training_rollup(all_histories, out_path, smooth=25):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    colors = {'scratch': '#adb5bd', 'warmstart': '#457b9d', 'regime': '#e63946'}
    for name, h in all_histories.items():
        pdr = np.array(h['flat']['pdr'])
        x = np.arange(1, len(pdr)+1)
        col = colors.get(name.split('_seed')[0], '#333')
        if len(pdr) >= smooth:
            sm = np.convolve(pdr, np.ones(smooth)/smooth, mode='valid')
            axes[0].plot(x[smooth-1:], sm, color=col, label=name, linewidth=2)
        axes[1].plot(x, h['flat']['epsilon'], color=col, label=name, linewidth=2)
    axes[0].set_xlabel('Episode'); axes[0].set_ylabel(f'Delivered rate (smoothed/{smooth})')
    axes[0].set_title('Training delivery rate', fontweight='bold')
    axes[0].set_ylim(0,1); axes[0].grid(alpha=0.3); axes[0].legend(fontsize=8)
    axes[1].set_xlabel('Episode'); axes[1].set_ylabel('Epsilon')
    axes[1].set_title('Exploration schedule', fontweight='bold')
    axes[1].grid(alpha=0.3); axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight'); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mlp',        default='models/mlp_warmstart_full.pt')
    ap.add_argument('--regime_pkl', default='models/regime_clustering_full.pkl')
    ap.add_argument('--out_dir',    default='models/rl_full')
    ap.add_argument('--variant',    default='all',
                    choices=['all', 'scratch', 'warmstart', 'regime'])
    ap.add_argument('--episodes',   type=int, default=5000)
    ap.add_argument('--duration',   type=int, default=60)
    ap.add_argument('--eval_duration', type=int, default=60)
    ap.add_argument('--eval_packets',  type=int, default=150)
    ap.add_argument('--max_steps_per_episode', type=int, default=30)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--buffer_size',type=int, default=15000,
                    help='Lowered from 50000 — gives recency weighting since '
                         'runs generate only ~15-20k transitions total.')
    ap.add_argument('--lr_head',     type=float, default=1e-4,
                    help='LR for the output layer (net.6). Used for ALL variants.')
    ap.add_argument('--lr_backbone', type=float, default=1e-5,
                    help='LR for the pretrained backbone (net.0/net.3) AFTER '
                         'unfreezing, for warmstart/regime only. 10x lower than '
                         'lr_head to protect pretrained routing knowledge. '
                         'Scratch ignores this and uses lr_head uniformly.')
    ap.add_argument('--freeze_episodes', type=int, default=1000,
                    help='For warmstart/regime: episodes during which the '
                         'pretrained backbone is frozen and only the head '
                         '(+ new regime columns, if applicable) trains. 0 to '
                         'disable. Scratch always uses 0 regardless of this flag.')
    ap.add_argument('--gamma',      type=float, default=0.95)
    ap.add_argument('--eps_start',      type=float, default=1.0)
    ap.add_argument('--eps_start_warm', type=float, default=0.15,
                    help='Lowered from 0.30 — less random-action contamination '
                         'of the replay buffer while the backbone is protected.')
    ap.add_argument('--eps_end',    type=float, default=0.05)
    ap.add_argument('--eps_decay',  type=int,   default=2500,
                    help='Epsilon decay horizon for SCRATCH.')
    ap.add_argument('--eps_decay_warm', type=int, default=1200,
                    help='Epsilon decay horizon for warmstart/regime — shorter '
                         'since they start with less exploration needed.')
    ap.add_argument('--target_sync_every', type=int, default=40,
                    help='Raised from 20 for a more stable bootstrap target.')
    ap.add_argument('--update_every',      type=int, default=1)
    ap.add_argument('--eval_every',        type=int, default=250)
    ap.add_argument('--print_every',       type=int, default=250)
    ap.add_argument('--seed',       type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # REPRODUCIBILITY FIX: torch's global RNG (controls QNetwork's random
    # initial weights) was never seeded by --seed. This meant two "different
    # seed" scratch runs differed in BOTH environment sampling AND random
    # weight initialization, conflating two separate sources of variance.
    # scratch_123's anomalously poor start (PDR 0.073-0.087 vs scratch_42's
    # 0.347-0.753 on the same task) may have been partly or wholly a bad
    # weight draw rather than genuine seed-to-seed learning variance.
    # Seeding torch here isolates weight-init as controlled-by-seed, same as
    # everything else in the pipeline.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Device: {device}")

    mlp_bundle = torch.load(args.mlp, weights_only=False)
    print(f"  MLP warmstart bundle: {len(mlp_bundle['feature_cols'])} features")
    with open(args.regime_pkl, 'rb') as f:
        regime_bundle = pickle.load(f)
    print(f"  Regime bundle: k={regime_bundle['k']} ({regime_bundle['regime_names']})")

    variants = ['scratch','warmstart','regime'] if args.variant=='all' else [args.variant]
    all_histories = {}
    all_metrics   = {}

    for v in variants:
        agent, hist = train_variant(v, mlp_bundle, regime_bundle, args, device)
        run_key = f"{v}_seed{args.seed}"
        all_histories[run_key] = hist

        torch.save({'q_net': agent.q_net.state_dict(), 'input_dim': agent.input_dim,
                    'variant': v, 'seed': args.seed, 'flat_history': hist['flat']},
                   os.path.join(args.out_dir, f'dqn_{run_key}.pt'))
        with open(os.path.join(args.out_dir, f'history_{run_key}.json'), 'w') as f:
            json.dump(hist, f, indent=2,
                      default=lambda o: float(o) if isinstance(o, np.floating) else int(o))

        m = cold_start_metrics(hist['eval'])
        all_metrics[run_key] = m
        print(f"\n  {run_key} — cold-start metrics (training-scenario eval):")
        for sc, mm in m.items():
            if mm:
                print(f"    {sc}: init={mm['initial_pdr']:.3f} conv={mm['converged_pdr']:.3f} "
                      f"T50={mm['T_50']} T90={mm['T_90']} AULC={mm['AULC']}")

    eval_names = list(EVAL_SCENARIOS.keys())
    gen_names  = list(GENERALIZATION_SCENARIOS.keys())

    if len(all_metrics) > 1:
        print(f"\n{'='*70}\n  COLD-START COMPARISON (training-scenario eval)\n{'='*70}")
        for sc in eval_names:
            print(f"\n  {sc}:")
            print(f"    {'Run':<20}{'Init':>8}{'Conv':>8}{'T_50':>7}{'T_90':>7}{'AULC':>9}")
            for rk in all_metrics:
                mm = all_metrics[rk].get(sc, {})
                if mm:
                    print(f"    {rk:<20}{mm['initial_pdr']:>8.3f}{mm['converged_pdr']:>8.3f}"
                          f"{mm['T_50']:>7}{mm['T_90']:>7}{mm['AULC']:>9.2f}")

        print(f"\n{'='*70}\n  GENERALISATION (held-out scenarios, post-training)\n{'='*70}")
        for sc in gen_names:
            print(f"\n  {sc}:")
            for rk, h in all_histories.items():
                gr = h.get('generalization', {}).get(sc, {})
                if gr:
                    print(f"    {rk:<20} PDR={gr['pdr']:.3f}")

    with open(os.path.join(args.out_dir, f'cold_start_metrics_seed{args.seed}.json'), 'w') as f:
        json.dump(all_metrics, f, indent=2,
                  default=lambda o: float(o) if isinstance(o, np.floating) else int(o))

    if len(all_histories) > 1:
        plot_eval_curves(all_histories,
            os.path.join(args.out_dir, f'rl_eval_curves_seed{args.seed}.png'), eval_names)
        plot_training_rollup(all_histories,
            os.path.join(args.out_dir, f'rl_training_rollup_seed{args.seed}.png'))
        print(f"\n  Plots saved to {args.out_dir}")

    print(f"\n{'='*70}\n  PHASE 5 RUN DONE\n{'='*70}\n")


if __name__ == '__main__':
    main()
