"""
train_dqn_v4c_all.py  — v4c: Dueling + Curriculum + reward shaping (all three combined).

LINEAGE
-------
Exactly train_dqn_v3_ddqn_per.py (Double DQN + Prioritized Experience Replay)
with THREE targeted additions — same three changes as v4a and v4b, combined:

  1. QNetwork -> DuelingQNetwork                          [from v4a, unchanged]
  2. Env import -> rl_env_v4 (link-quality-weighted r_prog) [from v4b, unchanged]
  3. Curriculum scenario sampling                          [from v4b, unchanged]

See train_dqn_v4a_dueling_rs.py and train_dqn_v4b_curriculum_rs.py docstrings
for full detail on each individual change. Nothing new is introduced here beyond
combining the two independently-verified diffs onto the same v3 base.

STAGE 1+2 RESULTS SUMMARY (context for interpreting v4c):
  RS-only:  no effect vs Stage 0 (warmstart or scratch).
  v4a (Dueling+RS): no effect vs Stage 0 (warmstart); scratch within noise.
  v4b (Curriculum+RS): no effect on warmstart; scratch sc10 conv PDR -4%
    (directionally consistent across 2/3 seeds, plausible real cost from
    reduced sc10 training exposure under the curriculum schedule, but n=3
    cannot confirm against ~6% seed noise).
  None of the three individual techniques improved on warmstart-only (Stage 0).
  v4c is unlikely to show a positive combined effect on that basis, but
  completes the ablation table: all 2^3 combinations relevant to the spec
  (baseline, RS-only, Dueling+RS, Curriculum+RS, all-three) will have been run.

v3 base carried forward verbatim: SumTree, PrioritizedReplayBuffer, Double DQN,
freeze-then-finetune, column-masked regime protection, Adam momentum reset,
rotating eval seeds, all hyperparameters. Eval protocol (EVAL_SCENARIOS,
eval seeds, eval_packets, GENERALIZATION_SCENARIOS) untouched throughout.

CHANGES vs v3 in argparse:
  --out_dir default: models/rl_v4c/
  --curriculum_p1 / --curriculum_p2: schedule breakpoints (default 1500 / 3000)
  --print_curriculum_only: print schedule table and exit

Usage:
    python src\\train_dqn_v4c_all.py --print_curriculum_only
    python src\\train_dqn_v4c_all.py --variant scratch   --seed 42
    python src\\train_dqn_v4c_all.py --variant warmstart --seed 42
    (Stage 3 = 3 seeds [42,123,7] x 2 variants = 6 runs)
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
from rl_env_v4 import (FANETRoutingEnv, FEATURE_COLS, MAX_NEIGHBORS,
                        TRAINING_SCENARIOS, EVAL_SCENARIOS,
                        GENERALIZATION_SCENARIOS, make_env)


# ─── v4b: Curriculum scenario sampling ────────────────────────────────────────

TIER_EASY   = ['rwp_sc01', 'rwp_sc02', 'rwp_sc03']
TIER_MEDIUM = ['rwp_sc10', 'rwp_sc04', 'rwp_sc07']
TIER_HARD   = ['rwp_sc15', 'rwp_sc12', 'rwp_sc08']

# Tier-weight anchor points (Easy, Medium, Hard), each summing to 1.0
_W_START = {'easy': 0.80, 'medium': 0.15, 'hard': 0.05}   # ep <= p1
_W_MID   = {'easy': 0.40, 'medium': 0.40, 'hard': 0.20}   # ep == p2
_W_END   = {'easy': 1/3,  'medium': 1/3,  'hard': 1/3}    # ep >= episodes (uniform-over-9)


def curriculum_tier_weights(episode, p1, p2, total_episodes):
    """Piecewise-linear tier weights. Held at _W_START through p1, linearly
    interpolates to _W_MID by p2, then linearly interpolates to _W_END
    (uniform) by total_episodes. Returns dict tier -> weight (sums to 1.0)."""
    if episode <= p1:
        return dict(_W_START)
    if episode <= p2:
        frac = (episode - p1) / max(p2 - p1, 1)
        return {k: _W_START[k] + frac * (_W_MID[k] - _W_START[k]) for k in _W_START}
    frac = min(1.0, (episode - p2) / max(total_episodes - p2, 1))
    return {k: _W_MID[k] + frac * (_W_END[k] - _W_MID[k]) for k in _W_MID}


def curriculum_scenario_probs(episode, p1, p2, total_episodes, scenario_order):
    """Per-scenario probability vector (aligned to scenario_order), splitting
    each tier's weight evenly across its 3 member scenarios."""
    w = curriculum_tier_weights(episode, p1, p2, total_episodes)
    tier_of = {}
    for s in TIER_EASY:   tier_of[s] = 'easy'
    for s in TIER_MEDIUM: tier_of[s] = 'medium'
    for s in TIER_HARD:   tier_of[s] = 'hard'
    tier_size = {'easy': len(TIER_EASY), 'medium': len(TIER_MEDIUM), 'hard': len(TIER_HARD)}
    probs = np.array([w[tier_of[s]] / tier_size[tier_of[s]] for s in scenario_order])
    probs = probs / probs.sum()   # renormalise for float safety
    return probs


def print_curriculum_schedule(p1, p2, total_episodes, scenario_order):
    print("\n" + "=" * 72)
    print("  CURRICULUM SCHEDULE  (tier weights at representative episodes) — v4c")
    print("=" * 72)
    print(f"  Easy   = {TIER_EASY}")
    print(f"  Medium = {TIER_MEDIUM}")
    print(f"  Hard   = {TIER_HARD}")
    print(f"  Breakpoints: p1={p1} (end of 'mostly Easy'), p2={p2} (end of Medium blend), "
          f"total_episodes={total_episodes} (uniform reached)")
    checkpoints = sorted(set([1, p1, (p1+p2)//2, p2, (p2+total_episodes)//2, total_episodes]))
    print(f"\n  {'episode':>8} {'Easy':>8} {'Medium':>8} {'Hard':>8}   (tier weights)")
    for ep in checkpoints:
        w = curriculum_tier_weights(ep, p1, p2, total_episodes)
        print(f"  {ep:>8} {w['easy']:>8.3f} {w['medium']:>8.3f} {w['hard']:>8.3f}")
    print(f"\n  Per-scenario probability at same checkpoints:")
    header = "  {:>8}".format("episode") + "".join(f"{s.replace('rwp_',''):>9}" for s in scenario_order)
    print(header)
    for ep in checkpoints:
        probs = curriculum_scenario_probs(ep, p1, p2, total_episodes, scenario_order)
        print(f"  {ep:>8}" + "".join(f"{p:>9.3f}" for p in probs))
    print("=" * 72 + "\n")


class DuelingQNetwork(nn.Module):
    """Shared backbone matches MLP net.0/net.3 for warmstart transfer; two heads
    output V and A per candidate slot. forward() returns (V, A) SEPARATELY — the
    mask-aware Q = V + A - mean_valid(A) is computed by the caller (see
    DQNAgent._combine_q) because averaging A requires the valid-candidate mask.
    Verbatim from train_dqn_v4a_dueling_rs.py."""
    def __init__(self, input_dim):
        super().__init__()
        # shared[0]=Linear(in,128) shared[1]=ReLU shared[2]=Dropout
        # shared[3]=Linear(128,64) shared[4]=ReLU
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),        nn.ReLU(),
        )
        self.value_head     = nn.Linear(64, 1)   # V(s)  — always random init
        self.advantage_head = nn.Linear(64, 1)   # A(s,a) — always random init

    def forward(self, x):
        h = self.shared(x)
        return self.value_head(h), self.advantage_head(h)   # (V, A), no combination here


# ─── Prioritized Experience Replay (SumTree) ─────────────────────────────────

class SumTree:
    """Binary tree: leaves hold priorities, internals hold sums.
    O(log n) sampling and update."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree  = np.zeros(2 * capacity)
        self.data  = [None] * capacity
        self.ptr   = 0
        self.size  = 0

    def _propagate(self, idx, delta):
        parent = (idx - 1) // 2
        self.tree[parent] += delta
        if parent != 0:
            self._propagate(parent, delta)

    def _retrieve(self, idx, s):
        left  = 2 * idx + 1
        right = left + 1
        if left >= 2 * self.capacity or right >= 2 * self.capacity:
            return idx
        return self._retrieve(left, s) if s <= self.tree[left] \
               else self._retrieve(right, s - self.tree[left])

    @property
    def total(self):
        return float(self.tree[0])

    def add(self, priority, data):
        leaf = self.ptr + self.capacity - 1
        self.data[self.ptr] = data
        self.update(leaf, priority)
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, leaf_idx, priority):
        delta = priority - self.tree[leaf_idx]
        self.tree[leaf_idx] = priority
        self._propagate(leaf_idx, delta)

    def get(self, s):
        leaf_idx = self._retrieve(0, s)
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    def __len__(self):
        return self.size


class PrioritizedReplayBuffer:
    """PER with SumTree. alpha controls prioritization strength,
    beta is the IS-correction exponent (annealed from beta_start to 1.0)."""
    def __init__(self, capacity=15000, alpha=0.6, beta_start=0.4, eps_prio=1e-6):
        self.tree       = SumTree(capacity)
        self.capacity   = capacity
        self.alpha      = alpha
        self.beta_start = beta_start
        self.eps_prio   = eps_prio
        self.max_prio   = 1.0

    def push(self, obs, action, reward, next_obs, done):
        # New transitions get max priority so they are sampled at least once
        self.tree.add(self.max_prio ** self.alpha,
                      (obs, action, reward, next_obs, done))

    def sample(self, batch_size, beta):
        indices, batch, weights = [], [], []
        segment = self.tree.total / batch_size
        for i in range(batch_size):
            s = np.random.uniform(segment * i, segment * (i + 1))
            leaf_idx, priority, data = self.tree.get(s)
            # Edge case: tree not yet full
            if data is None:
                s2 = np.random.uniform(0, self.tree.total)
                leaf_idx, priority, data = self.tree.get(s2)
            indices.append(leaf_idx)
            batch.append(data)
            prob = priority / (self.tree.total + 1e-10)
            weights.append((self.tree.size * prob) ** (-beta))
        weights = np.array(weights, dtype=np.float32)
        weights /= (weights.max() + 1e-10)
        return batch, indices, weights

    def update_priorities(self, indices, td_errors):
        for idx, err in zip(indices, td_errors):
            prio = (abs(float(err)) + self.eps_prio) ** self.alpha
            self.max_prio = max(self.max_prio, prio)
            self.tree.update(idx, prio)

    def beta_by_episode(self, episode, total_episodes):
        frac = min(1.0, episode / total_episodes)
        return self.beta_start + frac * (1.0 - self.beta_start)

    def __len__(self):
        return self.tree.size


class DQNAgent:
    def __init__(self, input_dim, lr_head=1e-4, lr_backbone=1e-4, gamma=0.95,
                 eps_start=1.0, eps_end=0.05, eps_decay_episodes=2500,
                 device='cpu'):
        self.device = device
        self.input_dim = input_dim
        self.q_net      = DuelingQNetwork(input_dim).to(device)
        self.target_net = DuelingQNetwork(input_dim).to(device)
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
        backbone_params = list(self.q_net.shared[0].parameters()) + \
                          list(self.q_net.shared[3].parameters())
        head_params     = list(self.q_net.value_head.parameters()) + \
                          list(self.q_net.advantage_head.parameters())
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

    # v4a/v4c: mask-aware dueling Q combination
    def _combine_q(self, net, feats, masks):
        """feats: (..., MN, feat_dim); masks: (..., MN) with 1=valid.
        Returns Q: (..., MN) = V + A - mean_over_valid(A)."""
        V, A = net(feats)                     # each (..., MN, 1)
        V = V.squeeze(-1); A = A.squeeze(-1)  # (..., MN)
        mf = (masks == 1).float()
        n_valid = mf.sum(dim=-1, keepdim=True).clamp(min=1)
        A_mean  = (A * mf).sum(dim=-1, keepdim=True) / n_valid
        return V + A - A_mean

    def load_warmstart(self, mlp_state_dict):
        """Map MLP backbone -> shared[0]/shared[3]. Both dueling heads keep their
        random init (the MLP's single net.6 head is intentionally not reused).
        Regime variant: pad shared[0].weight — copy the 22 pretrained input
        columns, leave the extra regime columns at random init.
        Verbatim mapping logic from train_dqn_v4a_dueling_rs.py."""
        src_w0 = mlp_state_dict['net.0.weight']
        src_b0 = mlp_state_dict['net.0.bias']
        src_w3 = mlp_state_dict['net.3.weight']
        src_b3 = mlp_state_dict['net.3.bias']
        mlp_in = src_w0.shape[1]
        if mlp_in > self.input_dim:
            raise ValueError(f"MLP input_dim ({mlp_in}) > agent ({self.input_dim})")

        own = self.q_net.state_dict()
        tgt_w0 = own['shared.0.weight'].clone()
        if mlp_in == self.input_dim:
            tgt_w0 = src_w0.clone()
            pad_msg = "EXACTLY"
        else:
            tgt_w0[:, :mlp_in] = src_w0
            pad_msg = f"WITH PADDING ({mlp_in}->{self.input_dim}, " \
                      f"{self.input_dim - mlp_in} regime cols random-init)"

        new_sd = dict(own)
        new_sd['shared.0.weight'] = tgt_w0
        new_sd['shared.0.bias']   = src_b0.clone()
        new_sd['shared.3.weight'] = src_w3.clone()
        new_sd['shared.3.bias']   = src_b3.clone()
        # value_head / advantage_head left at their existing random init.
        self.q_net.load_state_dict(new_sd)
        self.target_net.load_state_dict(new_sd)
        self.q_net.eval(); self.target_net.eval()
        print(f"  Warmstart backbone loaded {pad_msg}; both dueling heads random-init "
              f"({sum(p.numel() for p in self.q_net.parameters()):,} params)")

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

        self.q_net.shared[3].weight.requires_grad = False
        self.q_net.shared[3].bias.requires_grad   = False
        self.q_net.shared[0].bias.requires_grad   = False
        if protect_input_cols is None:
            self.q_net.shared[0].weight.requires_grad = False
        # else: shared[0].weight stays trainable; column protection happens
        # inside update() via snapshot/restore.

        mode = "whole-layer freeze (shared[0]+shared[3])" if protect_input_cols is None \
               else f"column-masked freeze (shared[0].weight[:, :{protect_input_cols}] protected)"
        print(f"  Continual-learning protection ON for {freeze_episodes} episodes: {mode}")

    def _maybe_unfreeze(self, episode):
        if self._unfroze_already or self.freeze_until_episode == 0:
            return
        if episode < self.freeze_until_episode:
            return
        # Transition: unfreeze everything.
        self.q_net.shared[3].weight.requires_grad = True
        self.q_net.shared[3].bias.requires_grad   = True
        self.q_net.shared[0].bias.requires_grad   = True
        if self.protect_input_cols is not None:
            self.q_net.shared[0].weight.requires_grad = True
            # Reset Adam momentum for the columns that were being
            # snapshot-restored, so there's no artificial "catch-up" jump
            # from momentum that accumulated while the values were being
            # forcibly held constant.
            p = self.q_net.shared[0].weight
            state = self.optimizer.state.get(p, None)
            if state and 'exp_avg' in state:
                state['exp_avg'][:, :self.protect_input_cols] = 0
                state['exp_avg_sq'][:, :self.protect_input_cols] = 0
        else:
            self.q_net.shared[0].weight.requires_grad = True
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
            feats = torch.tensor(obs['features'][None], dtype=torch.float32,
                                 device=self.device)          # (1, MN, feat)
            masks = torch.tensor(mask[None], dtype=torch.float32,
                                 device=self.device)          # (1, MN)
            q = self._combine_q(self.q_net, feats, masks)[0].cpu().numpy()  # (MN,)
            q[mask == 0] = -np.inf
            return int(np.argmax(q))

    def update(self, batch, episode, is_weights=None):
        if len(batch) == 0:
            return 0.0
        self._maybe_unfreeze(episode)

        feats     = torch.tensor(np.stack([b[0]['features'] for b in batch]),
                                  dtype=torch.float32, device=self.device)
        cur_masks = torch.tensor(np.stack([b[0]['mask'] for b in batch]),
                                 dtype=torch.float32, device=self.device)   # v4c: needed for dueling
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
            snapshot = self.q_net.shared[0].weight.data[:, :self.protect_input_cols].clone()

        q_all   = self._combine_q(self.q_net, feats, cur_masks)
        q_taken = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Double DQN: q_net selects the action, target_net evaluates it.
            # Vanilla DQN uses target_net for both → maximisation bias.
            # Double DQN: a* = argmax_{a'} Q_online(s', a')   (selection)
            #             target = r + gamma * Q_target(s', a*)  (evaluation)
            q_next_online = self._combine_q(self.q_net, nxt_feats, nxt_masks)  # selection
            q_next_online = q_next_online.masked_fill(nxt_masks == 0, -1e9)
            best_actions  = q_next_online.argmax(dim=1, keepdim=True)  # a*

            q_next_target = self._combine_q(self.target_net, nxt_feats, nxt_masks)  # evaluation
            q_next_target = q_next_target.masked_fill(nxt_masks == 0, -1e9)
            q_next_eval   = q_next_target.gather(1, best_actions).squeeze(1)

            target = rewards + self.gamma * q_next_eval * (1.0 - dones)

        td_errors = (q_taken.detach() - target).abs().cpu().numpy()  # for PER

        # IS-weighted loss (weights=None → uniform, for vanilla buffer compat)
        if is_weights is not None:
            wt = torch.tensor(is_weights, dtype=torch.float32, device=self.device)
            loss = (wt * nn.functional.smooth_l1_loss(q_taken, target,
                                                       reduction='none')).mean()
        else:
            loss = nn.functional.smooth_l1_loss(q_taken, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            [p for p in self.q_net.parameters() if p.requires_grad], 1.0)
        self.optimizer.step()

        if freezing_now:
            self.q_net.shared[0].weight.data[:, :self.protect_input_cols] = snapshot

        return loss.item(), td_errors

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

    buffer = PrioritizedReplayBuffer(
        capacity   = args.buffer_size,
        alpha      = args.per_alpha,
        beta_start = args.per_beta_start,
    )
    scenario_names = list(TRAINING_SCENARIOS.keys())
    eval_scenario_names = list(EVAL_SCENARIOS.keys())

    # v4b: dedicated curriculum-sampling RNG, decorrelated from ep_seed_rng
    # (drives env seeds) and from PER's global np.random stream (buffer sampling
    # + epsilon-greedy), so switching to curriculum sampling doesn't perturb
    # either of those reproducibility chains.
    curriculum_rng = np.random.default_rng(args.seed + 500000)

    per_scn_eval_history = {sc: {'episode': [], 'pdr': [], 'mean_delay_ms': [],
                                  'mean_hops': []} for sc in eval_scenario_names}
    flat_history = {'episode': [], 'pdr': [], 'reward': [], 'epsilon': [],
                    'scenario': [], 'loss': []}

    ep_seed_rng  = np.random.default_rng(args.seed)
    eval_seed_rng = {sc: np.random.default_rng(900000 + 5000*i)
                     for i, sc in enumerate(eval_scenario_names)}

    t0 = time.time()
    for ep in range(args.episodes):
        # v4b: curriculum-weighted draw replaces `scenario_names[ep % len(scenario_names)]`.
        # Eval protocol (EVAL_SCENARIOS, eval seeds, eval_packets) is untouched below —
        # only which TRAINING scenario is sampled this episode changes.
        probs = curriculum_scenario_probs(ep + 1, args.curriculum_p1, args.curriculum_p2,
                                          args.episodes, scenario_names)
        sc = curriculum_rng.choice(scenario_names, p=probs)
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
                beta = buffer.beta_by_episode(ep, args.episodes)
                batch, per_indices, is_weights = buffer.sample(args.batch_size, beta)
                loss_val, td_errors = agent.update(batch, ep, is_weights=is_weights)
                buffer.update_priorities(per_indices, td_errors)
                ep_loss.append(loss_val)

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
    ap.add_argument('--mlp',        default='results/checkpoints/mlp_warmstart_full.pt')
    ap.add_argument('--regime_pkl', default='results/checkpoints/regime_clustering_full.pkl')
    ap.add_argument('--out_dir',    default='models/rl_v4c')
    ap.add_argument('--variant',    default='all',
                    choices=['all', 'scratch', 'warmstart', 'regime'])
    ap.add_argument('--episodes',   type=int, default=5000)
    ap.add_argument('--duration',   type=int, default=60)
    ap.add_argument('--eval_duration', type=int, default=60)
    ap.add_argument('--eval_packets',  type=int, default=150)
    ap.add_argument('--max_steps_per_episode', type=int, default=30)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--buffer_size',type=int, default=15000,
                    help='PER buffer capacity. Lowered from 50000 — recency weighting.')
    ap.add_argument('--per_alpha',      type=float, default=0.6,
                    help='PER prioritization exponent (0=uniform, 1=full). '
                         'Standard value: 0.6 (Schaul et al. 2016).')
    ap.add_argument('--per_beta_start', type=float, default=0.4,
                    help='PER IS-weight exponent start. Annealed linearly to 1.0. '
                         'Standard value: 0.4.')
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
    ap.add_argument('--curriculum_p1', type=int, default=1500,
                    help='v4b: episode through which tier weights are held at '
                         '"mostly Easy" (0.80/0.15/0.05).')
    ap.add_argument('--curriculum_p2', type=int, default=3000,
                    help='v4b: episode by which tier weights reach the Medium-blend '
                         'point (0.40/0.40/0.20). Weights then blend on to uniform '
                         '(0.333/0.333/0.333) by --episodes.')
    ap.add_argument('--print_curriculum_only', action='store_true',
                    help='v4b: print the full curriculum schedule table and exit '
                         '(no training, no MLP/regime bundle loading needed).')
    args = ap.parse_args()

    if args.print_curriculum_only:
        print_curriculum_schedule(args.curriculum_p1, args.curriculum_p2,
                                  args.episodes, list(TRAINING_SCENARIOS.keys()))
        return

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

    print_curriculum_schedule(args.curriculum_p1, args.curriculum_p2,
                              args.episodes, list(TRAINING_SCENARIOS.keys()))

    variants = ['scratch','warmstart','regime'] if args.variant=='all' else [args.variant]
    all_histories = {}
    all_metrics   = {}

    for v in variants:
        agent, hist = train_variant(v, mlp_bundle, regime_bundle, args, device)
        run_key = f"{v}_seed{args.seed}"
        all_histories[run_key] = hist

        torch.save({'q_net': agent.q_net.state_dict(), 'input_dim': agent.input_dim,
                    'variant': v, 'seed': args.seed, 'arch': 'dueling',
                    'flat_history': hist['flat']},
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
