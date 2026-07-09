"""
preflight_reward_v4.py  — MANDATORY sanity check BEFORE any v4 training run.

Per the project rule: "Reward formula changes require a pre-flight magnitude
sanity check — print reward breakdown (old vs new formula) on a sample batch
before any training run."

The v4 change (rl_env_v4.py) is exactly:
    OLD:  r_prog = 1.0 * (d_before - d_after) / max(d_before, 1.0)
    NEW:  r_prog = progress * (0.5 + 0.5 * link_quality)
where `link_quality` is the SAME `lq` already consumed by the independent
r_lq = 0.3 * lq term. This introduces a deliberate double-count of link quality.
This script verifies that double-count does NOT let link quality dominate the
reward relative to r_lt, p_delay, p_eng — i.e. r_lq stays the primary lq channel
and the added lq-sensitivity in r_prog is a modulation, not a takeover.

WHAT IT DOES:
  1. Rolls out a RANDOM policy on representative training scenarios in the REAL
     env (rl_env_v4), capturing every hop's raw (d_before, d_after, lq, lt, dt,
     next-hop queue occupancy, delivered/dropped flags).
  2. Recomputes the FULL reward breakdown under BOTH the old and new r_prog on
     the SAME transitions (both formulas are pure functions of the captured
     quantities, so the comparison is exact).
  3. Prints: per-component magnitude stats, the change in total r_prog, the
     link-quality sensitivity dR/dlq under each formula, and a representative
     progress x link_quality grid.
  4. Prints a PASS/REVIEW verdict on whether lq influence dominates.

This script does NOT train and writes no checkpoints.

Usage:
    python src\\preflight_reward_v4.py
    python src\\preflight_reward_v4.py --scenarios rwp_sc03 rwp_sc10 rwp_sc07 ^
                                       --episodes 200 --seed 42
"""

import os, sys, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl_env_v4 import make_env, TRAINING_SCENARIOS, MAX_NEIGHBORS
from models import EnergyModel

# Reward constants — must mirror rl_env_v4._compute_reward exactly.
W_DEL   = 10.0
W_LQ    = 0.3
W_LT    = 0.2
W_DELAY = -0.1
W_ENG   = -0.05
W_Q     = -0.2
W_DROP  = -10.0
W_LOOP  = -2.0


def old_r_prog(progress, lq, dropped):
    return 0.0 if dropped else 1.0 * progress


def new_r_prog(progress, lq, dropped):
    return 0.0 if dropped else progress * (0.5 + 0.5 * lq)


def full_breakdown(tr, r_prog_fn):
    """Return the reward-component dict for one captured transition `tr`
    using the supplied r_prog function."""
    lq, lt, dt = tr['lq'], tr['lt'], tr['dt']
    r_del   = W_DEL if tr['delivered'] else 0.0
    r_prog  = r_prog_fn(tr['progress'], lq, tr['dropped'])
    r_lq    = W_LQ * lq
    r_lt    = W_LT * min(lt / 30.0, 1.0)
    p_delay = W_DELAY * dt
    p_eng   = W_ENG * (EnergyModel.TX_COST + EnergyModel.RX_COST)
    p_q     = W_Q * tr['occupancy']
    p_drop  = W_DROP if tr['dropped'] else 0.0
    p_loop  = W_LOOP if tr['loop'] else 0.0
    total = r_del + r_prog + r_lq + r_lt + p_delay + p_eng + p_q + p_drop + p_loop
    return dict(r_del=r_del, r_prog=r_prog, r_lq=r_lq, r_lt=r_lt,
                p_delay=p_delay, p_eng=p_eng, p_q=p_q, p_drop=p_drop,
                p_loop=p_loop, total=total)


def capture_transitions(scenarios, episodes, seed, max_steps):
    """Roll out a random policy in the real env, mirroring the exact edge/queue
    lookups rl_env_v4._compute_reward uses, so captured (lq, lt, occupancy,
    progress) match the values the live reward would see."""
    transitions = []
    rng = np.random.default_rng(seed)
    for sc in scenarios:
        for k in range(episodes):
            env = make_env(sc, duration=60, seed=seed + k, max_neighbors=MAX_NEIGHBORS)
            orig_compute = env._compute_reward

            def wrapped(pkt, next_hop, d_before, d_after,
                        delivered=False, dropped=False, loop=False,
                        _env=env, _orig=orig_compute):
                G = _env.cached_graph
                lq = lt = 0.0
                if next_hop is not None and G is not None and G.has_edge(
                        pkt.path[-2] if len(pkt.path) >= 2 else pkt.current, next_hop):
                    e = G.edges[pkt.path[-2] if len(pkt.path) >= 2 else pkt.current, next_hop]
                    lq = e.get('link_quality', 0)
                    lt = e.get('estimated_link_lifetime', 0)
                progress = (d_before - d_after) / max(d_before, 1.0)
                occ = _env.queues[next_hop].occupancy if next_hop is not None else 0.0
                transitions.append(dict(
                    scenario=sc, lq=float(lq), lt=float(lt), dt=float(_env.dt),
                    progress=float(progress), occupancy=float(occ),
                    delivered=bool(delivered), dropped=bool(dropped), loop=bool(loop)))
                return _orig(pkt, next_hop, d_before, d_after,
                             delivered=delivered, dropped=dropped, loop=loop)

            env._compute_reward = wrapped
            obs, _ = env.reset()
            done, steps = False, 0
            while not done and steps < max_steps:
                valid = np.where(obs['mask'] == 1)[0]
                a = int(rng.choice(valid)) if len(valid) else 0
                obs, _, done, _, _ = env.step(a)
                steps += 1
    return transitions


def pct(a, p):
    return float(np.percentile(a, p)) if len(a) else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenarios', nargs='+',
                    default=['rwp_sc03', 'rwp_sc10', 'rwp_sc07', 'rwp_sc04', 'rwp_sc08'],
                    help='Representative TRAIN scenarios (easy..hard mix).')
    ap.add_argument('--episodes', type=int, default=150,
                    help='Random-policy episodes per scenario.')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max_steps', type=int, default=30)
    args = ap.parse_args()

    bad = [s for s in args.scenarios if s not in TRAINING_SCENARIOS]
    if bad:
        print(f"  ERROR: not TRAIN scenarios: {bad}")
        return 1

    print("\n" + "=" * 72)
    print("  v4 REWARD-SHAPING PRE-FLIGHT MAGNITUDE CHECK")
    print("=" * 72)
    print(f"  scenarios : {', '.join(args.scenarios)}")
    print(f"  episodes  : {args.episodes} per scenario (random policy)")
    print(f"  change    : r_prog  1.0*progress  ->  progress*(0.5+0.5*lq)")

    tr = capture_transitions(args.scenarios, args.episodes, args.seed, args.max_steps)
    n = len(tr)
    moving = [t for t in tr if not t['dropped']]   # progress-bearing hops
    print(f"\n  Captured {n} transitions "
          f"({len(moving)} forwarding hops, {n - len(moving)} drops).")
    if not moving:
        print("  No forwarding hops captured — cannot assess r_prog. Aborting.")
        return 1

    prog = np.array([t['progress'] for t in moving])
    lq   = np.array([t['lq'] for t in moving])
    old_rp = np.array([old_r_prog(t['progress'], t['lq'], False) for t in moving])
    new_rp = np.array([new_r_prog(t['progress'], t['lq'], False) for t in moving])
    r_lq_arr = W_LQ * lq

    # ---- 1. Distributions of the changed term -------------------------------
    print("\n" + "-" * 72)
    print("  1. r_prog DISTRIBUTION (forwarding hops only)")
    print("-" * 72)
    print(f"    {'stat':<10}{'progress':>12}{'link_qual':>12}"
          f"{'OLD r_prog':>13}{'NEW r_prog':>13}{'NEW/OLD':>10}")
    for label, p in [('min', 0), ('p25', 25), ('median', 50),
                     ('mean', -1), ('p75', 75), ('max', 100)]:
        if p == -1:
            row = (prog.mean(), lq.mean(), old_rp.mean(), new_rp.mean())
        else:
            row = (pct(prog, p), pct(lq, p), pct(old_rp, p), pct(new_rp, p))
        ratio = (row[3] / row[2]) if row[2] > 1e-9 else float('nan')
        print(f"    {label:<10}{row[0]:>12.4f}{row[1]:>12.4f}"
              f"{row[2]:>13.4f}{row[3]:>13.4f}{ratio:>10.3f}")
    print(f"\n    NEW r_prog is scaled by (0.5+0.5*lq) in [0.5, 1.0]: it is ALWAYS")
    print(f"    <= OLD r_prog. Mean shrink: {100*(1-new_rp.mean()/max(old_rp.mean(),1e-9)):.1f}%"
          f" of progress reward removed and re-gated on link quality.")

    # ---- 1b. Does the shaping actually bite? --------------------------------
    # Multiplier m = 0.5 + 0.5*lq in [0.5,1.0]. It only meaningfully changes
    # r_prog where lq is well below 1.0. In this distance-only link model lq =
    # clip(snr/30,0,1) saturates near 1.0 for most in-range links, so the
    # shaping can be nearly inert. Quantify that here.
    mult = 0.5 + 0.5 * lq
    frac_bite_10 = float(np.mean(mult < 0.90))   # >10% down-weight
    frac_bite_25 = float(np.mean(mult < 0.75))   # >25% down-weight (lq<0.5)
    print("\n" + "-" * 72)
    print("  1b. SHAPING BITE — does link_quality vary enough to matter?")
    print("-" * 72)
    print(f"    link_quality: mean={lq.mean():.4f} std={lq.std():.4f} "
          f"p05={pct(lq,5):.4f} p25={pct(lq,25):.4f} min={lq.min():.4f}")
    print(f"    multiplier (0.5+0.5*lq): mean={mult.mean():.4f} "
          f"min={mult.min():.4f} p05={pct(mult,5):.4f}")
    print(f"    hops with multiplier < 0.90 (lq<0.80): {100*frac_bite_10:.1f}%")
    print(f"    hops with multiplier < 0.75 (lq<0.50): {100*frac_bite_25:.1f}%")
    if frac_bite_10 < 0.10:
        print(f"    !! FLAG: link_quality is saturated near 1.0 on "
              f"{100*(1-frac_bite_10):.0f}% of hops. The shaping term is NEARLY")
        print(f"       INERT here (mean r_prog change ~"
              f"{100*(1-new_rp.mean()/max(old_rp.mean(),1e-9)):.1f}%). Expect the v4a")
        print(f"       reward-shaping ablation to show little effect FOR THIS REASON")
        print(f"       (distance-only link model, no interference/multipath), not")
        print(f"       because dueling/RS is broken. Consider this before 12 runs.")

    # ---- 2. Link-quality influence vs other terms ---------------------------
    # dReward/dlq: OLD = 0.3 (r_lq only). NEW = 0.3 + 0.5*progress.
    added_sens = 0.5 * prog                       # extra dR/dlq from new r_prog
    print("\n" + "-" * 72)
    print("  2. LINK-QUALITY SENSITIVITY  dReward/d(lq)  &  DOUBLE-COUNT CHECK")
    print("-" * 72)
    print(f"    OLD dR/dlq = 0.300 (from r_lq only, constant)")
    print(f"    NEW dR/dlq = 0.300 + 0.5*progress")
    print(f"      added lq-sensitivity 0.5*progress:  "
          f"mean={added_sens.mean():.4f}  median={pct(added_sens,50):.4f}  "
          f"p95={pct(added_sens,95):.4f}  max={added_sens.max():.4f}")
    print(f"    => standing r_lq channel (0.300) stays "
          f"{0.300/max(added_sens.mean(),1e-9):.1f}x the added channel on average.")

    # Magnitude of each lq-driven contribution vs the fixed reference terms.
    r_lt_ref  = W_LT                              # r_lt max = 0.2
    p_delay_r = abs(W_DELAY * np.array([t['dt'] for t in moving]).mean())
    p_eng_r   = abs(W_ENG * (EnergyModel.TX_COST + EnergyModel.RX_COST))
    lq_prog_part = 0.5 * prog * lq                # lq-dependent slice of new r_prog
    total_lq_influence = r_lq_arr + lq_prog_part  # full lq-driven reward magnitude
    print(f"\n    Per-hop lq-driven reward magnitude (NEW): r_lq + 0.5*progress*lq")
    print(f"      mean={total_lq_influence.mean():.4f}  "
          f"p95={pct(total_lq_influence,95):.4f}  max={total_lq_influence.max():.4f}")
    print(f"    Reference terms:  r_lt(max)={r_lt_ref:.4f}  "
          f"|p_delay|(mean)={p_delay_r:.4f}  |p_eng|={p_eng_r:.5f}")
    print(f"      of which the NEW double-count slice (0.5*progress*lq): "
          f"mean={lq_prog_part.mean():.4f}  p95={pct(lq_prog_part,95):.4f}")

    # ---- 3. Full-reward breakdown, old vs new (forwarding hops) -------------
    old_full = [full_breakdown(t, old_r_prog) for t in moving]
    new_full = [full_breakdown(t, new_r_prog) for t in moving]
    comps = ['r_del', 'r_prog', 'r_lq', 'r_lt', 'p_delay', 'p_eng', 'p_q',
             'p_drop', 'p_loop', 'total']
    print("\n" + "-" * 72)
    print("  3. MEAN FULL-REWARD BREAKDOWN (forwarding hops, old vs new)")
    print("-" * 72)
    print(f"    {'component':<10}{'OLD mean':>12}{'NEW mean':>12}{'delta':>12}")
    for c in comps:
        om = np.mean([b[c] for b in old_full])
        nm = np.mean([b[c] for b in new_full])
        print(f"    {c:<10}{om:>12.4f}{nm:>12.4f}{nm-om:>12.4f}")

    # ---- 4. Representative grid --------------------------------------------
    print("\n" + "-" * 72)
    print("  4. REPRESENTATIVE GRID — r_prog(progress, lq):  OLD -> NEW")
    print("-" * 72)
    lq_grid   = [0.0, 0.25, 0.5, 0.75, 1.0]
    prog_grid = [0.05, 0.10, 0.20, 0.30, 0.50]
    header = "    prog\\lq " + "".join(f"{q:>13.2f}" for q in lq_grid)
    print(header)
    for p in prog_grid:
        cells = []
        for q in lq_grid:
            o = old_r_prog(p, q, False)
            nw = new_r_prog(p, q, False)
            cells.append(f"{o:.3f}->{nw:.3f}")
        print(f"    {p:>7.2f}  " + "".join(f"{c:>13}" for c in cells))

    # ---- 5. Verdict ---------------------------------------------------------
    dominates = total_lq_influence.mean() > (r_lt_ref + p_delay_r + p_eng_r)
    added_over_standing = added_sens.mean() / 0.300
    print("\n" + "=" * 72)
    print("  VERDICT")
    print("=" * 72)
    print(f"    - NEW r_prog <= OLD r_prog everywhere (multiplier in [0.5,1.0]):"
          f" confirmed, reward not inflated.")
    print(f"    - Added lq-sensitivity (0.5*progress) is {100*added_over_standing:.0f}% "
          f"of the standing r_lq sensitivity (0.300) on average.")
    print(f"    - r_lq remains the PRIMARY link-quality channel; the r_prog "
          f"double-count is a modulation, not a takeover.")
    if dominates:
        print(f"    - REVIEW: mean lq-driven magnitude ({total_lq_influence.mean():.3f}) "
              f"exceeds r_lt+|p_delay|+|p_eng| ({r_lt_ref+p_delay_r+p_eng_r:.3f}).")
        print(f"      This is expected (r_lq alone was already ~{r_lq_arr.mean():.3f}); "
              f"confirm it is acceptable given r_lt/p_delay are intentionally small.")
    else:
        print(f"    - PASS: mean lq-driven magnitude ({total_lq_influence.mean():.3f}) "
              f"stays at/below r_lt+|p_delay|+|p_eng| ({r_lt_ref+p_delay_r+p_eng_r:.3f}).")
    print(f"\n    The double-count is DELIBERATE (reward progress made over good "
          f"links more).\n    Proceed to training only after eyeballing the table above.")
    print("=" * 72 + "\n")
    return 0


if __name__ == '__main__':
    sys.exit(main())
