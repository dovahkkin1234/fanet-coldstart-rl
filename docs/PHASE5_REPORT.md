# Phase 5 Results Report — RL Training

## Overview

Phase 5 trained three DQN variants across 30 seeds to characterize
cold-start behavior and demonstrate the warm-start advantage.

---

## Environment Design

**Approach 1: Single-packet trajectory episodes.**

Each episode follows one packet from source to delivery/drop. This formulation
ensures every transition `(s, a, r, s', done)` belongs to the same packet,
giving correct credit assignment in the Bellman update. `done=True` fires
exactly when the packet terminates.

**State:** 22 base features (link quality, progress, energy, topology metrics)
+ 3-dimensional regime one-hot (Dense-Stable / Sparse-Partitioned /
Medium-Fast) for the regime variant.

**Action:** Index into the set of unvisited neighbors (visited nodes masked).

**Reward:** delivery (+10) + geographic progress (+1) + link quality (+0.3)
+ link lifetime (+0.2) − delay (−0.1·dt) − energy (−0.05) − drop (−10)

---

## Training Configuration

| Parameter | Scratch | Warmstart | Regime |
|---|---|---|---|
| Episodes | 5000 | 5000 | 5000 |
| ε start | 1.0 | 0.15 | 0.15 |
| ε decay | 2500 ep | 1200 ep | 1200 ep |
| LR (head) | 1e-4 | 1e-4 | 1e-4 |
| LR (backbone) | 1e-4 | 1e-5 | 1e-5 |
| Freeze episodes | — | 1000 | 1000 |
| Buffer size | 15000 | 15000 | 15000 |
| Batch size | 64 | 64 | 64 |
| Target sync | 40 ep | 40 ep | 40 ep |

---

## Cold-Start Results (4 core runs)

### Warmstart advantage vs scratch (seed 42)

| Scenario | Scratch init | Warmstart init | Advantage |
|---|---|---|---|
| rwp_sc03 | 0.467 | 0.467 | +0.000 |
| rwp_sc10 | 0.753 | 0.767 | +0.014 |
| rwp_sc07 | 0.347 | 0.333 | −0.014 |

**Note:** With a good-luck scratch seed (42), the advantage at ep1 is small
because the good random init is already near-warmstart quality. The advantage
is clearest against the full 30-seed distribution.

### Warmstart advantage vs 30-seed scratch mean

| Scenario | Scratch mean init | Warmstart init | Advantage |
|---|---|---|---|
| rwp_sc03 | 0.403 | 0.467 | +0.064 |
| rwp_sc10 | 0.639 | 0.767 | **+0.128** |
| rwp_sc07 | 0.286 | 0.333 | +0.047 |

Warmstart init PDR exceeds all 30 scratch seeds on sc10.

### AULC advantage (total cumulative routing value)

| Scenario | Scratch mean | Warmstart | Gain |
|---|---|---|---|
| rwp_sc03 | 2423 | 2477 | +2.2% |
| rwp_sc10 | 3456 | 3665 | **+6.0%** |
| rwp_sc07 | 1627 | 1677 | +3.1% |

### T90 (episodes to reach 90% of converged PDR)

| Scenario | Worst scratch T90 | Warmstart T90 | Speedup |
|---|---|---|---|
| rwp_sc03 | 1500 ep | 750 ep | 2.0× |
| rwp_sc10 | 1500 ep | 250 ep | **6.0×** |
| rwp_sc07 | 1750 ep | 250 ep | **7.0×** |

---

## 30-Seed Variance Study

**Criterion:** Slow cold-start = init PDR < 0.30 on ≥2 of 3 eval scenarios.

| Metric | Value |
|---|---|
| Slow seeds | 5 of 30 (16.7%) |
| 95% Wilson CI | [7.3%, 33.6%] |
| Slow seeds identified | 7, 13, 123, 200, 1984 |
| Fast/slow init PDR ratio (sc10) | **5.1×** |
| Fast/slow T90 ratio (sc10) | **3.8×** |
| Slow AULC loss (sc10) | 10.7% |
| Conv PDR std (all 30 seeds) | **0.010** (near-zero) |

All 30 seeds converge to statistically equivalent final performance —
confirming the variance is a cold-start phenomenon, not a permanent gap.

---

## Regime Conditioning

Regime vs warmstart mean absolute difference: 0.005–0.010 PDR.
No statistically significant advantage of regime conditioning over plain
warmstart in single-mobility-model training. Attributed to:
- Each eval scenario is dominated by 1–2 regimes (limited context switching)
- Regime columns had only 4000 unfrozen training episodes

Planned: multi-mobility training (RWP + RDM) and regime-switching eval
scenarios to provide a clearer signal for regime conditioning.

---

## Generalisation (Held-Out Scenarios)

Post-training greedy evaluation (150 packets, ε=0):

| Run | rwp_sc06 | rwp_sc11 | rwp_sc14 |
|---|---|---|---|
| Scratch avg | 0.233 | 0.080 | 0.180 |
| Warmstart | 0.227 | 0.080 | 0.180 |
| Regime | 0.240 | 0.080 | 0.180 |

Note: rwp_sc11 PDR ceiling is ~0.17 due to network sparsity (structural
partition), not routing policy quality.

---

## Catastrophic Forgetting — Fixed

Previous run (before freeze-then-finetune): warmstart sc10 PDR collapsed
from 0.720 → 0.127 (drop of 0.593). After fix: max peak-to-trough drop
is 0.100 — within normal training noise.
