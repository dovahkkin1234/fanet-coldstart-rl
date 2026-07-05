# RWP Full-Scale Dataset Specification
## For: FANET Cold-Start RL — Random Waypoint Mobility Model
## Prepared by: Shirish Giroti (CS23B2041)
## Status: Replaces the 10-scenario pilot dataset

---

## 1. Why Regenerate Instead of Using the Pilot?

The pilot dataset (10 scenarios × 30 episodes = 300 episodes) was sufficient to validate
Phases 1–4 (simulator, baselines, warm-start, regime clustering). However, it has five
known problems that must be fixed before the RL training and final paper results:

### Simulator Bugs (Must Fix Before Regeneration)

**Bug 1: `topology_change_rate` is always 0.0**
- Affected files: `topology_features.csv`, `routing_decisions.csv`
- Root cause: the `prev_links` set used to compute symmetric difference is never
  being updated correctly between timesteps, OR it's being reset every timestep
- Expected behavior: should be ~0.05–0.30 per timestep (fraction of links that
  changed since the previous snapshot). In a 30-drone network at 25 m/s, links
  break and form frequently
- Impact: one of the 22 training features is contributing zero information to
  the warm-start model and RL agent

**Bug 2: `avg_queue_occupancy` and `max_queue_occupancy` are always 0.0**
- Affected files: `topology_features.csv`
- Root cause: NodeQueue is likely never being populated because packets are
  forwarded instantly within a single timestep (multi-hop-per-step design)
- Expected behavior: queues should accumulate packets when packet_rate × num_flows
  exceeds the forwarding capacity, especially in congested scenarios
- Fix options:
  - (a) Carry packets across timesteps (realistic but changes the simulator significantly)
  - (b) Track per-step forwarding load as a proxy for queue pressure
  - Either way, the column should not be identically zero

**Bug 3: `candidate_queue_occupancy` and `current_queue_occupancy` always 0.0**
- Affected files: `routing_decisions.csv`
- Same root cause as Bug 2 — queues never fill because forwarding is instantaneous
- These are 2 of the 22 training features, both contributing nothing

### Dataset Design Limitations

**Limitation 1: Only 10 scenarios — gaps in the parameter space**
The pilot had no scenario with very high drone count + very high speed (stress test),
no scenario with moderate density + extreme speed, and no scenario specifically
designed for partition-recovery testing.

**Limitation 2: Only 30 episodes per scenario — borderline for RL**
RL training needs ~100+ episodes per difficulty level to learn a stable policy.
With 30 episodes per scenario, the RL agent cycles through each episode only
a few times during a 300-episode training run. More episodes per scenario
means more diverse seeds = more varied network trajectories = better RL
generalization.

**Limitation 3: No explicit RL evaluation scenarios**
The pilot test set (pilot_03, pilot_05) was designed for supervised evaluation.
RL evaluation needs scenarios that test specific capabilities: recovery from
partition, sustained high-throughput routing, energy-constrained late-episode
routing, and cross-speed generalization.

---

## 2. What Gets Fixed in the Full-Scale Version

| Issue | Pilot | Full-Scale |
|-------|-------|------------|
| topology_change_rate | Always 0.0 (bug) | Fixed — computed correctly per timestep |
| Queue occupancy | Always 0.0 (bug) | Fixed — packets carried across timesteps |
| Scenarios | 10 | 15 (better parameter coverage) |
| Episodes per scenario | 30 | 30 for train, 20 for val/test |
| Total episodes | 300 | 390 |
| Train/Val/Test split | 6/2/2 scenarios | 9/3/3 scenarios |
| RL evaluation scenarios | None | 3 dedicated test scenarios |
| Estimated routing_decisions rows | ~6.3M | ~10–14M |

---

## 3. Dataset Structure

### 3.1 Folder Layout

```
FANET_sim/
└── data/
    └── rwp_full/
        ├── rwp_sc01/                        ← scenario folder
        │   ├── episode_001/                 ← one episode
        │   │   ├── mobility_trace.csv
        │   │   ├── link_states.csv
        │   │   ├── topology_features.csv
        │   │   ├── traffic_flows.csv
        │   │   ├── packet_logs.csv
        │   │   ├── routing_decisions.csv
        │   │   └── rl_transitions.csv
        │   ├── episode_002/
        │   │   └── (same 7 CSVs)
        │   └── ... (30 episodes for train scenarios)
        ├── rwp_sc02/
        │   └── (30 episodes)
        └── ... (15 scenarios total)
```

### 3.2 Seed Formula
```python
seed = scenario_number * 10000 + episode_number
# Example: scenario 7, episode 15 → seed = 70015
```

---

## 4. The 15 Scenarios

Designed to cover five axes systematically: **drone count**, **area size**,
**speed**, **comm range**, and **pause behavior**.

### 4.1 Scenario Table

| ID | Name | Drones | Area (m) | Speed (m/s) | Comm Range | Pause | Expected PDR | Split |
|----|------|--------|----------|-------------|------------|-------|-------------|-------|
| rwp_sc01 | Dense-Slow | 10 | 500×500 | 5–15 | 200 | 0–5s | 0.85–0.95 | TRAIN |
| rwp_sc02 | Dense-Fast | 10 | 500×500 | 25–40 | 200 | 0–3s | 0.72–0.85 | TRAIN |
| rwp_sc03 | Medium-Slow | 20 | 1000×1000 | 5–15 | 250 | 0–5s | 0.75–0.88 | TRAIN |
| rwp_sc04 | Medium-Fast | 20 | 1000×1000 | 25–40 | 250 | 0–3s | 0.55–0.70 | TRAIN |
| rwp_sc05 | Large-Slow | 30 | 1500×1500 | 5–15 | 250 | 0–5s | 0.55–0.72 | **VAL** |
| rwp_sc06 | Large-Fast | 30 | 1500×1500 | 25–40 | 250 | 0–3s | 0.25–0.45 | **TEST** |
| rwp_sc07 | Swarm-Medium | 40 | 1500×1500 | 10–25 | 250 | 0–5s | 0.50–0.70 | TRAIN |
| rwp_sc08 | Sparse-Slow | 15 | 1500×1500 | 5–15 | 200 | 0–5s | 0.30–0.50 | TRAIN |
| rwp_sc09 | Sparse-Fast | 15 | 1500×1500 | 25–40 | 200 | 0–3s | 0.18–0.35 | **VAL** |
| rwp_sc10 | Dense-VeryFast | 25 | 800×800 | 35–50 | 200 | 0–2s | 0.40–0.60 | TRAIN |
| rwp_sc11 | VerySparse-Slow | 20 | 2000×2000 | 5–20 | 300 | 0–5s | 0.25–0.45 | **TEST** |
| rwp_sc12 | BigSwarm-Fast | 40 | 2000×2000 | 20–35 | 300 | 0–3s | 0.35–0.55 | TRAIN |
| rwp_sc13 | Compact-Extreme | 30 | 600×600 | 30–50 | 150 | 0–2s | 0.55–0.75 | **VAL** |
| rwp_sc14 | Ultra-Sparse | 12 | 2000×2000 | 5–15 | 350 | 0–8s | 0.15–0.35 | **TEST** |
| rwp_sc15 | Mixed-Variable | 25 | 1200×1200 | 5–50 | 250 | 0–5s | 0.45–0.65 | TRAIN |

### 4.2 Design Rationale

**Scenarios 01–02:** Isolate speed effect at small/dense scale. Same drone count and area,
different speed. The warm-start model must learn that fast networks need different routing
than slow networks even at the same density.

**Scenarios 03–04:** Same isolation at medium scale. These are the workhorse training
scenarios — most similar to "typical" FANET deployments.

**Scenarios 05–06:** Large-scale validation and test. sc05 (slow) validates the model on
unseen large networks. sc06 (fast) is the **primary test scenario** — the hardest
combination the model has never seen during training.

**Scenario 07:** High drone count (40 nodes). Tests scalability — the model must handle
graphs with 40 nodes and potentially 780 possible edges. Produces the largest
`routing_decisions.csv` files.

**Scenarios 08–09:** Sparse networks where partitions are frequent. sc09 (fast + sparse)
is a validation scenario that tests the model's ability to route in heavily partitioned
networks with short link lifetimes.

**Scenario 10:** Dense but extremely fast. Tests whether the model learned to prioritize
link lifetime over geographic progress when velocities are very high.

**Scenario 11:** Very sparse, slow. **Second test scenario.** Partitions are frequent but
long-lived — the model must learn to wait for routes rather than force packets through
marginal links.

**Scenario 12:** Large swarm at moderate-fast speed. Combines high drone count with
challenging speed — tests whether the model's knowledge from smaller swarms (sc07)
transfers to a harder setting.

**Scenario 13:** Compact area with extreme speed and short comm range. **Third validation
scenario.** Very high topology change rate — links form and break every few seconds.
Tests the model's ability to make rapid routing decisions under churn.

**Scenario 14:** Ultra-sparse with long comm range. **Third test scenario.** 12 drones
over 2000×2000m with 350m range — long pauses (up to 8s) mean temporary stability
windows exist, but finding them requires the model to understand pause behavior.

**Scenario 15:** Wide speed range (5–50 m/s). Some drones are slow, some fast.
Tests whether the model handles heterogeneous mobility within a single episode.

### 4.3 Train / Val / Test Split

```
TRAIN (9 scenarios × 30 episodes = 270 episodes):
  rwp_sc01, rwp_sc02, rwp_sc03, rwp_sc04, rwp_sc07,
  rwp_sc08, rwp_sc10, rwp_sc12, rwp_sc15

VAL (3 scenarios × 20 episodes = 60 episodes):
  rwp_sc05, rwp_sc09, rwp_sc13

TEST (3 scenarios × 20 episodes = 60 episodes):
  rwp_sc06, rwp_sc11, rwp_sc14
```

**Why 3 test scenarios instead of 2:** The RL ablation study needs to report
cold-start metrics per scenario. With 3 test scenarios covering different
difficulty profiles (Large-Fast, VerySparse-Slow, Ultra-Sparse), the paper
can show that warm-start helps across all unseen conditions, not just one.

---

## 5. Fixed Parameters

```yaml
# Shared across all 15 RWP scenarios
z_min:          50          # metres altitude floor
z_max:          150         # metres altitude ceiling
timestep:       0.5         # seconds per simulation step
duration:       300         # seconds per episode (5 minutes)
packet_rate:    1.0         # packets/sec per source-destination flow
mobility_model: random_waypoint
```

**Number of flows per episode:** `num_flows = max(1, num_drones // 4)`
This scales with drone count — a 40-drone swarm generates 10 flows,
a 10-drone network generates 2 flows.

---

## 6. Simulator Bug Fixes Required Before Generation

### Fix 1: topology_change_rate

In `simulator.py`, the `_build_graph()` or equivalent function computes
`topology_change_rate` as the symmetric difference between current and
previous link sets. The bug is that `self.prev_links` is either:
(a) not being updated after the comparison, or
(b) being reset to empty at the start of each timestep

**Correct implementation:**
```python
def _build_graph(self):
    # ... build graph G, collect cur_links as a set of (i,j) tuples ...

    # Topology change rate: fraction of links that changed since last step
    if len(self.prev_links) == 0 and len(cur_links) == 0:
        change_rate = 0.0
    else:
        sym_diff = self.prev_links.symmetric_difference(cur_links)
        union    = self.prev_links.union(cur_links)
        change_rate = len(sym_diff) / max(len(union), 1)

    # CRITICAL: update prev_links AFTER computing the rate
    self.prev_links = cur_links    # ← this line was missing or misplaced

    # ... store change_rate in topology dict ...
```

**Verification:** After fixing, `topology_change_rate` should be:
- ~0.0 in Dense-Slow scenarios during pause periods
- ~0.05–0.15 in Medium scenarios
- ~0.15–0.35 in Fast scenarios
- Never exactly 0.0 for an entire episode (unless all drones are paused)

### Fix 2: Queue occupancy

The root cause is that the multi-hop-per-timestep forwarding loop processes
all hops for a packet within a single `step()` call. Packets never sit in
queues across timesteps.

**Fix approach:** After each forwarding decision, if the packet is not
yet at its destination, leave it in the receiver's queue until the next
timestep. Only forward one hop per timestep per packet (or limit to 2–3
hops per timestep to keep some multi-hop behavior while allowing queue
buildup).

```python
# In the main simulation loop:
MAX_HOPS_PER_STEP = 2   # was effectively unlimited (TTL=20)

for pkt in active_packets:
    hops_this_step = 0
    while hops_this_step < MAX_HOPS_PER_STEP:
        # ... forward one hop ...
        hops_this_step += 1
        if pkt.delivered or pkt.dropped:
            break
    # If packet not delivered/dropped, it stays in active_packets
    # and will continue forwarding next timestep
```

**Impact on queue_occupancy:** With `MAX_HOPS_PER_STEP = 2`, packets in a
5-hop path will spend 2–3 timesteps in transit. During that time, they
occupy queue slots at intermediate nodes. The `candidate_queue_occupancy`
and `current_queue_occupancy` features in `routing_decisions.csv` will
become non-zero, giving the warm-start model and RL agent a congestion
signal they currently lack.

**Verification:** After fixing, `avg_queue_occupancy` should be:
- ~0.01–0.05 in easy scenarios (low traffic, short paths)
- ~0.05–0.20 in medium scenarios
- ~0.10–0.40 in hard scenarios with 40 drones and many flows
- max_queue_occupancy should spike to 0.5+ occasionally in congested episodes

### Fix 3: Verify topology_change_rate propagates to routing_decisions.csv

The `topology_change_rate` value computed in `topology_features.csv` must
also be written into the corresponding rows of `routing_decisions.csv`.
In the pilot, both were 0.0, so it's unclear if the propagation works
correctly even after fixing the computation. Verify that:
```python
routing_decisions_row['topology_change_rate'] = topo['topology_change_rate']
```
is executed for every candidate row written to `routing_decisions.csv`.

---

## 7. The 7 Output CSV Files Per Episode

Identical schema to the pilot and the RDM spec. Every column name, type,
and semantic must match exactly. Refer to `RDM_DATASET_SPEC.md` Section 6
for the complete column-by-column specification.

**Critical reminders:**
- `routing_decisions.csv` must have exactly one `is_best_label=1` per `decision_id`
- `consensus_confidence` values must be one of {0.2, 0.4, 0.6, 0.8, 1.0}
- `regime_id` column must be present and filled with -1 (placeholder)
- No NaN or Inf in any of the 22 feature columns
- All 5 teacher columns (`teacher_dijkstra`, `teacher_gpsr`, `teacher_aodv`,
  `teacher_stable_path`, `teacher_link_lifetime`) must be present

---

## 8. Scale Targets

| Metric | Pilot (existing) | Full-Scale (target) |
|--------|-----------------|---------------------|
| Scenarios | 10 | 15 |
| Episodes | 300 | 390 (270 train + 60 val + 60 test) |
| Duration per episode | 300s | 300s |
| routing_decisions rows | ~6.3M | ~10–14M |
| topology snapshots | 7,200 | ~11,700 |
| Disk size (estimated) | ~2.5 GB | ~4–7 GB |
| topology_change_rate | Always 0 (bug) | Non-zero, varying |
| queue_occupancy | Always 0 (bug) | Non-zero, varying |

---

## 9. Validation Checklist

After generation, run the existing `validate_all.py` (updated for the new
`rwp_full` data root path). Every check must pass.

**Per episode:**
- [ ] All 7 CSV files present and non-empty
- [ ] `routing_decisions.csv` has all required columns (22 features + labels + teachers)
- [ ] Exactly one `is_best_label=1` per `decision_id`
- [ ] `consensus_confidence` values ∈ {0.2, 0.4, 0.6, 0.8, 1.0}
- [ ] No NaN or Inf in any feature column
- [ ] PDR > 0.01 (episode is not completely broken)
- [ ] PDR < 0.99 for non-trivial scenarios (sc02–sc15)
- [ ] `residual_energy` monotonically non-increasing per drone
- [ ] `topology_change_rate` is NOT always 0.0 — verify mean > 0.01
- [ ] `avg_queue_occupancy` is NOT always 0.0 — verify mean > 0.001
- [ ] `regime_id` column present and filled with -1

**NEW checks (not in pilot validator):**
- [ ] `topology_change_rate` in `routing_decisions.csv` matches the
      corresponding value in `topology_features.csv` for the same timestamp
- [ ] `current_queue_occupancy` and `candidate_queue_occupancy` are not
      identically 0.0 across the entire episode
- [ ] At least 3 out of 5 teachers agree on `is_best_label` in > 50% of decisions
      (if consensus is too low, the labels are noisy)

**Across all episodes:**
- [ ] Each train scenario folder contains exactly 30 episode sub-folders
- [ ] Each val/test scenario folder contains exactly 20 episode sub-folders
- [ ] Seeds are unique within each scenario
- [ ] No duplicate `packet_id` within an episode

---

## 10. Things To Avoid

**Do not** run the old simulator without applying the three bug fixes from
Section 6. The resulting data will have the same degenerate features as the
pilot and will waste computation time.

**Do not** change the 22 feature column names or order in `routing_decisions.csv`.
The training scripts use exact column matching.

**Do not** use the same seeds as the pilot dataset. The pilot used seeds in
the range `config_index * 1000 + episode`. The full-scale uses
`scenario_number * 10000 + episode_number` to guarantee no overlap.

**Do not** let any scenario consistently produce PDR > 0.98. If sc01
(Dense-Slow) delivers everything, reduce `comm_range` from 200 to 180
or increase area to 600×600. The model learns nothing from trivially
easy episodes.

**Do not** let any scenario consistently produce PDR < 0.05. If sc14
(Ultra-Sparse) drops everything, increase `comm_range` from 350 to 400
or reduce area. An episode where 95%+ of packets are dropped produces
almost no positive training labels.

---

## 11. Estimated Generation Time

| Component | Time estimate |
|-----------|--------------|
| Per episode (300s sim, ~20 drones avg) | ~3–6 minutes |
| Per scenario (30 episodes) | ~1.5–3 hours |
| Full dataset (15 scenarios) | ~25–45 hours |
| With 4 parallel processes | ~7–12 hours |

**Parallelization strategy:**
```powershell
# Terminal 1: Dense scenarios (fast, small graphs)
python src\run_full_rwp.py --scenarios rwp_sc01 rwp_sc02 rwp_sc10 rwp_sc13

# Terminal 2: Medium scenarios
python src\run_full_rwp.py --scenarios rwp_sc03 rwp_sc04 rwp_sc05 rwp_sc15

# Terminal 3: Large/sparse scenarios
python src\run_full_rwp.py --scenarios rwp_sc06 rwp_sc08 rwp_sc09 rwp_sc11

# Terminal 4: Swarm scenarios (slowest — 40 drones)
python src\run_full_rwp.py --scenarios rwp_sc07 rwp_sc12 rwp_sc14
```

---

## 12. Handoff Checklist

When generation is complete:

- [ ] `data/rwp_full/` directory (all 390 episode folders across 15 scenarios)
- [ ] Confirm bug fixes: share 3 sample topology_features.csv files showing
      `topology_change_rate > 0` and `avg_queue_occupancy > 0`
- [ ] Run `python src\validate_all.py --data_root data/rwp_full --samples 3`
      and share terminal output
- [ ] Report total row count of `routing_decisions.csv` across all episodes
- [ ] Report disk size of `data/rwp_full/`
- [ ] Note any scenarios where PDR was consistently < 0.05 or > 0.98

---

*Document version: 1.0 — June 2026*
*Contact: Shirish Giroti, CS23B2041, IIITDM Kancheepuram*
