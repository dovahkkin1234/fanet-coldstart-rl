# RDM Full-Scale Dataset Specification
## For: FANET Cold-Start RL — Random Direction Mobility Model
## Prepared by: Shirish Giroti (CS23B2041)
## Purpose: Share with teammate responsible for dataset generation

---

## 1. Why We Need This Dataset

The existing pilot dataset uses the **Random Waypoint (RWP)** mobility model.
RWP drones pause at waypoints — this creates artificially stable link periods
that do not reflect real UAV behavior. The **Random Direction Model (RDM)**
is more realistic: drones pick a *direction* and fly until they hit the
boundary, then pick a new direction. There are no pauses. Links break faster
and more unpredictably.

We need a full-scale RDM dataset to:
1. Train the XGBoost and MLP warm-start models on RDM-specific routing patterns
2. Enable topology regime clustering with genuine variety (the pilot 5-episode
   RDM dataset had no parameter variation — all episodes were identical)
3. Provide RL training data under a harder, more realistic mobility model
4. Demonstrate that our approach generalises across mobility models
   (RWP → RDM transfer is a key paper result)

---

## 2. Random Direction Model (RDM) — How It Differs From RWP

| Property             | RWP (existing)             | RDM (new)                        |
|----------------------|----------------------------|----------------------------------|
| Movement target      | Random waypoint in area    | Random *direction* (angle)       |
| On arrival           | Pause, then new waypoint   | Bounce/wrap at boundary, new dir |
| Pause behaviour      | Yes (0–pause_max seconds)  | **No pauses at all**             |
| Link stability       | Moderate (pause = stable)  | Low (always moving)              |
| Boundary behaviour   | Clamp + new waypoint       | **Reflect off boundary**         |
| Speed profile        | Uniform random per leg     | Uniform random per direction leg |
| Realism for UAVs     | Medium                     | Higher (UAVs rarely hover)       |

### RDM Movement Logic (pseudo-code for your simulator)

```python
class DroneRDM:
    def __init__(self, drone_id, area_x, area_y, z_min, z_max,
                 speed_min, speed_max, seed=0):
        self.rng = np.random.default_rng(seed * 10000 + drone_id)
        # ... (same init as RWP except no pause_max needed)
        self._pick_new_direction()

    def _pick_new_direction(self):
        # Pick a random 3D direction (azimuth + elevation)
        azimuth   = self.rng.uniform(0, 2 * np.pi)
        elevation = self.rng.uniform(-np.pi/6, np.pi/6)  # ±30° vertical
        self.current_speed = self.rng.uniform(self.speed_min, self.speed_max)
        self.vx = self.current_speed * np.cos(elevation) * np.cos(azimuth)
        self.vy = self.current_speed * np.cos(elevation) * np.sin(azimuth)
        self.vz = self.current_speed * np.sin(elevation)

    def step(self, dt):
        # Move in current direction
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.z += self.vz * dt

        # Reflect off boundaries (NOT clamp+new waypoint like RWP)
        if self.x < 0 or self.x > self.area_x:
            self.vx *= -1
            self.x = np.clip(self.x, 0, self.area_x)
        if self.y < 0 or self.y > self.area_y:
            self.vy *= -1
            self.y = np.clip(self.y, 0, self.area_y)
        if self.z < self.z_min or self.z > self.z_max:
            self.vz *= -1
            self.z = np.clip(self.z, self.z_min, self.z_max)

        # Pick a new direction every direction_change_interval seconds
        # (use a countdown timer; reset after each direction change)
        self.direction_timer -= dt
        if self.direction_timer <= 0:
            self._pick_new_direction()
            self.direction_timer = self.rng.uniform(
                self.min_leg_duration, self.max_leg_duration)
```

**Key RDM parameters per drone:**
- `min_leg_duration`: minimum seconds before direction change (suggest 10s)
- `max_leg_duration`: maximum seconds before direction change (suggest 30s)
- No `pause_max` — RDM drones never stop

---

## 3. Dataset Structure

### 3.1 Folder Layout

```
FANET_sim/
└── data/
    └── rdm_full/
        ├── rdm_sc01_10n_500m_v10/          ← scenario folder
        │   ├── episode_001_seed0042/        ← one episode
        │   │   ├── mobility_trace.csv
        │   │   ├── link_states.csv
        │   │   ├── topology_features.csv
        │   │   ├── traffic_flows.csv
        │   │   ├── packet_logs.csv
        │   │   ├── routing_decisions.csv
        │   │   └── rl_transitions.csv
        │   ├── episode_002_seed0137/
        │   │   └── (same 7 CSVs)
        │   └── ... (20 episodes per scenario)
        ├── rdm_sc02_10n_500m_v30/
        │   └── (20 episodes)
        └── ... (12 scenarios total)
```

### 3.2 Naming Convention

**Scenario folder:** `rdm_sc{NN}_{N}n_{A}m_v{V}`
- `NN` = scenario number (01–12)
- `N`  = num_drones
- `A`  = area size in metres (square: area_x = area_y = A)
- `V`  = nominal speed (speed_min to speed_max centred around V)

**Episode folder:** `episode_{NNN}_seed{SSSS}`
- `NNN`  = zero-padded episode number within scenario (001–020)
- `SSSS` = 4-digit random seed used (ensures reproducibility)

**Example:** `rdm_sc05_30n_1500m_v25/episode_007_seed0891/`

---

## 4. The 12 Scenarios

This mirrors the RWP pilot structure: 12 scenarios × 20 episodes = **240 episodes total**.
The scenarios vary across three axes — **drone count**, **area size**, **speed** —
to produce the topological diversity needed for meaningful regime clustering.

| Scenario | Drones | Area (m²) | Speed (m/s) | Comm Range | Expected PDR | Label |
|----------|--------|-----------|-------------|------------|--------------|-------|
| rdm_sc01 | 10 | 500×500 | 5–15 | 200 | ~0.85–0.92 | Dense-Slow |
| rdm_sc02 | 10 | 500×500 | 25–40 | 200 | ~0.70–0.82 | Dense-Fast |
| rdm_sc03 | 20 | 1000×1000 | 5–15 | 250 | ~0.75–0.88 | Medium-Slow |
| rdm_sc04 | 20 | 1000×1000 | 25–40 | 250 | ~0.55–0.72 | Medium-Fast |
| rdm_sc05 | 30 | 1500×1500 | 5–15 | 250 | ~0.60–0.75 | Large-Slow |
| rdm_sc06 | 30 | 1500×1500 | 25–40 | 250 | ~0.30–0.50 | Large-Fast |
| rdm_sc07 | 40 | 2000×2000 | 10–25 | 300 | ~0.45–0.65 | Swarm-Medium |
| rdm_sc08 | 15 | 1500×1500 | 5–15 | 200 | ~0.35–0.55 | Sparse-Slow |
| rdm_sc09 | 15 | 1500×1500 | 25–40 | 200 | ~0.20–0.40 | Sparse-Fast |
| rdm_sc10 | 25 | 800×800 | 30–50 | 200 | ~0.40–0.60 | Dense-VeryFast |
| rdm_sc11 | 20 | 2000×2000 | 5–20 | 300 | ~0.25–0.45 | VerySparse-Slow |
| rdm_sc12 | 35 | 1200×1200 | 15–35 | 250 | ~0.50–0.70 | Mixed |

**Why these 12 and not more:**
- Scenarios 01–02: isolate speed effect at low drone count
- Scenarios 03–04: isolate speed effect at medium drone count
- Scenarios 05–06: isolate speed effect at high drone count (RDM stress test)
- Scenarios 07: large swarm (40 drones)
- Scenarios 08–09: sparse topology (few drones, large area)
- Scenario 10: dense but very fast (tests high-velocity regime)
- Scenario 11: very sparse and slow (partition-prone)
- Scenario 12: mixed parameters (generalisation scenario)

**Train/Val/Test split by scenario** (same strategy as RWP):
- **Train:** sc01, sc02, sc03, sc04, sc07, sc08, sc10, sc12 (8 scenarios × 20 = 160 episodes)
- **Val:**   sc05, sc09 (2 scenarios × 20 = 40 episodes)
- **Test:**  sc06, sc11 (2 scenarios × 20 = 40 episodes — hardest unseen conditions)

sc06 (Large-Fast) and sc11 (VerySparse-Slow) are held out as test because they
represent the most challenging conditions under each difficulty axis.

---

## 5. Fixed Parameters (Same For All Scenarios)

```yaml
# Shared across all 12 RDM scenarios
z_min:          50        # metres altitude floor
z_max:          150       # metres altitude ceiling
timestep:       0.5       # seconds per simulation step
duration:       300       # seconds per episode (5 minutes)
packet_rate:    1.0       # packets/sec per source-destination flow
min_leg_duration: 10      # RDM: minimum seconds on one direction
max_leg_duration: 30      # RDM: maximum seconds before new direction
episodes_per_scenario: 20
num_flows:      num_drones // 4   # scales with drone count
```

### Seed Strategy

To ensure reproducibility across the full dataset, generate seeds
**deterministically from the scenario number and episode number**:

```python
seed = scenario_number * 10000 + episode_number
# Example: scenario 5, episode 12 → seed = 50012
```

This means anyone can regenerate any specific episode exactly.

---

## 6. The 7 Output CSV Files Per Episode

These must match the RWP pilot format exactly so the existing
training scripts work without modification.

### 6.1 `mobility_trace.csv`
One row per (drone, timestep).

| Column | Type | Description |
|--------|------|-------------|
| timestamp | float | Simulation time in seconds |
| drone_id | int | 0-indexed |
| x, y, z | float | Position in metres |
| vx, vy, vz | float | Velocity components (m/s) |
| speed | float | \|\|v\|\| in m/s |
| heading | float | Azimuth angle (radians) |
| pitch | float | Elevation angle (radians) |
| residual_energy | float | Energy remaining (starts at 100.0) |

### 6.2 `link_states.csv`
One row per (drone pair, timestep) where a link exists.

| Column | Type | Description |
|--------|------|-------------|
| timestamp | float | |
| drone_i, drone_j | int | Always i < j |
| distance | float | Euclidean distance (m) |
| rssi | float | Received signal strength (dBm) |
| snr | float | Signal-to-noise ratio (dB) |
| link_quality | float | 0–1 (derived from SNR) |
| packet_error_rate | float | 0–1 (probability of bit error) |
| estimated_link_lifetime | float | Seconds until link likely breaks |
| relative_velocity | float | \|\|v_i - v_j\|\| (m/s) |

### 6.3 `topology_features.csv`
One row per timestep (network-level snapshot).

| Column | Type | Description |
|--------|------|-------------|
| timestamp | float | |
| num_nodes | int | Total drones |
| num_active_links | int | Links currently active |
| avg_degree | float | Mean degree of connected nodes |
| network_density | float | Actual / max possible links |
| num_connected_components | int | Number of disconnected groups |
| largest_component_size | int | Drones in the biggest component |
| largest_component_ratio | float | largest_component_size / num_nodes |
| avg_link_quality | float | Mean link_quality across active links |
| avg_link_lifetime | float | Mean estimated_link_lifetime |
| avg_relative_velocity | float | Mean relative velocity across links |
| topology_change_rate | float | Fraction of links changed since last step |
| avg_residual_energy | float | Mean battery across all drones |
| min_residual_energy | float | Lowest battery drone |
| avg_queue_occupancy | float | Mean queue fill ratio |
| max_queue_occupancy | float | Highest queue fill ratio |
| regime_id | int | -1 placeholder (filled by rdm_regimes.py later) |

### 6.4 `traffic_flows.csv`
One row per source-destination flow (generated once per episode).

| Column | Type | Description |
|--------|------|-------------|
| flow_id | int | |
| src_drone | int | Source drone ID |
| dst_drone | int | Destination drone ID |
| start_time | float | When flow begins (usually 0.0) |
| packet_rate | float | Packets per second |

### 6.5 `packet_logs.csv`
One row per packet generated during the episode.

| Column | Type | Description |
|--------|------|-------------|
| packet_id | int | Unique within episode |
| flow_id | int | Links to traffic_flows |
| src_drone, dst_drone | int | |
| gen_time | float | When packet was created (seconds) |
| delivered | int | 1 = success, 0 = dropped |
| drop_reason | str | null / ttl_expired / no_route / packet_error / energy_depleted |
| end_to_end_delay | float | Seconds from gen to delivery (NaN if dropped) |
| hop_count | int | Hops taken (NaN if dropped) |
| energy_consumed | float | Total energy for this packet's journey |

### 6.6 `routing_decisions.csv` ← Most Important For Training
One row per (candidate neighbour, forwarding decision). This is what
XGBoost and MLP train on. Each forwarding event produces N rows
(one per candidate neighbour), with exactly one `is_best_label = 1`.

| Column | Type | Description |
|--------|------|-------------|
| timestamp | float | When decision was made |
| decision_id | int | Unique per forwarding event |
| packet_id | int | |
| current_node | int | Drone making the forwarding decision |
| candidate_node | int | Candidate next-hop being scored |
| destination | int | Final destination |
| **is_best_label** | **int** | **1 = best choice, 0 = not best (consensus of 5 teachers)** |
| **consensus_confidence** | **float** | **Fraction of teachers that agreed (0.2–1.0)** |
| candidate_distance | float | Distance to candidate (m) |
| candidate_relative_velocity | float | Relative speed to candidate (m/s) |
| candidate_link_quality | float | Link quality score (0–1) |
| candidate_rssi | float | RSSI to candidate (dBm) |
| candidate_packet_error_rate | float | PER to candidate (0–1) |
| candidate_estimated_lifetime | float | Expected link lifetime (s) |
| candidate_energy | float | Candidate's residual energy |
| candidate_queue_occupancy | float | Candidate's queue fill ratio |
| candidate_dist_to_dest | float | Candidate's distance to destination (m) |
| candidate_progress | float | (current_dist - cand_dist) / current_dist |
| current_energy | float | Current node's residual energy |
| current_queue_occupancy | float | Current node's queue fill ratio |
| current_num_neighbors | int | How many candidates available |
| current_speed | float | Current node's speed (m/s) |
| distance_to_destination | float | Current node's distance to destination |
| bearing_to_destination | float | Angle to destination (radians) |
| destination_reachable | int | 1 if dest in same connected component |
| avg_degree | float | Network avg degree (topology snapshot) |
| network_density | float | Network density (topology snapshot) |
| num_components | int | Number of components (topology snapshot) |
| avg_link_quality | float | Network avg link quality |
| topology_change_rate | float | Fraction of links changed |
| teacher_dijkstra | int | Dijkstra's choice (1=this candidate) |
| teacher_gpsr | int | GPSR's choice |
| teacher_aodv | int | AODV-like choice |
| teacher_stable_path | int | Stable-Path choice |
| teacher_link_lifetime | int | Link-Lifetime-Aware choice |
| regime_id | int | -1 placeholder (filled later by rdm_regimes.py) |

**Critical:** `is_best_label` must be derived from teacher consensus:
```python
teacher_votes = [teacher_dijkstra, teacher_gpsr, teacher_aodv,
                 teacher_stable_path, teacher_link_lifetime]
vote_sum = sum(teacher_votes)
is_best_label = 1 if vote_sum >= 3 else 0        # majority vote
consensus_confidence = vote_sum / 5.0             # 0.2, 0.4, 0.6, 0.8, or 1.0
```

If no candidate gets majority, assign `is_best_label=1` to the candidate
with the most votes; break ties by `candidate_link_quality`.

### 6.7 `rl_transitions.csv`
One row per forwarding step taken during the episode (the actual action taken,
not all candidates). Used for RL training and CBR knowledge base.

| Column | Type | Description |
|--------|------|-------------|
| timestamp | float | |
| packet_id, flow_id | int | |
| state_node | int | Drone that made the decision |
| action_node | int | Chosen next hop |
| destination | int | Final destination |
| reward_total | float | Composite reward (see below) |
| reward_delivery | float | +10.0 if this hop delivered the packet |
| reward_progress | float | Geographic progress toward destination |
| reward_link_quality | float | Quality of the chosen link |
| penalty_delay | float | -0.1 × dt |
| penalty_energy | float | Energy cost of this hop |
| penalty_drop | float | -10.0 if packet dropped here |
| next_state_node | int | Where packet is now |
| done | int | 1 if packet delivered or dropped |
| regime_id | int | -1 placeholder |

**Reward formula:**
```python
reward_total = (
    10.0 * delivered          # delivery bonus
  + 1.0  * geographic_progress  # (d_before - d_after) / d_before
  + 0.3  * link_quality         # chosen link quality
  + 0.2  * min(link_lifetime/30, 1.0)  # stability reward
  - 0.1  * dt                   # delay penalty
  - 0.05 * energy_cost          # energy penalty
  - 0.2  * queue_occupancy      # congestion penalty
  - 10.0 * dropped              # drop penalty
  - 2.0  * loop_detected        # loop penalty
)
```

---

## 7. Scale Targets

| Metric | RWP Pilot (existing) | RDM Full Scale (target) |
|--------|---------------------|------------------------|
| Scenarios | 10 | 12 |
| Episodes | 300 (30 each) | 240 (20 each) |
| Duration per episode | 300s | 300s |
| routing_decisions rows | ~6.3M | ~8–12M (more drones avg) |
| topology snapshots | 7,200 | ~7,200 |
| Disk size (estimated) | ~2.5 GB | ~3–5 GB |

Note: 20 episodes per scenario (vs 30 in RWP) is intentional — the 12-scenario
variety compensates. We need breadth more than repetition within one scenario.

---

## 8. Validation Checklist

After generation, run `python src\rdm_warmstart.py` which includes
automatic validation. The following checks must all pass:

**Per episode:**
- [ ] All 7 CSV files present and non-empty
- [ ] `routing_decisions.csv` has all 32 columns listed in Section 6.6
- [ ] Exactly one `is_best_label=1` per `decision_id`
- [ ] `consensus_confidence` values are one of {0.2, 0.4, 0.6, 0.8, 1.0}
- [ ] No NaN in any of the 22 feature columns
- [ ] No infinite values in any feature column
- [ ] PDR > 0.01 (episode is not completely broken)
- [ ] PDR < 0.99 (episode is not trivially easy — if this fails, increase area)
- [ ] `residual_energy` is monotonically non-increasing per drone
- [ ] `topology_features.csv` has `num_active_links > 0` for > 50% of timesteps
- [ ] `regime_id` column exists and is filled with -1 (placeholder)
- [ ] `rl_transitions.csv` `reward_total` has mean > -5.0 (not all drops)

**Across all episodes:**
- [ ] Each scenario folder contains exactly 20 episode sub-folders
- [ ] Seeds are unique within each scenario
- [ ] No duplicate `packet_id` within an episode

---

## 9. Things To Avoid

**Do not** change the column names in any CSV. The training scripts
use exact column name matching. Adding extra columns is fine;
removing or renaming breaks everything.

**Do not** set `is_best_label=1` for multiple candidates in the same
`decision_id`. This corrupts the top-k accuracy metric.

**Do not** use the same seeds across different scenarios. Use the
deterministic formula: `seed = scenario_number * 10000 + episode_number`.

**Do not** let episodes run with PDR > 0.98 in easy scenarios. If
sc01 (Dense-Slow) consistently delivers everything, reduce `comm_range`
slightly or increase area until PDR sits in the 0.80–0.95 range.
The warm-start model learns nothing useful from trivially easy episodes.

**Do not** vary `z_min`, `z_max`, or `timestep` across scenarios.
These are fixed. Changing them silently corrupts the feature distributions.

---

## 10. Quick Reference — Scenario Config File

Save this as `configs/rdm_full_configs.yaml` in the FANET_sim directory:

```yaml
# rdm_full_configs.yaml
# Full-scale RDM dataset — 12 scenarios × 20 episodes = 240 episodes

defaults:
  z_min: 50
  z_max: 150
  timestep: 0.5
  duration: 300
  packet_rate: 1.0
  min_leg_duration: 10
  max_leg_duration: 30
  mobility_model: random_direction
  episodes_per_config: 20

scenarios:
  - name: rdm_sc01   # Dense-Slow
    num_drones: 10
    area_x: 500
    area_y: 500
    speed_min: 5
    speed_max: 15
    comm_range: 200

  - name: rdm_sc02   # Dense-Fast
    num_drones: 10
    area_x: 500
    area_y: 500
    speed_min: 25
    speed_max: 40
    comm_range: 200

  - name: rdm_sc03   # Medium-Slow
    num_drones: 20
    area_x: 1000
    area_y: 1000
    speed_min: 5
    speed_max: 15
    comm_range: 250

  - name: rdm_sc04   # Medium-Fast
    num_drones: 20
    area_x: 1000
    area_y: 1000
    speed_min: 25
    speed_max: 40
    comm_range: 250

  - name: rdm_sc05   # Large-Slow  [VAL]
    num_drones: 30
    area_x: 1500
    area_y: 1500
    speed_min: 5
    speed_max: 15
    comm_range: 250

  - name: rdm_sc06   # Large-Fast  [TEST]
    num_drones: 30
    area_x: 1500
    area_y: 1500
    speed_min: 25
    speed_max: 40
    comm_range: 250

  - name: rdm_sc07   # Swarm-Medium
    num_drones: 40
    area_x: 2000
    area_y: 2000
    speed_min: 10
    speed_max: 25
    comm_range: 300

  - name: rdm_sc08   # Sparse-Slow
    num_drones: 15
    area_x: 1500
    area_y: 1500
    speed_min: 5
    speed_max: 15
    comm_range: 200

  - name: rdm_sc09   # Sparse-Fast  [VAL]
    num_drones: 15
    area_x: 1500
    area_y: 1500
    speed_min: 25
    speed_max: 40
    comm_range: 200

  - name: rdm_sc10   # Dense-VeryFast
    num_drones: 25
    area_x: 800
    area_y: 800
    speed_min: 30
    speed_max: 50
    comm_range: 200

  - name: rdm_sc11   # VerySparse-Slow  [TEST]
    num_drones: 20
    area_x: 2000
    area_y: 2000
    speed_min: 5
    speed_max: 20
    comm_range: 300

  - name: rdm_sc12   # Mixed
    num_drones: 35
    area_x: 1200
    area_y: 1200
    speed_min: 15
    speed_max: 35
    comm_range: 250
```

---

## 11. Estimated Generation Time

On a modern laptop (no GPU needed — this is simulation not ML):

| Component | Time estimate |
|-----------|--------------|
| Per episode (300s sim, 30 drones) | ~2–5 minutes |
| Full dataset (240 episodes) | ~8–20 hours |
| Recommended: parallelize by scenario | Run 4 scenarios simultaneously |
| With 4 parallel processes | ~2–5 hours total |

**Parallelization suggestion:**
```python
# Run these 4 commands simultaneously in separate terminals:
python src/run_rdm.py --scenarios rdm_sc01 rdm_sc02 rdm_sc03
python src/run_rdm.py --scenarios rdm_sc04 rdm_sc05 rdm_sc06
python src/run_rdm.py --scenarios rdm_sc07 rdm_sc08 rdm_sc09
python src/run_rdm.py --scenarios rdm_sc10 rdm_sc11 rdm_sc12
```

---

## 12. Handoff Checklist

When generation is complete, share with Shirish:

- [ ] `data/rdm_full/` directory (all 240 episode folders)
- [ ] Run `python src/rdm_warmstart.py --data_root data/rdm_full` — share the terminal output
- [ ] Share `models/RDM/validation_report.txt`
- [ ] Confirm total row count of routing_decisions across all episodes
- [ ] Confirm disk size of `data/rdm_full/`
- [ ] Note any scenarios where PDR was consistently < 0.05 (need parameter tuning)

If any scenario has mean PDR < 0.05 across all its episodes, the scenario
parameters are too hard and routing is almost always failing — let Shirish
know and we will adjust `comm_range` up by 50m for that scenario.

---

*Document version: 1.0 — June 2026*
*Contact: Shirish Giroti, CS23B2041, IIITDM Kancheepuram*
