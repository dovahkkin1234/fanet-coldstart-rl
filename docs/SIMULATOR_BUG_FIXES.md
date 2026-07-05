# Simulator Bug Fixes

Three critical bugs were identified and fixed in `src/simulator.py` before
the full-scale dataset generation. A fourth bug was found in the RL
environment.

---

## Bug 1 — `topology_change_rate` Always 0.0

**Affected files:** `topology_features.csv`, `routing_decisions.csv`

**Root cause:** In the `run()` method, `change_rate = 0.0` was hardcoded and
passed to `_log_topology()`. The correct value was computed inside
`_build_graph()` via `self.prev_links` symmetric difference, but it was never
returned or stored — so it never reached the logged output.

**Fix:** Added `self._last_change_rate` attribute in `__init__()`. Modified
`_build_graph()` to compute and store the correct change rate after updating
`self.prev_links`. Updated `run()` to pass `self._last_change_rate` instead
of the hardcoded 0.0.

**Verification:** After fixing, mean `topology_change_rate` in pilot_02 was
0.050 (range 0.0–1.0). Full-scale RWP shows 0.02–0.16 depending on scenario
speed.

---

## Bug 2 — Queue Occupancy Always 0.0

**Affected files:** `topology_features.csv`, `routing_decisions.csv`

**Root cause:** The forwarding loop (`for _ in range(TTL)`) forwarded packets
through all hops within a single timestep. Packets never sat in queues between
timesteps, so `NodeQueue` never accumulated any load. `avg_queue_occupancy`
and `candidate_queue_occupancy` were always 0.0.

**Fix:** Introduced `MAX_HOPS_PER_STEP = 2`. Packets now forward at most 2
hops per simulator timestep, requiring multiple timesteps for longer paths.
A queue drain block at the start of each timestep properly tracks occupancy.

**Verification:** After fixing, `avg_queue_occupancy` in pilot_02 was 0.0057
(max 0.08). `routing_decisions.csv` `current_queue_occupancy` mean was 0.017.

---

## Bug 3 — topology_change_rate Propagation to routing_decisions.csv

**Status:** Automatically fixed by Bug 1 fix.

The propagation path `topo.get('topology_change_rate', 0.0)` was already
correct in `_log_routing_decision()`. Once Bug 1 was fixed and the topology
snapshot carried the real non-zero value, propagation worked without any
additional changes.

---

## Bug 4 — sc01 PDR > 0.98 (Parameter Issue)

**Affected config:** `rwp_full_configs.yaml`, scenario `rwp_sc01`

**Root cause:** 10 drones in a 500×500m area with 200m comm range — nearly
every source-destination pair was 1–2 hops apart. `MAX_HOPS_PER_STEP = 2`
had no effect because no packet required more than 2 hops anyway. PDR was
effectively 1.0, providing no learning signal for hard routing decisions.

**Fix:** Reduced `comm_range` from 200m to 180m. This forces slightly longer
paths (2–3 hops on average), allowing the hop-step limit to apply backpressure.
PDR dropped to 0.837 — within the target 0.70–0.95 range.

---

## RL Environment Bugs (Phase 5)

Four additional bugs were found in the original multi-packet RL environment
and fixed in `src/rl_env_full.py` (Approach 1, single-packet trajectories):

### Flaw 1 — Cross-Packet Credit Assignment
The `step()` function forwarded one packet then returned a **different**
packet's state as `next_obs`. The DQN bootstrapped packet A's Q-value from
packet B's state. Fixed by switching to single-packet trajectory episodes.

### Flaw 2 — Episode-Level `done` Flag
Delivery and drop (true terminal events) were stored with `done=False`, adding
spurious `γ·maxQ(next_obs)` to terminal rewards. Fixed: `done=True` fires
exactly when the packet is delivered or dropped.

### Flaw 3 — Action/Execution Mismatch
When the agent selected a visited node, the code silently substituted a
different next-hop but stored the agent's original action in the replay buffer.
Fixed by masking visited nodes out of the action space entirely — the agent
cannot select them, eliminating the need for any override.

### Flaw 4 — Dropout Active During Evaluation and Target Computation
`QNetwork` has `Dropout(0.2)` but `.eval()` was never called. Both "greedy"
evaluation and TD-target computation ran through a randomly-masked network,
making evaluation non-deterministic and bootstrap targets noisy.
Fixed: both `q_net` and `target_net` are permanently set to `.eval()` mode.
