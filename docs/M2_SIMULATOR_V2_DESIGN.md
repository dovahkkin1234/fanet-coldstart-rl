# M2 Design Spec — `simulator_v2.py` (multi-packet, interference-coupled)

**Status:** design locked before implementation. Nothing here is code yet.
**Gate to clear before M3:** G2 (defined at the end).

---

## 0. What changes vs `simulator.py`, in one paragraph

Today's simulator forwards packets in an instantaneous per-timestep Python loop,
drains every queue to empty at the top of each step, and computes link quality
from distance alone. Three of those are the reason Approach 1 had zero queue
occupancy and saturated link quality. `simulator_v2` replaces the forwarding
model with a **slotted channel** where transmissions in the same slot interfere
(feeding `link_model_v2`), gives nodes a **bounded per-slot service rate** so
queues actually build, derives each node's **transmission activity** from its
queue state (closing the load→interference loop), and tracks a **per-packet
trajectory** so RL credit assignment stays clean across N concurrent packets.

---

## 1. Time and channel model

### 1.1 Two-level time
- **Slot** = one MAC transmission opportunity, duration `SLOT_DT` (e.g. 0.01 s).
- **Frame** = mobility/topology update interval, `FRAME_DT` (e.g. 0.5 s, = today's `dt`).
- One frame contains `SLOTS_PER_FRAME = FRAME_DT / SLOT_DT` slots (e.g. 50).
- Drones move and the graph rebuilds once per **frame**. Packets are forwarded
  once per **slot**. This is what creates concurrency: many nodes transmit in
  the same slot → they interfere.

Rationale: interference is a slot-level phenomenon; mobility is a frame-level
one. Rebuilding the O(n²) graph every slot would be 50× more expensive for no
physical benefit (positions barely change in 10 ms).

### 1.2 Per-slot transmission set
In each slot, the set of **transmitting** nodes `TX(slot)` is determined by a
simplified CSMA contention (see §3). For a receiver `rx` of a transmission
`tx→rx`, the interference is:

```
interference_mw(rx) = Σ_{k ∈ TX(slot), k≠tx}  rx_power_mw(dist(k, rx), shadow_k)
                      for k in the hidden-terminal band of (tx, rx)
```

using `link_model_v2.rx_power_mw`. The hidden-terminal band is
`dist(k,tx) > CS_range AND dist(k,rx) ≤ IF_range` — nodes far enough from `tx`
to not defer, close enough to `rx` to corrupt reception. This is exactly the
model the G1 preflight validated.

`n_contenders` for the Bianchi term = number of nodes within `CS_range` of `tx`
that had a packet to send this slot.

---

## 2. Queue and service model

### 2.1 Delete the total drain
Today (simulator.py lines ~554-557):
```python
for q in self.queues:
    while q.length > 0:
        q.dequeue()          # <-- DELETE THIS ENTIRELY
```
This is why queues were always ~0. Gone.

### 2.2 Bounded service
Each node serves **at most `SERVICE_RATE` packets per slot** (default 1 — a node
has one radio, one transmission per slot). A node with a non-empty queue and a
valid forwarding decision transmits its head-of-line packet; the rest wait.
Queues now build up when arrival rate > service rate — i.e. under load, exactly
where we want congestion to emerge.

`NodeQueue` (from models.py) is reused unchanged: `enqueue` tail-drops on
overflow (already returns False + increments `dropped_overflow`), `dequeue`
pops head-of-line, `occupancy`/`length` as-is.

### 2.3 Drop taxonomy (must be distinguishable in logs)
| drop_reason | cause |
|---|---|
| `queue_overflow` | receiver queue full at enqueue |
| `link_error` | SINR-driven PER OR Bianchi collision (roll < per_total) |
| `ttl_expired` | hop count ≥ TTL |
| `no_route` | no unvisited neighbor / disconnected from dst |
| `energy_depleted` | forwarding node out of energy |
| `episode_end` | still in flight at episode end |

G2 checks that under load the mix shifts toward `queue_overflow` + `link_error`
(the two load-induced causes), not just `ttl`/`no_route` as in Approach 1.

---

## 3. The load → activity → interference loop (the core new mechanism)

This is the coupling that makes the whole thesis work, and the one calibration
the G1 preflight left open. Precise definition:

### 3.1 Effective transmission activity of a node
A node's per-slot transmission probability is a function of how backed-up it is:

```
activity(node) = clip( α · occupancy(node) + β · 1[queue nonempty], 0, A_MAX )
```

- `occupancy(node)` ∈ [0,1] from `NodeQueue.occupancy`.
- Interpretation: an idle node (empty queue) has activity ≈ 0 → contributes no
  interference. A saturated node (full queue) transmits nearly every slot →
  activity ≈ `A_MAX`. This is the physical truth: **only nodes with traffic
  interfere**, and congested nodes interfere most.
- Defaults to calibrate against G1: `α ≈ 1.0`, `β ≈ 0.05`, `A_MAX ≈ 0.5`.
  These map queue state onto the same `activity` axis the G1 preflight swept
  (0–0.2 was already enough to crater quality), so we know the operating range.

### 3.2 The loop, explicitly
```
bad routing  →  packets pile at some node  →  its occupancy ↑  →  its activity ↑
     ↑                                                                    │
     │                                                                    ▼
worse routing options  ←  local link_quality craters  ←  interference ↑ at neighbors
```
This closed loop is the **congestion-collapse mechanism**. A warmstarted agent
keeps occupancy low everywhere → activity low → quality high → survivable. A
cold-start agent concentrates load → local activity spikes → quality craters →
even harder to route → *may not recover within the horizon.* That "may not
recover" is Headline Result #2 (congestion-collapse as an absorbing state).

### 3.3 Calibration gate (part of G2)
After building, sweep `packet_rate` and confirm the **realized** node activity
distribution lands in the 0–0.2 band that G1 showed produces graded (not binary)
degradation. If saturated nodes hit activity 1.0 and the network dies instantly,
lower `A_MAX` / `α`. This is a knob, tuned once, then frozen.

---

## 4. Per-packet trajectory tracking (RL correctness)

The Approach-1 cross-packet credit-assignment bug, generalized. Each packet
owns its own transition sequence; transitions from different packets are NEVER
interleaved into one trajectory.

```python
class PacketV2:
    # identity + routing state (as today: pid, src, dst, current, path, hops, ttl...)
    # NEW:
    trajectory: list          # [(obs, action, reward, next_obs, done), ...] for THIS packet
    pending_obs: dict | None  # obs captured at the moment a decision was made,
                              # awaiting the outcome of that hop (delayed reward)
    enqueue_slot: int         # slot this packet entered its current node's queue
```

Flow per forwarding decision:
1. When packet reaches head-of-line at node `v` and a routing decision is needed,
   compute `obs` **at that slot** (not cached — §5), store as `pending_obs`.
2. Agent (or teacher) picks `action` → next hop `u`.
3. Resolve the hop against the channel: compute SINR/PER at `u` **using this
   slot's TX set**, roll for link_error, check `u`'s queue for overflow.
4. Compute reward (delivery/progress/queue/interference terms — reward design is
   its own mini-spec in M5, not here).
5. Append `(pending_obs, action, reward, next_obs, done)` to **this packet's**
   trajectory. `done=True` iff delivered or dropped.
6. On delivery/drop, the packet's completed trajectory is emitted for logging
   (and later, for the RL replay buffer). Clean, per-packet, no interleaving.

---

## 5. Observation timing (the stale-obs pitfall)

**Rule: an observation is computed at decision time, from the network state of
that slot — never cached across slots.** Between two of a packet's decisions,
other packets change queues and the TX set changes interference. A cached obs
would describe a network that no longer exists.

Cost control: the *node/edge features that don't depend on the querying packet*
(positions, base link geometry, queue occupancies at slot start) are computed
**once per slot** and shared across all decisions in that slot. Only the
packet-relative parts (dist-to-dst, progress, TTL) are recomputed per decision.
This is the same "cache the shared part, recompute the query part" split that
the GNN will need in M4 — so build the seam now.

---

## 6. Event-driven vs slot-scan (implementation choice)

Two valid structures; pick **slot-scan** for v2:

- **Slot-scan (chosen):** iterate slots; in each slot, determine the TX set,
  resolve all transmissions, advance queues. Simple, deterministic, easy to
  make reproducible with one RNG. Cost: O(slots × active_nodes).
- **Event-driven (deferred):** a priority queue of (time, event) — only step
  when something happens. Faster for sparse traffic but much harder to get
  reproducible and to reason about interference (need "what else is transmitting
  in this instant"). Not worth the complexity for v2.

Slot-scan also vectorizes cleanly for the A100 port later (§8).

---

## 7. Reproducibility (carry the Approach-1 discipline forward)

- One `np.random.default_rng(seed)` for the simulator; **all** stochastic draws
  (shadowing, PER rolls, CSMA contention resolution, flow generation) come from
  it or from cleanly-offset child generators.
- Given a seed, a full episode must be bit-reproducible (we just proved G1 is —
  keep that property). This is non-negotiable for the seed studies in M8–M11.

---

## 8. Performance budget and the A100 seam

- Target: **< 5 s per 60 s episode** on the 5060 for the *scalar* v2 (G4 later).
- The expensive parts are (a) per-frame O(n²) graph build, (b) per-slot
  hidden-terminal interference sums. Both are array-friendly.
- Build v2 scalar-first (correctness), but keep the per-slot interference
  computation as a function `interference_for_slot(positions, tx_set, ...)` that
  can later be swapped for a vectorized/torch version without touching the loop.
  Do **not** micro-optimize now; just don't wall yourself off from vectorizing.

---

## 9. Backward-compatible sanity mode

`simulator_v2` must support a **degenerate single-packet, interference-off mode**
(packet_rate → one packet at a time, `A_MAX=0`) that approximately reproduces
Approach-1 network PDR on the same scenarios. This is a regression check: if the
new sim can't reproduce the old regime when interference is disabled, something
in the rewrite is wrong. (Ties to the interference on/off ablation later.)

---

## 10. GATE G2 — what must be true before M3

Run classical teachers (Dijkstra + a queue-aware greedy) across a `packet_rate`
sweep and confirm ALL of:

1. **Queues are nonzero and load-dependent.** Mean/max queue occupancy rises
   monotonically with `packet_rate`. (Approach 1: flat ~0.)
2. **Network PDR degrades smoothly with load.** No cliff to 0, no flat line —
   a graded curve, matching the graded link_quality from G1.
3. **Drop taxonomy shifts sensibly.** At low load: mostly `no_route`/`ttl`.
   At high load: `queue_overflow` + `link_error` become dominant. (This proves
   congestion is actually happening, not just topology limits.)
4. **Realized activity lands in the graded band** (§3.3): node activity
   distribution sits in ~0–0.2, not pinned at 1.0. (Calibration check.)
5. **Interference-off degenerate mode ≈ Approach 1** network PDR (§9).
6. **Reproducible:** same seed → identical episode outputs.

A companion `preflight_simulator_v2_check.py` (the G2 harness) will print all
six and PASS/FAIL, same style as the G1 gate.

---

## 11. Files this milestone produces

| File | Purpose |
|---|---|
| `src/simulator_v2.py` | the multi-packet interference-coupled simulator |
| `src/preflight_simulator_v2_check.py` | G2 gate harness |
| `src/routing_teachers_v2.py` (M3, stub referenced here) | queue-aware greedy used by G2 |

Mobility stays `mobility.py` for now; the extra mobility models (Gauss-Markov,
RPGM, Flocking) are M3-adjacent and come after G2 passes.

---

## 12. Explicitly NOT in M2 (scope fence)

- Reward function design → M5 (only a placeholder reward for G2's teacher runs).
- GNN/attention observation → M4.
- Congestion-aware teachers beyond one queue-aware greedy → M3.
- Jamming, extra mobility models, CBR, CL → later milestones.
- Vectorization/A100 port → after G4.

If any of these start creeping into the M2 code, stop.
