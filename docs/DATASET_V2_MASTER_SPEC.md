# Dataset V2 — Master Specification

**Status: DESIGN, pre-patch.** Nothing here is implemented yet. This document exists so
the regeneration is designed once, against the requirements of every phase that will
consume it, rather than patched reactively after each phase discovers what it lacks.

**Scope.** Suite A only. Suites B (`SCENARIOS_DENSITY`) and C (`SCENARIOS_CONVERGECAST`)
serve the comparative analysis and are not on the M5 critical path.

**Precondition.** Repo at `63d5ee9`. Rate grid frozen at `[0.02, 0.05, 0.10, 0.25, 0.40]`
(`docs/PROBE_PREREGISTRATION.md` §6.3). v8/v9 staged, deliberately unapplied.

---

# 1. PROJECT STATUS SYNC

## 1.1 Completed — and what each consumed from the dataset

| milestone | gate | dataset dependency |
|---|---|---|
| **M1** interference-aware link model | G1 PASS | none — pre-dataset |
| **M2** multi-packet congestion simulator | G2 6/6 | none — produces the dataset |
| **M3** 8-teacher panel + oracle table | G3 6/6, Holm 12/12 | none — labels the dataset |
| **M3.5** Phase-B dataset | G3.5 8/8 + independent audit 7/7 | **the dataset itself** |
| **M4** supervised warmstart policy | G4 6/6 | **full consumer** — imitation learning |

M4 reached 98.0% of SP-BP rollout PDR (masked GNN) / 97.5% (masked MLP). Both gates
passed on the dataset as it stands.

## 1.2 Upcoming — and what each will demand

| phase | what it is | dataset demand |
|---|---|---|
| **Phase 0** | environment contract, credit assignment, replay representation, D-mapping | resumable mid-trajectory state (**§2.3 — not available**) |
| **Phase 2** | `run_iter()` steppable simulator | none (bit-identical gate) |
| **Phase 3** | `FANETEnvV2` + SP-BP parity 12/12 | feature-construction parity with `extract_decision`, incl. `resolve_mask` |
| **Phase 4** | warmstart transfer | identical feature schema + normalisation constants |
| **M5** | RL loop | transitions `(s,a,r,s′)` — **action, outcome, contiguity** |
| **M6** | decision gate: F1/F2/F3 separately measurable | headroom + queue value at the new operating point |
| **M7** | scratch DQN (= D1 arm b) | same as M5 |
| **M8–M9** | continual learning (D3) | task-boundary metadata; A→B→A |
| **M10** | CBR / episodic recall (D4) | `(state, action, outcome)` tuples for recall — **NOT NOVEL** per `NOVELTY_AUDIT_AND_PLAN.md` (MFEC 2016, NEC 2017, GEM 2021); reframe as application |
| **M11** | LOO ablation + write-up | all of the above, per-arm |

## 1.3 The five D-criteria and the cells they need

| | scenarios required | load |
|---|---|---|
| **D1 ESCAPE** | `dense_slow`, `very_dense`, **`sparse_fast`** | high |
| **D2 EXCEED ORACLE** | `dense_slow`, `very_dense` only | high |
| **D3 RETENTION** | A=`dense_slow` → B=**`sparse_fast`** → A | high |
| **D4 RECOVERY** | A → B → A, ±CBR | high |
| **D5 DEGRADATION** | **`sparse_fast`**, all loads | all |

**Three of five depend on `sparse_fast`, whose rate grid is undecided** (§4.2).

---

# 2. RETROSPECTIVE AUDIT

## 2.1 What the current dataset did well — keep all of it

These are not incidental; several were hard-won and must survive regeneration.

- **Ragged storage via flat buffers + offsets**, not object arrays. The comment at
  `generate_dataset_v2.py:524` records why: `np.array(list_of_arrays, dtype=object)`
  silently fails when inner arrays share a leading dimension. Pickle-free, faster, smaller.
- **Normalisation constants recorded per scenario in the manifest**, with the explicit
  contract that they must be reused. Without this, train/eval feature scaling silently
  diverges.
- **`feature_schema_version`** asserted by both checkers against the live `features_v2`
  module, aborting on mismatch. A dataset can never be validated by code that disagrees
  with it about column layout.
- **Fail-fast on orphaned split assignment.** The seed-range split raises rather than
  silently dropping decisions — it caught 65.2% orphaning with `--seeds 1 2 3`.
- **`local_horizon` (=2) recorded**, so the decentralisation claim is traceable to the
  dataset rather than asserted.
- **Canonical candidate ordering**, which makes "always pick slot 0" a meaningful trivial
  baseline (G3.5 check 3) rather than an arbitrary one.
- **8-teacher `votes` + `vote_agreement`** per decision, and `label_fallback` flagging.
- **Eight G3.5 checks + seven independent audit checks**, all passing.

## 2.2 What it dropped or lacked

Ordered by severity for the phases ahead.

### (a) The action actually taken is not saved — CRITICAL

```python
action_hop = (cands[random] if deviated else label_hop)   # line 323
'action': action_hop,                                     # line 346 — computed
```

`action` appears **zero times** in the `decisions.npz` save block. Only `label` (the
oracle's choice) and a `behaviour_deviated` bool are stored. With `EPSILON = 0.10`, on
~10% of decisions the simulator **took a random hop** while the record shows the oracle's.

For imitation learning this is harmless — the oracle label is what you want, which is why
M4 passed. **For anything transition-based it is fatal:** the state sequence was produced
by one policy and the recorded actions describe another. Roughly 1 in 10 transitions pairs
a state with an action that did not produce the successor state.

**Not recoverable post hoc.** The candidate list is stored, but the `ds_rng` draw is not,
so which candidate was taken cannot be reconstructed.

**Why no gate caught it.** Audit check G tests only
`abs(behaviour_deviated.mean() − epsilon) < 0.02` — a *rate* check. G3.5 check 4 is the
same. Neither ever asks which action was taken. The defect passed 8/8 and 7/7.

### (b) `drop_reason` computed then discarded — HIGH

Backfilled per decision at line 367 (`d['drop_reason'] = tr['drop_reason'] or ''`), absent
from the save block. This is the only field distinguishing a **congestion** failure from a
**link** failure — the exact distinction D2 rests on and the one credit assignment needs.
Without it the sole outcome signal is one bool.

### (c) `hop_succeeded` is a constant — MEDIUM

Line 370: `d['hop_succeeded'] = True` for every recorded decision, unconditionally. Also
not saved. As written it carries zero information; it should be computed honestly or
removed rather than existing as a misleading name.

### (d) Queue features are a stale start-of-frame snapshot — HIGH

`_build_graph` stamps `queue_occupancy` onto every node at **step 2** of the frame loop,
before that frame's packets are generated (step 3) and before any servicing (step 4).
Every decision in step 4 reads a snapshot from before its own traffic existed.

Measured (150 s, altitude 100–300, seed 1):

| scenario | rate | feat `cur>0` | live `cur>0` | feat `cand>0` | live `cand>0` |
|---|---|---|---|---|---|
| very_dense | 0.05 | **0.00%** | 12.37% | 83.51% | 100.00% |
| very_dense | 0.25 | **0.00%** | 11.74% | 94.38% | 100.00% |
| dense_slow | 0.25 | **0.00%** | 8.27% | 93.48% | 97.49% |
| medium_slow | 0.25 | **0.00%** | 7.84% | 86.42% | 89.48% |

This reproduces FILE1 §11's "0.000% nonzero across 129,302 decisions" and **identifies its
cause**. The §11 hypothesis — dequeue-before-decision — was wrong; that explains a deficit
of one packet, not a structural zero. The audit's own note at
`audit_dataset_v2.py:465` repeats the same incorrect explanation and should be corrected.

Affects every queue-derived feature: `queue_occupancy` (node), `current_queue_occupancy`,
`neigh_buffered_packets`, `neigh_mean_occupancy` (query), and candidate queue values.

Congestion stays *observable* — candidate occupancy is nonzero in 72–94% of decisions even
through the stale snapshot, which is why SP-BP works and M4 reached 98.0%. But the one
feature describing the agent's **own** node is identically zero, and that is precisely the
signal needed to attribute congestion to one's own forwarding choices.

### (e) Frame striding would destroy transition contiguity — HIGH

Transitions are reconstructible today: `packet_id` + `hop_index` orders a packet's hops,
`frame_id` locates the graph. The planned stride-10 patch records every 10th frame, and
decisions are recorded only on recorded frames — so consecutive hops mostly vanish. What
survives is not `(s_t, a_t, s_{t+1})` but pairs separated by an unknown number of missing
hops.

FILE2 §2.2 justified stride 10 purely on dataset size (~4.2× coverage). **The RL
consequence was never considered.** This is a direct conflict between two committed
decisions.

### (f) `eventual_delivered` carries survivorship bias — MEDIUM

Assigned per *packet* (line 368), so every hop of a delivered packet is `True`, including
poor choices on a route that happened to succeed. Correct starting vocabulary for delayed
reward — and already flagged in `train_supervised_v2.py` as "ablated, not assumed" — but
unusable alone without per-hop `drop_reason`.

### 2.3 What it structurally cannot do

**Mid-trajectory resumption.** There is **no state serialisation anywhere in
`simulator_v2.py`** — no `__getstate__`, no checkpoint, no restore. Frames store
*observations*, not simulator state. Absent from every frame: queue contents, in-flight
packet positions and TTLs, per-flow generation phase, energy accumulators, RNG state.
`node_feat` carries `queue_occupancy` as a scalar summary — a queue cannot be rebuilt from
it.

**Consequence:** Phase 0's proposed default — *"60–120 s windows sampled from
mid-trajectory states of the 1000 s dataset episodes"* — is not implementable as written.

---

# 3. FORWARD-LOOKING REQUIREMENTS (THE BLUEPRINT)

Requirements are `R-n`, each traced to the phase that needs it.

## 3.1 Per-decision record

| id | requirement | needed by |
|---|---|---|
| **R-1** | Save `action` — the candidate actually taken, as an index into the stored candidate list | M5, M7, D1, D3, D4, CBR |
| **R-2** | Save `drop_reason` per decision (categorical, from the packet's trajectory) | D2, credit assignment |
| **R-3** | `hop_succeeded` computed honestly per hop, or removed | reward design |
| **R-4** | Keep `label` unchanged and distinct from `action` — imitation must be unaffected | M4 retrain, Phase 4 |
| **R-5** | Keep `eventual_delivered`, `behaviour_deviated`, `votes`, `vote_agreement`, `label_fallback` | M4, CBR, audit G |
| **R-6** | Add `t_frame` (or make `frame_id → time` unambiguous) so windows can be sampled by simulated time | Phase 0 |

## 3.2 Transition structure

| id | requirement | needed by |
|---|---|---|
| **R-7** | Consecutive hops of a packet must be contiguous — no stride gaps within a trajectory | M5, M7, M8–M10 |
| **R-8** | Terminal flag per trajectory: delivered / dropped, with reason | M5 reward, D2 |
| **R-9** | Successor pointer derivable without ambiguity (`packet_id` + `hop_index` + `frame_id`) | M5, CBR |
| **R-10** | Episode/task identity per decision (`scenario`, `seed`, `packet_rate`) retained for A→B→A construction | D3, D4 |

## 3.3 Feature correctness

| id | requirement | needed by |
|---|---|---|
| **R-11** | Queue-derived features read at **decision time**, not start-of-frame | D2, env contract |
| **R-12** | Feature construction in `FANETEnvV2` must be byte-identical to `extract_decision` | Phase 3 gate, Phase 4 |
| **R-13** | `resolve_mask` replicated exactly in the env; parity asserted | D1 (silent-break risk) |
| **R-14** | Normalisation constants regenerated for the new `BASE` and recorded per scenario | Phase 4 |
| **R-15** | `feature_schema_version` bumped; both checkers assert against it | all |

## 3.4 Coverage

| id | requirement | needed by |
|---|---|---|
| **R-16** | Frozen grid `[0.02, 0.05, 0.10, 0.25, 0.40]` for the three connected scenarios | all |
| **R-17** | `sparse_fast` covered at a **defined** notion of "high load" | D1, D3, D5 |
| **R-18** | `load_bucket` re-thresholded to the new grid; discontinuity vs old results documented | D-criteria reporting |
| **R-19** | Dataset seeds (101–150) remain disjoint from evaluation seeds (1–30) | D1–D5 pairing |
| **R-20** | All 12 panel cells present, or the "12-cell" wording revised if `sparse_fast` diverges | Gate G-A criterion 3 |

## 3.5 Provenance and gates

| id | requirement | needed by |
|---|---|---|
| **R-21** | Manifest records resolved `BASE`, `RATES`, stride, epsilon, schema version, and config fingerprint | reproducibility |
| **R-22** | All 8 G3.5 checks and all 7 audit checks still pass | M3.5 re-gate |
| **R-23** | **New audit check: `action` consistency** — on `behaviour_deviated` rows, `action != label`; on others, `action == label` | closes the gap that let (a) through 15 checks |
| **R-24** | Audit's dequeue-before-decision note corrected to the real cause | accuracy |

---

# 4. ACTIONABLE RESOLUTION

## 4.1 Fixes a–e — all generation-time, one window

Every defect in §2.2 is fixed at generation. v8 already forces regeneration (altitude
change invalidates normalisation constants), so the marginal cost of fixing all of them is
approximately zero. **Fixing them separately means regenerating twice.**

| fix | change | risk |
|---|---|---|
| **a** | add `action` to the `decisions.npz` save block | none — additive |
| **b** | add `drop_reason` (encode categorically) | none — additive |
| **c** | compute `hop_succeeded` properly or drop it | low |
| **d** | queue features read live in `extract_decision` | **changes feature semantics — schema bump, M4 retrain** |
| **e** | striding — see §4.3 | see §4.3 |

Fix **d** is flag 5's **option 2**, and the earlier objection to it ("invalidates the M4
dataset, requires retraining") no longer costs anything: the dataset is already invalidated
by v8 and by fix (a), and the M4 retrain is already scheduled (FILE2 §13 step 9).

## 4.2 `sparse_fast` grid — DECISION REQUIRED

Blocks **D1, D3, D5**. The probe found: every cell in the floor band, sign consistency
never above 0.6, carried load flat at 0.06 from rate 0.05 onward, PDR non-monotonic in load
(0.750 at 0.02, **0.861** at 0.05), and ~22 routable packets/episode at the lowest rate.
It is **connectivity-limited, not congestion-limited** — its addressable budget is the
largest of any scenario (0.80) precisely because most loss is `no_route`.

Options:

1. **Inherit the global grid.** Simplest; "high load" (0.25, 0.40) is then 4–8× past
   saturation — a defensible *stress* definition, but D1's "high load in `sparse_fast`"
   would mean deep collapse.
2. **Own grid anchored on its own knee** (~0.05). Scientifically cleaner; costs a
   scenario-aware `load_bucket()`, touching `teacher_panel` and the dataset schema, and
   every "12-cell panel" reference needs **rewording**, not renumbering.
3. **Redefine D5** as connectivity-limited robustness and drop the congestion framing.
   Honest, and consistent with the measurement — but it is a change to a pre-registered
   criterion and must be recorded as such.

**Recommendation: 1 + 3.** Keep one global grid so the panel stays 12 cells, and restate
D5 (and `sparse_fast`'s role in D1/D3) as robustness under partition rather than under
congestion — which is what the data supports and what FILE2 §9.2 already half-says
("Robustness claim, explicitly NOT superiority").

## 4.3 Striding — DECISION REQUIRED

| option | imitation | RL transitions | size |
|---|---|---|---|
| stride 1 | full | **intact** | 1× |
| stride 10 | ~4.2× coverage | **destroyed** | ~0.1× |
| stride graph only, keep all decisions | full | intact but graphs up to 9 frames stale | middle |

The third option reintroduces exactly the staleness that fix (d) removes, so it is
self-defeating.

**Recommendation: stride 1 for Suite A.** RL is the entire point of M5–M11, and stride 10
was justified on size alone. Reserve striding for Suites B and C, where only imitation and
comparative analysis matter.

## 4.4 `medium_slow` holdout — DECISION REQUIRED

Current split: `train 101–135 | val 136–142 | test 143–150 | generalisation = all
medium_slow`. But **no D-criterion uses `medium_slow`**, so the generalisation split does
no work for the thesis's own criteria, and every D-criterion is evaluated on scenarios that
were in training.

Options: keep it (clean generalisation story, unused by D1–D5); hold out a *seed range*
instead of a scenario (generalisation then applies to the cells that matter); or drop the
holdout and rely on the val/test seed split.

**Recommendation: keep it, and state explicitly** that generalisation is measured on a
held-out *scenario* while D1–D5 measure in-distribution behaviour. That is a real
distinction worth reporting, not a flaw — provided it is stated rather than assumed.

## 4.5 Execution order

```
0.  freeze the three decisions above (4.2, 4.3, 4.4)          <- BLOCKS EVERYTHING
1.  RATES + load_bucket patch (grid already frozen)            R-16, R-18
2.  dataset patch set: fixes a,b,c,d + R-6 t_frame             R-1..R-6, R-11
    - bump FEATURE_SCHEMA_VERSION                              R-15
    - add audit check for action consistency                   R-23
    - correct the audit's dequeue note                         R-24
3.  striding decision implemented (recommend: stride 1)        R-7
4.  G1 anchor + remaining flag-6 work                          (independent)
5.  Phase 0 close: env contract with WARM-UP RE-SIMULATION,
    not mid-trajectory resumption                              §2.3
6.  run_iter() refactor, bit-identical gate
7.  FANETEnvV2, SP-BP parity 12/12 at the CURRENT 40 s point   R-12, R-13
8.  apply v8 + v9
9.  REGENERATE Suite A -- once                                 R-14, R-21
10. re-gate: G3.5 8/8 + audit 7/7 + new action check           R-22, R-23
11. retrain masked GNN; re-gate G4                             Phase 4
12. M5
```

**Step 7 precedes step 8 deliberately** — the parity gate needs the 40 s operating point as
its fixed reference (FILE2 §3). Step 9 happens **once**, after every generation-time fix is
in.

## 4.6 Open decisions, consolidated

| # | decision | blocks |
|---|---|---|
| 1 | `sparse_fast` grid — recommend option 1+3 | D1, D3, D5 |
| 2 | striding — recommend stride 1 for Suite A | M5–M11 |
| 3 | `medium_slow` holdout — recommend keep + state | reporting |
| 4 | continuing-task window: confirm warm-up re-simulation | Phase 0, D1 compute |
| 5 | `hop_succeeded` — compute or remove | reward design |
| 6 | D2's `+0.0645` reference — restate at the new operating point (measured 1.5–2.6 pp there) | D2 wording |
