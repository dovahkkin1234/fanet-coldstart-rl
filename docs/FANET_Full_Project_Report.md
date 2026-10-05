# FANET Lifelong-RL Routing — Full Project Report

### Developer / researcher walkthrough: working idea, complete progress record, every decision, everything dropped, everything suggested

**Project:** Cold-Start-Aware Lifelong Reinforcement Learning for Congestion-Aware Packet Routing in Flying Ad-hoc Networks
**Author:** Shirish Giroti (CS23B2041), IIITDM Kancheepuram · **Guide:** Dr. Preeth Raguraman
**Repository:** `github.com/dovahkkin1234/fanet-coldstart-rl` · **HEAD at time of writing:** `376d209` (2026-09-22) · 56 commits, 2026-07-05 → 2026-09-22
**Report date:** 2026-09-30

---

## 0. How to read this report

This document is meant to make you familiar with *everything* — not a pitch, not a summary. It is organised so that you can read it end to end once, and afterwards use it as a reference.

| Part | What it covers | Read when |
|---|---|---|
| **§1** | Where the project stands today, on one page | first, always |
| **§2** | The research problem, the thesis, novelty and positioning | to explain the project to anyone |
| **§3** | Timeline and repository map | to find anything in the code |
| **§4** | Approach 1 (complete prior work) and why it was not enough | to understand why Approach 2 exists |
| **§5** | The simulator, at developer level | before touching `simulator_v2.py` |
| **§6** | The teachers (hand-designed routers), with exact formulas | before reading any panel result |
| **§7** | Features, the dataset (M3.5) and its gates | before regenerating anything |
| **§8** | The learned model (M4) and its results | before M5 warmstart work |
| **§9** | The gates (G1–G4, G-A, parity) — what each tests, how each has changed | when a gate fails |
| **§10** | The verification era (pre-M5): chronological, stage by stage | to understand how the current state was reached |
| **§11** | The final operating point and per-scenario load bands | before choosing any rate |
| **§12** | The oracle panels and the revised understanding of congestion-awareness | before the teacher decision |
| **§13** | Status of every claim ever made: valid / superseded / void / retracted | before citing any number |
| **§14** | Decisions taken (locked) | before re-opening anything |
| **§15** | What was dropped, rejected or superseded, and why | before re-proposing an old idea |
| **§16** | What was suggested but not yet done | backlog |
| **§17** | Open decisions that need your call | now |
| **§18** | Critical path to dataset regeneration and RL start | planning |
| **§19** | The RL programme (M5–M11) and the D1–D5 pre-registration | planning |
| **§20** | Competitor positioning | writing |
| **§21** | The complete defect ledger | methodology section |
| **§22** | Documentation state — what is stale where | before trusting a doc |
| **§23** | Risks and an honest assessment | with your guide |
| **App.** | Glossary, file index, key-number sheet, PowerShell cheat-sheet | reference |

**Provenance tags used throughout** (the same convention as FILE1/FILE2):
**[MEASURED]** came out of a run · **[COMPUTED]** derived from parameters · **[VERIFIED IN SOURCE]** read from the code at `376d209` · **[DECIDED]** a project decision on record · **[ASSESSMENT]** my own judgement, stated as such. A number with no tag is quoted from a document that itself cites a run.

**Every figure in this report was taken from the repository, the result JSONs, or the terminal logs you shared — not from memory.** Where two sources disagree, the disagreement is stated rather than resolved silently.

---

## 1. Where the project stands — one page

**In one sentence:** the infrastructure (physics, simulator, teachers, supervised warmstart) is built and gated; a long verification campaign has since found and fixed defects that voided the old operating point, the old rate grid and the old oracle; a new, measured operating point and a new per-cell oracle now exist; the dataset has **not** been regenerated yet, and **no reinforcement learning has started**.

| Area | State | Detail |
|---|---|---|
| M1 interference physics | ✅ done, G1 passes (re-verified after v17) | §5.3, §9.1 |
| M2 multi-packet simulator | ✅ done; G2 **re-baselined** (v20–v22) and passes 6/6 | §5, §9.2 |
| M3 teacher panel | ✅ built — but its original verdict (SP-BP wins 12/12) is **void** (pre-leak-fix, wrong operating point) | §6, §13 |
| **Oracle of record** | ✅ **new, per-cell**: `dijkstra` / `gpsr` / `da_gpsr` — SP-BP displaced in 9/9 cells | §12 |
| M3.5 dataset | ⚠️ exists (533,237 decisions) but is **obsolete**: old operating point, old oracle, leak-era simulator, missing fields | §7 |
| M4 supervised warmstart | ✅ done on the old dataset; G4 6/6. **Must be retrained** after regeneration | §8 |
| Operating point | ✅ **measured**: 1000 s, altitude 100–300 m, battery 8000 — staged as v8b, **not applied** | §11 |
| Rate grids | ✅ **measured per scenario** (usable bands) — not yet written into `config_v2` | §11 |
| Dataset schema v2 | ✅ specified (`M3.5_Dataset_Schema.md`) with 9 gaps flagged; **not implemented** | §7.6, §18 |
| Phase 0 (env contract) | ❌ not started | §18 |
| Phase 2 (`run_iter()` refactor) | ❌ not started | §18 |
| Phase 3 (`FANETEnvV2` + parity gate) | ❌ not started | §18 |
| M5–M11 (all RL) | ❌ not started | §19 |
| **Open decision blocking the dataset** | ❓ **one global teacher vs per-cell teachers** — you are researching this | §17.1 |

**What is genuinely proven today** (and survives every correction): the congestion-coupling mechanism exists in the simulator (§5.4); no single hand-designed router is best across regimes (§12); four controlled mechanism comparisons with 9/9 or 0/9 consistency (§12.5); the matched-capacity GNN-vs-MLP reversal under feature masking on the old dataset (§8 — to be re-confirmed after regeneration).

**What is not proven:** all three thesis failure modes (F1 cold-start trap, F2 forgetting, F3 recall). **None has been measured.** Everything so far is infrastructure and characterisation.

**Your timeline (stated):** regenerate the dataset and start RL within ~2 weeks; RL (offline + online) over 2–3 months.

---

## 2. The research problem and the thesis

### 2.1 The problem

Deep-RL routing policies for FANETs are normally evaluated with a stationary protocol: train on a scenario until convergence, evaluate on the same distribution, report PDR/delay/energy. That protocol cannot see three things that happen to a policy over an operational lifetime:

| | Failure mode | Mechanism in this project | Proposed remedy |
|---|---|---|---|
| **F1** | **Cold start** — a fresh policy routes badly; under load, bad routing creates congestion, congestion degrades links, degraded links make routing harder. Claimed to be an **absorbing trap**, not just slow learning. | the congestion-coupling loop of §5.4 | **warmstart** from supervised imitation of hand-designed teachers |
| **F2** | **Catastrophic forgetting** — adapting to a new regime destroys competence on earlier regimes | regime shifts (density, mobility, load) | **continual learning** (EWC / replay) |
| **F3** | **No episodic recall** — a previously-solved situation must be re-learned from scratch | recurring conditions (e.g. a returning interference pattern) | **case-based reasoning / episodic memory** |

**The thesis-defining claim:** *only the combination survives lifelong deployment* — tested by a **leave-one-out ablation** (M10): full system vs −warmstart vs −CL vs −CBR.

**The single most important framing statement, repeated from FILE1:** *none of F1, F2 or F3 has been measured yet.* M1–M4 are infrastructure.

### 2.2 Why the congestion loop is the heart of it

F1 is only interesting if cold start is *dangerous*, not merely slow. The project's claim is that under realistic load a badly-initialised agent pushes the network into a self-reinforcing state:

```
bad routing → packets pile up → queue occupancy ↑ → transmission activity ↑
     ↑                                                      ↓
worse options ← link quality craters ← interference / collisions ↑
```

For this loop to exist the simulator needs three things Approach 1 did not have: load-dependent link quality (M1), concurrent packets with real queues (M2), and teachers that are good enough under congestion to warmstart from (M3). That is why the project is structured as it is.

### 2.3 Novelty audit (as recorded in `NOVELTY_AUDIT_AND_PLAN.md`, 2026-08-10)

| # | Claim | Verdict | Nearest prior work |
|---|---|---|---|
| N1 | GNN + DRL routing | **dead** — not novel | HCPMR, GNNPPOR, GraphSAGE-MAPPO |
| N2 | CBR / episodic-recall mechanism | **not novel** — reposition as application | MFEC 2016, NEC 2017, GEM 2021; CBR-MPLS 2003; CBR smart-grid |
| N3 | Continual-learning method | **not novel** — novel only as application to routing regimes | EWC, PackNet; Davaslioglu 2024 (jamming); STCL/G-MAPPO 2026 (UAV edge) |
| N4 | **Cold start as an absorbing trap in routing** | **novel** (no hit in 8 searches) | — |
| N5 | **Lifetime failure-mode decomposition for routing** | **novel** | surveys *name* forgetting as open; nobody *measures* it in FANET routing |
| N6 | **Three-way leave-one-out ablation** | **novel** | no combined system exists in routing to ablate |
| N7 | Matched-capacity GNN vs MLP control | promising, now evidenced (§8) | no FANET GNN paper runs a matched control |
| N8 | Reproducible artefact (gates, audits, pre-registration) | differentiator, not a contribution | — |

**Precision rule for the paper:** "stationary" must be stated surgically. HCPMR *does* evaluate across scenarios (urban/desert/mountain). The real gap is **sequential training with retention re-measured on earlier regimes**, not cross-scenario evaluation.

**A newer candidate contribution from the panel work (§12) [ASSESSMENT]:** a controlled characterisation of *when and how* congestion signals help FANET routing (formulation × density), with three unanimous 9-cell findings. This did not exist when the novelty audit was written.

### 2.4 Publication strategy [DECIDED]

- **Venue ceiling:** IEEE TNSM / TMC / IoT-J / Ad Hoc Networks (Elsevier) / TVT. MobiCom/INFOCOM ruled out for simulation-only work.
- **Two papers.** *Paper A* — the architecture result (matched-capacity MLP vs GNN, reversal under masking) — a GLOBECOM/ICC-class short paper. *Paper B* — the lifetime thesis, after M5–M11, targeting TNSM / Ad Hoc Networks.
- **Do not compete on absolute PDR.** Competitors route to a single ground station with no queue model; this project routes random UAV pairs with finite queues. Compare only against baselines re-implemented in the same simulator (§20).

---

## 3. Timeline and repository map

### 3.1 Timeline by phase (from `git log`)

| Dates | Phase | Key commits |
|---|---|---|
| 07-05 → 07-11 | **Approach 1** — single-packet DQN, warmstart, 30-seed study, v4a/b/c ablation | `44529da`, `b58ab1f`, `ccb2cbd`, `2bb35d1`, `2417374` |
| 07-16 | **M1** interference link model + G1; M2 design spec | `d40d9aa`, `d859cb5` |
| 07-18 → 07-23 | **M2** simulator + G2; **M3** 8-teacher panel, paired t-test fix, G3 | `55fcdfe`, `9d9c34a`, `a2d88da`, `7535376`, `e42f72e` |
| 07-28 | **M3.5** dataset, G3.5, independent audit | `d39c81e`, `98a0291`, `c89e02c` |
| 07-30 → 08-02 | Adversarial review (M-1…M-7) and answering experiments: locality, mechanism, headroom, calibration sensitivity, M-4 collision model | `2521318`, `175cab1`, `fa7e025`, `012ac5e`, `3ec4f15`, `bb1f0a1` |
| 08-05 → 08-08 | **Pre-M4 fix pass** v1–v6 (anchors, redundancy, SP-BP re-implementation divergences, normaliser clipping) | `36881dd`, `1769e57`, `870aaef`, `454b9f8`, `92dcb6b` |
| 08-13 | **M4** GNN+attention vs matched MLP; masking reversal; `id(G)` cache bug; novelty audit | `7abd5c6`, `f2b35fe`, `ebc013d`, `fa86609` |
| 09-02 → 09-08 | **Pre-M5**: v10 headroom cache fix, v11 `config_v2`, v7 rollout mask, v8/v9 staged, CLAUDE.md, FILE1/FILE2 v2, v11.2, rate probe + amendment, Dataset V2 spec, v8a | `96dd94d` … `2492297` |
| 09-18 | **v12 leak fix** + v14 energy override (plus v13, v15–v22 swept into the same commit), v23/v24 provenance, results index | `ea52d04`, `f512144` |
| 09-22 | **Oracle panels** (1,769 episodes), v25 rule fix, panel report, diagnostics moved | `aaa2e3b`, `10884ab`, `d6763ff`, `376d209` |

A note on `ea52d04`: the commit message says "v12/v14", but `git add -A` swept in v13 and v15–v22 plus the band-search tooling and results. The later commit attempts for v16–v22 then reported "nothing to commit". The code is all there; the history just attributes it to one commit.

### 3.2 Repository layout (at `376d209`)

```
fanet-coldstart-rl/
├── src/                       all source
│   ├── config_v2.py           SINGLE source of SCENARIOS / RATES / BASE / suites (v11, v8a)
│   ├── simulator_v2.py        M2 simulator (FANETSimulatorV2), TEACHERS registry, PANEL
│   ├── link_model_v2.py       M1 physics: SINR, Bianchi, shadowing, lethal-interferer model
│   ├── routing_teachers_v2.py the congestion-aware teachers (spbp, da_gpsr, car, dpp, ...)
│   ├── routing_teachers.py    Approach-1 teachers (dijkstra, gpsr reused from here)
│   ├── routing_teachers_v3_local.py  k-hop-limited SP-BP (locality experiment)
│   ├── teacher_panel.py       panel runner, load_bucket() (STALE thresholds)
│   ├── features_v2.py         feature schema v5, extract_frame / extract_decision, norm_constants
│   ├── generate_dataset_v2.py M3.5 generator (DatasetSimulator, spbp_pick_restricted, assert_no_drift)
│   ├── preflight_*_check.py   gates G1 (interference), G2 (simulator), G3 (teachers), G3.5 (dataset)
│   ├── audit_dataset_v2.py    independent dataset audit (7 checks)
│   ├── model_gnn_attn.py      M4 model: dense graph attention encoder + query/key scorer
│   ├── train_supervised_v2.py M4 trainer (masks, HP grid)
│   ├── rollout_eval_v2.py     M4 rollout evaluator (ModelActorSimulator)
│   ├── experiment_*.py        mechanism / locality / headroom / queue-weight / calibration / collision
│   ├── probe_rate_grid.py     pre-registered rate probe (void grid, retargeted by v13)
│   ├── map_operating_regimes.py, sweep_energy_ceiling.py      energy investigation
│   ├── find_usable_band.py, find_congestion_band_convergecast.py   usable-band search (current)
│   ├── panel_oracle_generalized_v1.py, panel_contenders_v2.py,
│   │   panel_extend_linkquality_v3.py, probe_oracle_agreement.py    oracle panels (current)
│   ├── scan_congestion_window.py, verify_oracle_congestion.py      (VOID-result scripts, kept)
│   └── rl_env*.py, train_dqn*.py, train_mlp*.py, ...    Approach-1 lineage — QUARANTINED
├── docs/                      specs, records, reports (see §22 for staleness)
├── results/                   JSON outputs + README.md index (CURRENT / SUPERSEDED / VOID)
├── diagnostics/               one-off probes behind the G2 re-baseline (v20–v22)
├── apply_*.py / verify_*.py / fix_*.py   every patch, with its verifier (repo root)
└── configs/, logs/, scripts/  Approach-1 artefacts
```

**Two files that do *not* exist in the repo but are referenced by it:** `docs/LOAD_DENSITY_DESIGN.md` (pointed to by `PROBE_PREREGISTRATION.md` §6.3) and `apply_v8b_operating_point_STAGED.py`. Both were written in this session and handed over but never committed (§22).

### 3.3 Hardware and environment [VERIFIED]

HP Z8 G5 · Xeon Gold 6426Y 16C/32T · 32 GB RAM · RTX A4000 16 GB · Windows 11 Pro · PowerShell only (no `&&`, no `unzip`) · conda env `fanet` · Python 3.11.15 · PyTorch 2.11.0+cu128 · working dir `C:\Users\PREETH\FANET_sim\`. PyTorch Geometric / DGL deliberately excluded: graphs are 20–50 nodes, and dense `torch.nn` attention is simpler and faster at that size. The simulator is pure Python / NetworkX and **CPU-bound**; parallelism is across seeds with `ProcessPoolExecutor`. On Windows, 16 spawn workers stalled the convergecast script, so band/panel runs use `--max_workers 4–8`.

---

## 4. Approach 1 — the complete prior work, and why it was not enough

### 4.1 What was built

Single-packet DRL routing. One packet is routed to completion, then the next.

- **Simulator** (`simulator.py`, `mobility.py`, `link_model.py`, `models.py`): 3-D random-waypoint drones, log-distance link model, energy model. 15 RWP scenarios (`rwp_sc01…sc15`), 9/3/3 train/val/test scenarios.
- **Teachers** (`routing_teachers.py`): Dijkstra, GPSR, AODV, Stable-Path, Link-Lifetime-Aware.
- **Warmstart:** a 22-feature binary classifier trained on consensus labels from the 5 teachers — **AUC 0.903, top-1 85.6%** — whose weights initialise the DQN Q-network.
- **Regime clustering:** KMeans (k=3, silhouette 0.243) on 9 topology features → Dense-Stable / Sparse-Partitioned / Medium-Fast.
- **RL:** Double DQN + prioritised replay; freeze-then-finetune (backbone frozen 1000 episodes, then 10× lower LR) to prevent the warmstart from being destroyed.
- **Ablation:** Dueling (v4a), curriculum + reward shaping (v4b), all three (v4c).

### 4.2 Results that stand (from `docs/PHASE5_REPORT.md`)

| Metric | Value |
|---|---|
| Slow cold-start incidence among 30 scratch seeds | **5/30 = 16.7%**, Wilson 95% CI [7.3%, 33.6%] |
| Fast/slow initial-PDR ratio (sc10) | **5.1×** |
| Fast/slow T90 ratio (sc10) | 3.8× |
| Warmstart T90 speed-up | up to **6×** (sc10), 7× (sc07) |
| Converged-PDR std across 30 seeds | **0.010** — a cold-start phenomenon, not a permanent gap |
| AULC advantage (sc10) | +6.0% |
| Catastrophic forgetting of the warmstart | fixed by freeze-then-finetune (drop 0.593 → ≤0.100) |

### 4.3 Why it could not carry the thesis

The Dueling / reward-shaping / curriculum ablation returned **uniformly null**, for a structural reason: **in a single-packet world queue occupancy is identically zero**. No packet ever contends with another. The congestion penalty in the reward never fired; `link_quality` was saturated near 1.0 on ~100% of hops because nothing interfered; Dueling had 3–6 actions over 3–8 steps and nothing to decompose. The environment could not exhibit the phenomenon the thesis is about. **That is the whole reason Approach 2 exists.**

### 4.4 Approach-1 artefacts are quarantined [DECIDED]

`rl_env.py`, `rl_env_full.py`, `rl_env_v4.py`, `train_dqn*.py`, `train_mlp*.py`, `eval_routers.py`, `preflight_reward_v4.py` and `results/checkpoints/*.pt` belong to a separate lineage: a flat 22-column feature vector (not `features_v2`'s graph schema), the v1 link model with no interference argument, and exactly one packet per episode (`p_q = −0.2·occupancy` is identically zero in every reward ever computed). The checkpoints are ~134-byte **git-LFS pointer files**, not weights. This is a scope mismatch, not a defect: the study stands and should be cited properly; the artefacts must never be loaded as an Approach-2 warmstart.

---

## 5. The simulator — developer-level specification (`simulator_v2.py`, `link_model_v2.py`)

### 5.1 Time model and main loop [VERIFIED IN SOURCE]

Two clocks. A **frame** (`FRAME_DT = 0.5 s`) is the mobility/topology tick; a **slot** (`SLOT_DT = 0.01 s`) is the channel/forwarding tick; `SLOTS_PER_FRAME = 50`. Rebuilding the O(N²) graph every slot would cost 50× for no physical benefit.

`FANETSimulatorV2.run()` executes, per frame:

```
1. mobility + idle energy                 (DroneRWP.step for each node)
2. _build_graph()                         topology rebuild; STAMPS queue_occupancy / queue_len onto nodes
3. generate this frame's packets          enqueue at source (suppressed during the drain phase)
4. for each of 50 slots:
       _update_activity()                 activity[k] = min(ACT_ALPHA·occ + ACT_BETA·[nonempty], ACT_MAX)
       for each active packet at the head of a queue (SERVICE_RATE = 1 per node per slot):
           _try_forward(G, t, pkt)        routing decision → ARQ → admission → commit
5. per-frame diagnostics
```

**Important consequence of step 2 happening before steps 3–4:** every routing decision in a frame reads the queue state from *before* that frame's own traffic existed. This is the root cause of the "`very_dense` zero-occupancy anomaly" (§7.5-d) and must be fixed before RL.

Episode = `duration` seconds of traffic generation + `drain_time` (10 s) with generation suppressed. `network_pdr = delivered / generated`; `pdr_predrain` excludes packets generated inside the final drain window — verified identical to 4 decimals in all 12 old cells, so in-flight packets are not miscounted.

### 5.2 Constants [VERIFIED IN SOURCE]

| Group | Constant | Value |
|---|---|---|
| Packet | `TTL` | 20 hops |
| Queue | `MAX_QUEUE` | 50 packets, tail-drop |
| Service | `SERVICE_RATE` | 1 pkt/slot/node = **100 pkt/s per node** |
| Delay | `HOP_DELAY_MS` | 2.0 ms per attempt (+2 ms on commit) |
| ARQ | `DEFAULT_MAX_RETX` | 5 retries (6 attempts) |
| Activity | `ACT_ALPHA` / `ACT_BETA` / `ACT_MAX` | 1.0 / **0.0 (locked)** / 0.5 |
| Interference | `LETHAL_SINR_LINEAR` | 10 dB decode knee |
| Energy | `INITIAL_ENERGY` | 100.0 default — **overridable per config since v14; 8000 at the new operating point** |
| Energy | `TX_COST` / `RX_COST` / `IDLE_COST` | 0.02 per attempt / 0.01 per receive / 0.001 per s (`models.py`) |
| PHY | TX / noise / freq / path-loss exp | 20 dBm / −95 dBm / 2.4 GHz / 2.0 (free space) |
| PHY | RX sensitivity / shadowing σ | −85 dBm / 5 dB |
| MAC | Bianchi CW_MIN / stages | 16 / 6; carrier-sense = 1× range, interference = 2× range |
| Packet | size | 1024 bytes |

At the default battery, a node dies after 100 / 0.02 = **5,000 transmit attempts**. That number is what created the "energy wall" (§10.6).

### 5.3 The link model (M1, `link_model_v2.py`)

- **SINR, not SNR:** `SINR = P_signal / (N0 + Σ P_interferer)`.
- **Hidden-terminal interference:** interferers at receiver *j* are nodes inside interference range of *j* but **outside carrier-sense range of the transmitter *i*** (nodes near *i* defer under CSMA). Of those, only interferers that individually push SINR below the 10 dB knee are **lethal**. The measured link state is `p_clear = Π (1 − activity_k)` over lethal candidates — i.e. **P(no lethal interferer fires this slot)**. This, not expected interference power, is the correct quantity: link quality is convex in power, so evaluating at the mean badly underestimates mean quality (Jensen). The first fix attempt used expected power and collapsed `link_quality` to ~0.12 everywhere.
- **MAC contention (Bianchi 2000):** `unsaturated` is the default since the M-4 flip (`bb1f0a1`): each carrier-sense neighbour's *own* activity is passed in, because heterogeneity (congested next to idle) is the congestion-collapse mechanism. The old `saturated` path rounded summed activity to an integer station count, returning exactly 0 for n ≤ 1 — so `p_collision` was **exactly zero** across activity 0.02–0.09, then jumped discontinuously. Retained only to reproduce pre-flip results. **The collision flip moved SP-BP's margin by −0.7% and changed no ranking** [MEASURED].
- **Log-normal shadowing** σ = 5 dB — drawn **per transmission attempt inside the ARQ loop**.
- **Link quality seen by teachers:** `lq = lq_clean · p_clear · (1 − p_coll)`; PER `= 1 − (1−per_clean)(1−p_interf)(1−p_coll)`. The `(1 − p_coll)` term in `lq` was added in the M3 audit; before it, every teacher scoring on `link_quality` was blind to MAC contention (`p_coll` is 0.10 at 2 contenders and 0.58 at 8).
- **Backward compatibility:** at zero interference, contention and shadowing, the model reduces to the Approach-1 model to **3.5×10⁻¹⁵** — which is what licenses a clean interference on/off ablation.

**The connectivity graph is a unit-disk graph** [VERIFIED IN SOURCE]. `link_exists()` passes `shadowing_db = 0.0` explicitly ("shadowing affects quality/PER, not existence"). Edge existence is a deterministic function of 3-D Euclidean distance. Consequences:
- hop distance is essentially a function of distance by construction — the measured geo-vs-hop correlation **r = 0.89–0.93** reflects the construction, not the channel;
- shadowing is **not obstacle modelling** on three counts: it does not gate connectivity, it is not spatially correlated, it does not persist (redrawn per retry).
- **Correct scope sentence for the paper:** *"the connectivity graph is a unit-disk graph in which hop distance is a deterministic function of Euclidean distance."* The natural robustness experiment (not yet run) is persistent geometric blockage.

**The simulator is 3-D** (`DroneRWP` has x, y, z, vx, vy, vz; vertical motion capped at 30%). An early report draft said "open 2-D space" — that was wrong (defect ledger #1). With altitude span ≪ 2×range the geometry is a thin slab, so connectivity is quasi-planar.

### 5.4 The congestion-coupling mechanism — proven, and unique to this work

```
queue fills → activity = f(occupancy) (ACT_ALPHA = 1.0) → Bianchi p_coll ↑ and lethal-interferer p ↑
          → link quality ↓ → more ARQ retries (more airtime, more energy) → queues fill further
```

**Evidence [MEASURED]:**
- **G1:** mean link quality 0.972 → 0.345 as activity rises 0 → 0.20; fraction below 0.90 rises 0.1% → 88.4% (current run, after v17).
- **G2, old operating point (40 s, `dijkstra`):** offered load rose 8× (0.25 → 2.0), link errors rose ~70×. Isolation: same load, interference **on** PDR 0.311 / 25 link errors vs **off** PDR 0.375 / **exactly 0**. *(Pre-v12 numbers; the qualitative isolation was re-confirmed after the fixes — see next line.)*
- **G2, current (post-v22):** at rate 50, interference on: PDR 0.620, 1,656 link errors; off: PDR 0.893, **0** link errors.
- **Panel work (§12.6):** at the final operating point, **link errors exceed queue overflows by 1.3× to 6× in every measured cell.** In this simulator congestion shows up *mainly as degraded links*, secondarily as full queues.

**Still unproven:** that a *worse routing policy at the same offered load* causes *more* congestion than a better one. That is F1 itself, and it is ahead.

**Two calibration decisions made during M2 bring-up** (both discretionary, both disclosed):
1. **ARQ (`max_retx = 5`).** Without retries, PER was a near-step function of SINR — one hidden-terminal firing ≈ packet death — so interference acted as spatially-uniform random loss no router could avoid; measured actor spread ~0.008. With ARQ, interference becomes a graded delay/energy cost; spread appeared immediately (up to +0.06 PDR). *This is the largest discretionary intervention in the environment.*
2. **`ACT_BETA = 0`.** Interference is driven only by real backlog, concentrating it at genuine hot-spots rather than as ambient noise.

The M-5 calibration sweep (one-at-a-time: `act_beta` ∈ {0.02, 0.05}, `max_retx` ∈ {3, 8}, `act_max` ∈ {0.3, 0.7}, interference range ∈ {1.5×, 2.5×}) kept the winner and all four checked qualitative claims in every variant [MEASURED, pre-v12 — see §13]. Interactions were not explored (stated limitation).

### 5.5 Forwarding, drops and invariants (`_try_forward`)

Per decision: `_select_next_hop` (actor call, then **loop-avoidance override** — if the actor picks an already-visited node, `unvisited[0]` in arbitrary NetworkX order is substituted and `n_overrides` is incremented); energy check at the forwarder; **ARQ loop** (each attempt re-samples interference and shadowing and charges TX energy); on success, **v12:** check `next_hop == dst` *before* queue admission; tail-drop admission at the receiver only for packets that still need forwarding; commit (RX energy, path, hops, delay).

**Drop taxonomy:** `queue_overflow`, `link_error`, `ttl_expired`, `no_route`, `energy_depleted`, `episode_end`.

**Invariants added in the verification era [VERIFIED IN SOURCE]:**
- `_assert_queue_conservation()` — **queue-slot** conservation: slots held == packets actually buffered (`n_phantom_slots` must be 0). Packet-*count* conservation passed 8/8 **with the leak present** — it was the wrong invariant.
- `_assert_node_id_invariant()` — graph node ids are `0..N−1`, because `cand_flat` global ids double as frame-local embedding rows.

**Loop-override rates [MEASURED, old point]:** dijkstra 0.000 · spbp 0.088–0.109 · gpsr ~0.179 · da_gpsr 0.194–0.258 · car 0.195–0.261 · spbp_lookahead 0.220–0.344 · dpp ~0.447 · backpressure 0.449–0.462. The override contaminates **teacher comparisons** (it replaces a teacher's choice with an arbitrary one, far more often for wandering-prone teachers) but **not the dataset**, because the dataset generator re-scores on the visited-excluded set and the override never fires there. The corrected panel (§12) did not condition on override — an open refinement (§16).

### 5.6 Traffic and scenarios (`config_v2.py`)

**Flows:** `num_flows = N // 4`, each flow a fixed `(src, dst)` pair drawn by `rng.choice(N, 2, replace=False)` with **no reachability check** (deliberate — it creates partition stress). **Convergecast (v8a):** if `sink_node` is set, every flow's destination is that node and the source is drawn from the rest; **v9** pins the sink at the area centre at `z_min` with zero velocity and a construction-time assertion (before v9 the "ground station" moved 156.4 m in 10 s from (517, 424, 215)). The sink branch leaves the Suite A `rng.choice` call byte-identical, so Suite A results are unchanged (verified bit-identical over 12 cells).

**Suite A** (the four original cells — node count, area, range and speed all vary together, a known confound):

| scenario | N | flows | area (m) | range (m) | speed (m/s) | old reachability | old mean candidates |
|---|---|---|---|---|---|---|---|
| `very_dense` | 45 | 11 | 700² | 250 | 5–15 | 100% | 17.92 |
| `dense_slow` | 30 | 7 | 800² | 250 | 5–15 | 99.2% | 9.14 |
| `medium_slow` | 30 | 7 | 1300² | 280 | 5–15 | 54.7% [UNSOURCED] | 4.01 |
| `sparse_fast` | 20 | 5 | 1500² | 300 | 35–50 | 21.4% [UNSOURCED] | 2.68 |

The 54.7% / 21.4% reachability figures appear only in prose; nothing at `ebc013d` computes them. Direct measurement gave pair reachability 0.83 / 0.46 and packet-level routable fraction 0.790 / 0.494 — flagged, not silently replaced (they may be a differently-defined quantity).

**Suite B** (`density_{50,100,150,200}`): fixed 1000×1000 m, range 250, speed 10–30; **node count only** varies — removes the four-factor confound. Expected neighbours 8.59 → 34.91 [COMPUTED]. **Evaluation-only** [DECIDED].
**Suite C** (`sink_{50,100,150,200}`): Suite B + one pinned ground sink. The only configuration directly comparable to HCPMR/CQMR/IQMR. **30% of training volume, added on top of Suite A** [DECIDED].
**Tall probe** (`tall_probe`, altitude 100–600): genuinely 3-D connectivity; report separately, never pool with Suite A.

**`BASE` today:** `z 50–150, duration 40 s, drain 10 s, interference on` — this is deliberately still the **parity reference**. The new operating point is staged in v8b (§11). `RATES` today: `[0.5, 2.0, 4.0]` (stale). Every consumer imports from `config_v2` (v11 — eight independent copies had existed; see defect ledger).

### 5.7 Extension points used by everything downstream

- `_select_next_hop(G, pkt, neighbors)` — overridable; the dataset generator, rollout evaluator and the future RL environment all plug in here without duplicating the ARQ/queue/energy logic.
- `_build_graph()` — overridable; the headroom experiment overrides it to maintain a frame counter for its reachability cache (v10).
- `_on_packet_generated(G, pkt)` — hook for per-packet bookkeeping (routability).
- **There is no state serialisation** (`__getstate__`, checkpoint, restore). Frames store *observations*, not simulator state. This blocks mid-trajectory sampling and is a Phase-0/Phase-2 item (§18).

---

## 6. The teachers — exact formulas and known pathologies

### 6.1 The full registry [VERIFIED IN SOURCE]

`simulator_v2.TEACHERS` registers: `dijkstra, gpsr, backpressure, spbp, da_gpsr, dpp, car, spbp_lookahead, lq_dijkstra, arq_etx, etx_dijkstra, spbp_k1..k4, spbp_kinf, queue_aware_greedy (alias of da_gpsr), random`. `PANEL` (the 8-teacher M3 panel) = `dijkstra, gpsr, backpressure, spbp, da_gpsr, dpp, car, spbp_lookahead`. The SP-BP ablations `spbp_ab_*` are registered by `experiment_spbp_mechanism` on import.

| teacher | base family | congestion signal | link quality | score (maximised unless stated) |
|---|---|---|---|---|
| `dijkstra` | shortest path | none | none | min hop count |
| `gpsr` | geographic | none | none | neighbour closest to destination |
| `da_gpsr` | geographic | **absolute**, candidate's own `queue_occupancy` (0–1) | additive | `1.0·progress − 1.0·occ(u) + 0.5·lq(v,u)`, progress = (d(c,dst) − d(u,dst)) / max(d(c,dst), 1) |
| `car` | geographic | **absolute**, mean occupancy of the candidate's **neighbourhood** | additive | `1.0·progress − 1.0·field(u) + 0.5·lq` — *identical to da_gpsr except the field* |
| `spbp` | shortest path | **differential** `Q_v − Q_u` on `queue_len` (0–50) | multiplier | `lq(v,u) · [(Q_v − Q_u) + 1.0·(h(v) − h(u))]`, h = BFS hop distance to dst; unreachable excluded; dst short-circuit |
| `backpressure` | **none** | differential | multiplier | `lq · (Q_v − Q_u)` — no destination term |
| `dpp` | **none** | differential | via expected attempts | `(Q_v − Q_u) − 2.0·(attempts − 1)` — no destination term |
| `spbp_lookahead` | shortest path | differential + 2-hop backlog | multiplier | SP-BP with second-hop visibility |
| `lq_dijkstra` | shortest path | none | **bounded** path weight | Dijkstra, `w = 1 / max(lq, 0.05)^1` |
| `etx_dijkstra` | shortest path | none | **unbounded** path weight | Dijkstra, `w = 1 / max(1 − PER, 1e-3)` |
| `arq_etx` | shortest path | none | ARQ-exact | retained for reporting only |
| `random` | — | — | — | uniform neighbour (G3 sanity floor) |

**SP-BP ablations** (`experiment_spbp_mechanism.py`): `spbp_ab_full` (must equal `spbp` exactly — it is a free regression check), `spbp_ab_noqueue` (queue term removed — becomes hop-count + lq), `spbp_ab_candqueue` (differential → absolute on the candidate), `spbp_ab_additive` (lq multiplicative → additive).

### 6.2 Pathologies found and fixed in the teachers (M3 rounds, full list in §21)

- **Blind teachers** (round 1): graph built with `interference_mw = 0` → PER identically 0 on every link; 4/6 teachers byte-identical.
- **Backpressure silently became GPSR**: a defensive `Q_v − Q_u > 0` gate almost never held at occupancies of 0.02–0.14, so it fell through to greedy progress ~100% of the time. Fix: unconditional argmax (Tassiulas–Ephremides). **DPP repeated the same mistake later** (85% fallback via a `score > 0` gate) — caught because it produced byte-identical PDR to GPSR.
- **Diagnostic measuring the wrong thing**: a "58% fallback rate" sampled on a frozen post-episode graph; rebuilt to count live during the rollout. A later rewrite then deleted the counter increments; cross-verified against the simulator's own counters.
- **Shadowed duplicate definition**: two `lq_dijkstra_next_hop` functions; the first (additive form, claimed to beat Dijkstra) was dead code. Quarantined as `lq_dijkstra_additive_unverified`, unregistered.
- **Three SP-BP re-implementations diverged from the panel on partitioned graphs**: `spbp_ab_full` (0.4061 vs 0.4128 → fixed in v3, 12/12 exact), `spbp_khop(k=∞)` (279/345 agreement, all 66 disagreements on partitioned cases → fixed in v4), and the dataset label picker (22.6% fallback → `spbp_pick_restricted`). **Standing rule since:** any script that re-implements a teacher must carry an equivalence control verified to fail on a broken variant.

### 6.3 The structural fact that explains most of the panel

**`backpressure` and `dpp` have no destination term at all.** In a lightly loaded network they have no idea which way the destination lies — their own source says "an empty network gives all-zero weights and the argmax degenerates to the first neighbour — effectively a random walk." Their losses (10–28 pp in every cell, §12) are a statement about that, **not** about congestion-awareness. Retraction 1 (§13.3) came from missing this.

### 6.4 Another structural fact: SP-BP's differential was inoperative at the old point

At the old (leak-era) operating point the packet is dequeued before its decision is recorded, so `Q_v ≈ 0` and `(Q_v − Q_u) ≈ −Q_u`: replacing the differential with candidate-only queue cost **exactly 0.0000 PDR in all 12 cells** [MEASURED, pre-v12]. G3's check 2 ("backpressure family tops congested high load") was passed by SP-BP *while performing no backpressure at all* — so it must never be cited as evidence that backpressure-style routing wins. At the new operating point the differential and the absolute forms **do** differ (+2.09 pp mean in favour of absolute, 9/9 cells, §12.5).

---

## 7. Features and the dataset (M3.5)

### 7.1 Feature schema v5 (`features_v2.py`) [VERIFIED IN SOURCE]

| Block | n | Features |
|---|---|---|
| **NODE** | 9 | `x, y, z, vx, vy, vz, queue_occupancy, energy, degree` |
| **EDGE** | 4 | `distance, link_quality, estimated_link_lifetime, relative_velocity` |
| **QUERY** | 6 | `ttl_left, dist_to_dest, current_queue_occupancy, neigh_buffered_packets, neigh_mean_occupancy, hop_distance_to_dst` |
| **CANDIDATE** | 4 | `progress, cand_hop_distance, is_destination, cand_reachable` |

`FEATURE_SCHEMA_VERSION = 5` · `LOCAL_HORIZON = 2` (the two neighbourhood aggregates are computed over the 2-hop neighbourhood — what OLSR-family HELLO exchange already provides; `None` would be the controller-assisted whole-network variant) · `BUFFERED_REF = 500` · `LIFETIME_REF = 60` · `HOP_CAP = 10`.

**Removed on measured evidence** (not inspection): `queue_len` (duplicate of `queue_occupancy`), `snr` (deterministic, invertible function of `distance`, r ≈ −0.93), `hops_so_far` (`= 1 − ttl_left` exactly, r = −1.000), `packet_error_rate` (≈ 1 − `link_quality`, |r| = 0.998, found by G3.5 check 8 on real data). **Added deliberately:** `cand_reachable` — adds no information but makes global reachability independently *maskable* (it is a global property a real drone cannot compute locally).

**Global-BFS features (the "genie" features):** `hop_distance_to_dst`, `cand_hop_distance`, `cand_reachable` — all computed by BFS from the destination. These are the three columns masked in the decentralised design (§8).

**Normalisation (`norm_constants`)** is persisted per scenario in the manifest and must be reused, never recomputed. **v17** fixed `initial_energy` (hardcoded 100 → read from cfg; at battery 8000 the energy feature would otherwise have read ≈80.0 instead of ≈1.0 in every row, undetectable by any gate); **v18** did the same for `max_queue`, `ttl`, `lifetime_ref`. `HOP_CAP` (a scoping decision) and `BUFFERED_REF` (ungrounded, not duplicated) were deliberately left alone.

### 7.2 Generation (`generate_dataset_v2.py`)

`DatasetSimulator` subclasses the simulator and overrides `_select_next_hop`. Per decision it: computes the canonical, visited-excluded candidate set; computes the oracle **label** with `spbp_pick_restricted(G, c, dst, cands, h_map)` — hop distances on the **full** graph, choice restricted to legal candidates (the first attempt deleted visited nodes and re-ran BFS on the pruned graph, severing paths in sparse topologies and sending 22.6% of labels to a nearest-to-destination fallback); queries all 8 panel teachers for `votes`; draws the **ε-deviation** (`EPSILON = 0.10`, measured 0.0992) — with probability ε the simulator takes a random candidate instead of the label; records the decision; returns the **action** actually taken. After the episode, `eventual_delivered` and `drop_reason` are backfilled per decision by `packet_id`. `assert_no_drift()` runs at import and pins the restricted scorer against the real `spbp_next_hop`.

**Storage:** ragged flat buffers + offsets (never object arrays, never a fixed candidate cap — max degree reached 39 against Approach 1's cap of 15). Files: `frames.npz`, `decisions.npz`, `manifest.json`.

### 7.3 The dataset on disk [MEASURED]

**533,237 decisions · 48,000 frames · 600 episodes** (4 scenarios × 3 rates `[0.5, 2.0, 4.0]` × 50 seeds 101–150), 40 s, altitude 50–150, battery 100, oracle SP-BP, `label_fallback = 0.0000`.
**Split by (scenario, seed), never by row:** train 101–135 → 236,541 · val 136–142 → 48,012 · test 143–150 → 57,206 · **generalisation = all of `medium_slow`** → 191,478 (36%). *(PROJECT_STATUS reports 533,200 / 236,531 / 47,983 / 57,044 / 191,642 — the pre-v6 regeneration; FILE1's figures are the later ones.)*

**Trivial baselines:** nearest-to-destination = **71.14%** top-1; excluding the 28.8% of decisions whose label *is* the destination, the **contested floor is 59.46%**. Cross-check from independent arrays: mean `is_destination` (0.035) × mean candidates (8.13) = 0.2846 vs measured 0.2881. **Per-scenario contested floors:** very_dense 0.2529 · dense_slow 0.4147 · medium_slow 0.6832 · sparse_fast 0.8035 — driven by candidate count. **Always score per-scenario**, or generalisation appears to beat test through candidate count alone.

### 7.4 Gates on the dataset

**G3.5** (8 checks, `preflight_dataset_v2_check.py`) — structure, label fallback, trivial baseline, ε rate, feature liveness, **check 8 pairwise redundancy** (Pearson for linear duplicates, Spearman for monotone-nonlinear — verified to fail on a planted |r|=1.000 duplicate and on a monotone one with linear |r| = 0.976, below the Pearson threshold), plus a **saturation diagnostic** (added in v6 after `neigh_buffered_packets` was clipped on 38.8% of decisions). **Independent audit** (7 checks, `audit_dataset_v2.py`) — rebuilds each frame's graph from disk and re-derives labels without calling the generator (agreement 0.9998; residual = float32 quantisation), referential integrity (0 violations in 20,000 samples), zero cross-split frame leakage. The audit caught what G3.5 missed: an oracle-vote inconsistency (label and `votes['spbp']` by different code paths) and a **42,000×** performance bug (lazy `.npz` decompression inside a per-decision loop — spotted from a Task Manager screenshot showing 4% CPU).

### 7.5 What is wrong with the current dataset — the regeneration blockers [VERIFIED IN SOURCE]

| | Defect | Severity | Why no gate caught it |
|---|---|---|---|
| a | **`action` computed but never saved** — only `label` and a `behaviour_deviated` bool reach `decisions.npz`; ~10% of transitions pair a state with an action that did not produce the successor. Unrecoverable post hoc (the RNG draw is not stored). | critical for RL | audit check G / G3.5 check 4 test only the deviation *rate* |
| b | **`drop_reason` computed then discarded** — the only field separating congestion from link failure | high | not checked |
| c | **`hop_succeeded = True` constant** ("a recorded decision always attempted a hop") and not saved | medium | not checked |
| d | **Queue features are a stale start-of-frame snapshot** (§5.1). Own-node occupancy reads 0.00% nonzero vs 4.5–12.4% live; candidate occupancy stayed 72–94% nonzero, which is why imitation/M4 still worked | high for RL, harmless for imitation | — (the FILE1 "dequeue-before-decision" hypothesis was wrong; `audit_dataset_v2.py:465` repeats it) |
| e | **Stride 10 would destroy `(s, a, s′)` contiguity** — was justified on size alone | high | design conflict, not a code bug |
| f | `eventual_delivered` has survivorship bias (per packet, not per hop) | medium | — |
| g | **No simulator-state serialisation** → mid-trajectory resumption impossible | structural | — |
| h | `ORACLE_TEACHER = 'spbp'  # G3: wins all 12 cells` — the comment is false, and editing the constant alone would silently mislabel everything, because the label is computed by the hard-wired `spbp_pick_restricted()` regardless | critical | — |
| i | `load_bucket()` thresholds `≤0.5 / ≤2.0 / else` — every new rate (30–160) maps to `high` | high | — |

### 7.6 The regenerated schema — superseded by `docs/DATASET_V3_SPEC.md` (implemented in v26)

> The paragraph below is the pre-2026-10-02 plan. V3 changed: single da_gpsr label (no
> `label_teacher`), schema v6, no `checkpoint_id`, `eps_fired` instead of the
> `action != label` check, de-duplicated contexts + packet-sampled steps.


Feature schema carries forward unchanged. Generation-time changes: save `action`, `drop_reason`, honest `hop_succeeded`; per-cell (or single global) oracle with a **generalised restricted picker per teacher** + drift pins; `label_teacher` (if per-cell); `behaviour_teacher` (70/30 behaviour mix); `checkpoint_id` (once checkpointing exists); scenario-relative `load_bucket`; live queue reads; stride 1; `run_params` and a new **`record_schema_version`** in the manifest (because `FEATURE_SCHEMA_VERSION` only covers the four feature lists, two structurally different datasets would otherwise both declare v5); a new **audit check for action consistency** (`behaviour_deviated == (action != label)` row-by-row, verified to fail on a broken variant); restate D2/D5 wording. Full list in §18.

---

## 8. The learned model (M4)

### 8.1 Architecture (`model_gnn_attn.py`) — plain `torch.nn`

```
Stage 1  encoder, ONCE per frame, cached:
           h = Linear(9 → d=128); repeat L=2: h = LayerNorm(h + DenseGATLayer(h, adj_mask, edge_feat))
           (4 heads × 32; edge features as a per-head scalar attention bias)
Stage 2  query  q   = MLP([h_current, h_dst, query_feat(6)]) → d
Stage 3  keys   k_u = MLP([h_u, edge_feat(v,u), h_dst − h_u]);  logit_u = ⟨q, k_u⟩/√d;  invalid → −∞;  softmax
Loss     cross-entropy against the oracle label (pure classification; no value head)
```

**Matched control:** `mixer='mlp'` replaces graph attention with a node-wise MLP of the same depth and width. Parameters: **336,168 (GNN) vs 335,872 (MLP)** — 0.1% apart. Message passing verified by **gradient reachability**: a node 2 hops away has |∂out/∂in| = 2.53×10⁻³ for the GNN and **exactly 0** for the control. A third mixer, `attention_edgekey` (338,176 params, edge features in keys/values), was built but **not run at scale**. Assertions in the model: adjacency symmetry (a triangular mask would silently make a directed message passer) and encoder cache hit rate ≈ frames, not decisions.

**Tuned config (final):** `lr = 1e-3, attn_dropout = 0.1, max_epochs = 100`. The default (`lr 3e-4, attn_dropout 0`) was ~3 pp suboptimal for both; the earlier dropout-parity change had inadvertently handicapped the GNN.

### 8.2 Results [MEASURED, n=30 paired seeds, old dataset]

**Accuracy (contested):**

| arm | GNN | MLP | GNN − MLP | 95% CI | p |
|---|---|---|---|---|---|
| all features | 0.9124 | 0.9229 | **−0.0105** | [−0.0141, −0.0069] | 1.8e-06 |
| global-BFS features masked | 0.7559 | 0.7160 | **+0.0399** | [+0.0363, +0.0435] | 5.2e-20 |
| **difference of differences** | | | **+0.0504** | [+0.0460, +0.0548] | 2.3e-20, Cohen's d = +4.27 |

Cost of masking: GNN −0.1565, MLP −0.2069 — the MLP depends on those three features **32% more**; it cannot recompute them, the GNN partly can. **Message passing and hand-computed graph features are substitutes, not complements.** Robustness: MLP wins 6/6 HP configs and 60/60 seeds unmasked (5/6 survive Holm); the interaction replicates at +0.0472 / +0.0472 / +0.0504 across two HP settings × two mask definitions; holds OOD on `medium_slow` (−0.0189 → +0.0162, both p < 1e-13). **Notable null:** also masking the two locally-computable congestion aggregates changed nothing (+0.0003, p = 0.64) — the whole effect is the three global-BFS features.

**Rollout PDR (as % of SP-BP):** unmasked GNN 97.8% / MLP 98.9%; **masked GNN 98.0% / MLP 97.5%**. Paired MLP − GNN: unmasked +0.90 pp (p = 6.1e-09); masked −0.47 pp (p = 5.4e-03); DoD −1.37 pp (p = 1.6e-08, flips in 27/30 seeds). **Masking cost the GNN nothing — it gained +0.39 pp (p = 3.5e-03)**; the MLP lost −0.98 pp. The prediction that masking would lower PDR was wrong (ledger #6). Scope: the PDR reversal is significant on trained scenarios (−0.53 pp, p = 2.5e-03) but **not** on held-out `medium_slow` (−0.23 pp, p = 0.613); do not claim "regime-dependent".

**Methodological finding (transferable):** in the masked arm the accuracy gap (+5.08 pp) is **10.8×** the PDR gap (+0.47 pp). Top-1 imitation accuracy overstates the routing cost of *information removal* by about an order of magnitude — a model that cannot reproduce the oracle's exact choice often picks an equally good alternative. The ~1:1 accuracy→PDR translation holds only for *architecture comparison at fixed information*. Corroborated: k=1 and k=∞ horizons disagree on 2–13% of decisions with identical PDR.

### 8.3 G4 — all six checks

1 beats trivial floor — PASS (+0.29/+0.30 above contested floor) · 2 GNN beats MLP — answered: MLP wins with features, reverses under masking · 3 advantage grows with load — original **no**, replacement inconclusive, reported honestly · 4 rollout ≥ 90% of SP-BP — PASS (98.0/97.5 masked; 97.8/98.9 unmasked) · 5 generalises — PASS, per-scenario · 6 bit-reproducible — PASS 0.00e+00 after two fixes: the **`id(G)` encoder-cache bug** (CPython reuses freed addresses; one frame was routed with the previous frame's encoding; signature: every "random" drift was identically 2.86×10⁻² = one packet in 35) and CUDA determinism (`warn_only=False`; `CUBLAS_WORKSPACE_CONFIG` set at module import, before CUDA initialises — setting it inside a function is a silent no-op).

### 8.4 Architecture decision [DECIDED]

**Fully decentralised:** GNN (`mixer='attention'`, L=2) with the three global-BFS features **masked at training and at rollout** (v7 added masking to the rollout path, with presets imported from the trainer so they cannot drift, and a startup zero-assertion). Justification in paper order: no ground controller exists to provide BFS; obtaining it needs flooding with delay and staleness; L layers of message passing are the distributed analogue (L rounds of HELLO exchange); all competitors use this state class; masking is free or better in PDR. **The teacher stays genie-aided** — name it *learning from privileged information* (teacher–student).

### 8.5 What M4 did *not* produce

`results/m5_masked/` holds `check4_summary.json` and per-seed rollout JSONs only — **no `.pt` weights** (the masked rollout path does not persist them). A checkpoint-save step is required before warmstart transfer, and the current masked model was trained on the obsolete dataset anyway: **it must be retrained after regeneration** and re-gated against its *own* new-operating-point rollout PDR (not against 98.0%).

**[ASSESSMENT] How much of M4 survives the leak fix?** The GNN-vs-MLP comparisons are *relative* comparisons on a fixed imitation dataset, so their internal validity is not affected by the simulator defect in the way absolute congestion measurements are. But the dataset was generated on the leak-era simulator at an operating point now known to contain almost no real congestion, imitating an oracle now known to be the wrong one. Treat M4's findings as **established on the old dataset and to be re-confirmed** on the regenerated one — especially the rollout-PDR reversal, which depends on the environment.

---

## 9. The gates — what each tests, and how each has changed

### 9.1 G1 — interference makes link quality load-dependent (`preflight_interference_check.py`)
40 drones, 1500×1500 m, range 250, 150 topologies, 8,622 feasible links. Current run (after v17): activity 0.00 → 0.20 → mean lq 0.972 → 0.345, std 0.034 → 0.352, fraction < 0.90 0.1% → 88.4%, mean PER 0 → 0.657. **PASS.** *G1 has no regression anchor* — its numbers changed materially with the v2 lq fold-in (0.295 → 0.345 at a = 0.20) and nothing would catch future drift (open item).

### 9.2 G2 — multi-packet congestion dynamics are real (`preflight_simulator_v2_check.py`)
Six checks: queues nonzero and load-dependent; PDR degrades smoothly; drop taxonomy shifts to congestion; activity in a graded band (0–0.30); interference on/off differ; bit-reproducible with **no drift vs pinned anchors**.

History: the original anchors (dijkstra 87/280, spbp 112/280 at rate 1.0) were armed in v2 — the second, lq-sensitive anchor was added because `dijkstra` is blind to link-model changes (the M-4 flip moved it by exactly 0 while spbp moved 111 → 112). **After v12/v17/v18, G2 failed 6/6**: at its sweep `[0.25…4.0]` and anchor rate 1.0 the corrected simulator delivered 280/280 with zero drops. Re-baselined in three patches, each from direct measurement on your machine (`diagnostics/g2_*.py`):
- **v20:** anchor moved to `REGRESSION_RATE = 60` (a dedicated constant, decoupled from `mid`), where losses are mixed-cause: **dijkstra 8015/16800, spbp 7161/16800**.
- **v21:** sweep rebuilt from measurement.
- **v22:** `CONGESTION_CAUSES = ('queue_overflow', 'energy_depleted')` (link_error is G1's domain; with it, the share *fell* 100% → 39.4% because energy death grew 11.7×); sweep `[35, 40, 50, 60, 70]` because `mean_activity` is exactly 0 for rates 5–30 (it tracks queue build-up, which starts at 35).

**Current G2: PASS 6/6** — PDR 0.886 → 0.374 monotone, congestion share 31.8% → 82.1%, activity 0.006 → 0.011, on/off link errors 1656 vs 0 (PDR 0.620 vs 0.893), anchors reproduce. *Note:* G2 runs at the default battery, so its "congestion" includes energy death; that is intended for a regression gate but means G2 is not a congestion-regime certificate.

### 9.3 G3 — teacher panel (`preflight_teachers_v2_check.py`)
Final form (after 8 rounds, §10.1): 1 all teachers beat random at every load · 2 renamed backpressure-family check (see §6.4) · 3 oracle pick statistically justified (≥ 90% of cells) · 4 panel non-degenerate (≥ 2 orderings, runner-up varies) · 5 teachers disagree (vote agreement < 0.90; measured ~0.65) · 6 reproducible. Passed 6/6 with Holm 12/12 at the old point. **Its verdict (SP-BP everywhere) is void** (§13) — G3 has not been re-run at the new point; the oracle panels of §12 replaced it as the oracle-selection instrument.

### 9.4 G3.5 + audit — §7.4. G4 — §8.3.

### 9.5 Gates that do not exist yet
- **Phase-2 equivalence gate:** `run_iter()` steppable simulator must be **bit-identical** to `run()` across all 12 Suite A cells × 30 seeds.
- **Phase-3 parity gate:** `FANETEnvV2` driven by a teacher must reproduce that teacher's PDR **exactly** at the parity reference point (12/12). Originally specified against SP-BP; **must be restated against the new oracle(s)** (§17.6).
- **Gate G-A** (baselines): each baseline beats random at every load; each has an equivalence control verified to fail on a broken variant; panel comparison Holm-corrected ("12 cells" wording must be revised for per-scenario grids); cold-start curve (PDR vs episode, 30 seeds) for QMR and DQN.
- **Grid-in-band verifier** (proposed in LOAD_DENSITY_DESIGN): every scenario's grid must lie inside its measured usable band.

---

## 10. The verification era — how the current state was reached (chronological)

This is the long middle of the project. The recurring signature of almost every defect below is the same: **a plausible-looking number sitting on top of a dead or broken mechanism**, caught only by testing the mechanism directly.

### 10.1 M3's eight rounds (07-18 → 07-23)

| Round | What was wrong | Fix / lesson |
|---|---|---|
| 1 | Teachers blind to congestion — graph built with `interference_mw = 0` | measure channel state; then Jensen error (expected power → lq 0.12); correct quantity is P(≥1 lethal interferer) → lq 0.67 → 0.50 |
| 2 | Backpressure ≡ GPSR (defensive positive-differential gate) | unconditional argmax; diagnostic rebuilt to count live |
| 3 | The live counters were deleted by round 2's rewrite; LQ-Dijkstra "path-stretching" hypothesis | cross-checked counters; a 3-way head-to-head (incl. `arq_etx`) showed identical hop counts 1.89–1.95 — hypothesis wrong; plain Dijkstra beat all three. Mechanism: lq is informative (r = −0.38 with retries) but noisy and frame-stale; errors compound over a *global path* but not in a *one-hop greedy* decision |
| — | `medium_slow` degree 7.4 fell in the same class band as `dense_slow` | area 1000 → 1300; fourth "very dense" band added |
| 4 | "Regime dependence" was noise at 2–3 seeds | 30 seeds, parallelised (1,800 runs in minutes on 16 cores) |
| 5 | SP-BP wins all 12 cells, 10/12 above the 0.02 margin | "information superset" explanation (later falsified — §10.2) |
| 6 | Three hybrids added (DPP, CAR, Lookahead-SPBP) to test one-sidedness; DPP repeated round 2's gate bug; shadowed duplicate `lq_dijkstra` found | same fix; dead code quarantined |
| 7 | **Wrong statistical test** — unpaired Welch on paired data (per-seed PDR r = 0.89–0.98 across teachers): 1/12 significant, +0.060 margins scoring p = 0.16 | paired t-test: 12/12, p ≤ 0.0007. Synthetic check: same +0.062 margin p = 0.085 unpaired vs 7×10⁻¹² paired; null control stayed p = 0.69 |
| 8 | Checks 3/4 ("winner changes across regimes") could not distinguish a structurally dominant teacher from a broken panel | reframed to "oracle pick justified" and "panel non-degenerate", both verified to fail on synthetic noise before being trusted; PASS message corrected when it contradicted the degeneracy note printed above it |

### 10.2 The adversarial review and its experiments (07-30 → 08-08)

The work was reviewed as a hostile referee would: **19 findings, two potentially fatal.**

- **M-1** god's-eye information at zero control-plane cost (every teacher sees exact global BFS every 0.5 s); **M-2** M4's design fed that same feature to the "novel" model, confounding the depth ablation. **Answered by the locality experiment:** `spbp_khop` restricts SP-BP to a *k*-hop horizon (exact hop distance within *k*, geographic proxy beyond). After the v4 fix (k=∞ had diverged from panel SP-BP on partitioned graphs), gap(k=3) = **+0.0065 PDR** — locality is approximately free. At k=1 the horizon constant cancels and SP-BP's hop term *is* geographic progress; that it matches exact BFS falsified the "information superset" explanation for SP-BP's dominance (**M-3**).
- **Mechanism ablation:** removing SP-BP's queue term cost **−0.0615** (→ −0.0645 after v3); differential → candidate-only cost **0.0000** in all 12 cells; multiplicative → additive lq cost ≈ 0. Conclusion at the time: the queue term is the only structural feature that matters, and the differential is inoperative (§6.4). *All pre-v12 — see §13.*
- **Queue-weight sweep** (7,560 runs): increasing DA-GPSR's queue weight monotonically destroys it (0.377 → 0.133); w = 1 is its optimum, bracketed both sides. SP-BP's own queue-scale plateau is flat from 0.5 to 10 (best gain over the shipped value +0.001). **Teacher tuning was declared exhausted.** Residual SP-BP − DA-GPSR gap 0.0358 unexplained (progress-metric hypothesis refuted, +0.0024).
- **Headroom** (`experiment_headroom.py`): share of *routable* packets lost to routing-addressable causes (`queue_overflow`, `ttl_expired`, `no_route`; `link_error` deliberately excluded). 1.6% / 18.1% / 35.6% at low / medium / high load — later corrected by v10 to **1.4% / 17.5% / 35.1%** (overall 18.0%). `ttl_expired` was exactly 0 in all 12 cells, so the metric is essentially queue-overflow share and **rises monotonically into collapse by construction** — it must never be used to pick rates.
- **M-4** Bianchi applied outside saturation → unsaturated model built and made default (§5.3). **M-5** calibration sensitivity (§5.4). **M-7** no multiple-comparison correction (~46% family-wise error across 12 tests) → Holm primary, BH alongside; 12/12 survive.

**Pre-M4 fix pass (v1–v6):** v1 dropped `snr` / `hops_so_far`, added `LOCAL_HORIZON` scoping, G3.5 check 8, manifest schema gate, JSON output for two experiments that could not write results; v2 armed the G2 anchors (dijkstra + lq-sensitive spbp), RNG parity and the lq fold-in in G1; v3 fixed `spbp_ab_full`; v4 fixed `spbp_khop(k=∞)` and added `cand_reachable`; v5 dropped `packet_error_rate`; v6 fixed the 38.8% clipping of `neigh_buffered_packets` (`BUFFERED_REF` 100 → 500, schema 5). The dataset was regenerated once and re-gated once, as the M4 plan required.

### 10.3 Pre-M5 consolidation (09-02 → 09-08)

- **v10 — the `id(G)` bug had a second home.** `HeadroomSimulator._reach_cache` was keyed on `(id(G), src)` under a comment claiming `(frame_marker, src)`; the cache cleared only above 4,000 entries against ≤ 45 sources, so it never cleared. Fixed with a frame-scoped key; `--cache_mode {fixed, legacy}` keeps archived numbers attributable. **Verifier revision 3** replaced an allocator-dependent check (Linux reuses addresses 17–30% of the time, Windows/conda 0–1.7%, so rev 2's "legacy must differ" check failed falsely on your machine) with a **deterministic forced-staleness signature**: under a constant cache key, very_dense/dense_slow must still give 0 wrong verdicts (reachability ~100%) and medium_slow/sparse_fast must give some (got 203 and 111) — plus a negative control that fires when the key is reverted. 7/7. Legacy re-run reproduced the archived numbers to 0.05 pp, so the defect is the whole difference; dense cells moved exactly 0. Lesson (ledger #13): *error 9 recorded the instance, not the class — grep for the class.*
- **v11 — `config_v2.py`.** The same `SCENARIOS` dict existed in **eight** files, still in sync by luck; v8 would have moved only one of them. Now one module; verified by a **positive control** (mutate `config_v2.BASE` in memory, all 8 modules observe it). v11.1 made the suite-selector check v8a-aware with a negative control.
- **v7** — `--mask` in the rollout path (§8.4).
- **v8 / v9 staged, then split.** Original v8 bundled BASE (1000 s, 100–300 m) with the suite dicts. **Sequencing correction [DECIDED]:** do not move the operating point until the new RL environment's parity gate passes at the *current* point — it is the only fixed reference for telling "the environment is right" from "the operating point moved". So **v8a** (suites + sink support, no BASE change) and **v9** (sink pinning) were applied at `2492297` with Suite A verified bit-identical in 12 cells; the BASE change became **v8b** (§11).
- **v11.2** — `--duration/--z_min/--z_max/--drain_time` flags on the headroom experiment, `provenance()` actually written into the output (it had been imported and never called), and **`--by rate`** aggregation (without it, `load_bucket()` mapped all six probe rates to `low` and 24 cells would have silently averaged into 4). Backward compatibility verified byte-identical against `headroom_v10.json`.
- **CLAUDE.md** behavioural contract and **FILE1/FILE2 v2** committed (`02c3ca1`).

### 10.4 The rate-grid probe and its amendment (09-02 → 09-05) — later voided

**Why:** the old grid `[0.5, 2.0, 4.0]` collapses at 1000 s (dense_slow rate 2.0: PDR 0.3286 at 40 s → 0.0250 at 1000 s). A proposed `(0.05, 0.15, 0.4, 0.7, 1.0)` was flagged as likely wrong.

**Design (frozen in `PROBE_PREREGISTRATION.md` before running):** 4 scenarios × 6 rates `(0.02 … 0.40)` × 10 **paired** seeds × 6 actors (`spbp`, `spbp_ab_full`, `spbp_ab_noqueue`, `dijkstra`, `gpsr`, `random`). **Primary metric = queue value** = paired `spbp_ab_full − spbp_ab_noqueue` (matched control). `spbp − dijkstra` was **rejected** because it conflates queue value with link-quality value, which move oppositely in load. Also rejected: `R_rel = ΔPDR / headroom` (divides by a quantity designed to reach 0 — division by zero at dense_slow 0.5; selected 6/6 cells, discriminated nothing) — this was the proposal in the external "response1" you brought in; relative spread (inflates as PDR → 0); per-cell spread filtering (deletes the low-load anchor the thesis needs). Bands on point estimate + paired sign consistency ≥ 0.8, not on the CI lower bound (unusable at n=5–10). Four predictions recorded (all held).

**Result (1,440 episodes):** relative queue value looked **flat** (~14–22%) across a 20× rate range; absolute value fell only because PDR fell. Amendment 1 fixed three rule gaps found by the run itself — **A1** a peak may not come from a floor-band cell (sparse_fast had emitted a grid from a "peak" where 9/10 paired seeds were negative); **A2** the knee is the last ≥ 2% carried-load rise, not the argmax; **A3** test flatness before acting on a boundary argmax. Frozen grid **`[0.02, 0.05, 0.10, 0.25, 0.40]`**; sparse_fast **NO MEASURABLE EFFECT**. Two post-probe claims were **retracted** with the measurements that refuted them ("the 17% is an episode-boundary artifact"; "§12's high-load framing is contradicted").

**Then the Dataset V2 Master Spec (`74b1362`)** audited the dataset (§7.5), wrote 24 traced requirements R-1…R-24, and listed six open decisions. **Nine regeneration decisions were then frozen with you** (§14).

**All of §10.4's numbers were later voided by v12.** The frozen grid tops out at 2.8 pkt/s offered *network-wide* — 2.8% of one node's capacity — so after the leak fix there is no congestion anywhere in it (PDR ~0.99, zero overflow, zero link error). The "flat response" was the leak flattening the signal. The rule amendments (A1–A3) remain correct as rules.

### 10.5 The v12 destination-queue leak (discovered 09-08 → 09-18) — the pivotal defect

**How it surfaced:** the first convergecast oracle panel (Suite C, `sink_50`/`sink_100`, 1000 s) returned **all 8 teachers byte-identical** in 10/10 cells, e.g. 0.0850 at rate 0.05 — and the verdict printed "KEEP spbp; the Suite A assumption is now a MEASUREMENT." Delivery was pinned at **exactly `MAX_QUEUE = 50`** in 400/400 episodes across a 43× range of offered traffic.

**The defect [VERIFIED IN SOURCE]:** `_try_forward` enqueued every packet at `next_hop` **before** checking whether `next_hop` was the destination; delivery was detected afterwards and the packet removed from `active`, but its queue slot was **never released** (the only `buffer.remove` was for forwarded packets). TTL-sweep drops leaked the same way. Destination queues filled permanently.

**Impact [MEASURED]:** Suite A PDR decayed with episode length with **zero** overflow drops — 96.8% (200 s) → 89.5% (400 s) → 71.0% (700 s) → 63.9% (1000 s); episode-end audit: 102–356 slots held, 0 packets in flight (100% phantom). At 40 s, **88.4% of all recorded drops (3,394 / 3,840) were phantom**, and dense_slow / very_dense had **no real congestion at any old rate**. It went unnoticed for four milestones because it degraded smoothly rather than hitting a wall.

**Semantics decided with you [DECIDED]:** a delivered packet is **consumed on arrival and occupies zero buffer** (what real stacks and NS-3 do; the alternative needs new draining machinery; per-hop delay is already modelled). Side effect, deliberate: an arriving packet is no longer tail-dropped by its own destination's queue.

**Verification (`verify_delivery_leak_fix_v12.py`, 6/6):** queue-slot conservation (phantom = 0 everywhere), convergecast cap gone, duration independence (PDR 0.984 / 0.993 at 200 / 400 s with zero overflow), capacity sanity (rate 60, 420 offered → PDR 0.644, 96 overflows), node-id invariant, and a **negative control** that re-creates the leak and confirms the check catches it (strict mode raises).

**Consequence:** every congestion measurement made before v12 is void: the 1000 s probe, the headroom triple, the queue-term value (+0.0645), the M3 oracle verdict, the first convergecast panels (§13).

### 10.6 Finding a congestion regime that actually exists (09-08 → 09-18)

- **v13** retargeted the probe to per-flow rates `[20, 35, 50, 65, 80, 100, 120]` (flows × rate = offered load, reported alongside because the same per-flow rate is 550 p/s in very_dense but 250 p/s in sparse_fast). A dense_slow measurement in its docstring showed the queue term worth **exactly 0 until overflow exists**, then +6.19 pp @50, +6.37 @60, +2.17 @80, −0.76 @100 — i.e. **the inverted-U model the original probe "refuted" was correct**; the refutation was a leak artifact.
- **The energy wall.** `map_operating_regimes.py` (full loss accounting) found **no congestion-dominated cell anywhere**: the network went *clean → interference/energy* without passing through a queue-limited regime. Analytical energy ceiling (perfect spreading): at 1000 s the maximum per-flow rate before nodes die is **6.56** (dense_slow) / **6.26** (very_dense), while congestion needs ~40–80. **1000 s episodes and a congestion regime were mutually exclusive at battery 100.** `scan_congestion_window.py` had printed `q_ovf` and `link_error` but hidden `energy_depleted`, which dominated — its verdicts are void. `verify_oracle_congestion.py` (20 s, rates 160–260) measured SP-BP's queue term *hurting* (−2.98 / −2.39 / −1.83 pp) — but inside the **saturation plateau** (delivery pinned constant, independent of offered load), so its three "rates" were one operating point measured three times — void.
- **v14** made `INITIAL_ENERGY` config-overridable (default unchanged, bit-identical). **`sweep_energy_ceiling.py`** (dense_slow, 1000 s, 2 seeds): at **4000**, dead nodes and nonzero energy share at every rate above 40; at **8000**, energy share 0.000 and 0 dead at rates 60/80/120/160 (usable: all four). **Battery 8000 = the smallest value that opens a congestion window at 1000 s** [MEASURED]. Justification: 5,000 transmissions per battery was a calibration choice, not physics; a real UAV radio draws milliwatts against watt-hours, so raising it is *more* defensible.
- **The usable-band criterion** (`find_usable_band.py`): a rate is usable if **delivery elasticity** ∈ [0.05, 0.85] (elasticity = relative change in delivered packets ÷ relative change in rate vs the previous rate — responsive but congested), **q_ovf ≥ 0.02** (congestion is real), **energy share ≤ 0.05** (not measuring batteries) and **dead == 0**. Self-tests: S1 v12 present, S2 determinism, S3 paired-seed traffic identity, S4 no delivery collapse (tolerance 0.10, direction-aware display since v24). Phase 2 runs the matched queue ablation **only inside the band**, with a pre-registered **Tier 1** (`spbp_ab_full` vs noqueue, Holm across rates) and **Tier 2** (`candqueue`, `additive`), flat Holm shown for transparency.

**The band search moved through three durations** (the transfer was verified, not assumed):

| run | result | status |
|---|---|---|
| dense_slow, 20 s | band 40/60/80; queue term −1.25 (ns), −3.00, −3.92 pp (Tier 1 sig at 60, 80) | superseded |
| very_dense, 20 s | band 40/60; **helps** +2.66, +1.32 pp | superseded |
| medium_slow, 20 s | band 40/60; hurts −4.25, −5.00 pp | superseded |
| very_dense, **200 s**, battery 4000 | band 60/80/100; helps +2.86, +2.12, then **hurts −0.98 at 100** — the "sign flip" | superseded |
| **all five scenarios, 1000 s, battery 8000, 15 paired seeds** | the current operating point | **CURRENT** — §11 |

`LOAD_DENSITY_DESIGN.md` (never committed) was written from the 20 s / 200 s runs: two independent axes (density and load position), per-scenario grids, scenario-relative buckets, and an open question about including very_dense's sign-flip region in training. The 1000 s runs changed its inputs (very_dense's band became 60/80 only, with 100 saturated and 8.7 dead nodes), so its *principles* stand but its *numbers* are superseded (§22).

### 10.7 Convergecast (Suite C) at the true operating point

`find_congestion_band_convergecast.py` (written fresh; `sink_energy_frac` guard; midpoint-rate self-tests after the max-rate self-tests appeared "stuck" and 16 Windows spawn workers stalled). At 40 s, battery 8000: band 30/40/50/60, queue term **helps** +1.94 / +3.33 / +2.87 / +1.88 pp (all Tier-1 significant). At **1000 s, battery 8000: only rate 30 is usable** (elasticity 0.30); from 40 upward elasticity is **negative** (delivery *falls* as load rises — congestion collapse at the sink). At 30: queue term +3.55 pp (p_holm 0.0029). Convergecast conclusions therefore rest on **one cell** — a stated limitation.

### 10.8 Fixes that followed the band search (v15–v24)

| patch | what | why it mattered |
|---|---|---|
| band-script fixes | `holm()` called on a dict → `holm_dict`; missing `significant` key → `setdefault`; tier label `is not None` always true → membership test; single-rate elasticity NaN guard; S4 strict-monotone false fail → collapse tolerance | crashes / mislabelled verdicts |
| **v15** | `--initial_energy` added to the band script | the 8000 battery could not otherwise be passed |
| **v16** | panel `delta` sign flip: `means[winner] − means[t]` left over from an earlier argmax design → `means[t] − means[incumbent]`; `fix_panel_json_v16.py` repaired stored JSON | **inverted a verdict** ("backpressure beats spbp" should have read the opposite); p-values were unaffected |
| **v17 / v18** | normalisation constants follow cfg (§7.1) | energy feature would have been 80× too large at battery 8000 |
| **v19** | superseded-note added to `PROBE_PREREGISTRATION.md` §6.3 | the frozen grid is void, not outdated |
| **v20–v22** | G2 re-baselined (§9.2) | G2 failed 6/6 on the corrected simulator |
| **v23** | `run_params` retro-fitted onto six result files | `provenance()` reports the **40 s parity reference**, not the run; four of five band files recorded no battery at all |
| **v24** | S4 display made direction-aware (a "−64.6%, tolerance 10%, OK" line was correct but read as a violated threshold); `run_params` written by future runs | misreading risk |
| results index | `results/README.md`: every file marked CURRENT / SUPERSEDED / VOID | results from three operating points sat side by side unmarked |

A pre-regeneration audit at this point also flagged — without acting — that warmstart quality must not be compromised ("error will move onto the GNN and then get amplified in the RL phase", your words) and that sign-flip regions must not be used manipulatively to make D2 pass (§17.3).

### 10.9 The oracle panels (09-18 → 09-22) — §12 has the full results

Four experiments, **1,769 episodes**: the 9-teacher generalised panel (990) → the contender panel with per-seed storage (550) → the v25 decision-rule correction (recompute, 0 episodes) → the link-metric extension (220, equivalence-controlled merge) → the oracle-agreement probe (9). Then the teacher audit (reading every teacher's source), the controlled-comparison analysis, the Panel Results Report, the schema specification and its re-check.

---

## 11. The operating point and the per-scenario load bands (current truth)

### 11.1 The operating point [MEASURED, staged as v8b, NOT applied]

| parameter | parity reference (today's `BASE`) | **new operating point** | basis |
|---|---|---|---|
| duration | 40 s | **1000 s** | matches HCPMR / CQMR / IQMR; unblocks the `energy` feature (std 0.0205 at 40 s). *Corrected 2026-10-02:* it does **not** unblock `estimated_link_lifetime` — 45–73% of edges sit at the 60 s cap at every time of a 1000 s episode, because the estimator returns the cap for any non-separating pair. `LIFETIME_REF` stays 60 |
| altitude | 50–150 m | **100–300 m** | matches all three competitors; costs ~8% of density (thin-slab geometry) [COMPUTED] |
| drain | 10 s | 10 s | a 2–3 hop packet completes in well under a second |
| battery | 100 | **8000** | smallest battery that opens a congestion window at 1000 s [MEASURED] |

`apply_v8b_operating_point_STAGED.py` bundles all three into one atomic edit of `config_v2.BASE` (v8's own anchor targets `generate_dataset_v2.py` and can no longer match) and adds `initial_energy` to `provenance()`. It **requires an explicit confirmation flag** and must run only after the Phase-3 parity gate passes. It deliberately does not touch `RATES`.

**Density at span 200** [COMPUTED]: very_dense 15.75 · dense_slow 7.95 · medium_slow 3.87 · sparse_fast 2.21 neighbours. **Caveat for the paper:** span 200 against 2R = 500–600 m is still slab-like (ratio 0.33–0.40) — it matches the literature, it does not make connectivity genuinely 3-D.

### 11.2 Usable bands at 1000 s / battery 8000 [MEASURED, 15 paired seeds — `results/band_*_1000s.json`]

> Altitude 100–300 m for the four Suite A scenarios; **sink_50's band was measured at 50–150 m** (the convergecast band script had no altitude flags — added in v26). Re-measure it before Dataset V3 uses it (DATASET_V3_SPEC §2).

> **v28 (2026-10-04): every number in this section was measured with the waypoint-trapping bug** — at 1000 s, 37–51% of the 5–15 m/s drones and every sparse_fast / sink_50 drone end the episode frozen at a waypoint (DATASET_V3_SPEC §0 #15). The bands are re-measured after the fix; treat these as pre-v28.

**Phase 1 — elasticity sweep (bold = usable):**

| scenario (N, flows) | rate → PDR (elasticity, q_ovf) |
|---|---|
| **dense_slow** (30, 7) | 40 → 0.833 (—, 0.037) · **60 → 0.639 (0.30, 0.099)** · **80 → 0.511 (0.20, 0.158)** · **100 → 0.436 (0.27, 0.192)** · 120 → 0.363 (≈0, saturated) · 160 → 0.272 (0, saturated) |
| **very_dense** (45, 11) | 40 → 0.714 (—, 0.023) · **60 → 0.515 (0.16, 0.069)** · **80 → 0.392 (0.05, 0.117)** · 100 → 0.314 (≈0; energy 0.049, 8.7 dead) · 120 → 0.261 (saturated, dead) |
| **medium_slow** (30, 7) | 30 → 0.689 (—, 0.051) · **40 → 0.597 (0.47, 0.081)** · **60 → 0.444 (0.23, 0.141)** · **80 → 0.349 (0.14, 0.198)** |
| **sparse_fast** (20, 5) | 40 → 0.335 (—, 0.004) · 60 → 0.319 (0.86, 0.039) · **80 → 0.303 (0.80, 0.081)** · **100 → 0.294 (0.85, 0.107)** · 120 → 0.245 (0, saturated) |
| **sink_50** (50, 12) | 20 → 0.776 (—, 0.106) · **30 → 0.595 (0.30, 0.144)** · 40 → 0.421 (−0.17) · 50 → 0.330 (−0.08) · 60 → 0.254 (−0.38) |

**Phase 2 — matched queue ablation inside the band** (percentage points vs `spbp_ab_noqueue`; Tier 1 = `full`):

| cell | full | candqueue | additive | Tier-1 verdict |
|---|---|---|---|---|
| dense_slow@60 | −0.43 (ns) | **+2.20** | +1.55 | ns |
| dense_slow@80 | **−1.84** | +1.57 | +0.85 | hurts |
| dense_slow@100 | **−3.98** | −0.38 (ns) | −1.00 | hurts |
| very_dense@60 | **+2.66** | +3.28 | +3.07 | helps |
| very_dense@80 | **+1.48** | +2.71 | +2.44 | helps |
| medium_slow@40 | **−4.15** | −2.31 | −2.70 | hurts |
| medium_slow@60 | **−5.68** | −2.71 | −3.23 | hurts |
| medium_slow@80 | **−5.43** | −2.59 | −3.04 | hurts |
| sparse_fast@80 | −3.15 (ns) | −2.05 (ns) | −2.72 (ns) | ns, CIs ±4–5 pp |
| sparse_fast@100 | −3.25 (ns) | −2.18 (ns) | −2.22 (ns) | ns |
| sink_50@30 | **+3.55** | +3.95 | +3.69 | helps |

Energy share 0.000 and zero dead nodes in every usable cell.

### 11.3 How to read these bands

- **The density axis is real:** SP-BP's queue differential hurts in medium_slow (degree ~5) and dense_slow (~10), helps in very_dense (~19) and sink_50 (~13).
- **The absolute (candidate-queue) form beats the differential everywhere** it was measured — foreshadowing §12's Finding 1.
- **sparse_fast is qualitatively different:** its usable cells sit against the elasticity ceiling (0.80, 0.85 vs 0.05–0.47 elsewhere), link error is ~0.007, PDR ~0.30 — the dominant loss is neither link nor queue (most likely route unavailability; it is connectivity-limited). It is measured and reported, **never used to set an oracle** [DECIDED].
- **very_dense's 200 s "sign flip" at rate 100 is outside the 1000 s band** (rate 100 is saturated with dead nodes at 1000 s). At 1000 s, the load-dependent sign change instead appears in **dense_slow** (−0.43 ns → −1.84 → −3.98 as load rises).
- **Per-scenario rate grids are required** — a single global grid cannot express genuine congestion in all scenarios at once (their capacities differ by node count and topology). This supersedes the Dataset V2 Master Spec's "one global grid" decision, which is recorded as *superseded by measurement* (§15).

**Not yet decided:** the exact 5-rate grid per scenario for the dataset (the bands give 2–3 usable rates per scenario; whether to add below-band "low-load anchor" rates and above-band stress rates is §17.4), and how `sparse_fast` and `sink_50` (single usable rate) are gridded.

---

## 12. The oracle panels and the revised understanding of congestion-awareness

All at 1000 s, battery 8000, **altitude 50–150 m** (config_v2.BASE: the panel scripts set duration and battery but never altitude — corrected 2026-10-02; the bands that chose these rates ran at 100–300 m, except sink_50), **10 paired seeds per cell** (every teacher faces byte-identical traffic — verified by self-test S3). Cells: medium_slow 40/60/80, dense_slow 60/80/100, very_dense 60/80, sink_50 30 (**9 oracle-setting cells**) + sparse_fast 80/100 (**flagged**, reported only). Full report: `docs/Panel_Results_Report.md`.

### 12.1 Experiment 1 — 9-teacher generalised panel (990 episodes, ~20.4 h)

Mean PDR (bold = cell top):

| cell | dijkstra | gpsr | backpr. | spbp | da_gpsr | dpp | car | spbp_la | candq |
|---|---|---|---|---|---|---|---|---|---|
| medium_slow@40 | **0.5458** | 0.5050 | 0.3125 | 0.5109 | 0.5115 | 0.3029 | 0.4877 | 0.4928 | 0.5281 |
| medium_slow@60 | **0.4057** | 0.3666 | 0.2162 | 0.3570 | 0.3655 | 0.2083 | 0.3379 | 0.3337 | 0.3839 |
| medium_slow@80 | **0.3156** | 0.2811 | 0.1643 | 0.2708 | 0.2750 | 0.1570 | 0.2489 | 0.2456 | 0.2965 |
| dense_slow@60 | 0.5947 | 0.6187 | 0.3903 | 0.5946 | **0.6337** | 0.3679 | 0.5879 | 0.5597 | 0.6209 |
| dense_slow@80 | 0.4733 | 0.4986 | 0.3040 | 0.4621 | **0.5062** | 0.2824 | 0.4599 | 0.4193 | 0.4941 |
| dense_slow@100 | 0.3940 | **0.4205** | 0.2520 | 0.3713 | 0.4156 | 0.2284 | 0.3753 | 0.3267 | 0.4046 |
| very_dense@60 | 0.5140 | 0.5489 | 0.3154 | 0.5483 | **0.5614** | 0.2864 | 0.5386 | 0.5379 | 0.5549 |
| very_dense@80 | 0.3966 | 0.4294 | 0.2403 | 0.4135 | **0.4458** | 0.2131 | 0.4146 | 0.3893 | 0.4277 |
| sink_50@30 | 0.6724 | 0.6848 | 0.4708 | 0.7038 | **0.7164** | 0.4265 | 0.6842 | 0.6832 | 0.7095 |
| sparse_fast@80 ⚑ | **0.1791** | 0.1346 | 0.1150 | 0.1586 | 0.1379 | 0.1312 | 0.1427 | 0.1478 | 0.1700 |
| sparse_fast@100 ⚑ | **0.1682** | 0.1142 | 0.0905 | 0.1464 | 0.1209 | 0.0830 | 0.1131 | 0.1327 | 0.1561 |

Its printed verdict (incumbent-replacement rule: SP-BP replaced in 7/11) is **superseded**; the means are sound. Two gaps: `spbp_ab_noqueue` was missing, and only means were stored (no per-seed data).

### 12.2 Experiment 2 — contender panel (550 episodes, ~10 h)

`dijkstra, gpsr, da_gpsr, spbp_ab_noqueue, spbp`; **per-seed PDR stored for the first time**. Question: is SP-BP's problem its queue term (H1) or its whole approach (H2)? **H2.** `spbp_ab_noqueue` was recommended in 0/9 cells — it topped all three medium_slow cells but only by 0.10–0.17 pp over `dijkstra`, i.e. stripped of its queue term SP-BP *converges to shortest path*.

**The decision-rule bug (v25).** v2's leading-group rule excluded a teacher only if it was *both* significantly worse *and* ≥ 1 pp worse. At n=10, a teacher 4 pp worse often fails significance, stays in the "tied" group, and parsimony (simplest-first order: dijkstra < gpsr < da_gpsr < spbp_ab_noqueue < spbp) then picks it. **4 of 9 cells were wrong:** dense_slow@60 dijkstra (−3.90 pp vs da_gpsr), dense_slow@80 dijkstra (−3.29), dense_slow@100 dijkstra (−2.65 vs gpsr), sink_50@30 dijkstra (−4.40 vs da_gpsr). **Corrected rule [DECIDED]:** leading group = every teacher whose **point estimate is within 1 pp of the top**; significance is reported as confirmation, not used as the gate; parsimony breaks ties only inside a genuine tie. Fixed by recompute from stored per-seed data (`fix_oracle_assignment_v25.py`) and at source (`apply_leading_group_fix_v25.py`) — no re-run. *"Not significantly worse at n = 10" does not mean "equally good."*

### 12.3 Experiment 3 — link-metric extension (220 new episodes, ~4.2 h)

`lq_dijkstra` and `etx_dijkstra` merged into the stored contender data. **Equivalence control before merging:** re-simulated dijkstra @ dense_slow|60 seeds 1–3 and reproduced the stored per-seed values to 10 decimal places. **Both lost in every one of the 11 cells**, and the ordering is monotone in every cell: plain hop count > bounded link-quality weighting > unbounded ETX weighting (e.g. sink_50: −4.12 and −5.79 pp vs dijkstra). **No new teacher won a cell** — da_gpsr's advantage is not reducible to link quality alone.

### 12.4 The oracle of record — `results/panel_extended_v3.json` [DECIDED]

| cells | oracle | leading group |
|---|---|---|
| medium_slow @ 40, 60, 80 | **dijkstra** | {dijkstra, spbp_ab_noqueue, lq_dijkstra} — dijkstra by parsimony |
| dense_slow @ 60 | **da_gpsr** | {da_gpsr} |
| dense_slow @ 80, 100 | **gpsr** | {gpsr, da_gpsr} — gpsr by parsimony |
| very_dense @ 60, 80 | **da_gpsr** | {da_gpsr} |
| sink_50 @ 30 | **da_gpsr** | {da_gpsr} |
| sparse_fast @ 80, 100 | **flagged — no oracle set** (best: dijkstra) | {dijkstra, lq/etx_dijkstra, spbp_ab_noqueue, …} |

**SP-BP is displaced in all 9 oracle-setting cells.**

### 12.5 Controlled comparisons — the strongest evidence in the project

Pairs of teachers that differ in exactly one respect, across the 9 oracle cells (cells in order: medium_slow 40/60/80 | dense_slow 60/80/100 | very_dense 60/80 | sink_50 30; values in pp):

| # | comparison | isolates | positive | mean | per cell |
|---|---|---|---|---|---|
| A | `da_gpsr − gpsr` | adding queue + lq terms to a geographic base | 6/9 | +0.86 | +0.65 −0.11 −0.62 \| +1.49 +0.76 −0.49 \| +1.25 +1.65 \| +3.16 |
| B | `spbp − spbp_ab_noqueue` | adding a queue **differential** to a shortest-path base | **3/9** | −1.33 | −3.65 −5.01 −4.59 \| −1.26 −2.00 −3.60 \| +2.39 +1.19 \| +4.55 |
| **C** | `spbp_ab_candqueue − spbp` | **absolute vs differential** | **9/9** | +2.09 | +1.72 +2.69 +2.58 \| +2.63 +3.20 +3.33 \| +0.66 +1.42 \| +0.56 |
| **D** | `lq_dijkstra − dijkstra` | lq **path weighting** | **0/9** | −1.25 | −0.78 −0.81 −0.67 \| −0.78 −1.05 −0.83 \| −0.89 −1.34 \| −4.12 |
| **E** | `etx_dijkstra − lq_dijkstra` | unbounded vs bounded | **0/9** | −1.17 | −0.88 −0.54 −0.50 \| −1.83 −1.17 −0.64 \| −1.96 −1.32 \| −1.68 |
| **F** | `car − da_gpsr` | neighbourhood vs **candidate-own** queue | **0/9** | −3.29 | −2.38 −2.76 −2.60 \| −4.57 −4.63 −4.03 \| −2.27 −3.12 \| −3.22 |
| G | `spbp_ab_noqueue − dijkstra` | SP-BP minus queue vs plain shortest path | 8/9 | +0.45 | +0.17 +0.13 +0.10 \| +1.25 +0.88 +1.32 \| +1.05 +0.51 \| −1.41 |

**Established findings:**
1. **Formulation matters more than presence (C, 9/9).** An absolute penalty on the candidate's own queue beats SP-BP's differential in every cell. *Plausible reason:* at occupancies of 2–20% neighbouring queues are nearly equal, so a gradient is small and noisy while the detours it induces are real; an absolute penalty acts as a filter that only fires when a neighbour is clearly full.
2. **The signal must be local to the candidate (F, 0/9).** `car` and `da_gpsr` are identical except for this; averaging over a neighbourhood dilutes the information that matters.
3. **Whether congestion-awareness helps depends on density (B, 3/9).** The differential hurts in medium_slow (degree 4.9) and dense_slow (10.1), helps in very_dense (19.2) and sink_50 (12.9). Comparison A shows the same pattern on the geographic base. *Working hypothesis, untested:* congestion-awareness works by spreading load onto alternative paths — cheap in dense networks, costly in sparse ones; convergecast concentrates load near the sink, where spreading pays most.
4. **Base family matters more than any added signal.** Every attempt to improve shortest-path by adding a signal (differential, lq weighting, ETX) made it worse in most cells; `da_gpsr` wins with both a queue and an lq term on a *geographic* base.

### 12.6 Link quality and its interaction with congestion

Share of generated packets lost, per cause, at the band search's operating point:

| cell | queue overflow | link error | link ÷ queue |
|---|---|---|---|
| medium_slow@40/60/80 | 0.081 / 0.141 / 0.198 | 0.117 / 0.215 / 0.258 | 1.4× / 1.5× / 1.3× |
| dense_slow@60/80/100 | 0.099 / 0.158 / 0.192 | 0.252 / 0.322 / 0.362 | 2.5× / 2.0× / 1.9× |
| very_dense@60/80 | 0.069 / 0.117 | 0.416 / 0.490 | 6.0× / 4.2× |

**Link errors exceed queue overflows in every cell.** A router that watches only queues is watching the smaller of the two ways congestion causes loss. The same lq information helps or hurts depending on where it enters the decision: as a path weight — harmful 9/9 (D); stronger/unbounded — worse (E); multiplier on hop progress — slightly helpful in dense cells (G); additive term in a local geographic score — part of the winning combination (A). *Hypothesis, untested:* decision-time lq is a stale snapshot of transmission-time lq; the more strongly a policy trusts it, the worse it does.

**Hop count is not the explanation** (Retraction 2): medium_slow@40 dijkstra (3.06 hops) beats gpsr (3.04) by 4.08 pp; dense_slow@60 da_gpsr takes *more* hops than dijkstra (2.27 vs 2.09) and wins by 3.90 pp. **Energy does not confound:** zero energy drops, zero dead nodes in every oracle cell.

### 12.7 Experiment 4 — oracle-agreement probe (9 episodes, ~29 min)

The assigned oracle drives each cell; **at every decision all three candidate oracles are queried on the identical state** (a `__probe__` actor), up to 20,000 decisions per cell.

| cell | oracle | 3-way | dij–gpsr | dij–da_gpsr | gpsr–da_gpsr | mean degree |
|---|---|---|---|---|---|---|
| medium_slow@40 / 60 / 80 | dijkstra | 0.652 / 0.592 / 0.544 | 0.802 / 0.789 / 0.756 | 0.664 / 0.606 / 0.562 | 0.830 / 0.770 / 0.746 | 5.3 / 4.8 / 4.5 |
| dense_slow@60 / 80 / 100 | da_gpsr / gpsr / gpsr | 0.747 / 0.695 / 0.649 | 0.775 / 0.792 / 0.801 | 0.757 / 0.707 / 0.676 | 0.947 / 0.866 / 0.796 | 10.7 / 10.0 / 9.5 |
| very_dense@60 / 80 | da_gpsr | 0.647 / 0.593 | 0.680 / 0.648 | 0.660 / 0.622 | 0.918 / 0.855 | 19.9 / 18.6 |
| sink_50@30 | da_gpsr | 0.723 | 0.738 | 0.732 | 0.965 | 12.9 |

**Overall 3-way agreement 0.649** — the oracles disagree on about one decision in three. The disagreement concentrates on the **shortest-path vs geographic** boundary (`gpsr`–`da_gpsr` agree 75–97%). **Agreement falls as load rises in every scenario** — policy choice matters most exactly where the network is most stressed. Mean degree separates medium_slow (~4.9) from the rest (10–19) by ~2×, but per-decision local degree ranges overlap, so degree alone may not resolve every conflict. *Known defect:* the probe's override counter read `n_loop_override` (the real attribute is `n_overrides`), so that field is null; agreement figures are unaffected.

### 12.8 The price of a single teacher [MEASURED, re-computed from `per_seed_pdr` for this report]

| labelling scheme | mean PDR over 9 cells | vs per-cell ceiling | worst cell |
|---|---|---|---|
| per-cell oracles (ceiling) | **0.5048** | — | — |
| `da_gpsr` everywhere | 0.4923 | **−1.25 pp** | medium_slow@80, −4.07 |
| `gpsr` everywhere | 0.4837 | −2.11 pp | medium_slow@40, −4.07 |
| `spbp_ab_noqueue` everywhere | 0.4836 | −2.13 pp | sink_50@30, −5.81 |
| `dijkstra` everywhere | 0.4791 | −2.57 pp | very_dense@80, −4.93 |
| `spbp` everywhere | 0.4703 | −3.46 pp | dense_slow@100, −4.92 |

### 12.9 What this changes

| before | after |
|---|---|
| Congestion-awareness helps routing. | The value of a congestion signal depends on its **formulation** and on **network density**. |
| SP-BP is the oracle. | No single policy is best; the best teacher is **regime-dependent**. |
| (implicit) Backpressure is a sound basis. | The backpressure-style differential is the wrong formulation for sparse-to-moderate FANETs; an **absolute, candidate-local** signal is consistently better. |

**For RL [ASSESSMENT, consistent with the panel report]:** this gives a principled reason to learn. No fixed hand-designed policy is best across regimes, the best one depends on locally observable conditions, and the winner is an untuned linear combination of three signals. The gap between the best single teacher and the per-cell ceiling is a *measured, located* target for D2.

---

## 13. Status of every claim — valid, superseded, void, retracted

The single most useful table in this report for writing. "Void" means produced by code with a known defect affecting that quantity; "superseded" means correct when measured but at an operating point the project has left.

### 13.1 Valid and current

| claim | source |
|---|---|
| Interference makes link quality load-dependent (G1) | G1, re-run post-v17 |
| Congestion coupling exists; interference on/off isolates it (G2, post-v22) | G2 |
| Operating point 1000 s / 100–300 m / battery 8000; per-scenario usable bands | `band_*_1000s.json`, `energy_range.json` |
| Queue-differential effect is density-dependent; absolute > differential (in the band search) | `band_*_1000s.json` |
| SP-BP displaced in 9/9 oracle cells; per-cell oracle table | `panel_extended_v3.json` |
| Controlled findings C (9/9), D (0/9), E (0/9), F (0/9); B density pattern; A 6/9 | panels |
| Link errors 1.3–6× queue overflows in every cell | band search |
| Oracle agreement 0.649; falls with load | `oracle_agreement.json` |
| Single-teacher cost (−1.25 pp for da_gpsr) | recomputed |
| Approach 1's warmstart results (§4.2) | PHASE5_REPORT |
| Competitor facts (§20) | full texts read |

### 13.2 Established on the old dataset — re-confirm after regeneration

| claim | why it needs re-confirmation |
|---|---|
| M4 accuracy reversal (DoD +0.0504) and PDR reversal (−1.37 pp DoD) | old dataset (leak-era simulator, SP-BP labels, 40 s) — relative comparison, likely robust, but not yet shown on the new data |
| Accuracy overstates the PDR cost of information removal ~10.8× | same |
| Contested floor 59.46%; per-scenario floors | property of the old dataset |
| Matched-capacity MLP beats GNN with features, 6/6 HP configs | same |

### 13.3 Void or superseded (do not cite as current)

| claim | status | reason |
|---|---|---|
| SP-BP wins 12/12 cells (G3) | **void** | pre-v12, 40 s, no genuine congestion in dense cells |
| SP-BP queue term worth +0.0645 PDR (D2's reference) | **void** | pre-v12; and SP-BP is no longer the oracle |
| Differential → candidate-only costs exactly 0.0000 | **superseded** | true at the old point because `Q_v ≈ 0`; at the new point absolute beats differential by +2.09 pp |
| Headroom 1.6/18.1/35.6 → 1.4/17.5/35.1 | **void** (both) | pre-v12; the metric is also monotone in load by construction |
| Locality: global BFS worth +0.0065 PDR | **pre-v12** | mechanism likely unaffected (geo ≈ hop by construction), but number not re-measured |
| Calibration sensitivity: claims survive every one-at-a-time variant | **pre-v12** | re-run later if cited |
| Frozen rate grid `[0.02, 0.05, 0.10, 0.25, 0.40]` | **void** | 2.8% of one node's capacity; no congestion post-v12 |
| "Relative queue value is flat; inverted-U refuted" | **void** | leak artifact; v13 measurement shows the inverted U |
| "The ~17% queue gain is genuine congestion avoidance" (the entry that retracted the episode-boundary claim) | **void** | measured pre-v12 |
| Unclaimed budget grows monotonically with load | **void** | pre-v12 probe |
| Convergecast panels (all teachers tied; "keep spbp") | **void** | leak pinned delivery at MAX_QUEUE |
| `panel_cc_v12` "keep spbp" (sink_50, rates 0.02–0.4) | **superseded** | stale rate grid (PDR ~0.98 everywhere, no signal) |
| `panel_cc_v2` verdicts | **superseded** | pre-v16 delta sign flip |
| `scan_congestion_window` verdicts | **void** | hid `energy_depleted` |
| `oracle_congestion` "queue term hurts at 160–260" | **void** | measured inside the saturation plateau |
| `operating_regimes` / `regimes_1000s` "no congestion regime" | **superseded** | correct at battery 100; the premise changed with v14 |
| 20 s / 200 s band results (incl. very_dense "sign flip at 100") | **superseded** | directions transferred, rates did not |
| `panel_oracle_generalized` verdict | **superseded** | incumbent rule, no sparse_fast filter; means valid |
| `panel_contenders_v2` printed assignment | **superseded** | 4/9 cells wrong under the v2 rule |
| Reachability 54.7% / 21.4% | **unsourced** | only prose; direct measures 0.83/0.46 pair, 0.790/0.494 packet-level |

### 13.4 Retracted claims (published in the same place as the confirmed ones, by convention)

1. **"Congestion-awareness hurts routing; the more a policy has, the worse it does."** Drawn from the SP-BP family and from backpressure/dpp without reading their code (they lack a destination term) and ignoring `da_gpsr`, which is congestion-aware and wins 4 cells.
2. **"PDR rank is inverse hop rank; fewest hops wins."** True within the four SP-BP ablations (only hops varied); false across families (§12.6).
3. **"Queue differentials are harmful; absolute penalties help."** Half right: absolute beats differential 9/9, but the differential *helps* in very_dense and sink_50.
4. (Probe era) "The 17% is mostly an episode-boundary artifact" — rested on an adjustment biased toward the arm with more stuck packets. *(Later the counter-claim was itself voided by v12.)*
5. (Probe era) "FILE1 §12's high-load framing is contradicted" — measured what SP-BP had already captured, not what remained. *(Also pre-v12.)*
6. (M3) "SP-BP dominates because it is an information superset" — falsified by the locality experiment.
7. (M3) "ETX loses because of path-stretching" — falsified by identical hop counts.
8. An earlier memory entry that "congestion-awareness does not help" — retracted once `da_gpsr`'s formula was read.

**Recorded prediction failures** (kept as a pattern): masked rollout PDR would fall (it rose for the GNN); the matched ablation would rise with load in both scenarios (it did not); `cand_hop_distance`/`cand_reachable` would be collinear (|r| 0.839, |ρ| 0.260 — not redundant; never exempt on a prediction); 1000 s episode cost from linear scaling (124 s actual vs 9 s predicted — 13× wrong).

---

## 14. Decisions taken (locked — do not re-litigate without new evidence)

| # | Decision | Basis |
|---|---|---|
| D-1 | Approach 2: multi-packet, interference-coupled simulator | Approach 1's structural null |
| D-2 | Gate-per-milestone discipline; design spec approved before implementation | project convention |
| D-3 | Oracle labelling by **measured performance**, never majority vote (correlated-bloc problem) | M3 design |
| D-4 | Paired tests + **Holm–Bonferroni**; 30+ seeds for headline claims; effect size and CI before p | M3 rounds 7, M-7 |
| D-5 | Unsaturated Bianchi as default collision model | M-4 |
| D-6 | `ACT_BETA = 0`, ARQ `max_retx = 5` (disclosed calibration choices) | M2 bring-up |
| D-7 | Ragged storage; full 8-teacher votes; split by (scenario, seed); `medium_slow` held out | M3.5 |
| D-8 | `LOCAL_HORIZON = 2` | pre-M4 |
| D-9 | **Fully decentralised model:** GNN (attention, L=2), three global-BFS features masked at train and rollout; genie-aided teacher (learning from privileged information) | M4 results |
| D-10 | Tuned M4 config `lr 1e-3, attn_dropout 0.1, max_epochs 100` | M4 grid |
| D-11 | Single shared config module (`config_v2`) | v11 |
| D-12 | Parity gate at the **current** 40 s point before moving the operating point; v8 split into v8a (applied) / v8b (staged) | sequencing correction |
| D-13 | Delivered packet consumed on arrival (zero buffer) | v12, confirmed by you |
| D-14 | Operating point: 1000 s, 100–300 m, battery 8000 (bundled atomically in v8b) | energy sweep |
| D-15 | Usable-band criterion (elasticity 0.05–0.85, q_ovf ≥ 0.02, energy ≤ 0.05, dead = 0) | band search |
| D-16 | **Per-scenario rate grids** and **scenario-relative load buckets** | band search; LOAD_DENSITY_DESIGN principles |
| D-17 | Leading-group rule: within 1 pp of top, parsimony tiebreak (v25) | panel |
| D-18 | **Per-cell oracle table** of §12.4 as the oracle of record; sparse_fast flagged, never oracle-setting | panels |
| D-19 | Dataset regeneration decisions (frozen with you): queue features read **live**; **stride 1** for Suite A; **behaviour mix 70% oracle / 30% other teachers**, tagged per episode; compute `hop_succeeded` honestly; keep `medium_slow` holdout and state D1–D5 are in-distribution; **Suite B evaluation-only**, **Suite C = 30% of training added on top**; **simulator-state checkpointing every 50 s** (~20 resume points per episode); save `action` and `drop_reason`; restate D2's reference | Dataset V2 spec + your decisions |
| D-20 | **D5 restated as partition-robustness** (not congestion-robustness) | sparse_fast measurements |
| D-21 | Warmstart quality must not be compromised; sign-flip / harmful regions must not be used manipulatively for D2 — D2 must "happen naturally" | your instruction |
| D-22 | Do not compete on absolute PDR; two-paper strategy | positioning |
| D-23 | RWP-only through M4; other mobility models (Gauss-Markov, RPGM, flocking) only at M8 | M2/M3 scope fence |
| D-24 | Timeline: regeneration + RL start in ~2 weeks; RL 2–3 months | your statement |

---

## 15. What was dropped, rejected or superseded — and why

| Item | Fate | Reason |
|---|---|---|
| Majority-vote labelling | rejected | correlated congestion-blind bloc wins by headcount, invisibly |
| AODV, Stable-Path, Link-Lifetime in the Approach-2 panel | dropped | same shortest-path family; adds bloc weight, no diversity |
| WCETT / ABC dynamic-metric teachers | not built | need bandwidth/channel-diversity state the simulator does not model; ETX used instead (and named ETX, not WCETT) |
| ETX-Dijkstra / LQ-Dijkstra as panel members | dropped from panel (M3), re-tested at the new point, lost 11/11 cells | noisy, stale signal compounds over a path |
| Additive-bounded `lq_dijkstra` | quarantined, unregistered | dead code with an unverifiable claim |
| Unpaired Welch t-test; fixed `THIN_MARGIN = 0.02` | replaced | wrong test for paired data; arbitrary threshold |
| G3 checks 3/4 original form ("winner changes") | reframed | could not distinguish a dominant teacher from a broken panel |
| "Information superset" explanation | falsified | locality experiment |
| Saturated Bianchi | kept only for reproduction | quantisation defect |
| `snr`, `hops_so_far`, `packet_error_rate`, `queue_len` features | removed | measured redundancy |
| Global aggregates (`n_inflight`, `network_mean_occupancy`) | replaced by 2-hop `neigh_*` | observability scoping |
| Default M4 config (`lr 3e-4`, attn_dropout 0) | replaced | ~3 pp suboptimal |
| Depth sweep L ∈ {0..3}, `attention_edgekey` at scale, HP grid expansion (Tier 2/3 of the M4 critique) | deferred | not needed for M5; "3 hours now, not 15" |
| Dueling value head in M4 | deferred to M5 | M4 is pure classification |
| Old rate grid `[0.5, 2.0, 4.0]` | superseded | saturates at 1000 s |
| `(0.05, 0.15, 0.4, 0.7, 1.0)` | rejected | designed before the inverted-U was understood |
| Frozen probe grid `[0.02 … 0.40]` | void | v12 |
| `spbp − dijkstra` as the grid-selection metric; `R_rel`; relative spread; per-cell spread filtering | rejected | conflation / divide-by-zero / inflation / deletes the anchor |
| Addressable-share headroom as a rate selector | forbidden | monotone in load by construction |
| **One global rate grid** (Dataset V2 Master Spec) | **superseded by measurement** | no single grid produces congestion in all scenarios |
| **Stride 10** | rejected for Suite A | destroys `(s, a, s′)` contiguity (may be used for Suites B/C, imitation only) |
| 16-seed subset | rejected | workaround for size; stride solved size |
| Mid-trajectory resumption from recorded frames | not implementable | no state serialisation; → warm-up re-simulation or checkpointing |
| Battery 100 (and 4000) at 1000 s | rejected | energy death precedes congestion |
| **SP-BP as oracle** | superseded | displaced 9/9 |
| Incumbent-replacement rule (v1 panel) | replaced | leading-group rule |
| v2 leading-group rule (significance-gated) | replaced | promoted underpowered losers (v25) |
| backpressure, dpp, car, spbp_lookahead, spbp_ab_candqueue as contenders | dropped after Exp. 1 | lost everywhere or never won outright |
| Approach-1 lineage as a warmstart source | quarantined | wrong feature space, v1 physics, single packet |
| DAR / PARS / CA-GAR / ParRouting as additional teachers (you asked, 1–2 days available) | not pursued — the re-run of the existing panel at the true operating point was done first | [record of the choice made; no measurement of those protocols exists] |
| `apply_sim_changes_v8.py` as written | cannot apply | its anchor targets the pre-v11 location of BASE |
| G2's old anchors (87/280, 112/280) and sweep | replaced (v20–v22) | uninformative at zero load |
| Claims "congestion-awareness hurts", "hop count explains rank", "differentials uniformly harmful" | retracted | §13.4 |

---

## 16. Suggested but not yet done (the backlog)

Ordered roughly by how much they matter to the thesis. None of these is started unless noted.

**Blocking the dataset / RL (see §18 for order):** Phase 0 env contract · `run_iter()` refactor + bit-identical gate · `FANETEnvV2` + parity gate (restated for the new oracle) · checkpoint-save step for masked M4 models and a load-path validation at 40 s (re-produce the 98.0% rollout) · v8b apply · `RATES` → per-scenario dict + `get_rates(scenario)` · scenario-relative `load_bucket(scenario, rate)` · generalised restricted pickers (`dijkstra`, `gpsr`, `da_gpsr`) with drift pins · dataset patch set (save `action`/`drop_reason`/honest `hop_succeeded`, live queue reads, `behaviour_teacher`, `label_teacher` if per-cell, `checkpoint_id`, `record_schema_version`, `t_frame`) · new audit check for action consistency (must fail on a broken variant) · correct the audit's dequeue note · simulator-state checkpointing · grid-in-band verifier.

**Mechanism hypotheses to test (cheap, high value for the paper):**
- **Path diversity** as the density mechanism — Suite B density sweep holding geometry fixed.
- **Link-quality staleness** — correlate decision-time lq with transmission-time success.
- **Interference-strength sensitivity** — where does loss tip from link-driven to queue-driven, and does congestion-awareness gain value there?
- **`da_gpsr` weight tuning** — its (1.0, 1.0, 0.5) weights were never tuned at the new point; a tuned version is the more honest bar for D2.
- **Loop-override conditioning** — compare teachers on non-overridden decisions only.
- **Regime-classifier feasibility** — can locally observable features (degree, occupancy) predict the per-cell oracle? Directly informs §17.1.

**Methodology / robustness:** G1 regression anchor · re-measure locality and calibration post-v12 if cited · more seeds (panels are at 10) · a second convergecast operating point (sink_50 has one usable rate; sink_100–200 not banded; timing: ~336 s/episode at N=100) · NS-3 link-model cross-check at one operating point (shape, pre-registered criterion) · persistent-blockage robustness experiment · M4 Tier-2/3 items (depth sweep, edge-key GAT) for Paper A · reading GNNPPOR in full.

**Baselines (Gate G-A):** `StatefulActor` protocol (`reset`, `select`, `update`) · `QMRRouter` (online Q-learning — its cold-start curve *is* F1) · `DQNRouter` (random init = M7's scratch arm = D1 arm a) · `PPORouter` optional.

**Documentation:** commit `LOAD_DENSITY_DESIGN.md` (updated to the 1000 s bands) and `apply_v8b_operating_point_STAGED.py`; update README, FILE1/FILE2, CLAUDE.md, DATASET_V2_MASTER_SPEC, APPROACH2 notes, Workflow Guide (§22).

---

## 17. Open decisions that need your call

### 17.1 One global teacher, or per-cell teachers? — **DECIDED 2026-10-01: single global da_gpsr**

> Teacher-choice test, pre-registered verdict SINGLE da_gpsr: dijkstra labels are not
> learnable locally (student −3.97 pp vs a da_gpsr student in medium_slow), mixing
> teachers did not hurt the dense cells, a da_gpsr student matches its teacher. The
> analysis below is the reasoning before the test. See DATASET_V3_SPEC §0-§1.

**What is at stake.** The label is what M4 imitates, so the choice sets the warmstart's prior — and you have said warmstart quality must not be compromised, because errors propagate into the GNN and are amplified by RL.

**The measured inputs:**
- Per-cell ceiling 0.5048 vs `da_gpsr` everywhere 0.4923 (**−1.25 pp**), vs `gpsr` −2.11, vs `dijkstra` −2.57 (§12.8).
- Oracle agreement 0.649 overall; the conflict is almost entirely **dijkstra vs geographic**; `gpsr`–`da_gpsr` agree 75–97%; agreement falls with load (§12.7).

**An observation computed for this report that reframes the question [MEASURED + ASSESSMENT].** The −1.25 pp cost of a single `da_gpsr` teacher sits **entirely in `medium_slow`** — and `medium_slow` is the **held-out generalisation scenario** (D-19), so it contributes *no training labels at all*. On the six oracle-setting cells that *are* trained on, `da_gpsr` is in the leading group of **every** cell:

| training cell | per-cell oracle | `da_gpsr` vs oracle (pp) |
|---|---|---|
| dense_slow@60 | da_gpsr | 0.00 |
| dense_slow@80 | gpsr (parsimony; da_gpsr is the top scorer) | **+0.76** |
| dense_slow@100 | gpsr | −0.49 (within the 1 pp tie) |
| very_dense@60, @80 | da_gpsr | 0.00 |
| sink_50@30 | da_gpsr | 0.00 |
| **mean over the 6 trained cells** | | **+0.04** |

So, **for the training set as currently split, "one global `da_gpsr`" and "per-cell" are practically identical in label quality**, while per-cell introduces a gpsr/da_gpsr switch inside `dense_slow` that exists only because of the parsimony tiebreak. The real sub-decisions hiding inside "teacher count" are therefore:

1. **Keep `medium_slow` held out?** If yes, `dijkstra` never labels a training row, and the generalisation test becomes "a `da_gpsr`-trained model on a `dijkstra`-optimal scenario" — a harder and more informative test, which must be *named* as such. If `medium_slow` enters training, the one-in-three dijkstra/geographic conflict enters the labels and the per-cell vs global question becomes real.
2. **What labels `sparse_fast`?** It is in training but flagged with no oracle. `dijkstra` is top there (not significantly); `da_gpsr` is 4.1–4.7 pp lower (also not significant). Options: label it with the global teacher (consistency), with `dijkstra` (best point estimate), or exclude it from imitation and use it only for D1/D5 RL evaluation.
3. **Suite C (`sink_50`) — 30% of training:** `da_gpsr` is the oracle; consistent with option "global da_gpsr".
4. **Tie-breaking:** if per-cell is kept, a *consistency-aware* tiebreak (prefer the teacher that is in the leading group of the most cells) would pick `da_gpsr` in dense_slow@80/100 and remove the artificial switch.

**Options on the table:**

| option | label consistency | warmstart quality on trained cells | cost / complexity | notes |
|---|---|---|---|---|
| **A. Global `da_gpsr`** | perfect | at the ceiling (+0.04 pp) | one restricted picker (`da_gpsr`) + drift pin | medium_slow held out → cost invisible in training; must state OOD test honestly |
| B. Per-cell, parsimony tiebreak | gpsr/da_gpsr switch in dense_slow (75–97% agreement) | ceiling by definition | three pickers + `label_teacher` field | adds conflict with no measured gain on trained cells |
| C. Per-cell, consistency-aware tiebreak | = A on the current split | ceiling | = A unless medium_slow is trained | collapses to A today; diverges only if medium_slow enters training |
| D. Multi-teacher distillation (soft labels / mixture over the leading group) | smooth | ≈ ceiling | new loss; audits must change | more research-y; defensible if framed as ensemble distillation |
| E. Regime-conditioned labels (per-cell) + let RL learn the switch | conflicts deliberately present | ceiling | needs a locally observable regime signal | this is the RL story (D2) — but it moves label conflict into the warmstart |

**[ASSESSMENT]** Given the held-out split and your warmstart-first constraint, option A (or C, which is A on the current split) is hard to beat on evidence: it gives the most consistent labels at no measured cost on trained cells, keeps one picker to verify, and leaves the dijkstra/geographic regime switch as a clean, *located* target for RL rather than a source of contradictory supervision. The decision remains yours; the points that would change this assessment are (i) bringing `medium_slow` into training, (ii) a regime-classifier result showing the per-cell oracle is predictable from local features (then E becomes attractive), or (iii) deciding that D2 should be demonstrated on medium_slow.

**What the dataset needs under each option:** A/C → `oracle_teacher` (manifest) stays a string, no `label_teacher`; B/D/E → per-cell mapping in the manifest + `label_teacher` per row. **Every option needs a new restricted picker** (full-graph distances, choice restricted to unvisited candidates) plus a drift pin analogous to `assert_no_drift()` — editing `ORACLE_TEACHER` alone would silently keep labelling with SP-BP.

### 17.2 The per-scenario rate grids for the dataset

The bands give 3 / 2 / 3 / 2 / 1 usable rates (dense_slow / very_dense / medium_slow / sparse_fast / sink_50). Decide whether the dataset grid = band only, or band + one low-load anchor below (e.g. dense_slow 40, very_dense 40, medium_slow 30) + possibly one stress rate above. The probe pre-registration's lesson applies: *the low-vs-high contrast is required evidence*, so a low anchor is worth keeping. `sink_50` has a single usable rate — decide whether Suite C trains on one rate or includes the (collapsing) 40.

### 17.3 Harmful regions and D2 — what "naturally" means (your instruction)

Within the bands some cells are ones where the oracle's own congestion formulation is locally worse (e.g. SP-BP's differential in dense_slow@100 at −3.98 pp). Two principles, consistent with what you asked: (1) the **dataset grid is fixed from the usable-band criterion before any RL result is seen** — never chosen to create room for D2; (2) **D2's cells and reference are fixed now**, in the pre-registration, against the per-cell oracle (not SP-BP), and reported whatever the outcome. Including a cell where the oracle is weak is legitimate *only if* it is in the band and was chosen by the band rule.

### 17.4 Other open items

| # | Decision | Recommendation on record | Blocks |
|---|---|---|---|
| 1 | `sparse_fast` grid | inherit coverage grid, restate D5 as partition-robustness (C3) | D1, D3, D5 |
| 2 | RL episode window | warm-up re-simulation / 50 s checkpoints, 60–120 s training windows (proposed default, not frozen) | Phase 0, D1 compute |
| 3 | Per-decision credit assignment with many packets in flight | open; `eventual_delivered` + per-hop `drop_reason` is the natural starting vocabulary | M5 reward |
| 4 | Replay-buffer representation | must store raw node/edge/adjacency per transition and re-encode at sample time (the encoder is part of the Q-network) — size the memory in Phase 0 | M5 |
| 5 | Warmstart transfer protocol | encoder + query/key MLPs transfer; the softmax imitation head is *not* a Q-function — define what is re-initialised | Phase 4 |
| 6 | `hop_succeeded` | compute from the ARQ loop's `delivered_hop`, or remove | reward design |
| 7 | `record_schema_version` vs bumping `FEATURE_SCHEMA_VERSION` | separate `record_schema_version` (cleaner) | regeneration |
| 8 | Parity-gate reference teacher | restate from SP-BP to the new oracle(s) — or keep SP-BP purely as a mechanical parity reference (it is deterministic and heavily verified); decide explicitly | Phase 3 |
| 9 | D2's reference number | replace +0.0645 with the per-cell oracle's headroom from the panel | pre-registration |
| 10 | Behaviour mix implementation | 70/30 per episode with `behaviour_teacher`; decide which "other" teachers (the losers carry coverage but also pathologies) | regeneration |

---

## 18. Critical path to dataset regeneration and RL start

### 18.1 The ordered plan (FILE2 §13 v2 + DATASET_V2 §4.5, reconciled with everything since)

```
PHASE 0  -- freeze decisions (1 day, no code)
  0a  teacher decision (17.1)            0b  per-scenario grids (17.2)
  0c  env contract: window / credit assignment / replay representation (17.4 #2-4)
  0d  D1-D5 -> (scenario, grid-position) mapping; D2 and D5 restated; pre-registration frozen
PHASE 1  -- config + labelling code (1-2 days)
  1a  RATES -> per-scenario dict + get_rates();  load_bucket(scenario, rate)
  1b  restricted picker(s) for the chosen teacher(s) + drift pins (assert_no_drift analogue)
  1c  dataset patch: action / drop_reason / honest hop_succeeded / live queue reads /
      behaviour_teacher (+ label_teacher if per-cell) / t_frame / record_schema_version / run_params
  1d  audit: action-consistency check (verified to FAIL on a broken variant); fix the dequeue note
  1e  grid-in-band verifier
PHASE 2  -- run_iter() steppable simulator; bit-identical to run() over 12 Suite A cells x 30 seeds
          + simulator-state checkpointing (50 s) if chosen in 0c
PHASE 3  -- FANETEnvV2 over run_iter() via _select_next_hop; teacher-parity gate 12/12 exact at the
          CURRENT 40 s reference (restated per 17.4 #8)
  3b  checkpoint-save for masked M4 models; load at 40 s, reproduce 98.0% rollout (validates the plumbing)
PHASE 4  -- apply v8b (1000 s / 100-300 m / 8000); re-run G1, G2; smoke-generate one episode per cell
PHASE 5  -- REGENERATE (once): Suite A + Suite C (30%); gate: G3.5 8/8 + audit 7/7 + action check
          + 40 s vs 1000 s feature comparison (energy std up, link-lifetime saturation down)
PHASE 6  -- retrain masked GNN on the new data; re-gate G4 against its OWN new rollout PDR
    || in parallel after Phase 3: StatefulActor + Gate G-A; QMRRouter; DQNRouter (= M7 scratch = D1 arm a)
--> smoke test --> M5 / D1
```

Phases 0, 2 and 3 were **confirmed never started**. Phase 2 + Phase 3 are roughly 3–4 days of build work before v8b and the dataset can move.

### 18.2 Cost realism [MEASURED where stated]

- 1000 s episodes at the new rates: dense_slow band episodes ~120–380 s each at 8 workers (the dense_slow band run: 18 Phase-1 episodes in 2,114 s; 180 Phase-2 episodes in ~13,300 s); very_dense Phase 2: 120 episodes in 17,333 s (~145 s/episode wall at 8 workers). Convergecast N=50 ~60–78 s/episode, **N=100 ~336 s/episode**.
- The last linear extrapolation of episode cost was **13× wrong** — timing-probe any new configuration before committing.
- Stride 1 at 1000 s makes `frames.npz` large (the audit loads it fully into 32 GB). Estimate from the first real run, not from the 40 s dataset (FILE2's ~5.5 GB at stride 1 was for a different rate grid).

### 18.3 The two-week window

Honest reading [ASSESSMENT]: two weeks covers Phase 0–1 comfortably and Phase 2–3 if nothing surprising surfaces; regeneration (~a day or more of unattended compute) and the M4 retrain fit at the end only if the parity gate passes first time. The history of this project suggests at least one gate will find something. Protect the order (parity before v8b, one regeneration after all generation-time fixes) rather than the date — regenerating twice costs more than a few days' slip.

---

## 19. The RL programme (M5–M11) and the pre-registration

### 19.1 Milestones

| # | Milestone | Gate / output |
|---|---|---|
| M5 | RL environment + DQN loop, warmstarted from the masked GNN | G5 (agent approaches teachers) |
| M6 | Cold-start study under load, 30 seeds — **decision gate**: F1, F2, F3 each separately measurable and non-zero | go/no-go for M8–M11 |
| M7 | Warmstart vs scratch across the load sweep | headline result 1 (D1) |
| M8 | Continual learning across mobility regimes (other mobility models enter here) | D3 |
| M9 | CBR / episodic recall (+ jamming regime) | D4 |
| M10 | Leave-one-out composability ablation | the thesis-defining result |
| M11 | Ablations (depth, teacher quality), write-up | — |

### 19.2 D1–D5, as pre-registered in FILE2 §9, with the restatements now required

| | Design | PASS | Needs restating because |
|---|---|---|---|
| **D1 ESCAPE (F1)** | high load in dense_slow, very_dense, sparse_fast; scratch DQN vs warmstart DQN; PDR at episode 1 and **K = 200**; 30 paired seeds | scratch improvement CI contains 0 AND warmstart's excludes 0 and is positive. PARTIAL: both improve, DoD > 0 ("warmstart accelerates escape"). FAIL: scratch improves and DoD CI contains 0 → the trap premise is wrong, report it | "high load" must become a **scenario-relative grid position**; sparse_fast is partition-limited |
| **D2 EXCEED ORACLE** | high-load dense_slow + very_dense only; RL PDR ÷ oracle PDR on identical held-out episodes; one-sample t vs 1.00, 30 seeds | mean > 1.00, CI excludes 1.00 | the oracle is now **per-cell** (da_gpsr/gpsr), not SP-BP; the +0.0645 reference is void — replace with the panel headroom |
| **D3 RETENTION (F2)** | A (dense_slow high) → B (sparse_fast high) → A; naive fine-tune vs EWC/replay | CL forgets significantly less. **Precondition at M6:** naive fine-tuning must actually forget | task identity must be (scenario, grid position), since A and B now run at different absolute rates |
| **D4 RECOVERY (F3)** | A → B → A; episodes to reach 95% of prior PDR on A; ±CBR | CBR significantly faster. **Precondition:** recovery without CBR must take a measurable number of episodes | as D3 |
| **D5 GRACEFUL DEGRADATION** | sparse_fast, all loads — **robustness, not superiority** | agent ≥ 90% of reference under partition; STRONGER: scratch collapses, warmstart does not | **restate as partition-robustness** (decided); no oracle is set in sparse_fast, so name the reference explicitly |

**Scoping rule:** D2 compares agent vs oracle → needs headroom → dense, high-load cells. D1 compares agent vs agent → can be most dramatic in partitioned cells. **All D-criteria are in-distribution** (medium_slow is held out) — state this.

**M10 outcome rule (pre-registered):** disproportionate collapse when any one mechanism is removed → "only the combination survives"; additive → "complementary but independent"; one dominates → report honestly as a negative result.

### 19.3 Baselines for the RL comparisons (Gate G-A)

`QMRRouter` (per-node Q-table over neighbours; reward from delay, link quality, energy; ε-greedy; online) — answers "how do you compare to prior DRL work" **and** its cold-start curve is F1 itself. `DQNRouter` — same features and architecture as the masked GNN, random init — M7's scratch arm and D1's arm (a). `PPORouter` optional (skip under schedule pressure; cite HCPMR/GNNPPOR and state cross-simulator comparison is invalid).

---

## 20. Competitor positioning (all three full texts read)

| | **CQMR** (PMC 2025) | **IQMR** (TNSM 2025) | **HCPMR** (IEEE Access 2026) | **This work** |
|---|---|---|---|---|
| learning | coordinated Q(λ) | independent Q(λ) | MAPPO + GNN, CTDE | imitation → DQN (M5) |
| simulator | MATLAB | MATLAB | NS-3.35 + ns3-gym | purpose-built Python, packet-level |
| geometry | cylinder R1000 H300 | cylinder R1000 H300 | box 1000×1000×300, fixed across terrains | per scenario; 100–300 m at the new point |
| UAVs | 50 | 50 | 50–200 | 20–50 (Suite A/C), up to 200 (Suites B/C) |
| **destination** | **1 TBS** | **1 TBS** | **1 GCC** | **random UAV pairs** (+ Suite C sink) |
| neighbours [COMPUTED] | 2.74 | 2.74 | 8.59–34.91 | 2.2–15.8 (span 200) |
| **queue model** | **none** | FSM triggers only | buffer in state, not in reward | **50-packet queue with overflow drop** |
| "collision" means | physical UAV collision | physical UAV collision | — | 802.11 packet collision (Bianchi) |
| sim time | 8000 episodes | 8000 episodes | 1000 s | 1000 s (new point) |
| reported delivery | 96.7% | 83.3% | > 92% (falls to ~84% at 200 UAVs) | per cell, not comparable |
| seeds / CIs | not stated | not stated | not stated | 10–50 paired, Holm |
| non-graph control | n/a | n/a | absent | present, matched capacity |

**Strongest differentiators, checkable from their own text:** neither CQMR nor IQMR models a finite buffer — congestion collapse **cannot occur** in their models; their "collision" is mid-air separation; density does not explain their high PDR (CQMR/IQMR are sparser than HCPMR) — the **single sink** does. **The positioning insight:** HCPMR's state `[E, B, position, GNN embedding, terrain]` has no BFS hop distance or reachability — exactly this project's **masked arm**, where the GNN wins. *We are not contradicting HCPMR; we supply the matched non-graph control that validates their design rationale, which their paper lacks* (they ablate embedding dimension, never the GNN's presence). **Metric ambiguity (flag, don't accuse):** HCPMR's reward counts delivery to "cluster leader or GCC"; whether reported PDR is end-to-end is not stated. **Error on record:** an earlier claim that HCPMR's 92% figure "does not exist" was wrong (abstract-only search); another that HCPMR repositions UAVs for connectivity was wrong (RWP + Gauss-Markov).

---

## 21. The complete defect ledger

**48 ledger entries (several of which group more than one defect — roughly 55 in all)** have been found across the project, almost all by a gate, an audit, an assertion or a negative control rather than by inspection. At least six were in code written to check other code.

| # | Defect | Found by | Consequence if missed |
|---|---|---|---|
| 1 | Approach-1 `topology_change_rate` always 0 | inspection of outputs | dead feature |
| 2 | Approach-1 queue occupancy always 0 (instant multi-hop) | same | dead features |
| 3–6 | Approach-1 RL env: cross-packet credit, wrong `done`, action/execution mismatch, dropout at eval | review | corrupted Bellman targets |
| 7 | Graph built with `interference_mw = 0` | identical teacher PDRs | congestion-blind teachers |
| 8 | Expected-power lq (Jensen) | implausible lq 0.12 | pessimistic physics |
| 9 | Backpressure positive-differential gate | byte-identical PDR | backpressure ≡ GPSR |
| 10 | Post-hoc "58% fallback" diagnostic on a frozen graph | cross-check | wrong mechanism story |
| 11 | Round-2 rewrite deleted the live counters | re-run showed 0.000 | dead diagnostic |
| 12 | `medium_slow` in the wrong degree class | check 4 thin evidence | weak regime claim |
| 13 | Unpaired Welch on paired data | synthetic check | 11/12 cells lost |
| 14 | DPP positive-score gate (repeat of #9) | smoke test | DPP ≡ GPSR |
| 15 | Shadowed duplicate `lq_dijkstra` | reading | dead code with a false claim |
| 16 | lq missing the Bianchi term | M3 audit | teachers blind to MAC contention |
| 17 | numpy object-array save crash | run | generation crash |
| 18 | Label via pruned-graph BFS: 22.6% fallback | G3.5 | a fifth of labels not the oracle's |
| 19 | Label vs vote by different code paths | independent audit | biased vote agreement |
| 20 | 42,000× lazy `.npz` decompression | Task Manager screenshot | audit infeasible |
| 21 | `NameError` in the audit | run | crash |
| 22 | Saturated Bianchi quantisation (p_coll exactly 0, then a jump) | M-4 review | wrong physics in the operating range |
| 23 | `spbp_ab_full` diverged on partitioned graphs | cross-experiment comparison | mechanism numbers off |
| 24 | `spbp_khop(k=∞)` diverged (279/345) | same | locality number off |
| 25 | `snr` ≈ f(distance); `hops_so_far` = 1 − `ttl_left`; `packet_error_rate` ≈ 1 − lq | check 8 | redundant features |
| 26 | `neigh_buffered_packets` clipped on 38.8% of decisions | saturation diagnostic | dead column at high load |
| 27 | Two experiment scripts could not write their results | review | unrecorded findings |
| 28 | Verifier pinned schema `== 4` after v6 made it 5 | run | false failure |
| 29 | `id(G)` encoder cache in rollout | determinism check (drift = exactly 1 packet) | stale encodings |
| 30 | `warn_only=True` + `CUBLAS_WORKSPACE_CONFIG` set after CUDA init | determinism check | cosmetic determinism |
| 31 | `id(G)` reachability cache in the headroom experiment (same class as #29) | config-duplication investigation | wrong headroom |
| 32 | Eight independent copies of SCENARIOS/RATES/BASE | same | v8 would have desynchronised seven scripts |
| 33 | v8 documented a pinned sink; code never pinned it (moved 156 m) | measurement | invalid Suite C |
| 34 | `load_bucket` would have collapsed 24 probe cells into 4 | v11.2 review | averaged-away curve |
| 35 | `provenance()` imported but never called | v11.2 review | undateable results |
| 36 | Probe rule gaps: floor-band peak, argmax knee, boundary-vs-flat | the probe's own output | grid from noise |
| 37 | `action` / `drop_reason` never saved; `hop_succeeded` constant | Dataset V2 audit | unusable RL transitions |
| 38 | Queue features stamped at frame start | Dataset V2 audit | own-node congestion invisible |
| 39 | **Destination / TTL queue-slot leak (v12)** | tied convergecast panel | voided every pre-fix congestion measurement |
| 40 | `scan_congestion_window` hid `energy_depleted` | regime map | measured batteries, not congestion |
| 41 | Energy normalisation hardcoded (and 3 siblings) | pre-regeneration scan | energy feature 80× too large |
| 42 | Panel `delta` sign flip | arithmetic re-check | inverted a verdict |
| 43 | Band script: `holm()` on a dict, missing key, always-true tier label, strict-monotone S4, single-rate NaN, no energy flag, misleading S4 display | runs | crashes / wrong verdicts / misreading |
| 44 | G2 stale anchors, stale sweep, check-3 definition diluted by energy, check-4 floor in the zero-activity regime | G2 failing 6/6 | a regression gate that could not detect a broken simulator |
| 45 | Result files' provenance reported the parity reference, not the run | audit | four files with no battery recorded |
| 46 | Leading-group rule promoted underpowered losers (4/9 cells) | per-seed recompute | wrong oracle in 4 cells |
| 47 | Agreement probe read `n_loop_override` (real: `n_overrides`) | output inspection | null override field (agreement unaffected) |
| 48 | Convergecast script "stuck": silent max-rate self-tests + 16 Windows spawn workers | observation | lost hours |

**Lessons that generalise** (all already in CLAUDE.md): verify by execution; equivalence controls must be shown to *fail* on a broken variant; grep for the defect *class*, not the instance; pin behaviour, not version numbers; never exempt a check on a prediction; never extrapolate cost; commit at verified checkpoints; a plausible number on a dead mechanism is the default failure mode.

---

## 22. Documentation state — what is stale where

| Document | State | What is wrong / missing |
|---|---|---|
| `README.md` | **stale** | says "M1–M3.5 gate-passed; M4 next"; describes Approach 1 as the headline |
| `docs/FILE1_APPROACH2_RECORD.md` | stale (HEAD `41ea7a4`) | pre-v12: SP-BP oracle, +0.0645, headroom triple, old grid; no v12–v25 |
| `docs/FILE2_PRE_M5_EXECUTION_PLAN.md` | stale | old ordering assumptions, single global grid, SP-BP parity, D2 +0.0645 |
| `docs/CLAUDE.md` | stale "where things stand" | next blocking piece is v11.2 (done long ago) |
| `docs/DATASET_V2_MASTER_SPEC.md` | partly superseded | "one global grid" and the frozen `[0.02…0.40]` grid; SP-BP oracle |
| `docs/PROBE_PREREGISTRATION.md` | correctly marked void (v19) | points to `LOAD_DENSITY_DESIGN.md`, which is **not in the repo** |
| `docs/APPROACH2_COMPLETE_NOTES.md`, `PROJECT_STATUS_AND_PLAN.md` | historical | pre-v10 headroom, SP-BP 12/12 |
| `docs/Panel_Results_Report.md` | **current** | — |
| `results/README.md` | **current** | does not yet mark pre-v12 M3/M4-era files (`headroom*.json`, `spbp_mechanism.json`, `queue_weight.json`, `locality_cost.json`, `calibration_sensitivity.json`, `collision_model.json`, `probe_rate_grid*.json`) as void/superseded |
| `LOAD_DENSITY_DESIGN.md` | **uncommitted**, numbers superseded | written on 20 s / 200 s bands; principles stand |
| `apply_v8b_operating_point_STAGED.py` | **uncommitted** | should be in the repo before anyone reconstructs v8b from memory |
| `M3.5_Dataset_Schema.md` | current (this session's output), uncommitted | the spec for regeneration |
| `FANET_Workflow_Guide.md/.pdf` | stale | says M3 is being re-run with SP-BP |
| Workflow flowchart + guide, progress report (.docx), 2-page summary (.docx), `fanet_workflow.html` | current as of the panel work | produced for your guide; not in the repo |
| Code comments | stale in places | `ORACLE_TEACHER = 'spbp'  # G3: wins all 12 cells`; `_build_graph` docstring "live queue state"; `audit_dataset_v2.py:465` dequeue explanation; `simulator_v2.PANEL` is the old 8 |

**[ASSESSMENT]** One doc refresh before regeneration pays for itself: FILE1/FILE2 (or a FILE3 that supersedes them), CLAUDE.md's status section, README's status line, and the two uncommitted files. The convention "HEAD must be current in FILE1/FILE2's headers" is currently broken.

---

## 23. Risks and an honest assessment

1. **Effect size is the biggest risk, and it has moved.** The old headroom measurement that justified "the thesis lives at high load" is void. What replaces it: the panel shows the best single teacher sits 1.25 pp below a per-cell ceiling on average, with a 4 pp located gap in medium_slow; the controlled comparisons move 1–5 pp. **These are small absolute effects.** D1 (agent vs agent) does not depend on them; D2 does. Measure the RL headroom early (M6) before M8–M11.
2. **The trap premise (F1) is unproven**, and the queue-driven version of congestion is secondary in this simulator — loss is mostly link degradation. If F1 exists it will likely manifest through interference, not queues. That is still the coupling loop, but the story should be told in those terms.
3. **The oracle is regime-dependent.** That is scientifically interesting and gives RL a principled target, but it makes the warmstart decision (§17.1) consequential.
4. **sparse_fast carries three D-criteria** (D1, D3, D5) but has no congestion regime and no oracle. D5 is already restated; D1/D3 in sparse_fast need the same care.
5. **Single-operating-point results.** Convergecast rests on one cell; panels on 10 seeds; bands on 15.
6. **Schedule.** Phase 2 and Phase 3 are real engineering; the parity gate is exactly the kind of check that has found something every time in this project.
7. **Methodological strength is real and publishable on its own terms:** gate-per-milestone validation, independent audits, pre-registration with published amendments and retractions, paired statistics with Holm, and a defect ledger that shows the checks do work. The environment (packet-level, queue-coupled, interference-coupled, bit-reproducible) is a contribution in its own right.

**The line to use with your guide, updated:** *"The environment, the operating point and the teacher set are now measured rather than assumed — at the cost of discovering that the original oracle and grid were artefacts of a simulator defect. What we now have is sharper: no hand-designed router is best across regimes, and the value of congestion information depends on its formulation and on density. The three-part thesis is still unproven; the next milestone tells us whether the effect is large enough, before the remaining milestones are committed."*

---

## Appendix A — Glossary

**PDR** packet delivery ratio (delivered / generated) · **pp** percentage points · **oracle** the teacher whose choices become labels · **label** oracle's choice; **action** what the trajectory actually did · **ε-deviation** random-candidate substitution with probability ε · **contested floor** trivial-rule accuracy excluding destination-neighbour decisions · **elasticity** relative change in delivered ÷ relative change in offered rate vs the previous rate · **usable band** rates with elasticity ∈ [0.05, 0.85], q_ovf ≥ 0.02, energy ≤ 0.05, 0 dead · **q_ovf** share of generated packets lost to queue overflow · **Tier 1 / Tier 2** pre-registered confirmatory / mechanistic comparisons · **Holm–Bonferroni** step-down family-wise correction: sort p-values, compare the k-th smallest to α/(m−k+1), stop at the first failure · **leading group** teachers within 1 pp of the cell top · **parsimony** simplest teacher in the leading group · **matched control** identical code path with one term neutralised (e.g. `spbp_ab_noqueue`) · **parity gate** a new component must reproduce a known reference exactly · **equivalence control** a check that a re-implementation matches the original, verified to fail on a broken variant · **unit-disk graph** edge iff distance ≤ range · **hidden terminal** interferer outside the transmitter's carrier-sense range but inside the receiver's interference range · **Bianchi model** fixed-point model of 802.11 DCF collision probability · **ARQ** automatic repeat request (retries) · **SP-BP** shortest-path backpressure · **DA-GPSR** delay-aware GPSR · **CAR** congestion-aware routing (neighbourhood field) · **DPP** drift-plus-penalty · **ETX** expected transmission count · **CBR** case-based reasoning · **CL** continual learning · **EWC** elastic weight consolidation · **LOO** leave-one-out · **F1/F2/F3** cold start / forgetting / no recall · **D1–D5** pre-registered pass criteria · **Suite A/B/C** original scenarios / density sweep / convergecast · **v8a/v8b** the split operating-point patch · **privileged information** a teacher that sees features the student does not.

## Appendix B — Key numbers on one page

| quantity | value |
|---|---|
| Approach 1 slow cold-start incidence | 5/30 (16.7%), CI [7.3, 33.6]% |
| G1 lq, activity 0 → 0.20 (current) | 0.972 → 0.345; <0.90: 0.1% → 88.4% |
| G2 (current) PDR over sweep 35 → 70 | 0.886 → 0.374; congestion share 31.8% → 82.1% |
| G2 anchors (rate 60) | dijkstra 8015/16800 · spbp 7161/16800 |
| Interference on/off, rate 50 | PDR 0.620 vs 0.893; link errors 1656 vs 0 |
| Old dataset | 533,237 decisions / 48,000 frames / 600 episodes |
| Contested floor (old) | 59.46% (raw 71.14%) |
| M4 params | 336,168 vs 335,872 |
| M4 accuracy DoD | +0.0504 [+0.0460, +0.0548], d = 4.27 |
| M4 rollout (masked) | GNN 98.0% / MLP 97.5% of SP-BP |
| Accuracy : PDR gap under masking | 10.8 : 1 |
| v12 phantom drops at 40 s | 88.4% (3,394 / 3,840) |
| Energy ceiling at 1000 s, battery 100 | per-flow ≤ 6.56 (dense_slow) |
| Operating point | 1000 s · 100–300 m · battery 8000 |
| Usable bands (1000 s) | dense_slow 60/80/100 · very_dense 60/80 · medium_slow 40/60/80 · sparse_fast 80/100 ⚑ · sink_50 30 |
| Oracle panels | 1,769 episodes; SP-BP displaced 9/9 |
| Oracle agreement | 0.649 |
| Single da_gpsr cost | −1.25 pp over 9 cells; +0.04 pp over the 6 trained cells |
| Link ÷ queue loss | 1.3× – 6.0× |

## Appendix C — File index (what to open for what)

| question | file |
|---|---|
| current oracle | `results/panel_extended_v3.json` → `oracle_assignment` |
| per-seed teacher PDRs | same → `per_seed_pdr` |
| current bands | `results/band_{dense_slow,very_dense,medium_slow,sparse_fast,sink50}_1000s.json` |
| battery choice | `results/energy_range.json` |
| which result files are safe | `results/README.md` |
| panel analysis | `docs/Panel_Results_Report.md` |
| regeneration schema | `M3.5_Dataset_Schema.md` (uncommitted) |
| operating-point patch | `apply_v8b_operating_point_STAGED.py` (uncommitted) |
| teacher formulas | `src/routing_teachers_v2.py` (+ `routing_teachers.py` for dijkstra/gpsr) |
| SP-BP ablations | `src/experiment_spbp_mechanism.py` |
| label picker | `src/generate_dataset_v2.py` → `spbp_pick_restricted`, `assert_no_drift` |
| scenarios / suites | `src/config_v2.py` |
| the leak fix | `src/simulator_v2.py` `_try_forward` (v12 block), `_assert_queue_conservation` |
| rules of the road | `docs/CLAUDE.md` |

## Appendix D — PowerShell cheat-sheet (current tools)

```powershell
conda activate fanet
cd $HOME\FANET_sim

# gates
python src\preflight_interference_check.py            # G1
python src\preflight_simulator_v2_check.py            # G2 (6/6 expected)

# usable-band search (one scenario per terminal; 4-8 workers on Windows)
python src\find_usable_band.py --scenario dense_slow --duration 1000 --initial_energy 8000 --rates 40 60 80 100 120 160 --measure-seeds 15 --max_workers 8 --out results\band_dense_slow_1000s.json
python src\find_congestion_band_convergecast.py --rates 20 30 40 50 60 --initial_energy 8000 --max_workers 8

# oracle panels (per-seed storage; extend without re-running)
python src\panel_contenders_v2.py --max_workers 8
python src\panel_extend_linkquality_v3.py --max_workers 8
python src\probe_oracle_agreement.py

# recompute an oracle assignment from stored per-seed data (no re-run)
python fix_oracle_assignment_v25.py results\panel_contenders_v2.json

# every patch: dry-run first, then apply, then its verifier
python apply_<name>.py --src src --dry-run
python apply_<name>.py --src src
```

*End of report.*
