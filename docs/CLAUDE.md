# CLAUDE.md — behavioral contract for this repo

FANET lifelong-RL routing research, IIITDM Kancheepuram. For project state, read
`docs/FANET_Full_Project_Report.md` (the current record and plan) and, for the dataset,
`docs/DATASET_V3_SPEC.md`. `docs/FILE1_APPROACH2_RECORD.md` / `FILE2_PRE_M5_EXECUTION_PLAN.md`
are history: their banners say what in them is superseded (FILE2's stride-10 patch,
rate grids and SP-BP oracle must NOT be implemented). Do not restate their content
here; this file is rules, not knowledge.

## Non-negotiable conventions

- **Design spec approved before implementation.** No M5-stage code without a written
  spec the user has seen.
- **Verify by execution, not inspection.** A script that aborts before its write step
  still compiles. Run it; read the output.
- **Patches are assertion-guarded `str.replace`.** Anchors must match the target file
  exactly once. Stage all edits in memory; write nothing unless every anchor in every
  file matches. Every patch script needs a `--dry-run` and an idempotency guard (detect
  "already applied", do nothing).
- **Equivalence controls must be verified to FAIL on a deliberately broken variant.** A
  checker that only ever passes proves nothing — this has caught real bugs in this
  project more than once. When you write a control, prove it can fail before trusting
  that it passes.
- **Pre-registered predictions and thresholds are never revised after seeing results.**
  If a result contradicts a pre-registered threshold, report the contradiction — do not
  adjust the threshold.
- **30+ seeds with Holm correction for any headline claim.** Report effect size and CI
  before the p-value.
- **Publish negative results, including refuted hypotheses**, in the same place as
  positive ones.
- **Never quote a grand-mean PDR alone.** It averages a deliberately designed stress
  grid; always pair a figure with its scenario.
- **Config comes from `src/config_v2.py` only.** `SCENARIOS`, `RATES`, `BASE` have one
  source. Do not add a local copy in a new script — that exact mistake (8 independent
  copies, one silently unpatched) already cost a full investigation once.
- **Commit and push at each verified checkpoint, not at milestone boundaries.** Work
  sitting uncommitted has already caused staleness regressions in this project.

## Environment

Windows 11, **PowerShell only** — `&&`, `unzip`, and other bash-isms do not work.
`conda activate fanet` before running anything. Repo root is `C:\Users\PREETH\FANET_sim\`.

```powershell
conda activate fanet
cd C:\Users\PREETH\FANET_sim
```

## Repo layout

- `src/` — all source. `config_v2.py` is the shared config; everything else imports
  from it.
- `docs/` — design specs and state (`FILE1`/`FILE2` above, plus per-milestone specs and
  `PRE_M5_POSITIONING_PLAN.md`, which has Gate G-A's detailed acceptance criteria).
- `results/` — run outputs. **`results/checkpoints/*.pt` is Approach-1-shaped and
  quarantined — do not load as a warmstart source.** See FILE1 §1 for why.
- `results/m5_masked/` — the real M4 warmstart checkpoints, but **no `.pt` files exist
  there yet** (FILE2 §10.5) — a save step is still needed before Phase 4 can load one.

## Where things stand (update this section, not the rest, as milestones close)

`HEAD` should always be current in FILE1/FILE2's headers — if it isn't, that's a sign
work was committed without updating them.

As of v28 (2026-10-04): the v26 dataset was generated and passed every gate, then the
mobility bug was found (drones trapped at their waypoints; DATASET_V3_SPEC §0 #15) and
fixed in v28. Results from episodes longer than ~40 s produced before v28 ran on freezing
networks. v29 (2026-10-06): on the fixed simulator the five bands were re-measured
(sparse_fast gained 60, sink_50 gained 40; DATASET_GRID follows -- 18 cells), the teacher
choice re-confirmed (SINGLE da_gpsr) and Point-1 re-run (DATA-LIMITED on the same weak
trigger). Next blocking pieces, in order: `verify_dataset_grid_v3.py` on the 18-cell grid,
smoke, regenerate into a new folder (data\v3m), G3.5 v3 + audit v3, export, then the
retrain gate (held-out cells reported by default). Never resume or mix outputs across a `MOBILITY_VERSION` / code-signature
change -- the scripts refuse it. Feature
schema is v6 (live own queue). Labels come from da_gpsr through `teacher_pickers_v3` only — never add
another label path. Training rows take `own_queue_live` from each context's own-queue
histogram (`export_phaseb_v3.py`), never from `c_query` (first occurrence, biased high);
an RL state built from a step uses `s_ownq`. Do not apply
`apply_v8b_operating_point_STAGED.py` before the new
RL environment's parity gate passes at the current 40 s point; the dataset does not
need v8b (it uses `config_v2.OPERATING_POINT`).
