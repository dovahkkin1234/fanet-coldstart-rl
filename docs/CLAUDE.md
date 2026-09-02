# CLAUDE.md — behavioral contract for this repo

FANET lifelong-RL routing research, IIITDM Kancheepuram. For project state, read
`docs/FILE1_APPROACH2_RECORD.md` (what was built and measured) and
`docs/FILE2_PRE_M5_EXECUTION_PLAN.md` (what's next, in order) — do not restate their
content here; this file is rules, not knowledge.

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

Next blocking piece: `experiment_headroom.py` needs `--duration`/`--z_min`/`--z_max`/
`--drain_time` CLI flags (v11.2, FILE2 §1.5.1) before the three-quantity rate probe can
run. Do not apply `apply_sim_changes_v8.py` before the new RL environment's SP-BP-parity
gate passes at the current 40 s operating point (FILE2 §3) — that gate is the only fixed
reference for telling "the environment is right" apart from "the operating point moved."
