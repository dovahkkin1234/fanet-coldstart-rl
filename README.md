# Cold-Start-Aware Continual RL for Adaptive Packet Routing in FANETs

**Author:** Shirish Giroti (CS23B2041) — IIITDM Kancheepuram  
**Target Venue:** IEEE ICNC / GLOBECOM 2026  
**Status:** Phase 5 complete — cold-start variance study (30 seeds)

---

## Overview

This repository contains the full implementation of a cold-start-aware deep reinforcement learning (DRL) pipeline for adaptive packet routing in Flying Ad-hoc Networks (FANETs). The core contribution is a warm-start initialization strategy that leverages supervised learning on expert routing demonstrations to eliminate the cold-start performance degradation observed in randomly-initialized RL agents.

### Key Contributions

1. **MLP Warm-Start Initialization** — A 22-feature binary classifier trained on consensus labels from 5 classical routing protocols (Dijkstra, GPSR, AODV, Stable-Path, Link-Lifetime-Aware) achieves AUC 0.903 and Top-1 accuracy 85.6%. Its weights initialize the DQN Q-network, providing immediate routing competence before any RL training.

2. **Cold-Start Risk Quantification** — Across 30 controlled scratch (random-init) runs, 5 (16.7%, 95% CI: 7–34%) exhibited slow cold-start, delivering 4.4–5.1× lower initial PDR and requiring 2.9–3.8× more training episodes to converge. Warm-start eliminates this risk entirely.

3. **Topology Regime Clustering** — KMeans (k=3, silhouette=0.243) on 9 network topology features identifies three regimes: Dense-Stable, Sparse-Partitioned, and Medium-Fast. Regime IDs are propagated to all training CSVs as an additional state feature for regime-conditioned routing.

4. **Simulator Bug Fixes** — Three critical bugs in the FANET simulator were identified and fixed: topology_change_rate always 0.0 (broken prev_links tracking), queue occupancy always 0.0 (instant multi-hop forwarding bypassing queues), and four RL environment flaws (cross-packet credit assignment, wrong done-flag, action/execution mismatch, dropout at eval time).

5. **Continual Learning Foundation** — Freeze-then-finetune protection (backbone frozen for 1000 episodes, then unfrozen at 10× lower LR) prevents catastrophic forgetting of pretrained routing knowledge during RL fine-tuning. Column-masked protection preserves the 22 pretrained feature columns while allowing the 3 new regime columns to train freely.

---

## Repository Structure

```
fanet-coldstart-rl/
├── src/                          # All Python source code
│   ├── simulator.py              # FANET simulator (DroneRWP, link model, routing)
│   ├── mobility.py               # DroneRWP mobility model
│   ├── link_model.py             # Path loss, RSSI, link lifetime estimation
│   ├── models.py                 # EnergyModel, NodeQueue
│   ├── routing_teachers.py       # 5 classical routing protocols
│   ├── run_pilots.py             # Pilot dataset batch runner
│   ├── run_full_rwp.py           # Full-scale RWP dataset generator (15 scenarios)
│   ├── validate_all.py           # Dataset validation
│   ├── eval_routers.py           # Router baseline comparison
│   ├── train_warmstart.py        # XGBoost warm-start (pilot)
│   ├── train_warmstart_full.py   # XGBoost warm-start (full-scale)
│   ├── train_mlp.py              # MLP warm-start (pilot)
│   ├── train_mlp_full.py         # MLP warm-start (full-scale, loss-stop)
│   ├── train_mlp_full_accstop.py # MLP warm-start (acc-stop variant, ablation)
│   ├── cluster_regimes.py        # Regime clustering (pilot)
│   ├── cluster_regimes_full.py   # Regime clustering (full-scale)
│   ├── rl_env_full.py            # Gymnasium FANET environment (Approach 1)
│   ├── train_dqn_full.py         # DQN trainer (scratch/warmstart/regime)
│   ├── rl_preflight_check.py     # Pre-flight checker before RL training
│   ├── rdm_warmstart.py          # RDM warm-start pipeline (Phases 1-3)
│   ├── rdm_regimes.py            # RDM regime clustering (Phase 4)
│   └── analyze_seed_variance.py  # 30-seed cold-start variance analysis
│
├── configs/                      # YAML configuration files
│   ├── pilot_configs.yaml        # 10-scenario pilot dataset config
│   ├── rwp_full_configs.yaml     # 15-scenario full-scale RWP config
│   └── rdm_full_configs.yaml     # 12-scenario full-scale RDM config (planned)
│
├── docs/                         # Documentation and specifications
│   ├── PROGRESS_REPORT.md        # Full phase-by-phase progress report
│   ├── RWP_FULL_DATASET_SPEC.md  # Full-scale RWP dataset specification
│   ├── RDM_DATASET_SPEC.md       # Full-scale RDM dataset specification
│   ├── SIMULATOR_BUG_FIXES.md    # Detailed simulator bug documentation
│   └── PHASE5_REPORT.md          # Phase 5 RL results report
│
├── scripts/                      # Utility scripts
│   ├── run_seeds.bat             # Windows batch for 30-seed variance study
│   └── setup_env.sh              # Conda environment setup
│
├── results/                      # Saved outputs (tracked, data not tracked)
│   ├── figures/                  # All plots (PNG)
│   ├── tables/                   # Cold-start metrics JSON files
│   └── checkpoints/              # Model weights (.pt, .pkl) via Git LFS
│
├── notebooks/                    # Analysis notebooks (optional)
│
├── .gitignore
├── .gitattributes                # Git LFS config for large files
├── README.md
├── requirements.txt
└── environment.yml               # Conda environment specification
```

---

## Installation

### Option 1: Conda (recommended)

```bash
conda env create -f environment.yml
conda activate fanet
```

### Option 2: pip

```bash
pip install -r requirements.txt
```

---

## Reproducing the Results

### Step 1: Generate the dataset

```powershell
# Dry-run PDR boundary check (< 2 minutes)
python src\run_full_rwp.py --dry_run

# Full dataset generation (7-12 hours, 4 parallel terminals)
python src\run_full_rwp.py --scenarios rwp_sc01 rwp_sc02 rwp_sc10 rwp_sc13
python src\run_full_rwp.py --scenarios rwp_sc03 rwp_sc04 rwp_sc05 rwp_sc15
python src\run_full_rwp.py --scenarios rwp_sc06 rwp_sc08 rwp_sc09 rwp_sc11
python src\run_full_rwp.py --scenarios rwp_sc07 rwp_sc12 rwp_sc14
```

### Step 2: Train warm-start models

```powershell
python src\train_warmstart_full.py    # XGBoost (~5 min)
python src\train_mlp_full.py          # MLP (~90 min on RTX 5060)
```

### Step 3: Regime clustering

```powershell
python src\cluster_regimes_full.py
```

### Step 4: Pre-flight check

```powershell
python src\rl_preflight_check.py
```

### Step 5: RL training

```powershell
# Core variants
python src\train_dqn_full.py --variant scratch   --seed 42
python src\train_dqn_full.py --variant scratch   --seed 123
python src\train_dqn_full.py --variant warmstart --seed 42
python src\train_dqn_full.py --variant regime    --seed 42

# 30-seed variance study
python src\analyze_seed_variance.py
```

---

## Key Results

### Warm-Start Advantage (vs 30-seed scratch mean)

| Scenario | Scratch init PDR | Warmstart init PDR | Advantage |
|---|---|---|---|
| rwp_sc03 (medium-slow) | 0.403 | 0.467 | +0.064 |
| rwp_sc10 (dense-fast) | 0.639 | 0.767 | **+0.128** |
| rwp_sc07 (40-drone swarm) | 0.286 | 0.333 | +0.047 |

### Cold-Start Risk Elimination

| Metric | Scratch (random init) | Warmstart |
|---|---|---|
| Slow-start probability | 16.7% (CI: 7–34%) | ~0% |
| sc10 init PDR spread | 0.680 | < 0.015 |
| Worst-case T90 (sc10) | 950 episodes | 250 episodes |
| AULC advantage (sc10) | — | +209 (+6.0%) |

### Convergence Properties

All 30 scratch seeds converge to statistically equivalent final performance (σ < 0.010), confirming the cold-start variance is a **cold-start phenomenon only** — not a permanent capability gap. Warm-start eliminates the risk of landing in the slow-convergence cluster.

---

## Hardware

- **GPU:** NVIDIA GeForce RTX 5060 Laptop (Blackwell sm_120, 8GB VRAM)
- **Training time:** ~5 min per 5000-episode RL run
- **Dataset generation:** ~7-12 hours (4 parallel processes)

---

## Citation

```bibtex
@inproceedings{giroti2026coldstart,
  title     = {Cold-Start-Aware Continual RL for Adaptive Packet Routing in FANETs},
  author    = {Giroti, Shirish},
  booktitle = {Proceedings of IEEE ICNC / GLOBECOM 2026},
  year      = {2026},
  note      = {Preprint}
}
```

---

## License

MIT License — see LICENSE file.
