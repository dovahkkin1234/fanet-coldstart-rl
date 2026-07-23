"""
teacher_panel.py  —  M3 oracle labeling machinery.

Implements docs/M3_TEACHERS_ORACLE_DESIGN.md S4: instead of majority-vote
labeling (which lets correlated congestion-blind teachers win by headcount
rather than by quality), the label is defined by MEASURED ACHIEVED PERFORMANCE
in the matching regime.

Two phases:
  Phase A  build_oracle_table()  — run every teacher standalone across the
           scenario x load grid, rank by achieved network PDR, produce
           oracle[(scenario_class, load_bucket)] -> ranked teacher list.
  Phase B  collect_votes()       — at a decision point, query all panel
           teachers, mark the oracle-best teacher's action as the label, and
           retain the full vote vector as a confidence weight.

The scheme cannot be fooled by correlated blocs because headcount never enters
the label.

PARALLELIZATION (M3 audit): build_oracle_table dispatches one process per
individual (teacher, scenario, rate, seed) simulator run via
ProcessPoolExecutor, sized for a 16-core/32-thread machine. At the grid sizes
this milestone needs (30 seeds x 5 teachers x 4 scenarios x 3 rates = 1800
runs, ~1.2s each), that is ~37 CPU-minutes serial and a few minutes wall-clock
in parallel. Only picklable data (strings, dicts, floats) crosses the process
boundary — teacher functions are looked up by name inside the worker via
TEACHERS, never passed as function objects.

UNIFIED DIAGNOSTICS (M3 audit): override_rate and bp_zerodiff_rate are now
recorded on EVERY run inside evaluate_teacher / the worker, so the harness's
diagnostic reporting reads them straight out of the same 'raw' results Phase A
already computed, rather than re-simulating separately. The earlier version's
disconnected re-simulation (a frozen graph, sampled after the episode ended)
was the root cause of a diagnostic that silently reported 0.0 regardless of
load — this design makes that class of bug structurally harder to reintroduce.
"""

import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

from simulator_v2 import FANETSimulatorV2, TEACHERS, PANEL
from routing_teachers_v2 import backpressure_next_hop

# ── Regime classification ────────────────────────────────────────────────────

def expected_degree(num_drones, area_x, area_y, comm_range):
    """Expected neighbour count for a uniform placement: n * pi r^2 / A."""
    return num_drones * np.pi * comm_range ** 2 / max(area_x * area_y, 1.0)


def scenario_class(cfg):
    """Coarse topology regime. Dense vs sparse is a physically distinct
    interference regime (G1 showed dense = congestion-limited, sparse =
    range/partition-limited), which is exactly the distinction the
    continual-learning-across-regimes thesis rests on. Four bands so the
    scenario grid maps to four distinct classes."""
    deg = expected_degree(cfg['num_drones'], cfg['area_x'], cfg['area_y'],
                          cfg['comm_range'])
    fast = cfg.get('speed_max', 15) >= 30
    if deg >= 12.0:
        base = 'very-dense'
    elif deg >= 6.0:
        base = 'dense'
    elif deg >= 3.0:
        base = 'medium'
    else:
        base = 'sparse'
    return f'{base}-fast' if fast else base


def load_bucket(packet_rate):
    if packet_rate <= 0.5:
        return 'low'
    if packet_rate <= 2.0:
        return 'medium'
    return 'high'


# ── Phase A worker (module-level: required for pickling under spawn) ────────

def _run_single(job):
    """Run exactly one (teacher, config-with-seed) episode. Returns a flat dict
    with the outcome metric plus every diagnostic the harness needs, so no
    separate re-simulation is ever required downstream."""
    teacher_name, cfg = job
    m = FANETSimulatorV2({**cfg, 'actor': teacher_name}).run()
    drops = m['drop_reasons']
    tot = max(sum(drops.values()), 1)
    cong_share = (drops.get('queue_overflow', 0) + drops.get('link_error', 0)) / tot
    return {
        'teacher': teacher_name,
        'seed': cfg['seed'],
        'pdr': m['pdr_predrain'],
        'mean_delay_ms': m['mean_delay_ms'],
        'congestion_drop_share': cong_share,
        'override_rate': m.get('override_rate', 0.0),
        'bp_zerodiff_rate': m.get('bp_zerodiff_rate', 0.0),
    }


def evaluate_teacher(teacher_name, cfg, seeds, metric='pdr_predrain'):
    """Serial single-teacher evaluation (kept for small/manual checks and for
    backward compatibility with earlier call sites). build_oracle_table uses
    the parallel path below for the full grid."""
    rows = [_run_single((teacher_name, {**cfg, 'seed': sd})) for sd in seeds]
    vals = [r['pdr'] for r in rows]
    return {
        'teacher': teacher_name,
        'mean': float(np.mean(vals)),
        'std': float(np.std(vals)),
        'mean_delay_ms': float(np.nanmean([r['mean_delay_ms'] for r in rows])),
        'congestion_drop_share': float(np.mean([r['congestion_drop_share'] for r in rows])),
        'override_rate': float(np.mean([r['override_rate'] for r in rows])),
        'bp_zerodiff_rate': float(np.mean([r['bp_zerodiff_rate'] for r in rows])),
        'n_seeds': len(seeds),
    }


# ── Phase A: oracle table (parallel) ─────────────────────────────────────────

def build_oracle_table(scenarios, packet_rates, seeds, panel=None,
                       base_cfg=None, verbose=True, max_workers=None):
    """Phase A, parallel. Returns:
        table       : {(scenario_class, load_bucket): [ranked (teacher, mean_pdr)]}
        raw         : list of per-(scenario, rate, teacher) AGGREGATE result dicts
        congestion  : {(scenario_class, load_bucket): mean congestion_drop_share}
        diagnostics : {key: {'override': {teacher: rate}, 'bp_zerodiff': rate}},
                      aggregated from the SAME runs as the PDR numbers.
        table_stats : {key: {teacher: {'mean','std','n'}}} -- per-teacher summary
                      statistics for each cell, letting the caller run a proper
                      significance test (e.g. Welch's t) between the top two
                      teachers instead of relying on a fixed margin heuristic.
    """
    panel = panel or PANEL
    base_cfg = base_cfg or {}

    # Build the full job list: one job per (scenario, rate, teacher, seed).
    jobs = []
    job_meta = []   # parallel list: (sc_name, pr, key, teacher) per job
    scenario_keys = {}
    for sc_name, sc in scenarios.items():
        for pr in packet_rates:
            cfg = {**base_cfg, **sc, 'packet_rate': pr}
            key = (scenario_class(cfg), load_bucket(pr))
            scenario_keys[(sc_name, pr)] = key
            for t in panel:
                for sd in seeds:
                    jobs.append((t, {**cfg, 'seed': sd}))
                    job_meta.append((sc_name, pr, key, t))

    n_jobs = len(jobs)
    if verbose:
        print(f"    dispatching {n_jobs} runs across up to "
              f"{max_workers or 'all available'} worker processes...")

    results = [None] * n_jobs
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_run_single, jobs[i]): i for i in range(n_jobs)}
        done = 0
        for fut in as_completed(futures):
            i = futures[fut]
            results[i] = fut.result()
            done += 1
            if verbose and done % max(n_jobs // 20, 1) == 0:
                print(f"      {done}/{n_jobs} runs complete")

    # Aggregate: per (scenario, rate, teacher) across seeds -> raw list
    # (kept in the original shape for any existing caller that expects it).
    grouped = {}
    for meta, r in zip(job_meta, results):
        sc_name, pr, key, t = meta
        grouped.setdefault((sc_name, pr, t), []).append(r)

    raw = []
    cells = {}
    cong_cells = {}
    diag_cells = {}   # key -> {'override': {teacher: [rates]}, 'bp_zerodiff': [rates]}
    for (sc_name, pr, t), rows in grouped.items():
        key = scenario_keys[(sc_name, pr)]
        vals = [r['pdr'] for r in rows]
        mean_pdr = float(np.mean(vals))
        agg = {
            'teacher': t, 'scenario': sc_name, 'packet_rate': pr,
            'scenario_class': key[0], 'load_bucket': key[1],
            'mean': mean_pdr, 'std': float(np.std(vals)),
            'mean_delay_ms': float(np.nanmean([r['mean_delay_ms'] for r in rows])),
            'congestion_drop_share': float(np.mean([r['congestion_drop_share'] for r in rows])),
            'override_rate': float(np.mean([r['override_rate'] for r in rows])),
            'bp_zerodiff_rate': float(np.mean([r['bp_zerodiff_rate'] for r in rows])),
            'n_seeds': len(rows),
        }
        raw.append(agg)
        cells.setdefault(key, {}).setdefault(t, []).append(mean_pdr)
        cong_cells.setdefault(key, []).append(agg['congestion_drop_share'])
        dc = diag_cells.setdefault(key, {'override': {}, 'bp_zerodiff': []})
        dc['override'].setdefault(t, []).append(agg['override_rate'])
        if t == 'backpressure':
            dc['bp_zerodiff'].append(agg['bp_zerodiff_rate'])

    if verbose:
        for sc_name, pr in scenario_keys:
            key = scenario_keys[(sc_name, pr)]
            best = max(panel, key=lambda t: np.mean(cells[key][t]))
            print(f"    {sc_name:<14} rate={pr:<5.2f} class={key[0]:<12}"
                  f" bucket={key[1]:<7} best={best}")

    table = {}
    table_stats = {}   # key -> {teacher: {'mean':.., 'std':.., 'n':..}} for significance testing
    for key, per_teacher in cells.items():
        ranked = sorted(((t, float(np.mean(v))) for t, v in per_teacher.items()),
                        key=lambda kv: kv[1], reverse=True)
        table[key] = ranked

    for agg in raw:
        key = (agg['scenario_class'], agg['load_bucket'])
        table_stats.setdefault(key, {})[agg['teacher']] = {
            'mean': agg['mean'], 'std': agg['std'], 'n': agg['n_seeds']}

    congestion = {k: float(np.mean(v)) for k, v in cong_cells.items()}
    diagnostics = {
        k: {'override': {t: float(np.mean(v)) for t, v in d['override'].items()},
            'bp_zerodiff': float(np.mean(d['bp_zerodiff'])) if d['bp_zerodiff'] else 0.0}
        for k, d in diag_cells.items()
    }
    return table, raw, congestion, diagnostics, table_stats


def welch_ttest(mean1, std1, n1, mean2, std2, n2):
    """Welch's t-test for two independent samples with unequal variance,
    computed from summary statistics (mean/std/n) rather than raw data --
    exactly what's available per teacher per cell after seeds are aggregated.

    Replaces a FIXED margin heuristic (THIN_MARGIN=0.02, set when only 2-3
    seeds were available) with an actual significance test now that n=30 gives
    enough samples to compute one properly. Returns (t_stat, dof, p_value),
    two-sided. Falls back to a conservative "not significant" (p=1.0) if either
    sample has zero variance and zero difference (degenerate case), or if
    scipy is unavailable.
    """
    if n1 < 2 or n2 < 2:
        return float('nan'), float('nan'), 1.0
    se1_sq = (std1 ** 2) / n1
    se2_sq = (std2 ** 2) / n2
    se = np.sqrt(se1_sq + se2_sq)
    if se == 0:
        return (float('inf') if mean1 != mean2 else 0.0), float('nan'), \
               (0.0 if mean1 != mean2 else 1.0)
    t_stat = (mean1 - mean2) / se
    dof_num = (se1_sq + se2_sq) ** 2
    dof_den = (se1_sq ** 2) / max(n1 - 1, 1) + (se2_sq ** 2) / max(n2 - 1, 1)
    dof = dof_num / dof_den if dof_den > 0 else (n1 + n2 - 2)
    try:
        from scipy import stats as _stats
        p_value = 2.0 * _stats.t.sf(abs(t_stat), dof)
    except ImportError:
        # Normal approximation, adequate for dof this large (n=30 per group).
        from math import erf, sqrt
        p_value = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(t_stat) / sqrt(2.0))))
    return float(t_stat), float(dof), float(p_value)


def oracle_teacher_for(table, sc_class, bucket, fallback='spbp'):
    """Look up the empirically best teacher for a regime, with graceful
    fallback to the nearest available bucket then to a sane default."""
    if (sc_class, bucket) in table:
        return table[(sc_class, bucket)][0][0]
    for b in ('medium', 'high', 'low'):
        if (sc_class, b) in table:
            return table[(sc_class, b)][0][0]
    for (c, b), ranked in table.items():
        if b == bucket:
            return ranked[0][0]
    return fallback


# ── Phase B: vote collection at a decision point ─────────────────────────────

def collect_votes(G, current, destination, panel=None):
    """Query every panel teacher for its proposed next hop.

    Returns (votes, bp_zerodiff):
      votes       : {teacher_name: next_hop | None}
      bp_zerodiff : True if classical backpressure had no queue gradient
                    (design spec S3.1 honesty diagnostic).
    """
    panel = panel or PANEL
    votes = {}
    bp_zerodiff = False
    for t in panel:
        if t == 'backpressure':
            hop, zd = backpressure_next_hop(G, current, destination,
                                            return_zerodiff_flag=True)
            votes[t] = hop
            bp_zerodiff = zd
        else:
            votes[t] = TEACHERS[t](G, current, destination)
    return votes, bp_zerodiff


def vote_agreement(votes, action):
    """Fraction of teachers proposing `action`. Retained as a CONFIDENCE WEIGHT,
    never as the label — that distinction is the whole point of S4."""
    valid = [v for v in votes.values() if v is not None]
    if not valid or action is None:
        return 0.0
    return sum(1 for v in valid if v == action) / len(valid)


def label_decision(G, current, destination, oracle_teacher, panel=None):
    """Produce the oracle-labeled decision record for one routing decision.

    The label is the action of the teacher empirically shown (Phase A) to
    achieve the highest network PDR in THIS regime — not the majority action.
    """
    votes, bp_zd = collect_votes(G, current, destination, panel)
    best_action = votes.get(oracle_teacher)
    return {
        'votes': votes,
        'oracle_teacher': oracle_teacher,
        'best_action': best_action,
        'vote_agreement': vote_agreement(votes, best_action),
        'bp_zerodiff': bp_zd,
    }
