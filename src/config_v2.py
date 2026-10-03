"""config_v2.py -- SINGLE SOURCE OF TRUTH for scenario grids and base config.

WHY THIS EXISTS

At ebc013d the same SCENARIOS dict was hardcoded independently in EIGHT files:

    generate_dataset_v2.py             experiment_spbp_mechanism.py
    experiment_headroom.py             experiment_collision_model.py
    experiment_locality_cost.py        experiment_calibration_sensitivity.py
    experiment_queue_weight.py         preflight_teachers_v2_check.py

They were still in sync -- verified by direct comparison, no drift. But v8
changes duration 40 -> 1000 and altitude 50-150 -> 100-300 in ONE of them.
Applied as written, the dataset would move to the new operating point while
seven experiments silently kept measuring the old one, and every comparison
against the dataset would be invalid without anything failing.

That is not hypothetical: results/headroom.json was already dated by git
archaeology rather than by content, because nothing recorded which config
produced it. Import from here instead, and apply v8 to this file only.

SUITES
    A / 'default'  the four-scenario grid M1-M4 was built on
    'tall'         span-500 altitude probe          (v8)
    'density'      Suite B, node-count sweep        (v8)
    'convergecast' Suite C, many-to-one to a pinned sink  (v8, v9)

Suites beyond 'default' are declared here but stay EMPTY until v8 lands, so
get_suite() raises a clear error instead of silently returning the wrong grid.
"""

# ── base episode configuration ─────────────────────────────────────────────
# v8 changes duration and altitude HERE and nowhere else.
BASE = dict(z_min=50, z_max=150, duration=40.0, drain_time=10.0,
            interference_on=True)

RATES = [0.5, 2.0, 4.0]

# ── Suite A: the grid every M1-M4 result rests on ──────────────────────────
SCENARIOS = {
    'very_dense':  dict(num_drones=45, area_x=700,  area_y=700,  comm_range=250,
                        speed_min=5,  speed_max=15, pause_max=5.0),
    'dense_slow':  dict(num_drones=30, area_x=800,  area_y=800,  comm_range=250,
                        speed_min=5,  speed_max=15, pause_max=5.0),
    'medium_slow': dict(num_drones=30, area_x=1300, area_y=1300, comm_range=280,
                        speed_min=5,  speed_max=15, pause_max=5.0),
    'sparse_fast': dict(num_drones=20, area_x=1500, area_y=1500, comm_range=300,
                        speed_min=35, speed_max=50, pause_max=2.0),
}

# ── v8a suites ─────────────────────────────────────────────────────────────
# Populated by apply_v8a_suites_v1.py. SCENARIOS (Suite A) is deliberately NOT
# touched, and BASE is NOT touched -- the operating-point change is v8b, which
# ships after the SP-BP parity gate.

# S3 -- ALTITUDE SCOPE-PROBE, span 500 m (100-600). REPORT SEPARATELY: this is
# a scope probe, not a Suite A cell, and must never be pooled with one.
SCENARIOS_TALL = {
    'tall_probe': dict(num_drones=30, area_x=1300, area_y=1300, comm_range=280,
                       speed_min=5, speed_max=15, pause_max=5.0,
                       z_min=100, z_max=600),
}

# S4 -- SUITE B, CLEAN DENSITY SWEEP. Fixed geometry; vary node count ONLY, so
# any density effect is attributable to density and nothing else. This is the
# control Suite A lacks (Suite A varies node count, area, range AND speed
# together -- FILE1 §4.3's recorded methodological weakness).
_DENSITY_FIXED = dict(area_x=1000, area_y=1000, comm_range=250,
                      speed_min=10, speed_max=30, pause_max=5.0)

SCENARIOS_DENSITY = {
    f'density_{n}': dict(_DENSITY_FIXED, num_drones=n)
    for n in (50, 100, 150, 200)
}

# S5 -- SUITE C, CONVERGECAST COMPARABILITY. Suite B plus one fixed ground
# sink. Suite B alone does NOT make our numbers comparable to HCPMR's: matching
# density leaves the TRAFFIC PATTERN different (random UAV pairs vs many-to-one
# to a fixed station). `sink_node=0` makes every flow terminate at node 0,
# which v9 pins stationary at the area centre at z_min -- a ground station does
# not fly.
SCENARIOS_CONVERGECAST = {
    f'sink_{n}': dict(_DENSITY_FIXED, num_drones=n, sink_node=0)
    for n in (50, 100, 150, 200)
}

# Convenience union for scripts that explicitly want everything. The default
# grid remains SCENARIOS alone -- nothing is silently averaged across suites.
SCENARIOS_ALL = {**SCENARIOS, **SCENARIOS_TALL,
                 **SCENARIOS_DENSITY, **SCENARIOS_CONVERGECAST}

SUITES = {
    'default':      SCENARIOS,
    'A':            SCENARIOS,
    'tall':         SCENARIOS_TALL,
    'density':      SCENARIOS_DENSITY,
    'B':            SCENARIOS_DENSITY,
    'convergecast': SCENARIOS_CONVERGECAST,
    'C':            SCENARIOS_CONVERGECAST,
}


def get_suite(name='default'):
    """Return a scenario grid by name. Raises rather than silently returning an
    empty grid, so a suite selected before v8 lands fails loudly."""
    if name not in SUITES:
        raise KeyError(f"unknown suite {name!r}; available: {sorted(SUITES)}")
    grid = SUITES[name]
    if not grid:
        raise RuntimeError(
            f"suite {name!r} is declared but empty -- v8 has not been applied. "
            f"Applying v8 populates SCENARIOS_TALL / SCENARIOS_DENSITY / "
            f"SCENARIOS_CONVERGECAST in config_v2.py.")
    return grid


def episode_config(scenario, rate, seed, suite='default', actor='spbp', **over):
    """Build a full episode config. Every caller should route through this so
    the provenance of a result is a function of its arguments, not of which
    file happened to define BASE."""
    return {**BASE, **get_suite(suite)[scenario],
            'packet_rate': rate, 'seed': seed, 'actor': actor, **over}


def provenance():
    """Config fingerprint to embed in every results file, so no future reader
    has to date a JSON by its git commit."""
    return {'duration': BASE['duration'], 'drain_time': BASE['drain_time'],
            'z_min': BASE['z_min'], 'z_max': BASE['z_max'],
            'interference_on': BASE['interference_on'],
            'rates': list(RATES), 'scenarios': sorted(SCENARIOS),
            'config_module': 'config_v2'}


# ── v26: DATASET V3 operating point and grid ───────────────────────────────
# BASE and RATES above stay the 40 s / 50-150 m PARITY REFERENCE until v8b
# (decision D-12). The dataset is generated at the measured operating point
# EXPLICITLY, through dataset_episode_config(), so the dataset never depends on
# BASE having moved and moving BASE never silently changes the dataset.
# docs/DATASET_V3_SPEC.md is the specification for everything below.
OPERATING_POINT = dict(z_min=100, z_max=300, duration=1000.0, drain_time=10.0,
                       interference_on=True, initial_energy=8000.0)

# Per-scenario grid (decision 2026-10-02: usable band + one low-load anchor).
#   band   = usable rates at 1000 s / battery 8000 / 100-300 m under the band
#            search's reference actor spbp_ab_noqueue (results/band_*_1000s.json,
#            criteria elasticity 0.05-0.85, q_ovf >= 0.02, energy <= 0.05).
#            The band is a property of the LOAD, measured with a fixed reference
#            policy; it is deliberately NOT re-defined under the dataset's own
#            behaviour policy (that would let a better policy move the band).
#   anchor = the lowest rate of that scenario's band sweep: below the band on
#            purpose, for the low-vs-high load contrast.
# Buckets are scenario-RELATIVE: anchor -> 'low', top band rate -> 'high',
# other band rates -> 'medium'. Never compare bucket labels with pre-v26 results
# (those used absolute thresholds 0.5 / 2.0 on the old grid).
# sink_50's band file was measured at 50-150 m (BASE); verify_dataset_grid_v3
# marks it UNVERIFIED until it is re-measured at 100-300 m, and the generator
# refuses unverified cells unless explicitly allowed.
DATASET_GRID = {
    'dense_slow':  {'suite': 'A', 'anchor': [40.0], 'band': [60.0, 80.0, 100.0]},
    'very_dense':  {'suite': 'A', 'anchor': [40.0], 'band': [60.0, 80.0]},
    'medium_slow': {'suite': 'A', 'anchor': [30.0], 'band': [40.0, 60.0, 80.0]},
    'sparse_fast': {'suite': 'A', 'anchor': [40.0], 'band': [80.0, 100.0]},
    'sink_50':     {'suite': 'C', 'anchor': [20.0], 'band': [30.0]},
}
DATASET_SEEDS = list(range(101, 151))
DATASET_SPLITS = {'train': (101, 135), 'val': (136, 142), 'test': (143, 150)}
GENERALISATION_SCENARIO = 'medium_slow'


def dataset_rates(scenario):
    g = DATASET_GRID[scenario]
    return sorted(g['anchor'] + g['band'])


def dataset_cells(scenarios=None):
    """[(scenario, rate)] in a fixed, documented order."""
    out = []
    for sc in DATASET_GRID:
        if scenarios is None or sc in scenarios:
            out += [(sc, r) for r in dataset_rates(sc)]
    return out


def rate_role(scenario, rate):
    g = DATASET_GRID[scenario]
    if float(rate) in g['anchor']:
        return 'anchor'
    if float(rate) in g['band']:
        return 'band'
    raise KeyError(f'{scenario}@{rate} is not in DATASET_GRID')


def rate_index(scenario, rate):
    return dataset_rates(scenario).index(float(rate))


def load_bucket_rel(scenario, rate):
    """Scenario-relative load bucket: 'low' / 'medium' / 'high'."""
    if rate_role(scenario, rate) == 'anchor':
        return 'low'
    return 'high' if float(rate) == max(DATASET_GRID[scenario]['band']) else 'medium'


def split_of_seed(seed):
    for name, (lo, hi) in DATASET_SPLITS.items():
        if lo <= int(seed) <= hi:
            return name
    raise KeyError(f'seed {seed} belongs to no dataset split {DATASET_SPLITS}')


def dataset_scenario_cfg(scenario):
    suite = 'C' if DATASET_GRID[scenario]['suite'] == 'C' else 'A'
    return get_suite(suite)[scenario]


def dataset_episode_config(scenario, rate, seed, **over):
    """Full episode config at the DATASET operating point (not BASE)."""
    return {**BASE, **OPERATING_POINT, **dataset_scenario_cfg(scenario),
            'packet_rate': float(rate), 'seed': int(seed), **over}


def dataset_provenance():
    return {'operating_point': dict(OPERATING_POINT),
            'grid': {k: dict(v) for k, v in DATASET_GRID.items()},
            'seeds': [DATASET_SEEDS[0], DATASET_SEEDS[-1]],
            'splits': {k: list(v) for k, v in DATASET_SPLITS.items()},
            'generalisation_scenario': GENERALISATION_SCENARIO,
            'parity_reference_base': dict(BASE),
            'config_module': 'config_v2'}
