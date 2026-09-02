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

# ── v8 suites: declared, deliberately empty until v8 is applied ────────────
SCENARIOS_TALL = {}
SCENARIOS_DENSITY = {}
SCENARIOS_CONVERGECAST = {}

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
