import sys, time, warnings
sys.path.insert(0, 'src')
warnings.simplefilter('ignore')
import experiment_spbp_mechanism   # registers spbp_ab_*
from simulator_v2 import FANETSimulatorV2
from config_v2 import BASE, get_suite

cfg = get_suite('convergecast')['sink_50']
VARIANTS = ['spbp_ab_noqueue', 'spbp_ab_full', 'spbp_ab_candqueue', 'spbp_ab_additive']

# Test all four variants at rate=8.0 -- the current self_tests() midpoint rate.
# spbp_ab_noqueue already confirmed ~6.4s here; this checks whether the OTHER
# three (never individually timed before) match that or are pathologically
# slower -- specifically spbp_ab_full, whose detour-seeking has shown up as
# extra hops in every scenario tested, and may have nowhere real to detour TO
# in a single-sink topology, driving retries/wall-clock time up sharply.
for actor in VARIANTS:
    print('starting %s ...' % actor, flush=True)
    t0 = time.time()
    m = FANETSimulatorV2({**BASE, **cfg, 'duration': 40.0, 'seed': 1,
                          'packet_rate': 8.0, 'actor': actor}).run()
    el = time.time() - t0
    print('  %-20s elapsed=%.1fs  gen=%d deliv=%d  pdr=%.4f' % (
        actor, el, m['n_generated'], m['n_delivered'], m['network_pdr']))
