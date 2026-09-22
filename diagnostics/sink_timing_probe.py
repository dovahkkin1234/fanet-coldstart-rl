import sys, time, warnings
sys.path.insert(0, 'src')
warnings.simplefilter('ignore')
import experiment_spbp_mechanism   # registers spbp_ab_* -- MISSING from the first draft
from simulator_v2 import FANETSimulatorV2
from config_v2 import BASE, get_suite

cfg = get_suite('convergecast')['sink_50']

# Start low and safe, step up toward the rate self_tests was stuck on (20.0).
# Prints BEFORE and AFTER each episode, with elapsed time -- if it hangs,
# you will see exactly which rate it hung at, not just "stuck" with no info.
for rate in [4.0, 8.0, 12.0, 16.0, 20.0]:
    print('starting rate=%.1f ...' % rate, flush=True)
    t0 = time.time()
    m = FANETSimulatorV2({**BASE, **cfg, 'duration': 40.0, 'seed': 1,
                          'packet_rate': rate, 'actor': 'spbp_ab_noqueue'}).run()
    el = time.time() - t0
    print('  rate=%5.1f  elapsed=%.1fs  gen=%d deliv=%d  pdr=%.4f  drops=%s' % (
        rate, el, m['n_generated'], m['n_delivered'], m['network_pdr'],
        m['drop_reasons']))
