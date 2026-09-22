import sys, warnings
sys.path.insert(0, 'src')
warnings.simplefilter('ignore')
from simulator_v2 import FANETSimulatorV2

# G2's exact scenario (confirmed by inspection: --area default 1000, comm_range 280)
g2_scenario = dict(num_drones=30, area_x=1000.0, area_y=1000.0,
                   z_min=50, z_max=150, comm_range=280.0,
                   speed_min=5.0, speed_max=15.0, pause_max=5.0)

# Candidate replacement for the stale [0.25, 0.5, 1.0, 2.0, 4.0] sweep.
# Probing wider before picking -- same discipline as the anchor fix.
for rate in [5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]:
    m = FANETSimulatorV2({**g2_scenario, 'duration': 40.0, 'seed': 42,
                          'packet_rate': rate, 'actor': 'dijkstra'}).run()
    print('rate=%6.1f  pdr=%.4f  gen=%d deliv=%d  drops=%s  phantom=%s' % (
        rate, m['network_pdr'], m['n_generated'], m['n_delivered'],
        m['drop_reasons'], m.get('n_phantom_slots')))
