import sys, warnings
sys.path.insert(0, 'src')
warnings.simplefilter('ignore')
from simulator_v2 import FANETSimulatorV2

# G2's OWN exact scenario, not config_v2's dense_slow -- these differ
# (area 1000 vs 800, comm_range 280 vs 250). Confirmed by inspection.
g2_scenario = dict(num_drones=30, area_x=1000.0, area_y=1000.0,
                   z_min=50, z_max=150, comm_range=280.0,
                   speed_min=5.0, speed_max=15.0, pause_max=5.0)

for rate in [0.25, 0.5, 1.0, 2.0, 4.0, 20.0, 60.0]:
    for actor in ['dijkstra', 'spbp']:
        m = FANETSimulatorV2({**g2_scenario, 'duration': 40.0, 'seed': 42,
                              'packet_rate': rate, 'actor': actor}).run()
        print('rate=%6.2f actor=%-9s pdr=%.4f  gen=%d deliv=%d  drops=%s  phantom=%s' % (
            rate, actor, m['network_pdr'], m['n_generated'], m['n_delivered'],
            m['drop_reasons'], m.get('n_phantom_slots')))
