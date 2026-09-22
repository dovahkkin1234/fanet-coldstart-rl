import sys, warnings
sys.path.insert(0, 'src')
warnings.simplefilter('ignore')
from simulator_v2 import FANETSimulatorV2

# G2's exact scenario. Fine-grained near the floor: does a rate exist with
# activity > 0 (needed by check 4) while total drops == 0 (needed by check 3's
# baseline)? Or do these two never coexist in this scenario/geometry?
g2_scenario = dict(num_drones=30, area_x=1000.0, area_y=1000.0,
                   z_min=50, z_max=150, comm_range=280.0,
                   speed_min=5.0, speed_max=15.0, pause_max=5.0)

for rate in [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 15.0, 18.0, 20.0, 25.0, 30.0, 35.0]:
    m = FANETSimulatorV2({**g2_scenario, 'duration': 40.0, 'seed': 42,
                          'packet_rate': rate, 'actor': 'dijkstra'}).run()
    tot_drops = sum(m['drop_reasons'].values())
    print('rate=%6.1f  activity=%.4f  tot_drops=%4d  drops=%s' % (
        rate, m['mean_activity'], tot_drops, m['drop_reasons']))
