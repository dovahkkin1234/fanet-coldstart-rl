import sys, warnings
sys.path.insert(0, 'src')
warnings.simplefilter('ignore')
from simulator_v2 import FANETSimulatorV2
from config_v2 import BASE, SCENARIOS

for rate in [4.0, 20.0, 60.0]:
    m = FANETSimulatorV2({**BASE, **SCENARIOS['dense_slow'],
                          'packet_rate': rate, 'seed': 42, 'actor': 'dijkstra'}).run()
    print('rate=%5.1f  pdr=%.4f  drops=%s  phantom=%s' % (
        rate, m['network_pdr'], m['drop_reasons'], m.get('n_phantom_slots')))
