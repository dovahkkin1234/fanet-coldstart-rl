import sys; sys.path.insert(0, 'src')
from simulator_v2 import FANETSimulatorV2

cfg = dict(num_drones=30, area_x=1300, area_y=1300, comm_range=280,
           speed_min=5, speed_max=15, pause_max=5.0, z_min=50, z_max=150,
           duration=40.0, drain_time=10.0, interference_on=True,
           packet_rate=2.0, seed=42, actor='spbp')

for model in ['saturated', 'unsaturated']:
    m = FANETSimulatorV2({**cfg, 'collision_model': model}).run()
    print(f"{model:<13} pdr={m['pdr_predrain']:.6f}  linkq={m['mean_link_quality']:.6f}")

d = FANETSimulatorV2(cfg).run()   # no override -> should equal 'unsaturated'
print(f"{'default':<13} pdr={d['pdr_predrain']:.6f}  linkq={d['mean_link_quality']:.6f}")