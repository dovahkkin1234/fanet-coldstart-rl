import sys
sys.path.insert(0, 'src')
from simulator_v2 import FANETSimulatorV2

s = dict(num_drones=30, area_x=1000, area_y=1000, comm_range=280,
         speed_min=5, speed_max=15, pause_max=5.0,
         z_min=50, z_max=150, duration=40, seed=42)

print("ACTOR SPREAD with locked defaults (ACT_BETA=0, max_retx=5):")
print(f"  {'rate':>5}{'dijkstra':>10}{'q_greedy':>10}{'backpr':>10}  spread")
for pr in [0.5, 1.0, 2.0]:
    vals = {}
    for actor in ['dijkstra', 'queue_aware_greedy', 'backpressure']:
        m = FANETSimulatorV2({**s, 'packet_rate': pr, 'interference_on': True, 'actor': actor}).run()
        vals[actor] = m['network_pdr']
    spread = max(vals.values()) - min(vals.values())
    print(f"  {pr:>5.1f}{vals['dijkstra']:>10.3f}{vals['queue_aware_greedy']:>10.3f}{vals['backpressure']:>10.3f}  {spread:+.3f}")