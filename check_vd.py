import numpy as np
d = np.load('data/phaseB/decisions.npz')
qf, scen = d['query_feat'], d['scenario']
import sys; sys.path.insert(0,'src')
import features_v2 as F
col = F.QUERY_FEATURES.index('current_queue_occupancy')
for sc in ['very_dense','dense_slow','medium_slow','sparse_fast']:
    v = qf[scen==sc, col]
    print(f"{sc:<13} n={len(v):>7}  mean={v.mean():.6f}  max={v.max():.4f}  "
          f"nonzero={100*(v>0).mean():.3f}%")