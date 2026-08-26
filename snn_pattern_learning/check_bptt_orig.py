import sys
sys.path.insert(0, '.')
import numpy as np
from run_three_conditions import run

for lr in [0.1]:
    r = run('bptt', 50, lr)
    out = np.array(r['final_spikes'])
    tgt = np.array(r['target_spikes'])
    o = (out > 0.5).astype(int)
    t = tgt.astype(int)
    missed = int(((t == 1) & (o == 0)).sum())
    extra = int(((t == 0) & (o == 1)).sum())
    print(f"orig bptt (T=20 seed10 lr{lr} 50ep): "
          f"best loss {r['best_loss']:.4f}, final VRD {r['vrds'][-1]:.4f}, "
          f"min VRD {min(r['vrds']):.4f}")
    print(f"raster(best-loss epoch): targets {t.sum()}, "
          f"missed {missed}, extra {extra}")
