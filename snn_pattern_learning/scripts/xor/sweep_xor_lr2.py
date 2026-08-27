import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # package root
from run_xor import run

print("--- lr decay sweep, seed 0 ---")
for lr, dec in [(0.1, 1.0), (0.1, 0.95), (0.1, 0.9), (0.2, 0.9), (0.2, 0.85)]:
    r = run('digital', 60, lr, seed=0, verbose=False, freeze_hidden=True,
            lr_decay=dec)
    tail = r['accs'][-20:]
    hold = sum(a == 1.0 for a in tail) / len(tail)
    first = next((i + 1 for i, a in enumerate(r['accs']) if a == 1.0), -1)
    print(f"lr={lr} decay={dec}: first100@ep{first} "
          f"hold100(last20) {hold:.2f}", flush=True)

print("--- more seeds at the best-looking setting ---")
for seed in range(8):
    r = run('digital', 60, 0.1, seed=seed, verbose=False, freeze_hidden=True,
            lr_decay=0.95)
    tail = r['accs'][-20:]
    hold = sum(a == 1.0 for a in tail) / len(tail)
    first = next((i + 1 for i, a in enumerate(r['accs']) if a == 1.0), -1)
    print(f"seed={seed}: best_acc {r['best_acc']:.2f} first100@ep{first} "
          f"hold100(last20) {hold:.2f}", flush=True)
