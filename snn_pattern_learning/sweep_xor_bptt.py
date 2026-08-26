"""Find a BPTT config that solves temporal XOR.

Fixes vs the failing first attempt:
  - wider Boxcar surrogate window (subthresh) -- repo default 0.1 passes
    gradient only within |mem-0.5|<0.1 and BPTT dies in the silent minimum
  - loss restricted to the response window (same task definition as the
    e-prop conditions' err_window)
  - pure BPTT (e-prop .grad buffers cleared before backward)
"""
import sys
sys.path.insert(0, '.')
from run_xor import run

print(f"{'subthr':>6} {'lr':>5} {'seed':>4}  best_acc first100 hold(last20)")
best = []
for subthr in [0.1, 0.25, 0.5]:
    for lr in [0.01, 0.05, 0.1]:
        for seed in [0, 1, 2]:
            r = run('bptt', 60, lr, seed=seed, verbose=False,
                    bptt_subthresh=subthr)
            tail = r['accs'][-20:]
            hold = sum(a == 1.0 for a in tail) / len(tail)
            first = next((i + 1 for i, a in enumerate(r['accs'])
                          if a == 1.0), -1)
            best.append((r['best_acc'], hold, subthr, lr, seed))
            print(f"{subthr:>6} {lr:>5} {seed:>4}  {r['best_acc']:.2f}"
                  f"     ep{first:<4} {hold:.2f}", flush=True)

best.sort(key=lambda t: (-t[0], -t[1]))
print("\nTOP:", best[:5])
