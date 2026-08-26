import sys
sys.path.insert(0, '.')
from run_xor import run

for lr in [0.02, 0.05, 0.1, 0.2]:
    for seed in [0, 1, 2]:
        r = run('digital', 80, lr, seed=seed, verbose=False,
                freeze_hidden=True)
        tail = r['accs'][-20:]
        hold = sum(a == 1.0 for a in tail) / len(tail)
        first = next((i + 1 for i, a in enumerate(r['accs']) if a == 1.0), -1)
        print(f"lr={lr} seed={seed}: best_acc {r['best_acc']:.2f} "
              f"first100@ep{first} hold100(last20) {hold:.2f}", flush=True)
