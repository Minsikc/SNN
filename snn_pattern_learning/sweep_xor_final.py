import sys
sys.path.insert(0, '.')
from run_xor import run

print("--- final operating point candidates (seed 0, 60 ep) ---")
for lr, dec in [(0.1, 1.0), (0.15, 1.0), (0.2, 1.0), (0.2, 0.97), (0.15, 0.97)]:
    r = run('digital', 60, lr, seed=0, verbose=False, freeze_hidden=True,
            lr_decay=dec)
    tail = r['accs'][-20:]
    hold = sum(a == 1.0 for a in tail) / len(tail)
    first = next((i + 1 for i, a in enumerate(r['accs']) if a == 1.0), -1)
    last10 = r['accs'][-10:]
    print(f"lr={lr} dec={dec}: first100@ep{first} hold(last20) {hold:.2f} "
          f"last10={[f'{a:.2f}' for a in last10]}", flush=True)

print("--- frozen control, seed 0 ---")
r = run('frozen', 60, 0.15, seed=0, verbose=False, freeze_hidden=True)
print(f"frozen: best_acc {r['best_acc']:.2f} final {r['final_acc']:.2f} "
      f"accs(last10)={[f'{a:.2f}' for a in r['accs'][-10:]]}")
