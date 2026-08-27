"""Find a learnable operating point for temporal XOR (digital e-prop).

Sweeps membrane tau, surrogate threshold, learning rate and the A->B gap.
Digital condition only (Mock interface = the exact pipeline the analog run
will use). Prints a ranked table; the winner becomes the hardware config.
"""
import os, sys, itertools, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # package root
import torch
from run_xor import run

EPOCHS = 60

grid = dict(
    tau=[0.6, 0.8, 0.9],
    thresh=[0.2, 0.5],
    lr=[0.05, 0.1, 0.3],
    gap=["long", "short"],   # long: B at 9-12 (gap 5), short: B at 5-8 (gap 1)
)

def ds_kw_for(gap):
    if gap == "long":
        return dict(a_window=(1, 4), b_window=(9, 12), response_window=(14, 20))
    return dict(a_window=(1, 4), b_window=(5, 8), response_window=(10, 16))

rows = []
for tau, thresh, lr, gap in itertools.product(*grid.values()):
    r = run("digital", EPOCHS, lr, seed=0, verbose=False,
            model_kw=dict(init_tau=tau, init_thresh=thresh),
            ds_kw=ds_kw_for(gap))
    # stability: fraction of the last 20 epochs at acc == 1.0
    tail = r["accs"][-20:]
    hold = sum(a == 1.0 for a in tail) / len(tail)
    rows.append((r["best_acc"], hold, r["best_loss"], tau, thresh, lr, gap))
    print(f"tau={tau} thr={thresh} lr={lr} gap={gap:5s} -> "
          f"best_acc {r['best_acc']:.2f}  hold100 {hold:.2f}  "
          f"best_loss {r['best_loss']:.3f}", flush=True)

rows.sort(key=lambda t: (-t[0], -t[1], t[2]))
print("\n=== TOP 5 ===")
for r in rows[:5]:
    print(f"acc {r[0]:.2f}  hold {r[1]:.2f}  loss {r[2]:.3f}  "
          f"tau={r[3]} thr={r[4]} lr={r[5]} gap={r[6]}")
json.dump([list(r) for r in rows], open("results/xor/sweep.json", "w"))
