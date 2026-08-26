#!/usr/bin/env python3
"""Software lr sweep for the paper's experiment-1 baselines.

Sweeps learning rate x seed for the three software conditions
(bptt / digital e-prop / frozen_wout) on the fixed teacher task at the
hardware operating point (thresh 0.4, tau 0.6, T=20, 50 epochs), so each
condition gets its own tuned lr before the analog runs. Reuses run() from
run_three_conditions.py -- same task, same init path (torch.manual_seed(seed)
-> model ctor), so seed s here starts from exactly the weights the analog
run with training.init_seed=s will copy.

Output: results/eprop_grad_log/sw_lr_sweep.json
  {condition: {lr: {seed: {best_loss, final_vrd, losses, vrds}}}}
"""
import json
import os
import time

from run_three_conditions import run

CONDITIONS = ["bptt", "digital", "frozen_wout"]
LRS = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3]
SEEDS = [0, 1, 2, 3, 4]
EPOCHS = 50
THRESH = 0.4
TAU = 0.6
OUT = "results/eprop_grad_log/sw_lr_sweep.json"

results = {c: {} for c in CONDITIONS}
t0 = time.time()
n_done, n_total = 0, len(CONDITIONS) * len(LRS) * len(SEEDS)
for cond in CONDITIONS:
    for lr in LRS:
        results[cond][str(lr)] = {}
        for seed in SEEDS:
            r = run(cond, EPOCHS, lr, seed=seed, thresh=THRESH, tau=TAU)
            results[cond][str(lr)][str(seed)] = dict(
                best_loss=r["best_loss"], final_vrd=r["vrds"][-1],
                losses=r["losses"], vrds=r["vrds"])
            n_done += 1
            print(f"[{n_done:3d}/{n_total}] {cond:>12} lr={lr:<5} seed={seed}"
                  f"  best={r['best_loss']:.4f}  vrd={r['vrds'][-1]:.3f}"
                  f"  ({time.time()-t0:.0f}s)", flush=True)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(results, open(OUT, "w"))
print(f"\nsaved -> {OUT}")

print(f"\n{'condition':>12} {'lr':>6}  mean_best  median  min    max")
for cond in CONDITIONS:
    for lr in LRS:
        v = sorted(results[cond][str(lr)][str(s)]["best_loss"] for s in SEEDS)
        mean = sum(v) / len(v)
        print(f"{cond:>12} {lr:>6}  {mean:9.4f}  {v[len(v)//2]:.4f}"
              f"  {v[0]:.4f} {v[-1]:.4f}")
