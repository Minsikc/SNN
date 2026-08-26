#!/usr/bin/env python3
"""Paper figure for experiment 1 (4x4, w_scale=1.5, dataset seed 25).

Panel A: loss curves of all four conditions for one representative seed
         (same initial weights).
Panel B: per-seed paired dot plot of best loss, seeds connected by lines,
         median bars per condition. Median, not mean: per-seed losses are
         basin-multimodal, a mean curve represents no actual run.

SW numbers: results/eprop_grad_log/sw_sweep_4x4_ds25.json (lr 0.05).
Analog numbers: parsed from _analog_4x4_ds25_s{seed}.log.
"""
import argparse
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SEEDS = [0, 1, 2, 3, 4]
LR = "0.05"
CONDS = ["bptt", "digital", "analog", "frozen_wout"]
LABELS = {"bptt": "BPTT (SW)", "digital": "e-prop digital (SW)",
          "analog": "e-prop analog (crossbar)", "frozen_wout": "W_out frozen"}
COLORS = {"bptt": "tab:gray", "digital": "tab:blue",
          "analog": "tab:red", "frozen_wout": "tab:green"}

ap = argparse.ArgumentParser()
ap.add_argument("--rep-seed", type=int, default=None,
                help="representative seed for panel A (default: the seed "
                     "whose analog best loss is closest to the median)")
ap.add_argument("--out", default="results/eprop_grad_log/fig_4x4_ds25_summary.png")
args = ap.parse_args()

sw = json.load(open("results/eprop_grad_log/sw_sweep_4x4_ds25.json"))


def analog_losses(seed):
    losses = []
    for line in open(f"_analog_4x4_ds25_s{seed}.log", encoding="utf-8",
                     errors="ignore"):
        m = re.match(r"Epoch (\d+)/50, Loss: ([\d.]+)", line)
        if m:
            losses.append(float(m.group(2)))
    return losses


curves = {c: {s: sw[c][LR][str(s)]["losses"] for s in SEEDS}
          for c in ["bptt", "digital", "frozen_wout"]}
curves["analog"] = {s: analog_losses(s) for s in SEEDS}
best = {c: {s: min(curves[c][s]) for s in SEEDS} for c in CONDS}

if args.rep_seed is None:
    med = np.median(list(best["analog"].values()))
    args.rep_seed = min(SEEDS, key=lambda s: abs(best["analog"][s] - med))

fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.2),
                               gridspec_kw={"width_ratios": [1.6, 1]})
for c in CONDS:
    axA.plot(range(1, len(curves[c][args.rep_seed]) + 1),
             curves[c][args.rep_seed], color=COLORS[c], label=LABELS[c],
             lw=1.8 if c == "analog" else 1.2,
             alpha=1.0 if c == "analog" else 0.85)
axA.set_xlabel("epoch")
axA.set_ylabel("loss")
axA.set_title(f"(A) training curves, seed {args.rep_seed} (shared init)")
axA.legend(fontsize=8)

xpos = {c: i for i, c in enumerate(CONDS)}
for s in SEEDS:
    ys = [best[c][s] for c in CONDS]
    axB.plot(list(xpos.values()), ys, color="lightgray", lw=0.8, zorder=1)
for c in CONDS:
    ys = [best[c][s] for s in SEEDS]
    axB.scatter([xpos[c]] * len(ys), ys, color=COLORS[c], s=28, zorder=2)
    axB.hlines(np.median(ys), xpos[c] - 0.22, xpos[c] + 0.22,
               color=COLORS[c], lw=2.5, zorder=3)
axB.set_xticks(list(xpos.values()))
axB.set_xticklabels(["BPTT", "digital", "analog", "frozen"], fontsize=9)
axB.set_ylabel("best loss (50 ep)")
axB.set_title("(B) per-seed best loss, n=5 (paired)")

fig.tight_layout()
fig.savefig(args.out, dpi=200)

print(f"representative seed: {args.rep_seed}")
print(f"{'seed':>4} " + " ".join(f"{c:>8}" for c in CONDS))
for s in SEEDS:
    print(f"{s:>4} " + " ".join(f"{best[c][s]:8.3f}" for c in CONDS))
for c in CONDS:
    v = [best[c][s] for s in SEEDS]
    print(f"{c}: median {np.median(v):.3f}  mean {np.mean(v):.3f}")
print(f"saved -> {args.out}")
