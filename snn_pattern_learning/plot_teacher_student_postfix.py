#!/usr/bin/env python3
"""Teacher-student results after the threshold fix -- software + analog hardware.

WHAT CHANGED, AND WHY THE OLD NUMBERS CANNOT BE COMPARED
--------------------------------------------------------
`Basic_RSNN_eprop_HW_forward` built its spiking nodes as LIF_Node() with no
`initial_thresh`, so the student always fired at the class default 0.5 while
the teacher generated its target at the yaml's init_thresh (0.2). The two were
different networks and the target was unreachable in principle: planting the
teacher's own weights into the student did NOT give loss 0.

The software models were fixed on 2026-08-13; the hardware model was missed.
So every earlier analog number (best_loss 0.231) sat on an unreachable floor,
and a sparse-firing student scored well by accident rather than by learning.

After the fix, planting the teacher's weights gives loss 0.000000 and
VRD 0.000000 with the output spike train EXACTLY equal to the target, for the
hardware model at thresh 0.2 and 0.4 alike. The optimum is now real, so the
gap each curve leaves is an honest optimisation gap.

LEFT   learning curves, all four conditions on one axis, matched settings.
RIGHT  analog gradient fidelity per epoch: r between the gradient asked for
       and the one read back off the array (25 cells), against mean |desired|
       on a log second axis. These two together explain any flattening --
       fidelity tracks gradient MAGNITUDE, and once |desired| falls under the
       device noise floor (~0.03) the correlation decays and learning stalls.
"""
import argparse
import csv
import json
import re
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, GRID = "#0b0b0b", "#e1e0d9"
COL = {"bptt": "#104281", "digital": "#2a78d6",
       "analog": "#eb6834", "frozen_wout": "#898781"}
LABEL = {"bptt": "BPTT (exact gradient)",
         "digital": "e-prop digital",
         "analog": "e-prop analog (W_out on 5x5 array)",
         "frozen_wout": "W_out frozen (control)"}


def load_analog(path):
    txt = open(path, encoding="utf-8", errors="ignore").read()
    ep = re.findall(r"Epoch (\d+)/\d+, Loss: ([0-9.]+), Metric: ([0-9.]+)", txt)
    return ([float(l) for _, l, _ in ep], [float(m) for _, _, m in ep])


def load_fidelity(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    by = defaultdict(lambda: ([], []))
    for r in rows:
        by[int(r["epoch"])][0].append(float(r["desired"]))
        by[int(r["epoch"])][1].append(float(r["hw_adc"]))
    eps, rs, mags = [], [], []
    for e in sorted(by):
        d, h = np.array(by[e][0]), np.array(by[e][1])
        if len(d) > 2 and d.std() > 0 and h.std() > 0:
            eps.append(e)
            rs.append(float(np.corrcoef(d, h)[0, 1]))
            mags.append(float(np.abs(d).mean()))
    return eps, rs, mags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sw", default="results/eprop_grad_log/sw_thr04_lr003_ep50.json")
    ap.add_argument("--analog-log", default="_analog_thr04_lr003.log")
    ap.add_argument("--grad-log",
                    default="results/eprop_grad_log/gradient_log_thr04_lr003.csv")
    ap.add_argument("--title", default="thr 0.4, lr 0.03, n_in 10, hidden 5, out 5")
    ap.add_argument("--out",
                    default="results/eprop_grad_log/teacher_student_postfix.png")
    args = ap.parse_args()

    sw = json.load(open(args.sw, encoding="utf-8"))
    a_loss, a_metric = load_analog(args.analog_log)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))

    ax = axes[0]
    for cond in ("bptt", "digital", "frozen_wout"):
        if cond not in sw:
            continue
        y = sw[cond]["losses"]
        ax.plot(range(1, len(y) + 1), y, lw=1.7, color=COL[cond],
                label=f"{LABEL[cond]} — best {sw[cond]['best_loss']:.3f}")
    if a_loss:
        ax.plot(range(1, len(a_loss) + 1), a_loss, lw=2.3, color=COL["analog"],
                label=f"{LABEL['analog']} — best {min(a_loss):.3f}")

    # The optimum is reachable, so mark it: without this the curves read as
    # "converged" when they have simply stopped improving.
    ax.axhline(0, color=INK, lw=1.2, ls="--", alpha=0.75)
    ax.annotate("realizable optimum (planted teacher: loss 0, VRD 0)",
                xy=(0.99, 0), xycoords=("axes fraction", "data"),
                xytext=(0, 7), textcoords="offset points",
                ha="right", fontsize=8, color=INK)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(f"Learning curves — {args.title}", fontsize=11)
    ax.grid(color=GRID, lw=0.8)
    ax.legend(fontsize=8, frameon=False)

    ax = axes[1]
    try:
        eps, rs, mags = load_fidelity(args.grad_log)
    except FileNotFoundError:
        eps = []
    if eps:
        ax.plot(eps, rs, "o-", ms=3.5, lw=1.6, color=COL["analog"],
                markerfacecolor="white", markeredgewidth=1.1,
                label="fidelity r(desired, hw_adc)")
        ax.set_ylim(0, 1.05)
        ax.axhline(0.9, color=GRID, lw=1.2)
        ax2 = ax.twinx()
        ax2.plot(eps, mags, lw=1.7, color="#104281", alpha=0.75,
                 label="mean |desired gradient|")
        # Below this the gradient is under the device noise floor and the
        # correlation collapses -- measured, not assumed (see README).
        ax2.axhline(0.03, color="#104281", lw=1.0, ls=":", alpha=0.8)
        ax2.set_yscale("log")
        ax2.set_ylabel("mean |desired gradient|", color="#104281")
        ax2.tick_params(axis="y", labelcolor="#104281")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2 + [], fontsize=8, frameon=False,
                  loc="lower left")
        ax.set_title("Analog gradient fidelity — array stores faithfully;\n"
                     "learning stalls when the gradient itself shrinks",
                     fontsize=11)
    else:
        ax.annotate("no gradient log", xy=(0.5, 0.5),
                    xycoords="axes fraction", ha="center", color="#898781")
    ax.set_xlabel("epoch")
    ax.set_ylabel("gradient fidelity  r")
    ax.grid(color=GRID, lw=0.8)

    fig.suptitle("Teacher-student after the threshold fix — the hardware model "
                 "now shares the teacher's threshold", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(args.out, dpi=110)
    print("saved ->", args.out)

    print(f"\n{'condition':>38} {'best loss':>10} {'final metric':>13}")
    for cond in ("bptt", "digital", "frozen_wout"):
        if cond in sw:
            print(f"{LABEL[cond]:>38} {sw[cond]['best_loss']:10.3f} "
                  f"{sw[cond]['vrds'][-1]:13.3f}")
    if a_loss:
        print(f"{LABEL['analog']:>38} {min(a_loss):10.3f} {a_metric[-1]:13.3f}")
    if eps:
        print(f"\nanalog fidelity r {rs[0]:.3f} (ep {eps[0]}) -> {rs[-1]:.3f} "
              f"(ep {eps[-1]});  mean|desired| {mags[0]:.4f} -> {mags[-1]:.4f}")


if __name__ == "__main__":
    main()
