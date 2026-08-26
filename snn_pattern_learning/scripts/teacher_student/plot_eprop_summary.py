#!/usr/bin/env python3
"""Summary of the teacher-student e-prop campaign: seeds and scale.

The individual runs each produced a raster, but the seed comparison and the
scale sweep only ever existed as numbers in JSON and run logs.  This collects
them.

Analog seed losses come from the hardware run log rather than the JSON,
because seed_sweep_sw.json holds only the software conditions -- the analog
runs drive the array and were logged separately.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, GRID, MUTED = "#0b0b0b", "#e1e0d9", "#898781"
DIG, ANA, FRZ, BPTT = "#2a78d6", "#eb6834", "#b8b6ae", "#2e9e6b"
HERE = os.path.join("results", "eprop_grad_log")

# analog runs, from the hardware run log (50 epochs, lr 0.1, seed-matched init)
ANALOG = {0: 0.0771, 1: 0.4941, 2: 0.6483}


def main():
    seeds = json.load(open(os.path.join(HERE, "seed_sweep_sw.json"),
                           encoding="utf-8"))
    scale = json.load(open(os.path.join(HERE, "scale_sw.json"),
                           encoding="utf-8"))
    analog_curve = json.load(open(os.path.join(HERE, "analog_curve.json"),
                                 encoding="utf-8"))
    three = json.load(open(os.path.join(HERE, "three_conditions.json"),
                          encoding="utf-8"))

    fig = plt.figure(figsize=(16.5, 9.4))
    gs = fig.add_gridspec(2, 3, hspace=0.34, wspace=0.26)

    # ---- 1. learning curves across seeds --------------------------------
    ax = fig.add_subplot(gs[0, 0])
    for s in sorted(seeds):
        ax.plot(np.arange(1, 51), seeds[s]["digital"]["losses"], color=DIG,
                lw=1.2, alpha=0.75, label="digital" if s == "0" else None)
        ax.plot(np.arange(1, 51), seeds[s]["frozen_wout"]["losses"],
                color=FRZ, lw=1.2, alpha=0.75,
                label="$W_{out}$ frozen" if s == "0" else None)
    ax.plot(np.arange(1, 51), analog_curve["losses"], color=ANA, lw=1.8,
            label="analog (seed 0)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("1. learning curves, 3 seeds\n"
                 "analog tracks digital; frozen control does not",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(color=GRID, lw=0.8)

    # ---- 2. best loss per seed ------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    ss = sorted(int(s) for s in seeds)
    x = np.arange(len(ss))
    w = 0.26
    dig = [seeds[str(s)]["digital"]["best_loss"] for s in ss]
    ana = [ANALOG[s] for s in ss]
    frz = [seeds[str(s)]["frozen_wout"]["best_loss"] for s in ss]
    ax.bar(x - w, dig, w, color=DIG, label="digital")
    ax.bar(x, ana, w, color=ANA, label="analog")
    ax.bar(x + w, frz, w, color=FRZ, label="$W_{out}$ frozen")
    for xi, vals in zip(x, zip(dig, ana, frz)):
        for off, v in zip((-w, 0, w), vals):
            ax.text(xi + off, v + 0.03, f"{v:.2f}", ha="center", fontsize=7.5)
    ax.set_xticks(x, [f"seed {s}" for s in ss])
    ax.set_ylabel("best loss")
    d, a = np.array(dig), np.array(ana)
    ax.set_title(f"2. best loss by seed\n"
                 f"digital {d.mean():.2f}$\\pm${d.std():.2f}, "
                 f"analog {a.mean():.2f}$\\pm${a.std():.2f}", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(color=GRID, lw=0.8, axis="y")

    # ---- 3. digital vs analog, paired -----------------------------------
    ax = fig.add_subplot(gs[0, 2])
    ax.plot([0, max(d.max(), a.max()) * 1.15],
            [0, max(d.max(), a.max()) * 1.15], color=INK, lw=1.2, ls="--",
            label="equal")
    ax.scatter(d, a, s=90, color=ANA, zorder=3, edgecolors="white")
    for s, xi, yi in zip(ss, d, a):
        ax.annotate(f"seed {s}", (xi, yi), textcoords="offset points",
                    xytext=(8, -3), fontsize=8)
    ax.set_xlabel("digital best loss")
    ax.set_ylabel("analog best loss")
    ax.set_title("3. paired comparison\n"
                 "points straddle the diagonal: analog is not\n"
                 "systematically worse (paired $p=0.51$)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(color=GRID, lw=0.8)

    # ---- 4. scale sweep --------------------------------------------------
    ax = fig.add_subplot(gs[1, :2])
    keys = ["1_20", "1_40", "1_80", "2_20", "5_20"]
    labs = [f"n_seq={k.split('_')[0]}\nT={k.split('_')[1]}" for k in keys]
    x = np.arange(len(keys))
    w = 0.26
    for off, cond, col, lab in ((-w, "bptt", BPTT, "BPTT"),
                                (0, "digital", DIG, "e-prop digital"),
                                (w, "frozen", FRZ, "$W_{out}$ frozen")):
        v = [scale[k][cond] for k in keys]
        ax.bar(x + off, v, w, color=col, label=lab)
        for xi, vi in zip(x, v):
            ax.text(xi + off, vi + 0.15, f"{vi:.2f}", ha="center",
                    fontsize=7.5)
    ax.set_xticks(x, labs, fontsize=9)
    ax.set_ylabel("best loss")
    ax.set_title("4. scale sweep — longer sequences and more of them\n"
                 "BPTT degrades alongside e-prop, so the limit is model "
                 "capacity (hidden=5), not the analog memory", fontsize=10)
    ax.legend(fontsize=8.5)
    ax.grid(color=GRID, lw=0.8, axis="y")

    # ---- 5. three conditions + analog, best loss -------------------------
    ax = fig.add_subplot(gs[1, 2])
    names = ["BPTT", "digital", "analog", "frozen"]
    vals = [min(three["bptt"]["losses"]), min(three["digital"]["losses"]),
            min(analog_curve["losses"]),
            min(three["frozen_wout"]["losses"])]
    cols = [BPTT, DIG, ANA, FRZ]
    ax.barh(range(4), vals, color=cols, height=0.6)
    for i, v in enumerate(vals):
        ax.text(v + 0.04, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_yticks(range(4), names)
    ax.invert_yaxis()
    ax.set_xlabel("best loss")
    ax.set_xlim(0, max(vals) * 1.25)
    ax.set_title("5. seed 0, all four conditions\n"
                 f"analog is {vals[2]/vals[1]:.2f}x digital; freezing "
                 f"$W_{{out}}$ costs {vals[3]/vals[1]:.1f}x", fontsize=10)
    ax.grid(color=GRID, lw=0.8, axis="x")

    fig.suptitle("Teacher-student e-prop demo: seed robustness and scaling",
                 fontsize=13)
    fig.savefig("eprop_seed_scale_summary.png", dpi=110,
                bbox_inches="tight")
    print("saved -> eprop_seed_scale_summary.png")
    print(f"  digital {d.mean():.3f} +- {d.std():.3f}")
    print(f"  analog  {a.mean():.3f} +- {a.std():.3f}")
    try:
        from scipy import stats
        t = stats.ttest_rel(d, a)
        print(f"  paired t-test p = {t.pvalue:.3f}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
