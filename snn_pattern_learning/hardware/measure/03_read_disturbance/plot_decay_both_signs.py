#!/usr/bin/env python3
"""The decay curves themselves: positive-start and negative-start.

plot_disturb_signed.py showed 5 traces per arm on one log axis, which was
enough to make the absolute-attractor point but is not a good look at the
decays. This draws them properly:

  1  all 25 cells, both arms, linear k -- the raw shapes side by side
  2  same on log k, so the early decay is visible too
  3  normalised to (V - Vinf)/(V0 - Vinf): if one exponential with a common
     tau governs both signs, every trace collapses onto exp(-k/tau)
  4  the 5x5 grid, both arms per cell

Panel 3 is the real test of "same mechanism in both directions".
"""
import csv
import glob
from collections import defaultdict

import numpy as np
from scipy.optimize import least_squares
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, GRID, MUTED = "#0b0b0b", "#e1e0d9", "#898781"
POT, DEP = "#2a78d6", "#eb6834"
N = 5


def load(path):
    d = defaultdict(dict)
    for r in csv.DictReader(open(path, encoding="utf-8")):
        d[(r["arm"], int(r["rep"]), int(r["row"]), int(r["col"]))][
            int(r["reads"])] = float(r["level"])
    return {k: (np.array(sorted(s), float),
                np.array([s[int(i)] for i in sorted(s)]))
            for k, s in d.items()}


def mB(p, k):
    return p[0] + (p[1] - p[0]) * np.exp(-k / p[2])


def fit(k, v):
    return least_squares(lambda p: mB(p, k) - v, [v[-1], v[0], 50.0],
                         max_nfev=20000).x


def main():
    path = sorted(glob.glob("*_disturb_signed.csv"))[-1]
    data = load(path)
    P = {t: fit(*data[t]) for t in data}

    # cell-averaged trace per arm (average over reps)
    cells = sorted({t[2:] for t in data})
    kgrid = data[next(iter(data))][0]
    avg = {}
    for arm in ("pot", "dep"):
        for c in cells:
            ts = [t for t in data if t[0] == arm and t[2:] == c]
            avg[(arm, c)] = np.mean([data[t][1] for t in ts], axis=0)

    fig = plt.figure(figsize=(16.5, 10.5))
    gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.26)

    # ---- 1. linear k ----------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    for arm, col in (("pot", POT), ("dep", DEP)):
        for i, c in enumerate(cells):
            ax.plot(kgrid, avg[(arm, c)], lw=0.9, color=col, alpha=0.45,
                    label=f"{'positive' if arm=='pot' else 'negative'} start"
                    if i == 0 else None)
    vinf = np.mean([P[t][0] for t in P])
    ax.axhline(vinf, color=INK, ls="--", lw=1.6,
               label=f"$V_\\infty$ {vinf:+.0f} LSB")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("read index k")
    ax.set_ylabel("level (LSB)")
    ax.set_title("1. all 25 cells, linear k\nboth signs relax to the same "
                 "level", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(color=GRID, lw=0.8)

    # ---- 2. log k -------------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    for arm, col in (("pot", POT), ("dep", DEP)):
        for c in cells:
            ax.plot(kgrid, avg[(arm, c)], lw=0.9, color=col, alpha=0.45)
    ax.axhline(vinf, color=INK, ls="--", lw=1.6)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("read index k")
    ax.set_ylabel("level (LSB)")
    ax.set_title("2. same, log k\nthe knee sits near k ~ 100 either way",
                 fontsize=10)
    ax.grid(color=GRID, lw=0.8, which="both")

    # ---- 3. normalised collapse ----------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    taus = {arm: np.mean([P[t][2] for t in P if t[0] == arm])
            for arm in ("pot", "dep")}
    for arm, col in (("pot", POT), ("dep", DEP)):
        for i, t in enumerate(sorted(t for t in data if t[0] == arm)):
            k, v = data[t]
            vi, v0 = P[t][0], P[t][1]
            if abs(v0 - vi) < 50:
                continue
            ax.plot(k, (v - vi) / (v0 - vi), lw=0.6, color=col, alpha=0.3,
                    label=f"{'positive' if arm=='pot' else 'negative'} "
                          f"($\\tau$={taus[arm]:.0f})" if i == 0 else None)
    kk = np.logspace(0, 3, 200)
    ax.plot(kk, np.exp(-kk / np.mean(list(taus.values()))), color=INK, lw=2,
            label=f"$e^{{-k/{np.mean(list(taus.values())):.0f}}}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-2, 1.4)
    ax.set_xlabel("read index k")
    ax.set_ylabel(r"$(V_k-V_\infty)/(V_0-V_\infty)$")
    ax.set_title("3. normalised: both signs collapse onto one\nexponential "
                 "-> same mechanism, one $\\tau$", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(color=GRID, lw=0.8, which="both")

    # ---- 4. per-cell grid ----------------------------------------------
    sub = gs[1, :].subgridspec(N, N, hspace=0.55, wspace=0.32)
    lo = min(avg[k].min() for k in avg)
    hi = max(avg[k].max() for k in avg)
    for r in range(N):
        for c in range(N):
            ax = fig.add_subplot(sub[r, c])
            cell = (r, c + 1)
            for arm, col in (("pot", POT), ("dep", DEP)):
                ax.plot(kgrid, avg[(arm, cell)], lw=1.1, color=col)
            ax.axhline(0, color=MUTED, lw=0.6)
            ax.set_xscale("log")
            ax.set_ylim(lo - 20, hi + 20)
            ax.set_title(f"({r+1},{c+1})", fontsize=7.5, pad=2)
            ax.tick_params(labelsize=6)
            if c:
                ax.set_yticklabels([])
            if r < N - 1:
                ax.set_xticklabels([])
            ax.grid(color=GRID, lw=0.5, which="both")

    fig.suptitle("Read-disturbance decay from positive and negative starts "
                 "(25 cells, 3 reps averaged)\nbottom: per-cell, both arms",
                 fontsize=13)
    fig.savefig("decay_both_signs.png", dpi=110, bbox_inches="tight")
    print("saved -> decay_both_signs.png")
    for arm in ("pot", "dep"):
        v = [P[t] for t in P if t[0] == arm]
        print(f"  {arm}: V0 {np.mean([x[1] for x in v]):+7.1f} -> "
              f"Vinf {np.mean([x[0] for x in v]):+6.1f} LSB, "
              f"tau {np.mean([x[2] for x in v]):.1f} reads")


if __name__ == "__main__":
    main()
