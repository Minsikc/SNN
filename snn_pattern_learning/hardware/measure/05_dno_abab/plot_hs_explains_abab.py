#!/usr/bin/env python3
"""Figure for verify_hs_explains_abab.py: is NORMAL-DNO the pair drive?

    python plot_hs_explains_abab.py 2026-08-20_12-01_abab_dno_cells.csv
"""
import sys

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import verify_hs_explains_abab as V


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "2026-08-20_12-01_abab_dno_cells.csv"
    rows = V.load(path)
    trials = V.regenerate()

    deltas = {}
    for x in rows:
        deltas[(x["u"], x["v"], x["rep"], x["mode"],
                x["cell_row"] - 1, x["cell_col"] - 1)] = x["delta"]

    D, P = [], []
    for (u, v, rep), (a, b) in sorted(trials.items()):
        pp = V.pot_pairs(a, b)
        for i in range(V.N):
            for j in range(V.N):
                D.append(deltas[(u, v, rep, "normal", i, j)]
                         - deltas[(u, v, rep, "dno", i, j)])
                P.append(pp[i, j])
    D, P = np.array(D), np.array(P, float)
    res = stats.linregress(P, D)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))

    ax = axes[0]
    jit = np.random.default_rng(0).uniform(-0.18, 0.18, P.size)
    ax.plot(P + jit, D, ".", ms=3, alpha=0.25, color="tab:blue")
    ks = np.arange(int(P.max()) + 1)
    means = [D[P == k].mean() for k in ks]
    sems = [D[P == k].std() / np.sqrt((P == k).sum()) for k in ks]
    ax.errorbar(ks, means, yerr=sems, fmt="o-", color="tab:red", lw=2,
                capsize=3, label="mean +- sem per pair count")
    xx = np.array([P.min(), P.max()])
    ax.plot(xx, res.slope * xx + res.intercept, "--", color="k",
            label=f"fit {res.slope:+.2f} LSB/pair (04: +1.65)")
    ax.axhline(0, color="gray", lw=1)
    ax.set_xlabel("half-select pairs in the trial's streams (regenerated)")
    ax.set_ylabel(r"$\Delta_{NORMAL} - \Delta_{DNO}$ per cell (LSB)")
    ax.set_title("Identical streams: mode difference vs pair exposure")
    ax.legend()

    ax = axes[1]
    labels, meas, pred = [], [], []
    uu = sorted({k[0] for k in trials})
    for u in uu:
        for v in uu:
            dd, pp2 = [], []
            for rep in range(V.REPEATS):
                a, b = trials[(u, v, rep)]
                p = V.pot_pairs(a, b)
                for i in range(V.N):
                    for j in range(V.N):
                        dd.append(deltas[(u, v, rep, "normal", i, j)]
                                  - deltas[(u, v, rep, "dno", i, j)])
                        pp2.append(p[i, j])
            labels.append(f"{u:g},{v:g}")
            meas.append(np.mean(dd))
            pred.append(res.slope * np.mean(pp2))
    xpos = np.arange(len(labels))
    ax.bar(xpos - 0.2, meas, 0.4, label="measured mean difference",
           color="tab:blue")
    ax.bar(xpos + 0.2, pred, 0.4, label="pair model  s x <P>",
           color="tab:orange")
    ax.set_xticks(xpos, labels)
    ax.set_xlabel("(u, v) grid point")
    ax.set_ylabel("LSB")
    ax.set_title("Per grid point: measured vs pair-model prediction")
    ax.legend()

    fig.suptitle("ABAB NORMAL-DNO difference is quantitatively the "
                 "half-select pair drive", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = path.replace("_cells.csv", "_hs_explain.png")
    fig.savefig(out, dpi=110)
    print("saved ->", out)


if __name__ == "__main__":
    main()
