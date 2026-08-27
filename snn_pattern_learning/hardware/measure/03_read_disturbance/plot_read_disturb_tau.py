#!/usr/bin/env python3
"""Read disturbance: is it V_0*exp(-k/tau), and what is tau's spread?"""
import csv
from collections import defaultdict

import numpy as np
from scipy.optimize import least_squares
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, GRID, MUTED = "#0b0b0b", "#e1e0d9", "#898781"
A_COL, B_COL, MEAS = "#eb6834", "#2e9e6b", "#2a78d6"
SRC = "2026-08-06_23-07_disturb_all25.csv"


def load():
    d = defaultdict(dict)
    for r in csv.DictReader(open(SRC, encoding="utf-8")):
        d[(int(r["rep"]), int(r["row"]), int(r["col"]))][int(r["reads"])] = \
            float(r["level"])
    return {k: (np.array(sorted(s), float),
                np.array([s[int(i)] for i in sorted(s)]))
            for k, s in d.items()}


def mA(p, k):
    return p[0] * np.exp(-k / p[1])


def mB(p, k):
    return p[0] + (p[1] - p[0]) * np.exp(-k / p[2])


def main():
    data = load()
    keys = sorted(data)

    pA, pB = {}, {}
    for t in keys:
        k, v = data[t]
        pA[t] = least_squares(lambda p: mA(p, k) - v, [v[0], 50.0],
                              max_nfev=20000).x
        pB[t] = least_squares(lambda p: mB(p, k) - v, [v[-1], v[0], 50.0],
                              max_nfev=20000).x

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4))

    # ---- 1. the two forms on real traces --------------------------------
    ax = axes[0]
    show = [t for t in keys if t[0] == 0][:6]
    kk = np.linspace(1, 1000, 400)
    for i, t in enumerate(show):
        k, v = data[t]
        ax.plot(k, v, "o", ms=3.5, color=MEAS, alpha=0.75,
                label="measured" if i == 0 else None)
        ax.plot(kk, mA(pA[t], kk), color=A_COL, lw=1.2, alpha=0.7,
                label="A: $V_0e^{-k/\\tau}$" if i == 0 else None)
        ax.plot(kk, mB(pB[t], kk), color=B_COL, lw=1.2, ls="--", alpha=0.85,
                label="B: $V_\\infty+(V_0-V_\\infty)e^{-k/\\tau}$"
                if i == 0 else None)
    ax.set_xscale("log")
    ax.set_xlabel("read index k")
    ax.set_ylabel("level (LSB)")
    ax.set_title("1. the proposed form is forced through 0\n"
                 "and misses the floor the cell settles at", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(color=GRID, lw=0.8, which="both")

    # ---- 2. the floor is directly observed ------------------------------
    ax = axes[1]
    vinf = np.array([pB[t][0] for t in keys])
    v0 = np.array([pB[t][1] for t in keys])
    last = np.array([data[t][1][-1] for t in keys])
    ax.hist(vinf, bins=16, color="#7fa8d9", edgecolor="white",
            label="fitted $V_\\infty$")
    ax.axvline(vinf.mean(), color=INK, lw=2,
               label=f"mean {vinf.mean():.0f} LSB")
    ax.axvline(0, color=A_COL, lw=2, ls="--",
               label="form A assumes 0")
    ax.set_xlabel("fitted floor $V_\\infty$ (LSB)")
    ax.set_ylabel("traces")
    ax.set_title(f"2. measured level at k=1000 is {last.mean():.0f} LSB;\n"
                 f"a pure exponential predicts "
                 f"{v0.mean()*np.exp(-1000/126.9):.1f} LSB", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(color=GRID, lw=0.8, axis="y")

    # ---- 3. tau spread, c2c vs d2d --------------------------------------
    ax = axes[2]
    cells = sorted({t[1:] for t in keys})
    reps = sorted({t[0] for t in keys})
    per = {c: [pB[(r,) + c][2] for r in reps if (r,) + c in pB] for c in cells}
    cm = np.array([np.mean(per[c]) for c in cells])
    within = [np.std(per[c], ddof=1) for c in cells if len(per[c]) > 1]
    c2c = float(np.sqrt(np.mean(np.square(within))))
    x = np.arange(len(cells))
    for i, c in enumerate(cells):
        ax.plot([i] * len(per[c]), per[c], "o", ms=4, color=MEAS, alpha=0.55)
    ax.plot(x, cm, "-", color=INK, lw=1.4, label="per-cell mean")
    ax.axhline(cm.mean(), color=B_COL, lw=2,
               label=f"grand mean {cm.mean():.1f} reads")
    ax.fill_between(x, cm.mean() - cm.std(ddof=1), cm.mean() + cm.std(ddof=1),
                    color=B_COL, alpha=0.15,
                    label=f"d2d sd {cm.std(ddof=1):.1f}")
    ax.set_xlabel("cell index (row-major)")
    ax.set_ylabel(r"$\tau$ (reads)")
    ax.set_title(f"3. $\\tau$ = {cm.mean():.1f} reads\n"
                 f"d2d sd {cm.std(ddof=1):.1f} (cv "
                 f"{cm.std(ddof=1)/cm.mean():.3f}), c2c sd {c2c:.1f} "
                 f"(cv {c2c/cm.mean():.3f})", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(color=GRID, lw=0.8)

    fig.suptitle("Read disturbance: testing $V_k=V_0e^{-k/\\tau}$ and "
                 "measuring $\\tau$", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig("read_disturb_tau.png", dpi=110)
    print("saved -> read_disturb_tau.png")
    print(f"tau {cm.mean():.2f} reads, d2d sd {cm.std(ddof=1):.2f}, "
          f"c2c sd {c2c:.2f}")


if __name__ == "__main__":
    main()
