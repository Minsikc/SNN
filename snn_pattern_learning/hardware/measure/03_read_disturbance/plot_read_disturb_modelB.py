#!/usr/bin/env python3
"""Model B parameters and their c2c / d2d spread."""
import csv
from collections import defaultdict

import numpy as np
from scipy.optimize import least_squares
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, GRID, MUTED = "#0b0b0b", "#e1e0d9", "#898781"
MEAS, TAU, VINF = "#2a78d6", "#2e9e6b", "#eb6834"
SRC = "2026-08-06_23-07_disturb_all25.csv"


def main():
    d = defaultdict(dict)
    for r in csv.DictReader(open(SRC, encoding="utf-8")):
        d[(int(r["rep"]), int(r["row"]), int(r["col"]))][int(r["reads"])] = \
            float(r["level"])
    data = {k: (np.array(sorted(s), float),
                np.array([s[int(i)] for i in sorted(s)]))
            for k, s in d.items()}
    mB = lambda p, k: p[0] + (p[1] - p[0]) * np.exp(-k / p[2])
    P = {t: least_squares(lambda q: mB(q, *[data[t][0]]) - data[t][1],
                          [data[t][1][-1], data[t][1][0], 50.0],
                          max_nfev=20000).x for t in data}
    keys = sorted(P)
    cells = sorted({k[1:] for k in keys})
    reps = sorted({k[0] for k in keys})

    def per(idx):
        return {c: [P[(r,) + c][idx] for r in reps if (r,) + c in P]
                for c in cells}

    def spread(pc):
        cm = np.array([np.mean(pc[c]) for c in cells])
        wi = [np.std(pc[c], ddof=1) for c in cells if len(pc[c]) > 1]
        c2c = float(np.sqrt(np.mean(np.square(wi))))
        n = np.mean([len(pc[c]) for c in cells])
        d2d = float(np.sqrt(max(cm.std(ddof=1) ** 2 - c2c ** 2 / n, 0)))
        return cm, d2d, c2c

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4))

    # ---- 1. fits over the traces ---------------------------------------
    ax = axes[0]
    kk = np.logspace(0, 3, 300)
    for i, t in enumerate([t for t in keys if t[0] == 0][:8]):
        k, v = data[t]
        ax.plot(k, v, "o", ms=3.2, color=MEAS, alpha=0.6,
                label="measured" if i == 0 else None)
        ax.plot(kk, mB(P[t], kk), color=TAU, lw=1.3, alpha=0.85,
                label="model B" if i == 0 else None)
    vi, _, _ = spread(per(0))
    ax.axhline(vi.mean(), color=VINF, ls="--", lw=1.8,
               label=f"$V_\\infty$ mean {vi.mean():.0f} LSB")
    ax.set_xscale("log")
    ax.set_xlabel("read index k")
    ax.set_ylabel("level (LSB)")
    ax.set_title("1. $V_k=V_\\infty+(V_0-V_\\infty)e^{-k/\\tau}$\n"
                 "rms 4.5 LSB over 75 traces", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(color=GRID, lw=0.8, which="both")

    # ---- 2 & 3. per-parameter spread ------------------------------------
    for ax, idx, nm, col, unit in ((axes[1], 2, r"$\tau$", TAU, " reads"),
                                   (axes[2], 0, r"$V_\infty$", VINF, " LSB")):
        pc = per(idx)
        cm, d2d, c2c = spread(pc)
        x = np.arange(len(cells))
        for i, c in enumerate(cells):
            ax.plot([i] * len(pc[c]), pc[c], "o", ms=4, color=MEAS,
                    alpha=0.5)
        ax.plot(x, cm, "-", color=INK, lw=1.4, label="per-cell mean")
        ax.axhline(cm.mean(), color=col, lw=2,
                   label=f"mean {cm.mean():.1f}{unit}")
        ax.fill_between(x, cm.mean() - d2d, cm.mean() + d2d, color=col,
                        alpha=0.16,
                        label=f"d2d {d2d:.1f} (cv {d2d/abs(cm.mean()):.3f})")
        ax.errorbar([len(cells) + 1.2], [cm.mean()], yerr=[c2c], fmt="s",
                    ms=6, color=INK, capsize=5,
                    label=f"c2c {c2c:.1f} (cv {c2c/abs(cm.mean()):.3f})")
        ax.set_xlabel("cell index (row-major)")
        ax.set_ylabel(f"{nm}{unit}")
        ax.set_title(f"{2 if idx==2 else 3}. {nm}: d2d "
                     f"{'>' if d2d>c2c else '<'} c2c "
                     f"({d2d/c2c:.1f}x)", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(color=GRID, lw=0.8)

    fig.suptitle("Read disturbance, model B: parameters and their "
                 "cell-to-cell / cycle-to-cycle spread", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig("read_disturb_modelB.png", dpi=110)
    print("saved -> read_disturb_modelB.png")


if __name__ == "__main__":
    main()
