#!/usr/bin/env python3
"""Plot the four ordered OVERLAP half-select pairs, both orderings paired up.

Layout is 2x2: each ROW is one line pair, each COLUMN one rise order, so the
two orderings of the same pair sit side by side and any asymmetry is read off
directly. That is the whole question -- the alternating measurement averaged
the two orderings together and returned zero.

Conventions (colours, x=0 pre-pulse point, convergence band) match
plot_seq_attractor.py so the two figures can be compared panel to panel.
"""
import csv
import glob
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAMP = {
    "full_pot": "#104281",
    "mid_pot": "#2a78d6",
    "reset": "#898781",
    "mid_dep": "#eb6834",
    "full_dep": "#8a2f13",
}
ORDER = ["full_pot", "mid_pot", "reset", "mid_dep", "full_dep"]
INK, GRID = "#0b0b0b", "#e1e0d9"

# rows = line pair, cols = rise order. Left column rises on the line the
# firmware's original primitive raised first.
GRID_PAIRS = [["N1+N4", "N4+N1"],
              ["N3+N2", "N2+N3"]]


def curve(sub, st, cycles):
    xs, ys, es = [], [], []
    for c in cycles:
        v = [float(r["after"]) for r in sub
             if r["start"] == st and int(r["cycles"]) == c]
        if v:
            xs.append(c)
            ys.append(np.mean(v))
            es.append(np.std(v))
    return xs, ys, es


def main():
    paths = sys.argv[1:] or sorted(glob.glob("*_overlap_seq_attractor.csv"))
    if not paths:
        raise SystemExit("no *_overlap_seq_attractor.csv found -- run "
                         "overlap_seq_attractor.py first")
    rows = []
    for p in paths:
        rows += list(csv.DictReader(open(p, encoding="utf-8")))
    print(f"loaded {len(rows)} rows from {len(paths)} file(s)")

    present = {r["combo"] for r in rows}
    cycles = sorted({int(r["cycles"]) for r in rows})
    starts = [s for s in ORDER if s in {r["start"] for r in rows}]

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.4), squeeze=False,
                             sharex=True, sharey=True)

    for i, row_pairs in enumerate(GRID_PAIRS):
        for j, combo in enumerate(row_pairs):
            ax = axes[i][j]
            if combo not in present:
                ax.annotate(f"{combo}\nnot measured", xy=(0.5, 0.5),
                            xycoords="axes fraction", ha="center",
                            fontsize=11, color="#898781")
                ax.set_title(combo, fontsize=13)
                ax.grid(color=GRID, lw=0.8)
                continue
            sub = [r for r in rows if r["combo"] == combo]
            finals, start_levels = [], []
            for st in starts:
                b0 = np.mean([float(r["before"]) for r in sub
                              if r["start"] == st])
                xs, ys, es = curve(sub, st, cycles)
                if not xs:
                    continue
                finals.append(ys[-1])
                start_levels.append(b0)
                col = RAMP.get(st, INK)
                ax.plot([0] + xs, [b0] + ys, "o-", ms=3.5, lw=1.6, color=col,
                        markerfacecolor="white", markeredgewidth=1.2,
                        label=f"{st}  (start {b0:+.0f})")
                ax.errorbar(xs, ys, yerr=es, fmt="none", ecolor=col,
                            elinewidth=0.9, capsize=2, alpha=0.6)

            if finals:
                lo, hi = min(finals), max(finals)
                span = max(start_levels) - min(start_levels)
                if (hi - lo) < 0.35 * span:
                    ax.axhspan(lo, hi, color=INK, alpha=0.06, zorder=0)
                    ax.axhline(np.mean(finals), color=INK, lw=1.2, ls="--",
                               zorder=1)
                    ax.annotate(f"$\\rightarrow$ {np.mean(finals):+.0f} LSB",
                                xy=(cycles[-1], np.mean(finals)),
                                xytext=(-6, 12), textcoords="offset points",
                                ha="right", fontsize=9, color=INK)
                else:
                    ax.annotate("no effect", xy=(0.5, 0.52),
                                xycoords="axes fraction", ha="center",
                                fontsize=12, color="#898781")

            ax.set_xlabel("overlap repetitions")
            ax.set_xlim(left=-8)
            ax.set_ylabel("ADC level, N5$-$N6 (LSB)")
            ax.set_title(f"{combo}   ({combo.split('+')[0]} rises first)",
                         fontsize=12)
            ax.grid(color=GRID, lw=0.8)
            ax.axhline(0, color=GRID, lw=1.2)
            if i == 0 and j == 0:
                ax.legend(fontsize=7, frameon=False, loc="best")

    fig.suptitle("Ordered overlap half-select pairs — level reached vs "
                 "repetition count\n"
                 "rows = line pair, columns = rise order; five starting "
                 "states, 5x5 array mean",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = "overlap_seq_attractor.png"
    fig.savefig(out, dpi=110)
    print("saved ->", out)

    # Numeric counterpart of the figure: does flipping the order flip the sign?
    cmax = cycles[-1]
    ctrl = [float(r["delta"]) for r in rows
            if r["combo"] == "control" and int(r["cycles"]) == cmax]
    c0 = np.mean(ctrl) if ctrl else 0.0
    print(f"\nat {cmax} repetitions, control {c0:+.2f} LSB subtracted")
    print(f"{'pair':>8} {'endpoint spread':>16} {'net delta':>11}   verdict")
    for row_pairs in GRID_PAIRS:
        for combo in row_pairs:
            sub = [r for r in rows if r["combo"] == combo]
            if not sub:
                continue
            f, b = [], []
            for st in starts:
                v = [float(r["after"]) for r in sub if r["start"] == st
                     and int(r["cycles"]) == cmax]
                s0 = [float(r["before"]) for r in sub if r["start"] == st]
                if v:
                    f.append(np.mean(v))
                    b.append(np.mean(s0))
            if not f:
                continue
            d = [float(r["delta"]) for r in sub if int(r["cycles"]) == cmax]
            spread, span = max(f) - min(f), max(b) - min(b)
            verdict = (f"converges to {np.mean(f):+.0f} LSB"
                       if spread < 0.35 * span
                       else "no effect (every start unchanged)")
            print(f"{combo:>8} {spread:>16.1f} {np.mean(d)-c0:>11.2f}   "
                  f"{verdict}")

    for a, b in (("N1+N4", "N4+N1"), ("N3+N2", "N2+N3")):
        va = [float(r["delta"]) for r in rows
              if r["combo"] == a and int(r["cycles"]) == cmax]
        vb = [float(r["delta"]) for r in rows
              if r["combo"] == b and int(r["cycles"]) == cmax]
        if not va or not vb:
            continue
        ma, mb = np.mean(va) - c0, np.mean(vb) - c0
        tag = "OPPOSITE SIGN -> alternating would cancel" if ma * mb < 0 \
            else "same sign"
        print(f"  {a} {ma:+8.2f}  vs  {b} {mb:+8.2f}   {tag}")


if __name__ == "__main__":
    main()
