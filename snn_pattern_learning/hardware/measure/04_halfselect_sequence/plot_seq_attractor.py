#!/usr/bin/env python3
"""Plot where each alternating half-select pair drives the cell.

One panel per pair: ADC level versus cycle count, one line per starting
state. Convergence of the five curves onto a common level is the signature
of an attractor; parallel curves would mean a fixed per-cycle increment
instead.

The x axis is linear, so the shape of the approach to the attractor is read
directly. The sampled cycle counts are log-spaced, so points crowd near the
origin; the pre-pulse level is drawn at x=0.
"""
import csv
import glob
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# one sequential ramp, dark = charged start, light = depressed start, so the
# ordering of the legend matches the physical ordering of the start levels
RAMP = {
    "full_pot": "#104281",
    "mid_pot": "#2a78d6",
    "reset": "#898781",
    "mid_dep": "#eb6834",
    "full_dep": "#8a2f13",
}
ORDER = ["full_pot", "mid_pot", "reset", "mid_dep", "full_dep"]
INK, GRID = "#0b0b0b", "#e1e0d9"


ALL_PAIRS = [f"{a}+{b}" for a in ["N1", "N2", "N3", "N4"]
             for b in ["N1", "N2", "N3", "N4"]]


def main():
    # The 16 pairs were measured in two runs (the four select pairs first,
    # then the remaining twelve), so merge every matching file rather than
    # taking only the newest one.
    paths = sys.argv[1:] or sorted(glob.glob("*_seq_attractor.csv"))
    rows = []
    for p in paths:
        rows += list(csv.DictReader(open(p, encoding="utf-8")))
    print(f"loaded {len(rows)} rows from {len(paths)} file(s)")
    present = {r["combo"] for r in rows}
    combos = [c for c in ALL_PAIRS if c in present]
    cycles = sorted({int(r["cycles"]) for r in rows})
    starts = [s for s in ORDER if s in {r["start"] for r in rows}]

    # 4x4 grid mirroring the pair matrix: row = first pulse, col = second
    ncol = 4 if len(combos) > 4 else min(len(combos), 2)
    nrow = int(np.ceil(len(combos) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 4.0 * nrow),
                             squeeze=False, sharex=True, sharey=True)

    for k, combo in enumerate(combos):
        ax = axes[k // ncol][k % ncol]
        sub = [r for r in rows if r["combo"] == combo]
        finals, start_levels = [], []
        for st in starts:
            b0 = np.mean([float(r["before"]) for r in sub
                          if r["start"] == st])
            xs, ys, es = [], [], []
            for c in cycles:
                v = [float(r["after"]) for r in sub
                     if r["start"] == st and int(r["cycles"]) == c]
                if v:
                    xs.append(c)
                    ys.append(np.mean(v))
                    es.append(np.std(v))
            if not xs:
                continue
            finals.append(ys[-1])
            start_levels.append(b0)
            col = RAMP.get(st, INK)
            # pre-pulse level sits at x=0 (no cycles applied yet)
            ax.plot([0] + xs, [b0] + ys, "o-", ms=3.5, lw=1.6, color=col,
                    markerfacecolor="white", markeredgewidth=1.2,
                    label=f"{st}  (start {b0:+.0f})")
            ax.errorbar(xs, ys, yerr=es, fmt="none", ecolor=col,
                        elinewidth=0.9, capsize=2, alpha=0.6)

        # A pair only converges if the five endpoints agree. When the
        # endpoint spread stays near the ~900 LSB span of the starting
        # states, nothing happened -- labelling that with a "convergence
        # level" (the mean of five unchanged starts) would be meaningless.
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

        ax.set_xlabel("alternating cycles")
        ax.set_xlim(left=-8)
        ax.set_ylabel("ADC level, N5$-$N6 (LSB)")
        ax.set_title(combo, fontsize=13)
        ax.grid(color=GRID, lw=0.8)
        ax.axhline(0, color=GRID, lw=1.2)
        if k == 0:
            ax.legend(fontsize=7, frameon=False, loc="best")

    for k in range(len(combos), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")

    fig.suptitle("Alternating half-select pairs act as attractors — level "
                 "reached vs cycle count\n"
                 "five starting states, 5x5 array mean",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = "seq_attractor_map.png"
    fig.savefig(out, dpi=110)
    print("saved ->", out)

    print(f"\n{'pair':>8} {'endpoint spread':>16}   verdict")
    for combo in combos:
        sub = [r for r in rows if r["combo"] == combo]
        f, b = [], []
        for st in starts:
            v = [float(r["after"]) for r in sub if r["start"] == st
                 and int(r["cycles"]) == cycles[-1]]
            s0 = [float(r["before"]) for r in sub if r["start"] == st]
            if v:
                f.append(np.mean(v))
                b.append(np.mean(s0))
        if not f:
            continue
        spread = max(f) - min(f)
        span = max(b) - min(b)
        verdict = (f"converges to {np.mean(f):+.0f} LSB"
                   if spread < 0.35 * span
                   else "no effect (every start unchanged)")
        print(f"{combo:>8} {spread:>16.1f}   {verdict}")


if __name__ == "__main__":
    main()
