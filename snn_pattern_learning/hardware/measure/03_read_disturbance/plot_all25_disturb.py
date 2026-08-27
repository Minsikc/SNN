#!/usr/bin/env python3
"""Per-cell read-disturbance decay, all 25 cells.

Layout mirrors the array: a 5x5 grid of small multiples, one cell per panel,
so a bad device is located by position rather than hunted in a legend. Each
panel carries its own single-exponential fit; the fitted rate is what the
grid heatmap at the end summarises.

Points to read the figure by:
  * every cell is fitted individually -- averaging cells with differing rates
    fabricates a double exponential (a simulated 0.4% rate spread fits 2exp
    11000x better than 1exp despite every constituent being a pure 1exp)
  * the fitted asymptote is NOT stored charge: per-column values ran from
    -41 to +49 LSB, and a negative one is impossible. It is a read-path
    offset, so curves are drawn raw and the offset is only a fit nuisance.
"""
import csv
import glob
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
MARK, FITC = "#3987e5", "#1c5cab"


def one_exp(n, vinf, v0, f):
    return vinf + (v0 - vinf) * f ** n


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else sorted(
        glob.glob("*_disturb_all25.csv"))[-1]
    rows = list(csv.DictReader(open(path, encoding="utf-8")))

    data = {}
    per_rep = {}
    for r in rows:
        key = (int(r["row"]), int(r["col"]))
        data.setdefault(key, {}).setdefault(int(r["reads"]), []).append(
            float(r["level"]))
        per_rep.setdefault((key, int(r["rep"])), {})[int(r["reads"])] = \
            float(r["level"])
    n_reps = len({int(r["rep"]) for r in rows})
    n_max = max(int(r["reads"]) for r in rows)

    rate = np.full((5, 5), np.nan)
    floor = np.full((5, 5), np.nan)
    v0g = np.full((5, 5), np.nan)

    fig, axes = plt.subplots(5, 5, figsize=(15, 13), sharex=True)
    for i in range(5):
        for j in range(5):
            ax = axes[i][j]
            d = data.get((i, j + 1))
            if not d:
                ax.axis("off")
                continue
            n = np.array(sorted(d), float)
            y = np.array([np.mean(d[k]) for k in sorted(d)])
            ax.plot(n, y, "o", ms=3, color=MARK, markerfacecolor="white",
                    markeredgewidth=1.1, zorder=4)
            try:
                p, _ = curve_fit(one_exp, n, y, p0=[0, y[0], 0.99],
                                 maxfev=200000)
                nn = np.linspace(n.min(), n.max(), 200)
                ax.plot(nn, one_exp(nn, *p), lw=1.8, color=FITC, zorder=3)
                rate[i, j] = (1 - p[2]) * 100
                floor[i, j] = p[0]
                v0g[i, j] = y[0]
                ax.set_title(f"({i+1},{j+1})  {rate[i,j]:.2f}%/read",
                             fontsize=9)
            except Exception:
                ax.set_title(f"({i+1},{j+1})  fit failed", fontsize=9)
            ax.axhline(0, color=MUTED, lw=0.7, ls=":")
            ax.grid(color=GRID, lw=0.7)
            ax.tick_params(labelsize=7)
            if j == 0:
                ax.set_ylabel("LSB", fontsize=8)
            if i == 4:
                ax.set_xlabel("reads of this row", fontsize=8)

    fig.suptitle("Read-disturbance decay, all 25 cells — each row exposed "
                 "separately via READ_ROW\n"
                 "one single-exponential fit per cell (panel title = fitted "
                 "rate)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    out1 = "all25_disturb_grid.png"
    fig.savefig(out1, dpi=105)
    print("saved ->", out1)

    # summary figure: rate map + spread + starting level
    fig2, ax2 = plt.subplots(1, 3, figsize=(15, 4.4))
    im = ax2[0].imshow(rate, cmap="magma_r")
    ax2[0].set_title(f"decay rate (%/read)\nmean "
                     f"{np.nanmean(rate):.3f}, spread "
                     f"{np.nanstd(rate):.3f}", fontsize=11)
    for i in range(5):
        for j in range(5):
            if np.isfinite(rate[i, j]):
                ax2[0].text(j, i, f"{rate[i,j]:.2f}", ha="center",
                            va="center", fontsize=8,
                            color="white" if rate[i, j] >
                            np.nanmean(rate) else "black")
    fig2.colorbar(im, ax=ax2[0], fraction=0.046)

    im = ax2[1].imshow(floor, cmap="coolwarm",
                       vmin=-np.nanmax(np.abs(floor)),
                       vmax=np.nanmax(np.abs(floor)))
    ax2[1].set_title("fitted asymptote (LSB)\nnegative values prove this is "
                     "read offset,\nnot stored charge", fontsize=10)
    for i in range(5):
        for j in range(5):
            if np.isfinite(floor[i, j]):
                ax2[1].text(j, i, f"{floor[i,j]:.0f}", ha="center",
                            va="center", fontsize=8)
    fig2.colorbar(im, ax=ax2[1], fraction=0.046)

    r = rate[np.isfinite(rate)]
    ax2[2].hist(r, bins=12, color=MARK, edgecolor="white")
    ax2[2].axvline(r.mean(), color=INK, lw=2,
                   label=f"mean {r.mean():.3f}%/read")
    ax2[2].set_xlabel("decay rate (%/read)")
    ax2[2].set_ylabel("cells")
    ax2[2].set_title(f"rate spread across 25 cells\n"
                     f"CV = {r.std()/r.mean()*100:.1f}%", fontsize=11)
    ax2[2].grid(color=GRID, lw=0.7)
    ax2[2].legend(fontsize=9, frameon=False)

    for a in ax2[:2]:
        a.set_xticks(range(5), [f"col{c+1}" for c in range(5)], fontsize=8)
        a.set_yticks(range(5), [f"row{r_+1}" for r_ in range(5)], fontsize=8)

    fig2.suptitle("Per-cell disturbance summary", fontsize=13)
    fig2.tight_layout(rect=[0, 0, 1, 0.92])
    out2 = "all25_disturb_summary.png"
    fig2.savefig(out2, dpi=110)
    print("saved ->", out2)

    print(f"\nrate: mean {r.mean():.3f} %/read, sd {r.std():.3f}, "
          f"min {r.min():.3f}, max {r.max():.3f}, CV {r.std()/r.mean()*100:.1f}%")
    print(f"asymptote: {np.nanmin(floor):.0f} .. {np.nanmax(floor):.0f} LSB")

    # ---- is the asymptote actually resolved, or still extrapolated? ----
    # A 300-read window gave asymptotes that drifted by up to 42 LSB as the
    # fit window grew, while each fit reported a +/-2 LSB CI -- the CI only
    # describes noise, not extrapolation error. The honest checks are how
    # far the decay actually got, and whether the estimate has stopped
    # moving.
    print(f"\n--- asymptote reliability ({n_reps} reps, {n_max} reads) ---")
    print(f"{'cell':>7} {'V_last':>8} {'Vinf':>8} {'decayed':>8} "
          f"{'drift K/2->K':>13} {'rep sd':>8}")
    completions, drifts = [], []
    for i in range(5):
        for j in range(1, 6):
            d = data.get((i, j))
            if not d:
                continue
            n = np.array(sorted(d), float)
            y = np.array([np.mean(d[k]) for k in sorted(d)])
            try:
                p_full, _ = curve_fit(one_exp, n, y, p0=[0, y[0], 0.99],
                                      maxfev=200000)
                m = n <= n_max / 2
                p_half, _ = curve_fit(one_exp, n[m], y[m],
                                      p0=[0, y[0], 0.99], maxfev=200000)
            except Exception:
                continue
            done = (y[0] - y[-1]) / (y[0] - p_full[0]) * 100
            drift = p_full[0] - p_half[0]
            completions.append(done)
            drifts.append(drift)
            # spread of the per-rep fitted asymptote
            reps_vinf = []
            for rp in range(n_reps):
                dd = per_rep.get(((i, j), rp))
                if not dd or len(dd) < 5:
                    continue
                nn = np.array(sorted(dd), float)
                yy = np.array([dd[k] for k in sorted(dd)])
                try:
                    pp, _ = curve_fit(one_exp, nn, yy, p0=[0, yy[0], 0.99],
                                      maxfev=200000)
                    reps_vinf.append(pp[0])
                except Exception:
                    pass
            sd = np.std(reps_vinf, ddof=1) if len(reps_vinf) > 1 else np.nan
            if (i, j) in [(0, 3), (0, 4), (1, 5), (3, 4)] or abs(drift) > 15:
                print(f"{f'({i+1},{j})':>7} {y[-1]:>8.1f} {p_full[0]:>8.1f} "
                      f"{done:>7.0f}% {drift:>+13.1f} {sd:>8.1f}")
    c = np.array(completions)
    dr = np.array(drifts)
    print(f"\ndecay completion: mean {c.mean():.0f}%, min {c.min():.0f}%")
    print(f"asymptote drift (half-window -> full): mean {dr.mean():+.1f} LSB, "
          f"max |{np.abs(dr).max():.1f}|")
    if np.abs(dr).max() < 5:
        print("  -> stable: the asymptote is resolved by this window")
    else:
        print("  -> STILL MOVING: the asymptote remains an extrapolation; "
              "quote the rate, not the floor")
    print("\nslowest / fastest cells:")
    flat = [((i + 1, j + 1), rate[i, j]) for i in range(5) for j in range(5)
            if np.isfinite(rate[i, j])]
    flat.sort(key=lambda x: x[1])
    for c, v in flat[:3]:
        print(f"  slowest {c}: {v:.3f} %/read")
    for c, v in flat[-3:]:
        print(f"  fastest {c}: {v:.3f} %/read")


if __name__ == "__main__":
    main()
