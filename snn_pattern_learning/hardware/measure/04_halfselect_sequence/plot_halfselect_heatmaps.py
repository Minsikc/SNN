#!/usr/bin/env python3
"""Per-cell heatmaps: half-select exposure by direction, next to the update.

Top row    the three classes of ACTIVE half-select sequence, counted per cell
           for one trial.  Class comes from the attractor sweep -- a pair is
           POT-driving / DEP-driving / reset-driving according to the level
           its five starting states collapse to.
Bottom row what the update asked for, what the array did, and the residual
           after removing the desired-update fit.

The point of putting them side by side: the residual map should track the net
half-select drive (POT count minus DEP count), and it does.
"""
import csv
import glob
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N = 5
INK, GRID, MUTED = "#0b0b0b", "#e1e0d9", "#898781"
STARTS = ["full_pot", "mid_pot", "reset", "mid_dep", "full_dep"]


def classify():
    rows = []
    for p in sorted(glob.glob("*_seq_attractor.csv")):
        rows += list(csv.DictReader(open(p, encoding="utf-8")))
    out = {}
    for c in sorted({r["combo"] for r in rows if r["combo"] != "control"}):
        sub = [r for r in rows if r["combo"] == c]
        mx = max(int(r["cycles"]) for r in sub)
        ends = [np.mean([float(r["after"]) for r in sub
                         if r["start"] == s and int(r["cycles"]) == mx])
                for s in STARTS]
        spread = max(ends) - min(ends)
        lvl = float(np.mean(ends))
        if spread >= 250:
            grp = None                       # inert
        elif lvl > 200:
            grp = "pot"
        elif lvl < -200:
            grp = "dep"
        else:
            grp = "reset"
        out[c] = dict(level=lvl, spread=spread, group=grp)
    return out


def grid(sub, key):
    M = np.zeros((N, N))
    for r in sub:
        M[int(r["cell_row"]) - 1, int(r["cell_col"]) - 1] = float(r[key])
    return M


def draw(ax, M, title, cmap, fig, cbl, fmt="{:+.0f}", diverging=True,
         vmax=None):
    if diverging:
        lim = vmax or (max(abs(M.min()), abs(M.max())) or 1)
        im = ax.imshow(M, cmap=cmap, vmin=-lim, vmax=lim)
        hot = lambda v: abs(v) > 0.6 * lim
    else:
        hi = vmax or (M.max() or 1)
        im = ax.imshow(M, cmap=cmap, vmin=0, vmax=hi)
        hot = lambda v: v > 0.6 * hi
    ax.set_title(title, fontsize=9.5)
    ax.set_xticks(range(N), [f"c{c+1}" for c in range(N)], fontsize=7)
    ax.set_yticks(range(N), [f"r{r+1}" for r in range(N)], fontsize=7)
    for i in range(N):
        for j in range(N):
            ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center",
                    fontsize=8, color="white" if hot(M[i, j]) else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.046)
    cb.set_label(cbl, fontsize=7)
    cb.ax.tick_params(labelsize=7)


def main():
    trial = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    rows = list(csv.DictReader(open("halfselect_seq_in_update.csv",
                                    encoding="utf-8")))
    cls = classify()
    pairs = {g: [k for k, d in cls.items() if d["group"] == g]
             for g in ("pot", "reset", "dep")}
    sub = [r for r in rows if int(r["trial"]) == trial]

    # global fit, so the residual means the same thing as in the pooled run
    des_all = np.array([float(r["desired"]) for r in rows])
    dl_all = np.array([float(r["delta"]) for r in rows])
    g, b = np.polyfit(des_all, dl_all, 1)

    HP, HR, HD = (grid(sub, "act_pot"), grid(sub, "act_reset"),
                  grid(sub, "act_dep"))
    D, A = grid(sub, "desired"), grid(sub, "delta")
    R = A - (g * D + b)
    cmax = max(HP.max(), HR.max(), HD.max(), 1)

    fig = plt.figure(figsize=(19.5, 9.6))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.05], wspace=0.35,
                          hspace=0.3)
    axes = np.array([[fig.add_subplot(gs[i, j]) for j in range(3)]
                     for i in range(2)])

    lv = lambda g_: ", ".join(f"{k} ({cls[k]['level']:+.0f})"
                              for k in sorted(pairs[g_],
                                              key=lambda x: -cls[x]["level"]))
    draw(axes[0, 0], HP,
         f"POT-driving half-selects\n{lv('pot')}\ntotal {HP.sum():.0f}",
         "Blues", fig, "count", "{:.0f}", diverging=False, vmax=cmax)
    draw(axes[0, 1], HR,
         f"RESET-driving half-selects\n{lv('reset')}\ntotal {HR.sum():.0f}",
         "Greys", fig, "count", "{:.0f}", diverging=False, vmax=cmax)
    draw(axes[0, 2], HD,
         f"DEP-driving half-selects\n{lv('dep')}\ntotal {HD.sum():.0f}",
         "Oranges", fig, "count", "{:.0f}", diverging=False, vmax=cmax)

    draw(axes[1, 0], D, "desired update (realized coincidences, signed)",
         "coolwarm", fig, "coincidences")
    draw(axes[1, 1], A,
         f"actual update measured\nr(desired, actual) = "
         f"{np.corrcoef(D.ravel(), A.ravel())[0,1]:+.3f}",
         "coolwarm", fig, "LSB")
    net = HP - HD
    draw(axes[1, 2], R,
         f"residual = actual - {g:.1f}x desired\n"
         f"r(residual, net POT-DEP drive) = "
         f"{np.corrcoef(R.ravel(), net.ravel())[0,1]:+.3f}",
         "PuOr", fig, "LSB")

    # ---- pooled evidence, so one trial is never over-read ---------------
    per = []
    for t in sorted({int(r["trial"]) for r in rows}):
        s = [r for r in rows if int(r["trial"]) == t]
        d_, a_ = grid(s, "desired"), grid(s, "delta")
        n_ = (grid(s, "act_pot") - grid(s, "act_dep")).ravel()
        R_ = (a_ - (g * d_ + b)).ravel()
        if n_.std() > 1e-9:
            per.append(np.corrcoef(R_, n_)[0, 1])
    per = np.array(per)
    this_r = np.corrcoef(R.ravel(), net.ravel())[0, 1]

    ax = fig.add_subplot(gs[:, 3])
    ax.hist(per, bins=12, color="#7fa8d9", edgecolor="white")
    ax.axvline(0, color=MUTED, lw=1)
    ax.axvline(per.mean(), color=INK, lw=2,
               label=f"mean {per.mean():+.2f} $\\pm$ "
                     f"{per.std()/np.sqrt(len(per)):.2f}")
    ax.axvline(this_r, color="#eb6834", lw=2, ls="--",
               label=f"trial {trial}: {this_r:+.2f}")
    ax.set_xlabel("r(residual, net POT$-$DEP drive)")
    ax.set_ylabel("trials")
    ax.set_title(f"per-trial correlation, all {len(per)} trials\n"
                 f"25 cells per trial gives sd {per.std():.2f}, so a single\n"
                 f"trial says little; the mean is {per.mean()/(per.std()/np.sqrt(len(per))):.1f}"
                 f"$\\sigma$ from zero", fontsize=9.5)
    ax.legend(fontsize=8.5)
    ax.grid(color=GRID, lw=0.8, axis="y")

    fig.suptitle(f"Half-select exposure by class (top) vs the update itself "
                 f"(bottom) — trial {trial}, representative of the {len(per)} "
                 f"trials", fontsize=13)
    fig.savefig("halfselect_heatmaps.png", dpi=110, bbox_inches="tight")
    print("saved -> halfselect_heatmaps.png")
    print(f"trial {trial}: POT-pairs {HP.sum():.0f}, reset {HR.sum():.0f}, "
          f"DEP {HD.sum():.0f}")
    print(f"r(residual, net drive) trial {trial} = {this_r:+.3f}")
    print(f"per-trial mean {per.mean():+.3f} +- {per.std()/np.sqrt(len(per)):.3f}"
          f"  (sd {per.std():.3f}, negative in {(per<0).sum()}/{len(per)})")


if __name__ == "__main__":
    main()
