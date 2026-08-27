#!/usr/bin/env python3
"""u,v grid correlation heatmap computed from the 15 devices in columns 3-5.

Same view as the sweep's primary panel -- r(delta, C) as a function of drive
level -- but each grid point's r is recomputed from only the cells in columns
3-5 (the other 10 include column 2, which is faulty and dominates the array
average).
"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SWEEPS = [("NORMAL", "2026-08-06_14-40_uv_grid_cells.csv"),
          ("DNO",    "2026-08-06_14-39_uv_grid_dno_cells.csv")]
COLS = {3, 4, 5}


def grids(path):
    """Return (levels, r_grid, gain_grid, n_grid) using only COLS."""
    obs = {}
    for x in csv.DictReader(open(path, encoding="utf-8")):
        if int(x["cell_col"]) not in COLS:
            continue
        key = (float(x["u"]), float(x["v"]))
        obs.setdefault(key, []).append((float(x["C"]), float(x["delta"])))

    us = sorted({k[0] for k in obs})
    vs = sorted({k[1] for k in obs})
    R = np.full((len(us), len(vs)), np.nan)
    G = np.full((len(us), len(vs)), np.nan)
    for i, u in enumerate(us):
        for j, v in enumerate(vs):
            pts = obs.get((u, v), [])
            if len(pts) < 3:
                continue
            C = np.array([a for a, _ in pts])
            d = np.array([b for _, b in pts])
            if C.std() > 1e-9 and d.std() > 1e-9:
                R[i, j] = np.corrcoef(C, d)[0, 1]
                G[i, j] = np.polyfit(C, d, 1)[0]
    return us, vs, R, G


def panel(ax, us, vs, M, title, cmap, vmin, vmax, fmt="{:.3f}"):
    im = ax.imshow(M, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect="equal")
    ax.set_title(title, fontsize=11)
    ax.set_xticks(range(len(vs)), [f"{v:g}" for v in vs])
    ax.set_yticks(range(len(us)), [f"{u:g}" for u in us])
    ax.set_xlabel("v  (column drive)")
    ax.set_ylabel("u  (row drive)")
    for i in range(len(us)):
        for j in range(len(vs)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center",
                        fontsize=10)
    return im


def main():
    data = [(lbl,) + grids(p) for lbl, p in SWEEPS]
    us, vs = data[0][1], data[0][2]
    Rs = [d[3] for d in data]
    lo = float(np.nanmin(Rs)) - 0.005
    hi = float(np.nanmax(Rs)) + 0.005

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    for k, (lbl, _, _, R, G) in enumerate(data):
        im = panel(axes[k], us, vs, R,
                   f"{lbl}\nr(delta, C)   median = {np.nanmedian(R):.3f}",
                   "RdYlGn", lo, hi)
        fig.colorbar(im, ax=axes[k], fraction=0.046)

    D = Rs[1] - Rs[0]
    lim = float(np.nanmax(np.abs(D)))
    im = panel(axes[2], us, vs, D,
               f"DNO - NORMAL\nmean {np.nanmean(D):+.4f}, "
               f"DNO better in {(D > 0).sum()}/{np.isfinite(D).sum()} points",
               "coolwarm", -lim, lim, fmt="{:+.3f}")
    fig.colorbar(im, ax=axes[2], fraction=0.046)

    fig.suptitle("Desired vs actual update correlation across the (u, v) drive "
                 "plane — 15 devices (rows 1-5, cols 3-5)\n"
                 "r = Pearson(measured delta, commanded coincidences C); "
                 "BL=10, 5 repeats, reset per trial, identical pulse streams",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out = "uv_grid_15cells_normal_vs_dno.png"
    fig.savefig(out, dpi=110)
    print("saved ->", out)
    for lbl, _, _, R, G in data:
        print(f"{lbl:7s} r median {np.nanmedian(R):.4f}  "
              f"range {np.nanmin(R):.3f}-{np.nanmax(R):.3f}")


if __name__ == "__main__":
    main()
