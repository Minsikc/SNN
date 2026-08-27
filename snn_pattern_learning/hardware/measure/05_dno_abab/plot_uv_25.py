#!/usr/bin/env python3
"""u,v grid correlation heatmap computed from all 25 devices.

Same view as plot_uv_15.py -- r(delta, C) per grid point, NORMAL vs DNO,
plus their difference -- but with no column filter. plot_uv_15.py excluded
columns 1-2 because column 2 was faulty on 2026-08-06; the 2026-08-20 session
shows all five columns healthy (per-cell r 0.90-0.99), so the full array is
used here.

    python plot_uv_25.py --normal <normal_cells.csv> --dno <dno_cells.csv>
"""
import argparse
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def grids(path):
    """Return (levels_u, levels_v, r_grid, gain_grid) over all cells."""
    obs = {}
    for x in csv.DictReader(open(path, encoding="utf-8")):
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--normal", required=True, help="NORMAL-mode *_cells.csv")
    ap.add_argument("--dno", required=True, help="DNO-mode *_cells.csv")
    ap.add_argument("--out", default="uv_grid_25cells_normal_vs_dno.png")
    args = ap.parse_args()

    sweeps = [("NORMAL", args.normal), ("DNO", args.dno)]
    data = [(lbl,) + grids(p) for lbl, p in sweeps]
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
                 "plane — all 25 devices (rows 1-5, cols 1-5)\n"
                 "r = Pearson(measured delta, commanded coincidences C); "
                 "BL=10, 5 repeats, reset per trial, identical pulse streams",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(args.out, dpi=110)
    print("saved ->", args.out)
    for lbl, _, _, R, G in data:
        print(f"{lbl:7s} r median {np.nanmedian(R):.4f}  "
              f"range {np.nanmin(R):.3f}-{np.nanmax(R):.3f}")


if __name__ == "__main__":
    main()
