#!/usr/bin/env python3
"""Per-cell correlation heatmap for the 15 devices in columns 3-5.

Same layout as the array plots, just cropped to the requested cell range.
r is computed per cell from the raw (C, delta) observations pooled over all
grid points and repeats in the *_cells.csv files.
"""
import csv
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DNO = "2026-08-06_14-39_uv_grid_dno_cells.csv"
NORMAL = "2026-08-06_14-40_uv_grid_cells.csv"
COLS = [3, 4, 5]
ROWS = [1, 2, 3, 4, 5]


def per_cell_r(path):
    obs = {}
    for x in csv.DictReader(open(path, encoding="utf-8")):
        key = (int(x["cell_row"]), int(x["cell_col"]))
        obs.setdefault(key, []).append((float(x["C"]), float(x["delta"])))
    R = np.full((len(ROWS), len(COLS)), np.nan)
    for i, ri in enumerate(ROWS):
        for j, cj in enumerate(COLS):
            v = obs.get((ri, cj), [])
            if len(v) < 3:
                continue
            C = np.array([a for a, _ in v])
            d = np.array([b for _, b in v])
            if C.std() > 1e-9 and d.std() > 1e-9:
                R[i, j] = np.corrcoef(C, d)[0, 1]
    return R


def panel(ax, M, title, cmap, vmin, vmax, fmt="{:.3f}"):
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_title(title, fontsize=11)
    ax.set_xticks(range(len(COLS)), [f"col {c}" for c in COLS])
    ax.set_yticks(range(len(ROWS)), [f"row {r}" for r in ROWS])
    for i in range(len(ROWS)):
        for j in range(len(COLS)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center",
                        fontsize=9)
    return im


def main():
    rn = per_cell_r(NORMAL)
    rd = per_cell_r(DNO)
    diff = rd - rn

    fig, axes = plt.subplots(1, 3, figsize=(13, 5.2))
    lo = float(np.nanmin([rn, rd])) - 0.005
    hi = float(np.nanmax([rn, rd])) + 0.005

    im0 = panel(axes[0], rn, f"NORMAL\nmedian r = {np.nanmedian(rn):.3f}",
                "viridis", lo, hi)
    fig.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = panel(axes[1], rd, f"DNO\nmedian r = {np.nanmedian(rd):.3f}",
                "viridis", lo, hi)
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    lim = float(np.nanmax(np.abs(diff)))
    im2 = panel(axes[2], diff,
                f"DNO - NORMAL\nmean {np.nanmean(diff):+.4f}, "
                f"DNO better in {(diff > 0).sum()}/{np.isfinite(diff).sum()}",
                "coolwarm", -lim, lim, fmt="{:+.3f}")
    fig.colorbar(im2, ax=axes[2], fraction=0.046)

    fig.suptitle("Per-cell update fidelity r(delta, C) — 15 devices, columns 3-5\n"
                 "BL=10, u,v in {0.3,0.5,0.7}, 5 repeats, reset per trial, "
                 "identical pulse streams",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = "cells15_r_normal_vs_dno.png"
    fig.savefig(out, dpi=110)
    print("saved ->", out)
    print(f"normal median {np.nanmedian(rn):.4f} | DNO median {np.nanmedian(rd):.4f}")


if __name__ == "__main__":
    main()
