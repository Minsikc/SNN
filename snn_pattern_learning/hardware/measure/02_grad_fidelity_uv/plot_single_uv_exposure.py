#!/usr/bin/env python3
"""Anatomy of one stochastic command: streams, coincidence, half-selects.

For a single (u, v) trial this shows what each of the 25 cells actually
received, decomposed into the three exposure types that a slot can produce:

    coincidence  n1[r]=1 and n2[c]=1   the intended update
    N1 alone     n1[r]=1, n2[c]=0      row line only
    N2 alone     n1[r]=0, n2[c]=1      column line only

The two half-select types are drawn in separate panels rather than summed,
because the pooled fit over 1125 cell-observations gives

    delta = +13.10*C  -0.72*(N1 alone)  +0.40*(N2 alone)  +5.85

i.e. they push the weight in OPPOSITE directions. Summing them hides that
and makes the half-select contribution look smaller than it is -- which is
part of why correcting for it barely moved the correlation (+0.003).

Panels are annotated with each term's LSB contribution using those fitted
coefficients, so the coincidence column and the half-select columns can be
compared on the same scale.
"""
import argparse
import csv
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N = 5
# fitted on 2026-08-07_16-43_uv_halfselect_corrected.csv
G_C, A_H1, A_H2 = 13.104, -0.721, 0.395

INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"


def cell_grid(ax, M, title, cmap, fmt="{:.0f}", vmin=None, vmax=None,
              cbar_label=None, fig=None):
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xticks(range(N), [f"c{c+1}" for c in range(N)], fontsize=8)
    ax.set_yticks(range(N), [f"r{r+1}" for r in range(N)], fontsize=8)
    span = (np.nanmax(M) - np.nanmin(M)) or 1
    for r in range(N):
        for c in range(N):
            v = M[r, c]
            light = (v - np.nanmin(M)) / span > 0.55
            ax.text(c, r, fmt.format(v), ha="center", va="center",
                    fontsize=8.5, color="white" if light else "black")
    if fig is not None:
        cb = fig.colorbar(im, ax=ax, fraction=0.046)
        if cbar_label:
            cb.set_label(cbar_label, fontsize=8)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--u", type=float, default=0.7)
    ap.add_argument("--v", type=float, default=0.5)
    ap.add_argument("--rep", type=int, default=0)
    args = ap.parse_args()

    path = args.csv or sorted(
        glob.glob("*_uv_halfselect_corrected.csv"))[-1]
    rows = [r for r in csv.DictReader(open(path, encoding="utf-8"))
            if float(r["u"]) == args.u and float(r["v"]) == args.v
            and int(r["rep"]) == args.rep]
    if not rows:
        raise SystemExit(f"no rows for u={args.u} v={args.v} rep={args.rep}")

    BL = len(rows[0]["n1"])
    n1 = np.zeros((N, BL), int)
    n2 = np.zeros((N, BL), int)
    C = np.zeros((N, N))
    H1 = np.zeros((N, N))
    H2 = np.zeros((N, N))
    D = np.zeros((N, N))
    for r in rows:
        i, j = int(r["cell_row"]) - 1, int(r["cell_col"]) - 1
        n1[i] = [int(x) for x in r["n1"]]
        n2[j] = [int(x) for x in r["n2"]]
        C[i, j] = float(r["C"])
        H1[i, j] = float(r["H1"])
        H2[i, j] = float(r["H2"])
        D[i, j] = float(r["delta"])

    fig = plt.figure(figsize=(17, 8.4))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.25], hspace=0.42,
                          wspace=0.32)

    # ---- top row: the pulse streams themselves ----
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(n1, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_title(f"N1 row streams  (u={args.u})", fontsize=10)
    ax.set_xlabel("slot")
    ax.set_yticks(range(N), [f"r{r+1}" for r in range(N)], fontsize=8)
    ax.set_xticks(range(BL), [str(i + 1) for i in range(BL)], fontsize=7)
    for r in range(N):
        for i in range(BL):
            ax.text(i, r, n1[r, i], ha="center", va="center", fontsize=7,
                    color="white" if n1[r, i] else MUTED)

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(n2, cmap="Oranges", vmin=0, vmax=1, aspect="auto")
    ax.set_title(f"N2 column streams  (v={args.v})", fontsize=10)
    ax.set_xlabel("slot")
    ax.set_yticks(range(N), [f"c{c+1}" for c in range(N)], fontsize=8)
    ax.set_xticks(range(BL), [str(i + 1) for i in range(BL)], fontsize=7)
    for c in range(N):
        for i in range(BL):
            ax.text(i, c, n2[c, i], ha="center", va="center", fontsize=7,
                    color="white" if n2[c, i] else MUTED)

    ax = fig.add_subplot(gs[0, 2])
    ax.axis("off")
    ax.text(0, 0.95, "per slot, cell (r,c) sees one of:", fontsize=10,
            va="top", color=INK)
    ax.text(0.03, 0.72, f"n1=1, n2=1  coincidence   x{G_C:+.2f} LSB",
            fontsize=9.5, va="top", color="#1c5cab")
    ax.text(0.03, 0.55, f"n1=1, n2=0  N1 alone      x{A_H1:+.2f} LSB",
            fontsize=9.5, va="top", color="#8a2f13")
    ax.text(0.03, 0.38, f"n1=0, n2=1  N2 alone      x{A_H2:+.2f} LSB",
            fontsize=9.5, va="top", color="#199e70")
    ax.text(0.03, 0.21, "n1=0, n2=0  nothing", fontsize=9.5, va="top",
            color=MUTED)
    ax.text(0, 0.04, "coefficients fitted over 1125 cell-observations",
            fontsize=8, va="top", color=MUTED)

    ax = fig.add_subplot(gs[0, 3])
    ax.axis("off")
    tot_c, tot_1, tot_2 = C.sum(), H1.sum(), H2.sum()
    ax.text(0, 0.95, "array totals for this command", fontsize=10, va="top",
            color=INK)
    ax.text(0.03, 0.74, f"coincidences   {tot_c:6.0f}  "
                        f"-> {G_C*tot_c:+8.0f} LSB", fontsize=9.5,
            va="top", color="#1c5cab", family="monospace")
    ax.text(0.03, 0.57, f"N1-alone       {tot_1:6.0f}  "
                        f"-> {A_H1*tot_1:+8.0f} LSB", fontsize=9.5,
            va="top", color="#8a2f13", family="monospace")
    ax.text(0.03, 0.40, f"N2-alone       {tot_2:6.0f}  "
                        f"-> {A_H2*tot_2:+8.0f} LSB", fontsize=9.5,
            va="top", color="#199e70", family="monospace")
    net_hs = A_H1 * tot_1 + A_H2 * tot_2
    ax.text(0.03, 0.20, f"half-select net {net_hs:+.0f} LSB, "
                        f"{abs(net_hs)/(G_C*tot_c)*100:.1f}% of coincidence",
            fontsize=9, va="top", color=INK)

    # ---- bottom row: the 5x5 exposure maps ----
    ax = fig.add_subplot(gs[1, 0])
    cell_grid(ax, C, "coincidences C  (raises weight)", "Blues",
              cbar_label="slots", fig=fig)
    ax = fig.add_subplot(gs[1, 1])
    cell_grid(ax, H1, f"N1-alone  ({A_H1:+.2f} LSB each -> LOWERS)",
              "Reds", cbar_label="slots", fig=fig)
    ax = fig.add_subplot(gs[1, 2])
    cell_grid(ax, H2, f"N2-alone  ({A_H2:+.2f} LSB each -> raises)",
              "Greens", cbar_label="slots", fig=fig)
    ax = fig.add_subplot(gs[1, 3])
    lim = np.nanmax(np.abs(D))
    cell_grid(ax, D, "measured delta (LSB)", "coolwarm",
              vmin=-lim, vmax=lim, cbar_label="LSB", fig=fig)

    rr = np.corrcoef(C.ravel(), D.ravel())[0, 1]
    fig.suptitle(f"One stochastic command, u={args.u} v={args.v} "
                 f"(rep {args.rep}, bit_length {BL}) — "
                 f"r(delta, C) = {rr:+.3f}\n"
                 "half-selects are split by direction: the row line lowers "
                 "the weight, the column line raises it",
                 fontsize=13)
    out = f"single_uv_exposure_u{args.u}_v{args.v}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print("saved ->", out)

    print(f"\ncell totals: C={C.sum():.0f}  H1={H1.sum():.0f}  "
          f"H2={H2.sum():.0f}  (25 cells x {BL} slots = {25*BL})")
    print(f"check: C+H1+H2+none = {C.sum()+H1.sum()+H2.sum():.0f} + "
          f"{25*BL-C.sum()-H1.sum()-H2.sum():.0f} idle")
    print(f"r(delta, C) = {rr:+.4f}")


if __name__ == "__main__":
    main()
